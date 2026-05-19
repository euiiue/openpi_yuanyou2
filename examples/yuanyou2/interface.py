#!/usr/bin/env python3
"""
ROS1 interface wrapper for Yuanyou2 robot.

Responsibilities:
1. Subscribe to three camera topics.
2. Subscribe to /joint_states.
3. Build 14-dim Yuanyou2 state:
   [left_arm_6, left_gripper, right_arm_6, right_gripper]
4. Publish 14-dim Yuanyou2 action to real robot command topics.
"""

import logging
import time
from threading import Lock
from typing import Dict, Optional

import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState


class Yuanyou2ROS1Interface:
    """Thread-safe ROS1 interface for Yuanyou2 dual-arm robot."""

    def __init__(self, node_name: str = "yuanyou2_openpi_interface"):
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        self.logger = logging.getLogger(__name__)
        self.cv_bridge = CvBridge()
        self.lock = Lock()

        self.latest_images: Dict[str, np.ndarray] = {}
        self.latest_joint_state: Optional[JointState] = None

        self.image_timestamps: Dict[str, float] = {}
        self.joint_timestamp: Optional[float] = None

        # Camera topics: modify only if your real topic names change.
        self.image_topics = {
            "head": "/head_camera/usb_cam/image_raw",
            "left_wrist": "/left_wrist_d435/color/image_raw",
            "right_wrist": "/right_wrist_d435/color/image_raw",
        }

        # Yuanyou2 14-dim layout:
        # [left_arm_6, left_gripper, right_arm_6, right_gripper]
        self.left_arm_joints = [
            "left_joint1",
            "left_joint2",
            "left_joint3",
            "left_joint4",
            "left_joint5",
            "left_joint6",
        ]

        self.right_arm_joints = [
            "right_joint1",
            "right_joint2",
            "right_joint3",
            "right_joint4",
            "right_joint5",
            "right_joint6",
        ]

        # You need to confirm whether joint7 or joint8 is the better gripper state.
        self.left_gripper_joint = "left_joint7"
        self.right_gripper_joint = "right_joint7"

        self._setup_subscribers()
        self._setup_publishers()

        self.logger.info("Yuanyou2ROS1Interface initialized.")

    def _setup_subscribers(self):
        rospy.Subscriber(
            self.image_topics["head"],
            Image,
            lambda msg: self._image_callback(msg, "head"),
            queue_size=1,
        )

        rospy.Subscriber(
            self.image_topics["left_wrist"],
            Image,
            lambda msg: self._image_callback(msg, "left_wrist"),
            queue_size=1,
        )

        rospy.Subscriber(
            self.image_topics["right_wrist"],
            Image,
            lambda msg: self._image_callback(msg, "right_wrist"),
            queue_size=1,
        )

        rospy.Subscriber(
            "/joint_states",
            JointState,
            self._joint_state_callback,
            queue_size=1,
        )

        self.logger.info("ROS1 subscribers initialized.")

    def _setup_publishers(self):
        # Confirm actual message type with:
        # rostopic type /left/joint_cmd
        # rostopic type /right/joint_cmd
        self.left_cmd_pub = rospy.Publisher(
            "/left/joint_cmd",
            JointState,
            queue_size=1,
        )

        self.right_cmd_pub = rospy.Publisher(
            "/right/joint_cmd",
            JointState,
            queue_size=1,
        )

        self.logger.info("ROS1 publishers initialized.")

    def _image_callback(self, msg: Image, camera_name: str):
        try:
            image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

            with self.lock:
                self.latest_images[camera_name] = image
                self.image_timestamps[camera_name] = time.time()

        except Exception as exc:
            self.logger.error(f"Failed to process {camera_name} image: {exc}")

    def _joint_state_callback(self, msg: JointState):
        with self.lock:
            self.latest_joint_state = msg
            self.joint_timestamp = time.time()

    def wait_for_initial_data(self, timeout: float = 10.0) -> bool:
        required_images = ["head", "left_wrist", "right_wrist"]

        start_time = time.time()

        while time.time() - start_time < timeout:
            with self.lock:
                images_ready = all(k in self.latest_images for k in required_images)
                joints_ready = self.latest_joint_state is not None

                if images_ready and joints_ready:
                    self.logger.info("All Yuanyou2 sensor data received.")
                    return True

            time.sleep(0.1)

        with self.lock:
            missing_images = [k for k in required_images if k not in self.latest_images]
            joints_ready = self.latest_joint_state is not None

        self.logger.error(
            f"Timeout waiting for sensor data. "
            f"missing_images={missing_images}, joints_ready={joints_ready}"
        )
        return False

    def get_observation(self) -> Optional[Dict]:
        with self.lock:
            if self.latest_joint_state is None:
                return None

            required_images = ["head", "left_wrist", "right_wrist"]
            if not all(k in self.latest_images for k in required_images):
                return None

            state_14d = self._extract_state_14d(self.latest_joint_state)

            return {
                "images": {
                    "head": self.latest_images["head"].copy(),
                    "left_wrist": self.latest_images["left_wrist"].copy(),
                    "right_wrist": self.latest_images["right_wrist"].copy(),
                },
                "state": state_14d,
            }

    def _extract_joint_position(self, msg: JointState, joint_name: str) -> float:
        if joint_name not in msg.name:
            raise ValueError(
                f"Joint {joint_name} not found in /joint_states. "
                f"Available joints: {list(msg.name)}"
            )

        idx = msg.name.index(joint_name)

        if idx >= len(msg.position):
            raise ValueError(f"Joint {joint_name} has no position value.")

        return float(msg.position[idx])

    def _extract_state_14d(self, msg: JointState) -> np.ndarray:
        left_arm = [
            self._extract_joint_position(msg, name)
            for name in self.left_arm_joints
        ]

        left_gripper = self._extract_joint_position(
            msg,
            self.left_gripper_joint,
        )

        right_arm = [
            self._extract_joint_position(msg, name)
            for name in self.right_arm_joints
        ]

        right_gripper = self._extract_joint_position(
            msg,
            self.right_gripper_joint,
        )

        state = np.asarray(
            left_arm + [left_gripper] + right_arm + [right_gripper],
            dtype=np.float32,
        )

        if state.shape != (14,):
            raise ValueError(f"Expected Yuanyou2 state shape (14,), got {state.shape}")

        return state

    def publish_action(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.float32)

        if actions.shape != (14,):
            self.logger.error(f"Expected 14-dim action, got shape {actions.shape}")
            return

        left_arm_cmd = actions[0:6]
        left_gripper_cmd = float(actions[6])
        right_arm_cmd = actions[7:13]
        right_gripper_cmd = float(actions[13])

        now = rospy.Time.now()

        left_msg = JointState()
        left_msg.header.stamp = now
        left_msg.name = self.left_arm_joints
        left_msg.position = left_arm_cmd.tolist()
        left_msg.velocity = [0.0] * 6
        left_msg.effort = [0.0] * 6

        right_msg = JointState()
        right_msg.header.stamp = now
        right_msg.name = self.right_arm_joints
        right_msg.position = right_arm_cmd.tolist()
        right_msg.velocity = [0.0] * 6
        right_msg.effort = [0.0] * 6

        self.left_cmd_pub.publish(left_msg)
        self.right_cmd_pub.publish(right_msg)

        # Gripper handling:
        # If your Piper gripper is controlled through joint_cmd joint7/joint8,
        # you should integrate gripper command into the correct topic/message.
        # Do not ignore this in final deployment.
        self.logger.debug(
            f"Gripper commands not yet published separately: "
            f"left={left_gripper_cmd}, right={right_gripper_cmd}"
        )