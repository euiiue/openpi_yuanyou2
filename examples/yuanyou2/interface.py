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

from __future__ import annotations

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

    def __init__(
        self,
        node_name: str = "yuanyou2_openpi_interface",
        image_topics: dict[str, str] | None = None,
    ):
        if not rospy.core.is_initialized():
            rospy.init_node(node_name, anonymous=True)

        self.logger = logging.getLogger(__name__)
        self.cv_bridge = CvBridge()
        self.lock = Lock()

        self.latest_images: Dict[str, np.ndarray] = {}
        self.latest_joint_state: Optional[JointState] = None

        self.image_timestamps: Dict[str, float] = {}
        self.joint_timestamp: Optional[float] = None

        self._warn_if_system_time_unset()

        default_image_topics = {
            "head": "/head_camera/usb_cam/image_raw",
            "left_wrist": "/left_wrist_d435/color/image_raw",
            "right_wrist": "/right_wrist_d435/color/image_raw",
        }
        configured_image_topics = dict(default_image_topics)
        if image_topics:
            configured_image_topics.update(image_topics)

        self.image_topics = {
            "head": rospy.get_param("~head_image_topic", configured_image_topics["head"]),
            "left_wrist": rospy.get_param("~left_wrist_image_topic", configured_image_topics["left_wrist"]),
            "right_wrist": rospy.get_param("~right_wrist_image_topic", configured_image_topics["right_wrist"]),
        }
        self.joint_state_topic = rospy.get_param("~joint_state_topic", "/joint_states")
        self.left_cmd_topic = rospy.get_param("~left_cmd_topic", "/left/joint_cmd")
        self.right_cmd_topic = rospy.get_param("~right_cmd_topic", "/right/joint_cmd")

        self.use_gripper = bool(rospy.get_param("~use_gripper", True))
        self.clamp_gripper = bool(rospy.get_param("~clamp_gripper", True))
        self.gripper_min = float(rospy.get_param("~gripper_min", 0.0))
        self.gripper_max = float(rospy.get_param("~gripper_max", 0.035))
        self.arm_velocity_default = float(rospy.get_param("~arm_velocity_default", 0.0))
        self.arm_effort_default = float(rospy.get_param("~arm_effort_default", 0.0))
        self.gripper_velocity_default = float(rospy.get_param("~gripper_velocity_default", 10.0))
        self.gripper_effort_default = float(rospy.get_param("~gripper_effort_default", 0.5))

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

        # The URDF exposes joint8 as the coupled opposite finger; command joint7 only.
        self.left_gripper_joint = "left_joint7"
        self.right_gripper_joint = "right_joint7"

        self._setup_subscribers()
        self._setup_publishers()

        self.logger.info("Yuanyou2ROS1Interface initialized.")
        self.logger.info("image_topics=%s", self.image_topics)
        self.logger.info("joint_state_topic=%s", self.joint_state_topic)
        self.logger.info("left_cmd_topic=%s right_cmd_topic=%s", self.left_cmd_topic, self.right_cmd_topic)

    def _warn_if_system_time_unset(self):
        # Jan 1 2020. A Jetson stuck near 1970 makes bag filenames and debugging painful.
        if time.time() < 1577836800:
            self.logger.warning(
                "System clock appears unset. Fix NTP/date before collecting serious rosbag data."
            )

    def _setup_subscribers(self):
        for camera_name, topic in self.image_topics.items():
            rospy.Subscriber(
                topic,
                Image,
                lambda msg, name=camera_name: self._image_callback(msg, name),
                queue_size=1,
            )

        rospy.Subscriber(
            self.joint_state_topic,
            JointState,
            self._joint_state_callback,
            queue_size=1,
        )

        self.logger.info("ROS1 subscribers initialized.")

    def _setup_publishers(self):
        self.left_cmd_pub = rospy.Publisher(
            self.left_cmd_topic,
            JointState,
            queue_size=1,
        )

        self.right_cmd_pub = rospy.Publisher(
            self.right_cmd_topic,
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
        self._log_published_topic_hint()
        return False

    def _log_published_topic_hint(self):
        try:
            published_topics = {name for name, _type in rospy.get_published_topics()}
        except Exception as exc:
            self.logger.warning("Could not query published topics: %s", exc)
            return

        expected_topics = [self.joint_state_topic, *self.image_topics.values()]
        missing_topics = [topic for topic in expected_topics if topic not in published_topics]
        if missing_topics:
            self.logger.error("Expected topics not currently published: %s", missing_topics)

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

        if self.clamp_gripper:
            left_gripper_cmd = float(np.clip(left_gripper_cmd, self.gripper_min, self.gripper_max))
            right_gripper_cmd = float(np.clip(right_gripper_cmd, self.gripper_min, self.gripper_max))

        left_msg = self._build_cmd_msg(
            now,
            self.left_arm_joints,
            self.left_gripper_joint,
            left_arm_cmd,
            left_gripper_cmd,
        )
        right_msg = self._build_cmd_msg(
            now,
            self.right_arm_joints,
            self.right_gripper_joint,
            right_arm_cmd,
            right_gripper_cmd,
        )

        self.left_cmd_pub.publish(left_msg)
        self.right_cmd_pub.publish(right_msg)

        self.logger.debug(
            f"Published Yuanyou2 action: left_gripper={left_gripper_cmd}, right_gripper={right_gripper_cmd}"
        )

    def _build_cmd_msg(
        self,
        stamp: rospy.Time,
        arm_joints: list[str],
        gripper_joint: str,
        arm_positions: np.ndarray,
        gripper_position: float,
    ) -> JointState:
        msg = JointState()
        msg.header.stamp = stamp
        msg.name = list(arm_joints)
        msg.position = list(np.asarray(arm_positions, dtype=np.float32))
        msg.velocity = [self.arm_velocity_default] * len(arm_joints)
        msg.effort = [self.arm_effort_default] * len(arm_joints)

        if self.use_gripper:
            msg.name.append(gripper_joint)
            msg.position.append(float(gripper_position))
            msg.velocity.append(self.gripper_velocity_default)
            msg.effort.append(self.gripper_effort_default)

        return msg
