import argparse
import shutil
from pathlib import Path
from bisect import bisect_left

import cv2
import numpy as np
import rosbag
from cv_bridge import CvBridge

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


IMAGE_TOPICS = {
    "head": "/head_camera/usb_cam/image_raw",
    "left_wrist": "/left_wrist_d435/color/image_raw",
    "right_wrist": "/right_wrist_d435/color/image_raw",
}

JOINT_STATE_TOPIC = "/joint_states"

# 先按你的 14 维 Yuanyou2 结构定义。
# gripper 到底用 joint7 还是 joint8，需要你最终实机确认。
LEFT_ARM_JOINTS = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
]

RIGHT_ARM_JOINTS = [
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
]

LEFT_GRIPPER_JOINT = "left_joint7"
RIGHT_GRIPPER_JOINT = "right_joint7"


def get_stamp_sec(msg, fallback_time):
    if hasattr(msg, "header") and msg.header.stamp:
        return msg.header.stamp.to_sec()
    return fallback_time.to_sec()


def resize_rgb(image, size=(224, 224)):
    image = cv2.resize(image, size)
    return image.astype(np.uint8)


def nearest_item(items, target_time):
    """
    items: list of (timestamp, data), sorted by timestamp
    """
    if not items:
        return None

    times = [x[0] for x in items]
    idx = bisect_left(times, target_time)

    if idx <= 0:
        return items[0][1]
    if idx >= len(items):
        return items[-1][1]

    before = items[idx - 1]
    after = items[idx]

    if abs(before[0] - target_time) <= abs(after[0] - target_time):
        return before[1]
    return after[1]


def extract_joint_position(msg, joint_name):
    if joint_name not in msg.name:
        raise ValueError(f"Joint {joint_name} not found in /joint_states. Names: {msg.name}")

    idx = msg.name.index(joint_name)

    if len(msg.position) <= idx:
        raise ValueError(f"Joint {joint_name} has no position value.")

    return float(msg.position[idx])


def extract_state_14d(joint_msg):
    left_arm = [extract_joint_position(joint_msg, name) for name in LEFT_ARM_JOINTS]
    left_gripper = extract_joint_position(joint_msg, LEFT_GRIPPER_JOINT)

    right_arm = [extract_joint_position(joint_msg, name) for name in RIGHT_ARM_JOINTS]
    right_gripper = extract_joint_position(joint_msg, RIGHT_GRIPPER_JOINT)

    state = np.array(
        left_arm + [left_gripper] + right_arm + [right_gripper],
        dtype=np.float32,
    )

    if state.shape != (14,):
        raise ValueError(f"Expected state shape (14,), got {state.shape}")

    return state


def read_ros1_bag(bag_path):
    bridge = CvBridge()

    image_buffers = {
        "head": [],
        "left_wrist": [],
        "right_wrist": [],
    }
    joint_states = []

    topics = list(IMAGE_TOPICS.values()) + [JOINT_STATE_TOPIC]

    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            timestamp = get_stamp_sec(msg, t)

            if topic == JOINT_STATE_TOPIC:
                joint_states.append((timestamp, msg))

            elif topic in IMAGE_TOPICS.values():
                camera_key = None
                for key, ros_topic in IMAGE_TOPICS.items():
                    if topic == ros_topic:
                        camera_key = key
                        break

                if camera_key is None:
                    continue

                # ROS image 转 OpenCV
                # 如果你的图像本来是 rgb8，可以直接用 rgb8；
                # 如果显示颜色反了，可以改成 bgr8 后再 cvtColor。
                image = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
                image = resize_rgb(image, size=(224, 224))

                image_buffers[camera_key].append((timestamp, image))

    for key in image_buffers:
        image_buffers[key].sort(key=lambda x: x[0])

    joint_states.sort(key=lambda x: x[0])

    return image_buffers, joint_states


def create_lerobot_dataset(repo_id, fps):
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="yuanyou2",
        fps=fps,
        features={
            "observation.images.head": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_wrist": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_wrist": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    return dataset


def convert_bag_to_episode(dataset, bag_path, fps, task):
    image_buffers, joint_states = read_ros1_bag(bag_path)

    if not joint_states:
        raise RuntimeError(f"No /joint_states found in {bag_path}")

    start_time = joint_states[0][0]
    end_time = joint_states[-1][0]
    dt = 1.0 / fps

    current_time = start_time

    while current_time < end_time - dt:
        joint_msg = nearest_item(joint_states, current_time)
        next_joint_msg = nearest_item(joint_states, current_time + dt)

        if joint_msg is None or next_joint_msg is None:
            current_time += dt
            continue

        head = nearest_item(image_buffers["head"], current_time)
        left_wrist = nearest_item(image_buffers["left_wrist"], current_time)
        right_wrist = nearest_item(image_buffers["right_wrist"], current_time)

        if head is None or left_wrist is None or right_wrist is None:
            current_time += dt
            continue

        state_14d = extract_state_14d(joint_msg)

        # 第一版先用下一帧状态作为 action。
        # 后面如果你录到了真实 command topic，可以改成 command_14d。
        action_14d = extract_state_14d(next_joint_msg)

        dataset.add_frame(
            {
                "observation.images.head": head,
                "observation.images.left_wrist": left_wrist,
                "observation.images.right_wrist": right_wrist,
                "observation.state": state_14d,
                "action": action_14d,
                "task": task,
            }
        )

        current_time += dt

    dataset.save_episode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag_dir", type=str)
    parser.add_argument("--output", "-o", type=str, default="eeason396/yuanyou2_test")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--task", type=str, default="pick a cube and place it on another cube")
    args = parser.parse_args()

    bag_dir = Path(args.bag_dir)
    bag_files = sorted(bag_dir.glob("*.bag"))

    if not bag_files:
        raise RuntimeError(f"No .bag files found in {bag_dir}")

    dataset = create_lerobot_dataset(args.output, args.fps)

    for bag_path in bag_files:
        print(f"Converting {bag_path}")
        convert_bag_to_episode(dataset, str(bag_path), args.fps, args.task)

    print(f"Done. Dataset saved to {args.output}")


if __name__ == "__main__":
    main()