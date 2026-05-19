#!/usr/bin/env python3
"""Preflight checks before Yuanyou2 data collection or policy rollout."""

import argparse
import sys
import time

import rospy
from sensor_msgs.msg import CameraInfo, Image, JointState


IMAGE_TOPICS = {
    "head": "/head_camera/usb_cam/image_raw",
    "left_wrist": "/left_wrist_d435/color/image_raw",
    "right_wrist": "/right_wrist_d435/color/image_raw",
}

CAMERA_INFO_TOPICS = {
    "head": "/head_camera/usb_cam/camera_info",
    "left_wrist": "/left_wrist_d435/color/camera_info",
    "right_wrist": "/right_wrist_d435/color/camera_info",
}

REQUIRED_JOINTS = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_joint7",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_joint7",
]


def wait_msg(topic, msg_type, timeout):
    try:
        return rospy.wait_for_message(topic, msg_type, timeout=timeout)
    except Exception as exc:
        raise RuntimeError(f"{topic}: {exc}") from exc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--joint-state-topic", default="/joint_states")
    parser.add_argument("--head-image-topic", default=IMAGE_TOPICS["head"])
    parser.add_argument("--left-wrist-image-topic", default=IMAGE_TOPICS["left_wrist"])
    parser.add_argument("--right-wrist-image-topic", default=IMAGE_TOPICS["right_wrist"])
    parser.add_argument("--head-camera-info-topic", default=CAMERA_INFO_TOPICS["head"])
    parser.add_argument("--left-wrist-camera-info-topic", default=CAMERA_INFO_TOPICS["left_wrist"])
    parser.add_argument("--right-wrist-camera-info-topic", default=CAMERA_INFO_TOPICS["right_wrist"])
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("yuanyou2_openpi_preflight", anonymous=True)

    failures = []

    if time.time() < 1577836800:
        failures.append("System clock appears unset; fix Jetson date/NTP before serious bag collection.")
        print("[WARN] System clock appears unset.")

    try:
        joint_msg = wait_msg(args.joint_state_topic, JointState, args.timeout)
        missing = [joint for joint in REQUIRED_JOINTS if joint not in joint_msg.name]
        if missing:
            raise RuntimeError(f"missing joints: {missing}; available={list(joint_msg.name)}")
        print(f"[OK] {args.joint_state_topic}: {len(joint_msg.name)} joints")
    except Exception as exc:
        failures.append(str(exc))
        print(f"[FAIL] {args.joint_state_topic}: {exc}")

    image_topics = {
        "head": args.head_image_topic,
        "left_wrist": args.left_wrist_image_topic,
        "right_wrist": args.right_wrist_image_topic,
    }
    camera_info_topics = {
        "head": args.head_camera_info_topic,
        "left_wrist": args.left_wrist_camera_info_topic,
        "right_wrist": args.right_wrist_camera_info_topic,
    }

    for name, topic in image_topics.items():
        try:
            msg = wait_msg(topic, Image, args.timeout)
            print(f"[OK] {name} image {topic}: {msg.width}x{msg.height} frame={msg.header.frame_id}")
        except Exception as exc:
            failures.append(str(exc))
            print(f"[FAIL] {name} image {topic}: {exc}")

    for name, topic in camera_info_topics.items():
        try:
            msg = wait_msg(topic, CameraInfo, args.timeout)
            print(f"[OK] {name} camera_info {topic}: frame={msg.header.frame_id}")
        except Exception as exc:
            failures.append(str(exc))
            print(f"[FAIL] {name} camera_info {topic}: {exc}")

    if failures:
        print(f"\n[FAIL] Yuanyou2 OpenPI preflight found {len(failures)} issue(s).")
        sys.exit(1)

    print("\n[PASS] Yuanyou2 OpenPI ROS topics look ready.")


if __name__ == "__main__":
    main()
