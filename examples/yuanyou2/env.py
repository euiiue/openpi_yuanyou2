#!/usr/bin/env python3
"""
Environment wrapper for Yuanyou2 robot using openpi_client.runtime framework.
ROS1 version.
"""

import logging
from typing import Optional

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.yuanyou2 import ros1_interface


class Yuanyou2Environment(_environment.Environment):
    """Environment for Yuanyou2 dual-arm robot."""

    def __init__(
        self,
        prompt: str = "pick a cube and place it on another cube",
    ):
        self._prompt = prompt

        self._ros_interface: Optional[ros1_interface.Yuanyou2ROS1Interface] = (
            ros1_interface.Yuanyou2ROS1Interface()
        )

        if not self._ros_interface.wait_for_initial_data(timeout=30.0):
            raise RuntimeError(
                "Failed to receive initial sensor data. "
                "Please check ROS1 topics:\n"
                "  rostopic list\n"
                "  rostopic hz /joint_states\n"
                "  rostopic hz /head_camera/usb_cam/image_raw\n"
                "  rostopic hz /left_wrist_d435/color/image_raw\n"
                "  rostopic hz /right_wrist_d435/color/image_raw"
            )

        logging.info(f"Yuanyou2Environment initialized with prompt: '{prompt}'")

    @override
    def reset(self) -> None:
        logging.info("Environment reset called. No-op for Yuanyou2.")

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        if self._ros_interface is None:
            raise RuntimeError("ROS1 interface not initialized")

        raw_obs = self._ros_interface.get_observation()
        if raw_obs is None:
            raise RuntimeError("Failed to get observation from ROS1 interface")

        return {
            "observation/state": raw_obs["state"],
            "observation/images/head": image_tools.convert_to_uint8(raw_obs["images"]["head"]),
            "observation/images/left_wrist": image_tools.convert_to_uint8(raw_obs["images"]["left_wrist"]),
            "observation/images/right_wrist": image_tools.convert_to_uint8(raw_obs["images"]["right_wrist"]),
            "prompt": self._prompt,
        }

    @override
    def apply_action(self, action: dict) -> None:
        if self._ros_interface is None:
            raise RuntimeError("ROS1 interface not initialized")

        if "actions" not in action:
            raise ValueError(f"Action dict must contain 'actions' key, got: {action.keys()}")

        actions = action["actions"]

        if not isinstance(actions, np.ndarray):
            actions = np.asarray(actions, dtype=np.float32)

        # Policy may return [action_horizon, 14].
        # Runtime execution uses the first action.
        if actions.ndim == 2:
            actions = actions[0]

        if actions.shape != (14,):
            raise ValueError(f"Expected 14-dim Yuanyou2 action, got shape {actions.shape}")

        self._ros_interface.publish_action(actions)

    def set_prompt(self, prompt: str):
        self._prompt = prompt
        logging.info(f"Updated prompt to: '{prompt}'")