# Yuanyou2 OpenPI Checklist

This directory contains the ROS1 client and rosbag conversion tools for the Yuanyou2 dual-Piper robot.

## Runtime Convention

The policy state and action are 14-dimensional:

```text
[left_joint1..left_joint6, left_gripper,
 right_joint1..right_joint6, right_gripper]
```

The gripper scalar is `joint7` on each arm. `joint8` is the opposite finger in the URDF and is not commanded separately by this OpenPI interface.

## Before Recording Bags

On the Jetson:

```bash
source /opt/ros/noetic/setup.bash
source ~/yuanyou2_ws/devel/setup.bash

roslaunch yuanyou2 three_cameras.launch
roslaunch yuanyou2 camera_tf.launch
```

Run the preflight check:

```bash
python3 examples/yuanyou2/check_ros_setup.py
```

Fix any failed camera topic, camera_info topic, `/joint_states`, or clock warning before collecting real data.

## Bag Topics

Record at least:

```bash
rosbag record -O yuanyou2_demo.bag \
  /joint_states \
  /left/joint_cmd \
  /right/joint_cmd \
  /head_camera/usb_cam/image_raw \
  /head_camera/usb_cam/camera_info \
  /left_wrist_d435/color/image_raw \
  /left_wrist_d435/color/camera_info \
  /right_wrist_d435/color/image_raw \
  /right_wrist_d435/color/camera_info \
  /tf \
  /tf_static
```

The converter uses `/left/joint_cmd` and `/right/joint_cmd` as actions when present. If they are missing, it falls back to the next sampled `/joint_states` value.

## Convert Bags

Run conversion in a Python environment that can import ROS1 `rosbag` and `cv_bridge`. On the Jetson this usually means sourcing ROS Noetic first; on another machine, use a ROS-aware virtual environment or container.

```bash
source /opt/ros/noetic/setup.bash
uv run examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py /path/to/bag_dir \
  --output eeason396/yuanyou2_test \
  --fps 10 \
  --task "pick a cube and place it on another cube"
```

For strict command-action conversion:

```bash
uv run examples/yuanyou2/convert_yuanyou2_data_to_lerobot.py /path/to/bag_dir \
  --action-source command
```

## Train

After conversion, compute normalization stats and train:

```bash
uv run scripts/compute_norm_stats.py pi05_yuanyou2_lora_finetune
uv run scripts/train.py pi05_yuanyou2_lora_finetune --exp_name first_run
```

Update the Yuanyou2 training config repo id to your real LeRobot dataset before this step.

## Serve And Run

On the policy server machine:

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_yuanyou2_lora_finetune \
  --policy.dir=checkpoints/pi05_yuanyou2_lora_finetune/first_run/20000
```

On the Jetson:

```bash
python3 examples/yuanyou2/main.py \
  --remote-host <policy-server-ip> \
  --prompt "pick a cube and place it on another cube"
```
