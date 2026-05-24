# ros2-cpp-teleop-safety-node

A ROS 2 C++ teleoperation safety node that sits between a raw velocity source and a robot base.

It subscribes to a raw `geometry_msgs/msg/Twist`, applies safety limits, and publishes a guarded `Twist` for the base controller. The node is intentionally small, dependency-light, and written in C++ with `rclcpp`.

## Features

- C++17 `rclcpp` node with an installable `ament_cmake` package.
- Velocity clamps for forward, lateral, and yaw commands.
- Linear and angular acceleration limiting.
- Deadman timeout that publishes zero velocity when teleop input goes stale.
- External e-stop topic using `std_msgs/msg/Bool`.
- Runtime speed scaling topic using `std_msgs/msg/Float64`.
- Optional forward obstacle stop from `sensor_msgs/msg/LaserScan`.
- Launch files for standalone use, keyboard teleop, and an optional Gazebo Sim demo.

## Topics

| Direction | Topic | Type | Purpose |
| --- | --- | --- | --- |
| Subscribe | `/cmd_vel_raw` | `geometry_msgs/msg/Twist` | Raw teleop command input |
| Publish | `/cmd_vel` | `geometry_msgs/msg/Twist` | Safety-filtered base command |
| Subscribe | `/scan` | `sensor_msgs/msg/LaserScan` | Optional obstacle stop input |
| Subscribe | `/e_stop` | `std_msgs/msg/Bool` | `true` forces zero output |
| Subscribe | `/speed_limit` | `std_msgs/msg/Float64` | Scale in the range `0.0` to `1.0` |

## Build

Clone this repository into a ROS 2 workspace:

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <repo-url> ros2-cpp-teleop-safety-node
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select ros2_cpp_teleop_safety_node
source install/setup.bash
```

## Run

Start only the safety node:

```bash
ros2 launch ros2_cpp_teleop_safety_node teleop_safety.launch.py
```

Run with keyboard teleop, where keyboard commands are remapped through the safety node:

```bash
sudo apt install ros-${ROS_DISTRO}-teleop-twist-keyboard
ros2 launch ros2_cpp_teleop_safety_node keyboard_safe_teleop.launch.py
```

Publish an e-stop:

```bash
ros2 topic pub /e_stop std_msgs/msg/Bool "{data: true}" --once
```

Set a 40% speed limit:

```bash
ros2 topic pub /speed_limit std_msgs/msg/Float64 "{data: 0.4}" --once
```

## Configuration

Default parameters live in `config/teleop_safety.yaml`.

Key parameters:

| Parameter | Default | Description |
| --- | --- | --- |
| `input_topic` | `/cmd_vel_raw` | Raw command topic |
| `output_topic` | `/cmd_vel` | Filtered output topic |
| `command_timeout` | `0.35` | Seconds before stale input is treated as zero |
| `max_linear_speed` | `0.6` | Absolute cap for forward velocity |
| `max_lateral_speed` | `0.25` | Absolute cap for lateral velocity when enabled |
| `max_angular_speed` | `1.4` | Absolute cap for yaw velocity |
| `max_linear_accel` | `1.0` | Linear acceleration limit in m/s^2 |
| `max_angular_accel` | `2.0` | Angular acceleration limit in rad/s^2 |
| `allow_lateral_motion` | `false` | Keep `linear.y` or force it to zero |
| `obstacle_stop_enabled` | `true` | Stop forward motion when an obstacle is close |
| `min_obstacle_distance` | `0.45` | Stop distance in meters |
| `front_arc_degrees` | `70.0` | LaserScan arc checked around straight ahead |

## Optional Gazebo Sim Demo

The repository includes an optional Gazebo Sim world under `sim/` and a launch file:

```bash
ros2 launch ros2_cpp_teleop_safety_node gazebo_demo.launch.py
```

This requires Gazebo Sim plus the ROS/Gazebo bridge packages for your ROS 2 distribution, commonly provided by packages such as `ros-${ROS_DISTRO}-ros-gz`, `ros-${ROS_DISTRO}-ros-gz-bridge`, and `ros-${ROS_DISTRO}-ros-gz-sim`.

```bash
sudo apt install ros-${ROS_DISTRO}-ros-gz ros-${ROS_DISTRO}-ros-gz-bridge ros-${ROS_DISTRO}-ros-gz-sim
```

The demo starts a tiny differential-drive robot in a simple world and bridges the safe `/cmd_vel` output into the Gazebo model command topic. If those simulator packages are not installed, the core safety node still builds and runs normally.

## Safety Notes

This project is a software guardrail for development and demos. Real robots still need hardware-level emergency stops, controller-level limits, watchdogs, and platform-specific safety validation.

## Safety-state output

![ros2-cpp-teleop-safety-node result screenshot](docs/results/result-screenshot.png)

Safety-filter command flow for watchdog, emergency-stop, and speed-limit behavior.


## Command-gating notes

- A C++ ROS 2 safety wrapper for teleoperation velocity commands.
- Runtime safety states for command limiting, stale input, and emergency stop.
- Configurable topics and parameters suitable for integration into a ROS workspace.


## Robot validation plan

- The node is a software safety demo and not a certified safety controller.
- Full validation requires a ROS 2 runtime and robot/simulator integration tests.
- Next steps: add rosbag-based regression tests and a short Gazebo demonstration video.

