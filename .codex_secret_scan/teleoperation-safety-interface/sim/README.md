# Gazebo Simulation Integration

Gazebo is applicable for this project because the interface is about remote robot command gating, telemetry, and safety interlocks. Use modern Gazebo Sim with ROS 2, not Gazebo Classic.

Recommended stack:

- Gazebo Sim Harmonic or newer.
- ROS 2 Jazzy or newer.
- `ros_gz_bridge` for Gazebo Transport to ROS 2 topics.
- `rosbridge_suite` for browser WebSocket access.

The browser UI can run without these tools. Select `Gazebo / ROS 2 bridge` only when rosbridge is running.

## Topics

| Direction | Topic | Type | Purpose |
| --- | --- | --- | --- |
| Browser to ROS 2 | `/cmd_vel` | `geometry_msgs/msg/Twist` | Gated velocity command from the UI |
| Gazebo to ROS 2 | `/odom` | `nav_msgs/msg/Odometry` | Simulated robot speed |
| Gazebo to ROS 2 | `/scan` | `sensor_msgs/msg/LaserScan` | Nearest obstacle or human range |
| ROS 2 to browser | `/tsi/telemetry` | `std_msgs/msg/String` | JSON telemetry consumed by the UI |

## Run

From the repository root:

```bash
gz sim sim/gazebo/worlds/teleop_safety_lab.sdf
```

In another shell:

```bash
ros2 run ros_gz_bridge parameter_bridge --ros-args -p config_file:=sim/gazebo/bridge.yaml
```

Build and run the telemetry bridge:

```bash
cd sim/ros2/tsi_bridge
colcon build --symlink-install
source install/setup.bash
ros2 run tsi_bridge telemetry_bridge
```

Start rosbridge:

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

Run the web app, choose `Gazebo / ROS 2 bridge`, and connect to `ws://localhost:9090`.

## Better simulation choice

For this repository, Gazebo Sim is the better physical simulation choice because it supports robot dynamics, sensors, obstacles, worlds, and ROS 2 bridge workflows. For UI-only demos or classroom presentations, the built-in browser simulator is better because it runs anywhere without robotics dependencies.
