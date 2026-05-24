# ROS2 Wrapper

Optional ROS2 package scaffold for connecting the GNSS-denied visual-inertial estimator to live or bagged topics. The Python simulation is the reference implementation; this wrapper shows the production-facing topic shape.

![Trajectory overview](../results/example/trajectory_overview.png)

## ROS2 wrapper scope

- A ROS2 node boundary for IMU prediction plus GNSS and visual pose updates.
- Topic names that mirror common robotics datasets and autonomy stacks.
- A path from the deterministic Python simulation toward bag replay and hardware integration.

## Topics

Subscriptions:

- `/imu/data` as `sensor_msgs/msg/Imu`
- `/wheel/odom` as `nav_msgs/msg/Odometry`
- `/visual_odometry` as `nav_msgs/msg/Odometry`
- `/gnss/pose` as `geometry_msgs/msg/PoseWithCovarianceStamped`

Publication:

- `/localization/ekf_odom` as `nav_msgs/msg/Odometry`

## Build

From a ROS2 workspace:

```bash
mkdir -p ~/vio_ws/src
cp -r ros2/gnss_denied_vio_cpp ~/vio_ws/src/
cd ~/vio_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select gnss_denied_vio_cpp
source install/setup.bash
ros2 run gnss_denied_vio_cpp vio_ekf_wrapper_node
```

## Integration notes

- The C++ wrapper is a compact node scaffold, not a replacement for the full Python simulation.
- It keeps a planar state and simple covariance model for readability.
- It should be validated against rosbag playback before any live robot use.
- Next steps are parameter files, diagnostics, transform publishing, and a bag-driven regression test.

