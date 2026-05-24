from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("ros2_cpp_teleop_safety_node")
    world_file = LaunchConfiguration("world_file")
    config_file = LaunchConfiguration("config_file")

    safety_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "teleop_safety.launch.py"])
        ),
        launch_arguments={"config_file": config_file}.items(),
    )

    gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", world_file],
        output="screen",
    )

    cmd_vel_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="cmd_vel_bridge",
        output="screen",
        arguments=[
            "/model/safety_bot/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist"
        ],
        remappings=[
            ("/model/safety_bot/cmd_vel", "/cmd_vel"),
        ],
    )

    scan_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="scan_bridge",
        output="screen",
        arguments=[
            "/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "world_file",
                default_value=PathJoinSubstitution(
                    [package_share, "sim", "worlds", "teleop_safety_demo.sdf"]
                ),
            ),
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [package_share, "config", "teleop_safety.yaml"]
                ),
            ),
            gazebo,
            safety_launch,
            cmd_vel_bridge,
            scan_bridge,
        ]
    )
