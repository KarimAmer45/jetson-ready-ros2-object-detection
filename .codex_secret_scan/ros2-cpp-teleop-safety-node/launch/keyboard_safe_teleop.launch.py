from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_share = FindPackageShare("ros2_cpp_teleop_safety_node")
    config_file = LaunchConfiguration("config_file")

    safety_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([package_share, "launch", "teleop_safety.launch.py"])
        ),
        launch_arguments={"config_file": config_file}.items(),
    )

    keyboard = Node(
        package="teleop_twist_keyboard",
        executable="teleop_twist_keyboard",
        name="teleop_twist_keyboard",
        output="screen",
        prefix="xterm -e",
        remappings=[("/cmd_vel", "/cmd_vel_raw")],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [package_share, "config", "teleop_safety.yaml"]
                ),
            ),
            safety_launch,
            keyboard,
        ]
    )
