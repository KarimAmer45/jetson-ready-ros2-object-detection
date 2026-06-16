from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
            DeclareLaunchArgument("detections_topic", default_value="/detections"),
            DeclareLaunchArgument("annotated_topic", default_value="/detections/annotated"),
            DeclareLaunchArgument("backend", default_value="torchvision"),
            DeclareLaunchArgument("model", default_value=""),
            DeclareLaunchArgument("device", default_value="auto"),
            DeclareLaunchArgument("score_threshold", default_value="0.5"),
            DeclareLaunchArgument("max_detections", default_value="50"),
            DeclareLaunchArgument("queue_size", default_value="2"),
            DeclareLaunchArgument("log_interval_sec", default_value="5.0"),
            DeclareLaunchArgument("publish_annotated", default_value="true"),
            DeclareLaunchArgument("stats_topic", default_value="/detections/stats"),
            Node(
                package="jetson_ready_ros2_object_detection",
                executable="detector_node",
                name="object_detector",
                output="screen",
                parameters=[
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "detections_topic": LaunchConfiguration("detections_topic"),
                        "annotated_topic": LaunchConfiguration("annotated_topic"),
                        "backend": LaunchConfiguration("backend"),
                        "model": LaunchConfiguration("model"),
                        "device": LaunchConfiguration("device"),
                        "score_threshold": ParameterValue(
                            LaunchConfiguration("score_threshold"), value_type=float
                        ),
                        "max_detections": ParameterValue(
                            LaunchConfiguration("max_detections"), value_type=int
                        ),
                        "queue_size": ParameterValue(
                            LaunchConfiguration("queue_size"), value_type=int
                        ),
                        "log_interval_sec": ParameterValue(
                            LaunchConfiguration("log_interval_sec"), value_type=float
                        ),
                        "publish_annotated": ParameterValue(
                            LaunchConfiguration("publish_annotated"), value_type=bool
                        ),
                        "stats_topic": LaunchConfiguration("stats_topic"),
                    }
                ],
            ),
        ]
    )
