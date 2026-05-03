from glob import glob

from setuptools import setup


package_name = "jetson_ready_ros2_object_detection"
python_package = "object_detection_ros2"


def package_files(pattern):
    return [str(path) for path in glob(pattern)]


setup(
    name=package_name,
    version="0.1.0",
    packages=[python_package],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", package_files("config/*.yaml")),
        (f"share/{package_name}/launch", package_files("launch/*.py")),
        (f"share/{package_name}/docs/images", package_files("docs/images/*")),
    ],
    install_requires=["setuptools", "numpy"],
    extras_require={
        "torchvision": ["torch", "torchvision"],
        "yolo": ["ultralytics"],
    },
    zip_safe=True,
    maintainer="Karim",
    maintainer_email="karim@example.com",
    description="Jetson-oriented ROS2 object detection with PyTorch backends.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "detector_node = object_detection_ros2.detector_node:main",
            "benchmark = object_detection_ros2.benchmark:main",
        ],
    },
)
