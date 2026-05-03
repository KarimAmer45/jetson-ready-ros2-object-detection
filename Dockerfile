ARG ROS_DISTRO=humble
FROM ros:${ROS_DISTRO}-ros-base-jammy

ARG ROS_DISTRO=humble
ARG TORCH_INSTALL="torch torchvision --index-url https://download.pytorch.org/whl/cpu"
ARG INSTALL_ULTRALYTICS=true

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ROS_DISTRO=${ROS_DISTRO}

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-pip \
    python3-opencv \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-diagnostic-msgs \
    ros-${ROS_DISTRO}-vision-msgs \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir ${TORCH_INSTALL} \
    && if [ "${INSTALL_ULTRALYTICS}" = "true" ]; then \
        python3 -m pip install --no-cache-dir ultralytics; \
    fi

WORKDIR /ros2_ws/src/jetson_ready_ros2_object_detection
COPY . .

WORKDIR /ros2_ws
RUN . "/opt/ros/${ROS_DISTRO}/setup.sh" && colcon build --symlink-install

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "jetson_ready_ros2_object_detection", "object_detection.launch.py"]
