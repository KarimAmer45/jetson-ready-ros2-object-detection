#!/usr/bin/env bash
set -euo pipefail

echo "== Jetson release =="
cat /etc/nv_tegra_release || true

echo "== Docker =="
docker --version

echo "== ROS 2 package benchmark =="
WORKSPACE_SETUP="/ros2_ws/install/setup.bash"
if [ -f "${WORKSPACE_SETUP}" ]; then
  . "${WORKSPACE_SETUP}"
fi

ros2 run jetson_ready_ros2_object_detection benchmark \
  --backend torchvision \
  --device auto \
  --warmup 1 \
  --iterations 3