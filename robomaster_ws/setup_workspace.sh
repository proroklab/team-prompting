#!/bin/bash
# Assemble the colcon workspace for the real-robot deployment (paper Sec. V-D).
# Links the language_driven_exploration package and clones the third-party ROS 2
# stacks that are NOT redistributed here (Freyja, RoboMaster messages).
set -euo pipefail
WS_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${WS_DIR}/src"
mkdir -p "${SRC}"

# 1) Author deployment package (single-stage policy + ROS 2 node), symlinked from the repo
ln -sfn ../../language_driven_exploration "${SRC}/language_driven_exploration"

# 2) Third-party flight-control stack (GPLv3, unl-nimbus-lab) — required, not vendored here
[ -d "${SRC}/Freyja" ] || git clone --branch ros2-devel \
  https://github.com/unl-nimbus-lab/Freyja.git "${SRC}/Freyja"

# 3) RoboMaster ROS 2 message/service definitions (proroklab)
[ -d "${SRC}/ros2_robomaster_msgs" ] || git clone \
  https://github.com/proroklab/ros2_robomaster_msgs.git "${SRC}/ros2_robomaster_msgs"

echo "Workspace ready. Now build with:"
echo "  cd ${WS_DIR} && colcon build --symlink-install --cmake-args -DNO_PIXHAWK=True"
