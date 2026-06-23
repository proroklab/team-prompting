#!/bin/bash
# Assemble the colcon workspace for the real-robot deployment (paper Sec. V-D,
# retrieve-the-flag). This clones the third-party ROS 2 stacks that are NOT
# redistributed here. The deployment node itself lives in the repo at
# ../sequential_tasks/scenarios/four_flags/four_flags_deployment.py and is
# launched after building and sourcing this workspace.
set -euo pipefail
WS_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="${WS_DIR}/src"
mkdir -p "${SRC}"

# 1) Freyja flight-control stack (GPLv3, unl-nimbus-lab) — provides freyja_msgs + controllers
[ -d "${SRC}/Freyja" ] || git clone --branch ros2-devel \
  https://github.com/unl-nimbus-lab/Freyja.git "${SRC}/Freyja"

# 2) RoboMaster ROS 2 message/service definitions (proroklab)
[ -d "${SRC}/ros2_robomaster_msgs" ] || git clone \
  https://github.com/proroklab/ros2_robomaster_msgs.git "${SRC}/ros2_robomaster_msgs"

echo "Workspace ready. Build, source, then launch the deployment node:"
echo "  cd ${WS_DIR} && colcon build --symlink-install --cmake-args -DNO_PIXHAWK=True"
echo "  source install/setup.bash"
echo "  cd ../sequential_tasks && PYTHONPATH=. python scenarios/four_flags/four_flags_deployment.py"
