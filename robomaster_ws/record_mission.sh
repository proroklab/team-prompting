#!/bin/bash
# Record per-robot mission logs to a rosbag.
#   ./record_mission.sh <output_bag> [robot_id ...]
# Example: ./record_mission.sh run1 4 6 7
set -euo pipefail
OUT="${1:?usage: record_mission.sh <output_bag> [robot_id ...]}"; shift
ROBOTS=("$@"); [ ${#ROBOTS[@]} -eq 0 ] && ROBOTS=(4 6 7)
TOPICS=(); for id in "${ROBOTS[@]}"; do TOPICS+=("/robomaster_${id}/mission_log"); done
exec ros2 bag record "${TOPICS[@]}" -o "${OUT}"
