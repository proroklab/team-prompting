# robomaster_ws — real-robot deployment workspace (Sec. V-D)

A minimal ROS 2 (colcon) workspace to run the trained policy on **DJI RoboMaster EP** robots,
as in the retrieve-the-flag deployment of the paper. Tested on **ROS 2 Humble / Ubuntu 22.04**,
running onboard an Nvidia Jetson Orin with motion-capture positioning.

The deployment **node itself is not here** — it lives in the main package at
`../sequential_tasks/scenarios/four_flags/four_flags_deployment.py`. It loads the distilled
`EventRNN` automaton plus the team GNN policy, prompts for a natural-language instruction, and
drives the team through Freyja. This workspace only provides the third-party ROS 2 stacks it
talks to:

| Package | Source | Notes |
| --- | --- | --- |
| `Freyja` | https://github.com/unl-nimbus-lab/Freyja (`ros2-devel`) | low-level flight control + `freyja_msgs` — **GPLv3**, not redistributed here |
| `ros2_robomaster_msgs` | https://github.com/proroklab/ros2_robomaster_msgs | RoboMaster message/service defs |

## Setup

```bash
cd robomaster_ws
./setup_workspace.sh        # git clone the two third-party stacks into src/
colcon build --symlink-install --cmake-args -DNO_PIXHAWK=True
source install/setup.bash   # puts freyja_msgs on the Python path
```

## Run

```bash
# 1) Low-level controller (one per robot frame, e.g. robomaster_2)
ros2 launch src/Freyja/freyja_robomaster.launch.yaml tf_myframe:=robomaster_2

# 2) Deployment node (from the main package). Hydra app; defaults to
#    conf/four_flags/deployment/four_flags_deployment.yaml. It prompts for the
#    task instruction at startup and drives the robots in deployment.id_list.
cd ../sequential_tasks
PYTHONPATH=. python scenarios/four_flags/four_flags_deployment.py

# 3) (optional) record per-robot mission logs to a rosbag
cd ../robomaster_ws && ./record_mission.sh my_run 4 6 7
```

## Notes before running

- **Checkpoint path.** The deployment config loads a trained team-GNN policy via a
  `policy_restore_path` (a `/path/to/...` placeholder in the `four_flags/deployment` configs) —
  set it to your checkpoint.
- **freyja_msgs on the Python path.** `four_flags_deployment.py` contains a
  `sys.path.insert(0, "/path/to/robomaster_ws/install/freyja_msgs/.../site-packages")` line.
  After `source install/setup.bash` the message package is already importable, so you can point
  this at your workspace's `install/` or remove the line.
- **Robot IDs / namespaces** (`robomaster_4/6/7`, `deployment.id_list`) and motion-capture
  frames are deployment specific — adjust to your setup.
