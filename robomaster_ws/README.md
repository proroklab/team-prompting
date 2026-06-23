# robomaster_ws — real-robot deployment workspace (Sec. V-D)

A minimal ROS 2 (colcon) workspace to run the trained policy on **DJI RoboMaster EP** robots,
as in the retrieve-the-flag deployment of the paper. Tested on **ROS 2 Humble / Ubuntu 22.04**,
running onboard an Nvidia Jetson Orin with motion-capture positioning.

To keep the release self-contained and license-clean, this workspace ships **only** the
bootstrap scripts. The actual packages are assembled on first setup:

| Package | Source | Notes |
| --- | --- | --- |
| `language_driven_exploration` | this repo (symlinked) | the policy + ROS 2 deployment node |
| `Freyja` | https://github.com/unl-nimbus-lab/Freyja (`ros2-devel`) | low-level flight control — **GPLv3**, not redistributed here |
| `ros2_robomaster_msgs` | https://github.com/proroklab/ros2_robomaster_msgs | RoboMaster message/service defs |

## Setup

```bash
cd robomaster_ws
./setup_workspace.sh        # symlink the package + git clone the two third-party stacks
colcon build --symlink-install --cmake-args -DNO_PIXHAWK=True
source install/setup.bash
```

## Run

```bash
# 1) Low-level controller (one per robot frame, e.g. robomaster_2)
ros2 launch src/Freyja/freyja_robomaster.launch.yaml tf_myframe:=robomaster_2

# 2) Policy node: loads the RNN + GNN policy, subscribes to /robomaster_*/current_state,
#    publishes /robomaster_*/reference_state
python src/language_driven_exploration/deployment/my_deployment.py

# 3) (optional) record per-robot mission logs to a rosbag
./record_mission.sh my_run 4 6 7
```

## Notes before running

- **Checkpoint path.** `my_deployment.py` and the deployment Hydra configs load a trained
  policy via a `/path/to/...` placeholder — set it to your checkpoint location.
- **freyja_msgs on the Python path.** The deployment scripts contain a
  `sys.path.insert(0, "/path/to/robomaster_ws/install/freyja_msgs/.../site-packages")` line.
  After `source install/setup.bash` the message package is already importable, so you can point
  this at your workspace's `install/` or remove the line.
- **Robot IDs / namespaces** (`robomaster_2/4/6/7`) and motion-capture frames are deployment
  specific — adjust to your setup.
