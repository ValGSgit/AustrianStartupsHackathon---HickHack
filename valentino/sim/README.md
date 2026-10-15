# SO-101 simulation (MuJoCo + ROS 2 Jazzy in Docker)

Practice the full **teleop → record → train → rollout** loop before touching the real arm.

> **Read this first: what the sim is (and isn't) for.**
> ACT learns from camera pixels, and simulated pixels don't look like the real camera.
> A policy trained here **will not work on the real arm**. The sim is for practicing the workflow,
> checking that training and inference work end to end, tuning settings, and seeing policies behave before
> risking hardware. For the stage you still record real demos with the leader arm ([real arm guide](../docs/real_arm_guide.md)).

**Why MuJoCo and not Gazebo:** the official SO-101 simulation model (TheRobotStudio/SO-ARM100) ships as
MuJoCo files with tuned STS3215 servo parameters. MuJoCo is also much more reliable at gripping a thin pen.
ROS 2 wraps it, so you get standard topics, Foxglove visualization and a ROS-shaped control interface.

## What's here

| Path | What |
|---|---|
| `sim/so101_sim/env.py` | the simulation: SO-101 + table + pen/cube + front camera (640x480), 30 Hz control, LeRobot units |
| `sim/so101_sim/expert.py` | scripted top-down grasp with inverse kinematics; generates demos |
| `sim/gen_demos.py` | scripted demos → LeRobot dataset in `data/sim_pick_<obj>` (same feature names as the real robot) |
| `sim/eval_policy.py` | roll out a trained policy in the sim and report its success rate (+ optional video) |
| `ros2_ws/src/so101_sim_ros` | ROS 2 package: `sim_node`, `teleop_udp_node`, `policy_node`, `sim.launch.py` |
| `docker/Dockerfile.sim`, `docker-compose.yml` | ROS 2 Jazzy + MuJoCo + LeRobot 0.6.1 (CPU) image |
| `host/leader_to_sim.py` | streams the **real leader arm** from the host into the container sim |

**Units = the real robot's LeRobot units** (what `lerobot-record` stores with the default settings): arm joints
−100..100 over the calibrated range (0 = middle), wrist roll −100..100 = ±180° (calibrated as a full turn),
gripper 0..100. Sim datasets and real recordings therefore mean the same thing, and the leader arm drives the
sim with its normal settings. On ROS topics, joints are in radians (ROS standard).

How this was matched to the real arm (`env.py` `lerobot_to_rad`):
* LeRobot normalizes each motor linearly over the range swept during calibration, with zero at the middle.
  Its kinematics code feeds joint degrees straight into this same SO-101 URDF, so directions match one-to-one.
* Checked on real Hugging Face datasets: replayed through the sim's kinematics, the gripper is at table height
  at the moments the real gripper closes on an object:

  | dataset | robot_type | frames | grasp moments | gripper height at grasp (median) |
  |---|---|---|---|---|
  | `youliangtan/so101-table-cleanup` | `so101_follower` | 46,963 | 5,018 | −1.2 cm |
  | `lerobot/svla_so101_pickplace` | `so100_follower` (SO-101 arm per its name) | 11,939 | 737 | −0.5 cm |

  The episode start pose maps to the folded rest pose with the gripper resting on the table.
* Careful when mixing Hub datasets: some are recorded with `use_degrees=True` (values beyond ±100, e.g.
  `hungho77/so101_left_arm_pick_red_cube_into_pot`). Those are in different units from ours.
* The scripted expert seeds its IK with the **median real grasp pose** (from `svla_so101_pickplace`), so it uses the same elbow configuration
  as human teleoperators. Objects are placed where real grasps happen (13–23 cm in front, ±10 cm sideways).

Pad-based gripper contact: the jaw meshes collide as convex hulls, which fill the gap between the
fingers, so box "finger pads" were added for gripping (see `_add_finger_pads` in `env.py`).

## Fast path: no Docker, on the Windows host

Uses the project `.venv` (`uv pip install mujoco==3.3.7` once; `trimesh fast-simplification` for `--lowpoly`).

```powershell
.venv\Scripts\python sim\fetch_model.py                                   # official SO-101 model, pinned commit
.venv\Scripts\python sim\gen_demos.py --obj pen --episodes 50             # ~3.5 min -> data\sim_pick_pen
.venv\Scripts\python robot.py train --dataset sim_pick_pen --policy act_sim_pick_pen --steps 10000 --no-push
.venv\Scripts\python sim\eval_policy.py --obj pen --episodes 20 --video outputs\sim_eval.mp4 `
    --policy outputs\train\act_sim_pick_pen\checkpoints\last\pretrained_model
```

## ROS 2 in Docker

Requires Docker Desktop (WSL2 backend on Windows). Build once, ~10 min:

```powershell
docker compose build sim
```

Run one of the modes. Each mode starts the sim plus Foxglove on `ws://localhost:8765`:

```powershell
# expert picks the pen in a loop - check that everything works
docker compose run --rm --service-ports sim

# leader arm drives the sim (in a second terminal on the host: the streamer)
docker compose run --rm --service-ports sim ros2 launch so101_sim_ros sim.launch.py mode:=teleop
.venv\Scripts\python host\leader_to_sim.py            # real leader arm
.venv\Scripts\python host\leader_to_sim.py --wave     # no hardware: test motion

# a trained policy drives the sim (CPU inference inside the container)
docker compose run --rm --service-ports sim ros2 launch so101_sim_ros sim.launch.py mode:=policy `
    policy_path:=/ws/outputs/train/act_sim_pick_pen/checkpoints/last/pretrained_model

# generate demos inside the container instead of on the host
docker compose run --rm sim python sim/gen_demos.py --obj pen --episodes 50
```

`data/` and `outputs/` are mounted into the container. Datasets made in either place train with
`robot.py train` on the host GPU, and policies trained on the host run in the container.

**View it:** install [Foxglove](https://foxglove.dev/download) (or use the web app) → *Open connection* →
*Foxglove WebSocket* → `ws://localhost:8765`. Add an **Image** panel on `/so101/camera/front/image_raw/compressed`
and a **Plot** panel on `/joint_states.position[0..5]`.

### Topics

| Topic | Type | Direction |
|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` (rad) | sim → |
| `/so101/camera/front/image_raw` | `sensor_msgs/Image` rgb8 640x480 | sim → |
| `/so101/camera/front/image_raw/compressed` | `sensor_msgs/CompressedImage` jpeg | sim → |
| `/so101/object_pose` | `geometry_msgs/PoseStamped` | sim → |
| `/so101/lifted` | `std_msgs/Bool` (object > 4 cm up) | sim → |
| `/so101/joint_commands` | `sensor_msgs/JointState` (rad, by name) | → sim |
| `/so101/reset` | `std_srvs/Trigger` | service |

Drive it yourself from another shell in the container:
```bash
docker compose exec sim bash   # or: docker exec -it <container> /entrypoint.sh bash
ros2 topic pub --once /so101/joint_commands sensor_msgs/msg/JointState "{name: [shoulder_pan], position: [0.5]}"
ros2 service call /so101/reset std_srvs/srv/Trigger
```

## Leader arm → sim: direction check

`host/leader_to_sim.py` reads the leader with `use_degrees=True`. If a sim joint moves the **opposite** way
to your hand, flip it in the `SIGN` table at the top of that file. If it's offset, adjust `OFFSET_DEG`.
Check once per joint: move only that joint and watch Foxglove.

## Results

Measured on 2026-10-14:

| What | Result |
|---|---|
| scripted expert, pen / cube (20 seeds each) | 20/20 / 19/20 lifted |
| host sim speed (GTX 1650 rendering) | 2–3 ms per 30 Hz step incl. 640x480 render |
| Docker sim (2 vCPU, CPU rendering, low-poly meshes) | `/joint_states` 30.0 Hz, camera 4.95 Hz |
| ROS scripted mode | `/so101/lifted` true during each pick |
| leader → sim over UDP (wave test motion) | 30 packets/s, joints follow |
| dataset made in the container → loaded on the host | OK (decodes, same features) |
| ACT trained on 50 sim demos, 5k steps, 20 unseen pen positions | **12/20 = 60 %** (earlier run, degree units and a farther workspace) |
| same, after switching to real-robot units + realistic workspace, 10k steps (`act_sim_pick_pen`) | **15/20 = 75 %** lifted ≥ 4 cm; 11/20 still held at the end of the episode (4 lifted, then dropped). 5 misses never lifted it |

Docker Desktop on this laptop only gives the container 2 CPUs. More CPUs (Docker Desktop → Settings →
Resources, or `.wslconfig`) make the camera faster.
