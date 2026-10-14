# SO-101 pick & place — Austrian Startups Hackathon (Track 1: Hack the Robot)

Teach the SO-101 follower arm a pick & place task from ~50 teleoperated demos
(leader arm + camera), train an **ACT** policy on a local GPU with
[LeRobot](https://huggingface.co/docs/lerobot), and run it live on stage.

| Where | What |
|---|---|
| [docs/real_arm_guide.md](docs/real_arm_guide.md) | **connect + test the real arm**, follow a movement, grab a pen |
| [sim/README.md](sim/README.md) | MuJoCo + ROS 2 (Docker) simulation: practice the whole loop without hardware |
| [cloudflare/robot-router](cloudflare/robot-router) | voice/text → Workers AI → the right skill (Worker + dashboard) |
| [docs/architecture.md](docs/architecture.md) | how the Cloudflare layer, camera and policies fit together (diagrams) |

Everything goes through `robot.py`, which builds the `lerobot-*` commands from
`config.toml`. That keeps ports, calibration ids and camera settings identical
between recording and rollout, so you never retype the long flag lists.
Add `--dry-run` to print a command without running it.

## 1. Install (Windows, once per laptop)

LeRobot 0.6.1 needs **Python 3.12** and a CUDA build of PyTorch.

```powershell
python -m pip install --user uv
python -m uv python install 3.12
python -m uv venv --python 3.12 .venv
# CUDA torch first. The cu128 index has no Windows torchcodec, so lerobot comes from PyPI afterwards.
python -m uv pip install --python .venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -m uv pip install --python .venv\Scripts\python.exe "lerobot[core-scripts,feetech,training]==0.6.1"
# torchcodec can't find FFmpeg DLLs on Windows and floods every command with errors.
# Without it LeRobot quietly uses pyav (bundled FFmpeg) - verified to record + decode fine.
python -m uv pip uninstall --python .venv\Scripts\python.exe torchcodec
# sim + Cloudflare agent extras
python -m uv pip install --python .venv\Scripts\python.exe mujoco==3.3.7 websockets==15.0.1 trimesh==4.8.3 fast-simplification==0.1.11
.venv\Scripts\python robot.py check        # expect cuda=True and your GPU
hf auth login                               # write token from huggingface.co/settings/tokens (.venv\Scripts\hf.exe)
```

Then set `[hf] user` in `config.toml`.

## 2. Hardware setup

**Full step-by-step bring-up with checks: [docs/real_arm_guide.md](docs/real_arm_guide.md).** Short version:

```powershell
.venv\Scripts\python robot.py find-port          # unplug/replug when asked -> put COM ports in config.toml
.venv\Scripts\python robot.py find-cameras       # pick camera index -> config.toml [cameras.front]
.venv\Scripts\python robot.py calibrate follower # middle pose, Enter, move every joint end to end, Enter
.venv\Scripts\python robot.py calibrate leader
.venv\Scripts\python robot.py teleop             # move the leader, the follower copies
```

* Arms from the organizers are pre-assembled, so **skip `setup-motors`**. It rewrites motor ids.
* Calibration is saved to `calibration/follower.json` and `calibration/leader.json` **in this repo**. Commit both after calibrating.
  Then every laptop maps the joints exactly as they were during recording. A different calibration at rollout time makes grasps miss.
* Windows may reorder camera indexes after a replug. Re-run `find-cameras` if the picture looks wrong.

## 3. Record demos

Do a **smoke test of the whole loop first**: 5 episodes, a short training, then a rollout.
That catches camera, codec and Hub problems early. Then record the real dataset.

```powershell
.venv\Scripts\python robot.py record --episodes 10                # first session
.venv\Scripts\python robot.py record --episodes 10 --resume       # add 10 MORE (repeat to reach ~50)
```

Keys during recording: `→`/`n` = episode done, `←`/`r` = redo episode, `Esc`/`q` = stop and save.
Data lands in `data/<dataset>` and is pushed (private) to the Hub as a backup.

What makes a good dataset:
* Fix the camera rigidly, and fix the lighting (bring a lamp; the stage light will be different).
* Use ~5 cube start positions with ~10 demos each, and move the same smooth way every time.
* You should be able to do the task yourself by looking only at the camera image.
* A second camera (wrist or side) helps grasping a lot. Add `[cameras.wrist]` in the config **before** recording.

## 4. Train (local GPU)

```powershell
.venv\Scripts\python robot.py train                 # ACT, settings from [train] in config.toml
.venv\Scripts\python robot.py train --resume        # continue after a crash / Ctrl-C
.venv\Scripts\python robot.py train --resume --steps 30000   # train an existing run for longer
```

`robot.py train` runs `train_win.py`, which wraps `lerobot-train`. Plain `lerobot-train` crashes on Windows at the
first checkpoint (`WinError 1314`), because it creates a `checkpoints/last` symlink and that needs admin rights.
The wrapper creates a directory junction instead. Train, save, resume and loading through `last/` have all been verified.

Checkpoints are saved to `outputs/train/<policy>/checkpoints/`, one every 5k steps. You can roll out an intermediate one while training continues.
Watch the log line `updt_s:… mem_gb:…`. If `updt_s` is well above ~0.6 s, GPU memory is spilling into system RAM.
Stop the run, lower `batch_size` in `config.toml`, and start again with a new `--policy` name.

## 5. Run it

```powershell
.venv\Scripts\python robot.py rollout --duration 60     # uses the local checkpoint, else the Hub copy
.venv\Scripts\python robot.py rollout --policy outputs\train\act_so101_pick_place\checkpoints\010000\pretrained_model
```

Use the same camera, the same position and the same lighting as when you recorded.

**Stage fallback:** `robot.py replay --episode N` replays a good recorded demo exactly.

## GPU benchmark

Measured on 2026-10-14 on a GTX 1650 (4 GB), training ACT on `lerobot/svla_so101_pickplace`
(2 cameras at 640x480), LeRobot 0.6.1, torch 2.11 cu128:

| batch | AMP | s/step | GPU mem | note |
|---|---|---|---|---|
| 8 | on  | 4.9  | 3.55 GB | spills to system RAM |
| 8 | off | 5.6  | 3.73 GB | spills to system RAM |
| 4 | on  | 2.2  | 2.14 GB | fp16 is slow on GTX 16xx |
| **4** | **off** | **0.51** | 2.17 GB | use this with 2 cameras |
| 2 | off | 0.27 | 1.40 GB | |
| **8** (1 cam) | off | **0.48** | 1.99 GB | **default in config.toml (1 camera)** |
| 16 (1 cam) | off | 6.4 | 3.41 GB | spills |

Takeaways:
* **Keep AMP off on this GPU.** The GTX 16xx cards have no tensor cores, and fp16 convolutions run about 5x slower than fp32.
* **Stay under ~3.3 GB.** The 1650 also drives the display. When it runs out of VRAM, Windows silently spills into system RAM and each step gets about 10x slower instead of failing.
  To make it fail fast instead: NVIDIA Control Panel → Manage 3D settings → *CUDA – Sysmem Fallback Policy* → *Prefer No Sysmem Fallback*.
* Expect about **2.7 h for 20k steps** (one camera, batch 8, 160k samples, about 5 passes over 50 episodes of 20 s).
  Two cameras at batch 4 run at the same speed but see half the samples, so give them more steps.
  Start training early, and test intermediate checkpoints.
* Data loading is not a bottleneck (`data_s` ≈ 0.004 s with pyav).
