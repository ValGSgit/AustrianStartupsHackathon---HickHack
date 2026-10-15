# Get the SO-101 working: bring-up guide

From "arms on the table" to "say *grab the pen* and it does". Work through it top to bottom; every step
has a check, so you know it worked before moving on.

**"Controller" = the leader arm**, the one you move by hand. The follower copies it.
Everything runs from the repo folder in PowerShell. Every robot command has this shape:

```powershell
.venv\Scripts\python robot.py <command>        # add --dry-run before <command> to only print the lerobot command
```

| Step | What | Time |
|---|---|---|
| [0](#0-laptop-setup-once-per-laptop) | laptop setup | 10 min |
| [1](#1-safety-before-plugging-in) | safety | 2 min |
| [2](#2-connect-and-find-the-ports) | connect, find ports | 5 min |
| [3](#3-calibrate-both-arms-then-commit) | calibrate both arms | 10 min |
| [4](#4-check-the-numbers) | check the joint numbers | 5 min |
| [5](#5-teleoperate-the-follower-copies-the-leader) | teleoperate | 5 min |
| [6](#6-camera) | camera | 5 min |
| [7](#7-make-it-follow-a-specific-movement) | follow a specific movement | 10 min |
| [8](#8-grab-a-pen) | grab a pen: record → train → run | ~1 h demos + ~2.7 h training |
| [9](#9-voice-control-through-cloudflare) | voice control through Cloudflare | 15 min |
| [10](#10-on-stage) | stage checklist | — |

---

## 0. Laptop setup (once per laptop)

```powershell
git clone https://github.com/ValGSgit/AustrianStartupsHackathon---HickHack.git
cd AustrianStartupsHackathon---HickHack
```
Then follow **Install** in the [README](../README.md) (Python 3.12 venv, CUDA torch, LeRobot 0.6.1, extras).

✅ **Check:** `.venv\Scripts\python robot.py check` prints `cuda=True` (on the training laptop) and your HF user.

> Use **one laptop for the arm**: the same one for recording, training and rollout. If you must switch laptops,
> pull the repo first so `calibration/` is identical.

## 1. Safety (before plugging in)

- [ ] Both arms on a stable table, **bases clamped or taped**. The follower tips over when it reaches far.
- [ ] Follower power supply matches the label on its servo board (STS3215 servos).
- [ ] Nothing breakable within ~40 cm of the follower.
- [ ] **E-stop = pull the follower's power plug.** Decide who holds it. Unplugging USB does *not* stop the motors.

## 2. Connect and find the ports

1. Plug in **power** (follower board, then the leader's if it has its own supply), then **USB** for both.
2. Find which COM port belongs to which arm:
   ```powershell
   .venv\Scripts\python robot.py find-port
   ```
   It asks you to unplug one USB cable and tells you which port disappeared. Repeat for the other arm.
3. Put the ports into `config.toml`:
   ```toml
   [follower]
   port = "COM5"      # <- yours
   [leader]
   port = "COM6"      # <- yours
   ```

✅ **Check:** `robot.py check` lists both COM ports.

> The arms come assembled, so **never run `setup-motors`**. It rewrites the motor ids and breaks the daisy chain.

### Getting the arm back from another team (shared arm)

A calibration lives **inside the motors**, so whoever calibrated last wins. That includes other teams and
`arman/calibrate.py`. Our `calibration/*.json` is still right; it just has to be written back.
**Don't recalibrate**: that changes the joint mapping, and recordings and policies stop matching.

```powershell
.venv\Scripts\python robot.py restore-calibration --check   # compare motors vs calibration/*.json, writes nothing
.venv\Scripts\python robot.py restore-calibration           # arms RESTING on the table: writes our files back
.venv\Scripts\python robot.py positions --follower          # rest pose should read like before (step 4)
```

`teleop`, `record`, `replay`, `rollout` and `stage.py` run this check first. On a mismatch they stop with the
fix, instead of hanging on lerobot's "Press ENTER to use provided calibration file" prompt.
If the restore reports a large **end-stop shift**, it's a different arm or a re-mounted motor. Check the COM port first.

## 3. Calibrate both arms, then commit

```powershell
.venv\Scripts\python robot.py calibrate follower
.venv\Scripts\python robot.py calibrate leader
```

For each arm:
1. Put **every joint in the middle of its range** (arm upright and slightly bent, gripper half open) → **Enter**.
   This pose becomes "0" for every joint, so take a moment to get it right.
2. Move **every joint except the wrist roll** slowly from one end stop to the other → **Enter**.
   (LeRobot treats the wrist roll as a full turn, so it needs no sweep.)

This writes `calibration/follower.json` and `calibration/leader.json`. **Commit and push them right away:**
```powershell
git add calibration/*.json; git commit -m "Calibrate SO-101 arms"; git push
```
Recalibrating later changes what the numbers mean, and recorded demos and trained policies stop matching.
Only recalibrate if step 4 fails.

## 4. Check the numbers

Datasets and policies work with the numbers LeRobot reads from the motors. Look at them once:

```powershell
.venv\Scripts\python robot.py positions --watch              # leader only
.venv\Scripts\python robot.py positions --follower --watch   # both (the follower keeps holding its pose)
```

Move the leader slowly and check:

| Joint | Expected |
|---|---|
| all arm joints | about **0 in the middle pose** from step 3, about **±100 at the end stops** |
| `wrist_roll` | changes smoothly while you twist it (±100 = ±180°) |
| `gripper` | about **0 closed**, rising as it opens (up to 100) |
| folded **rest pose** | about `[0, -99, 99, 74, *, 2]`, the same as in real Hugging Face SO-101 datasets |

The sim uses exactly these conventions (checked against real Hub datasets, see [sim/README.md](../sim/README.md)).
If your rest pose is far off, or a value doesn't change or jumps, recalibrate that arm (step 3).

✅ **Check:** leader and follower show similar numbers when they're in a similar pose.

## 5. Teleoperate: the follower copies the leader

```powershell
.venv\Scripts\python robot.py teleop --no-cameras
```

Start with both arms in a **similar pose**. Then move the leader **slowly**:

| Check | How | If wrong |
|---|---|---|
| each joint moves the same way | one joint at a time | recalibrate that arm |
| no jump at start | follower glides to the leader's pose | start with both arms in a similar pose |
| gripper fully opens and closes | squeeze and release the leader's trigger | recalibrate, sweep the gripper end to end |
| no buzzing or hot motor at rest | listen, touch after 1 min | a joint is pushing into a limit: recalibrate |

`Ctrl-C` stops. **Hold the follower** when you stop: its torque turns off and it drops.

**Optional: mirror the leader into the simulation.** This shows the sim matches your arm (needs Docker; see [sim/README.md](../sim/README.md)):
```powershell
docker compose run --rm --service-ports sim ros2 launch so101_sim_ros sim.launch.py mode:=teleop
.venv\Scripts\python host\leader_to_sim.py          # second terminal; view in Foxglove ws://localhost:8765
```
If a sim joint moves mirrored compared with the real leader, flip it in `SIGN` at the top of `host/leader_to_sim.py`.

## 6. Camera

1. Mount the camera **rigidly** (tape or clamp), looking at the workspace from the front/top. Add a **lamp** too, because stage light differs.
2. Find its index and look at the picture:
   ```powershell
   .venv\Scripts\python robot.py find-cameras        # saves outputs\captured_images\opencv_<i>.png
   ```
   Open the PNG. You must see the table and the follower clearly. A black or dark image means a covered lens or the wrong index.
3. Set `[cameras.front] index = <i>` in `config.toml` (640x480 @ 30 fps). The table name `front` becomes part of
   the dataset, so **never rename it after recording**.
4. Teleop with the live view: `robot.py teleop` (the rerun viewer shows camera + joints).

✅ **Check:** rerun shows a sharp, well-lit image while you teleoperate.

> Windows can renumber webcams after a replug. Re-run `find-cameras` whenever you replug.

## 7. Make it follow a specific movement

**Live:** `robot.py teleop`. The follower mirrors your hand in real time.

**Record once, replay exactly:** good for a fixed move such as a wave or a bow. It's also the **stage fallback**, because it always works.

```powershell
.venv\Scripts\python robot.py record --dataset moves_wave --episodes 1 --no-push
#   perform the movement with the leader, press → (right arrow) when done
.venv\Scripts\python robot.py replay --dataset moves_wave --episode 0
```
- Start the replay from the same pose you recorded from. Replay doesn't look at the camera.
- More moves: other `--dataset` names (`moves_bow`, …), or add episodes with `--resume` and choose them with `--episode N`.

## 8. Grab a pen

Grabbing is learned from your demos: **record ~50 demos → train ACT → run it.**

### Set up so it can succeed
- A **thick marker** (≥ 12 mm), not a thin ballpoint. Tape on the jaw tips helps grip.
- A dark, matte mat under it for contrast. Camera and lamp **fixed**.
- The pen lies **flat**, roughly **across** the arm's reach direction, **10–25 cm in front of the base**.
  That's where real SO-101 users grasp, and where the sim practices.
- In `config.toml`: `[task] description = "Pick up the pen"` and `episode_time_s = 15`.

### Record
```powershell
.venv\Scripts\python robot.py record --dataset pick_pen --episodes 10 --no-push
.venv\Scripts\python robot.py record --dataset pick_pen --episodes 10 --resume --no-push   # repeat to ~50
```
- **5 start positions × ~10 demos each.** Between demos, move the pen a few cm and rotate it ±20°.
- Use the same calm motion every time: above the pen → down → close → lift ~10 cm → hold.
- Bad demo? Press `←` to redo it. Keys: `→` next, `←` redo, `Esc` stop and save.

### Train (~2.7 h on the GTX 1650)
```powershell
.venv\Scripts\python robot.py train --dataset pick_pen --policy act_pick_pen --no-push
```
Checkpoints are written every 5k steps to `outputs\train\act_pick_pen\checkpoints\`. You can try one while training continues.

### Run it: slow and supervised first
```powershell
.venv\Scripts\python robot.py rollout --policy act_pick_pen --max-step 2 --duration 30
.venv\Scripts\python robot.py rollout --policy outputs\train\act_pick_pen\checkpoints\005000\pretrained_model --max-step 2
```
- `--max-step 2` caps every joint at 2 units per 30 Hz step (about 60 units/s), so there are no sudden jumps. Drop it once it behaves.
- Same camera position, lighting and mat as during recording. Hand on the power plug.

| Symptom | Likely cause |
|---|---|
| reaches the right area but misses | too few demos at that spot, or the camera moved |
| freezes or jitters | camera name/resolution or calibration differs from recording |
| grasps, then drops | pen too thin or slippery: use a marker, tape the jaws |
| random motions | lighting changed: record ~10 more demos under the new light |

## 9. Voice control through Cloudflare

The [robot-router Worker](../cloudflare/robot-router) turns "please grab the pen" into the right skill.

1. Check `[skills.*]` in `config.toml` matches what you trained and recorded:
   `pick_pen.policy = "act_pick_pen"` (step 8), `wave.replay_dataset = "moves_wave"` (step 7).
2. Start the Worker. Locally:
   ```powershell
   cd cloudflare\robot-router; npx wrangler dev        # http://127.0.0.1:8787, token from .dev.vars
   ```
   Or deploy it ([Worker README](../cloudflare/robot-router/README.md#deploy)).
3. Connect the arm to it, from the repo root:
   ```powershell
   .venv\Scripts\python host\robot_agent.py --url http://127.0.0.1:8787 --token <ROBOT_TOKEN>
   ```
4. Open the dashboard and enter the token. First check the hardware from the **status line** at the top:
   - **Camera → Start preview**: the Live panel shows the camera. Fix the framing and lighting now.
   - **Look (vision AI)**: the vision model says what it sees. The pen should be named, with its position.
   - **Arm → Connect** (read-only, torque untouched): the bars show the follower's joints. Rest pose ≈
     `[0,-99,99,74,*,2]`. If torque is off (e.g. right after calibrating), move it gently by hand and the bars
     must follow in the same directions as in step 4. If a joint resists, torque is on: don't force it.
5. Type or say: "please grab the pen", "wave at everyone". The Live panel shows each step (heard → vision →
   router → inference → joint targets) while the arm moves, then the result and the vision check.

- Only one program can hold the camera or the COM port. Stop teleop/record before using the status line;
  the agent releases both by itself before a skill runs and reopens them afterwards.
- **STOP** in the dashboard ends the running skill. The power plug is still the real e-stop.
- No arm yet? Add `--sim` to `robot_agent.py` and the MuJoCo sim stands in for the arm.

## 10. On stage

- [ ] Same laptop and `calibration/` as for recording. Camera taped in place. Run `find-cameras` once on stage.
- [ ] Lamp on (stage light ≠ hacking-table light).
- [ ] Rehearse once: `robot.py rollout --policy act_pick_pen --max-step 2`.
- [ ] Fallback ready: `robot.py replay --dataset moves_wave --episode 0`.
- [ ] Someone's hand on the follower's power plug.

---

## Cheat sheet

| Goal | Command |
|---|---|
| status / ports / GPU | `robot.py check` |
| which COM port is which | `robot.py find-port` |
| camera index + test image | `robot.py find-cameras` |
| calibrate | `robot.py calibrate follower` / `robot.py calibrate leader` |
| live joint numbers | `robot.py positions --watch [--follower]` |
| follower copies leader | `robot.py teleop [--no-cameras]` |
| record demos | `robot.py record --dataset NAME --episodes N [--resume] [--no-push]` |
| replay a recording | `robot.py replay --dataset NAME --episode N` |
| train ACT | `robot.py train --dataset NAME --policy POLICY [--resume] [--steps N]` |
| run a policy | `robot.py rollout --policy POLICY [--max-step 2] [--duration S]` |
| voice / text control | `host\robot_agent.py --url … --token …` + dashboard |
| practice in sim | [sim/README.md](../sim/README.md) |

## Troubleshooting

| Message / symptom | Fix |
|---|---|
| `Could not connect on port 'COMx'` | wrong port in `config.toml` (`find-port`), USB unplugged, or the board has no power |
| calibration prompt appears during teleop/record | `calibration/<id>.json` missing or ids changed: pull the repo, or calibrate |
| `Set [hf] user in config.toml` | set your HF username, or add `--no-push` to record/train locally |
| a motor buzzes, gets hot, or the arm sags | joint pushed into a limit or overloaded: lighter object, recalibrate |
| camera image black | lens covered, wrong index, or the camera is used by another program (close Zoom/Teams) |
| training very slow (`updt_s` ≫ 0.6) | GPU memory spills to RAM: lower `[train] batch_size` (8 → 4) |
| `WinError 1314` while training | you ran plain `lerobot-train`; use `robot.py train` |
