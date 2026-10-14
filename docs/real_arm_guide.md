# Real arm guide: connect, test, follow a movement, grab a pen

"**Controller**" here means the **leader arm**, the arm you move by hand. The follower copies it.
That's the setup the organizers hand out. All commands run from the repo folder in PowerShell:

```powershell
.venv\Scripts\python robot.py <command>
```

Do the sim first if you haven't ([sim/README.md](../sim/README.md)). It lets you practice the same
record → train → rollout loop without risking the hardware.

---

## 0. Before you plug anything in

- [ ] Follower and leader on a stable table, bases clamped or taped down. The follower tips over when it reaches far.
- [ ] Follower power supply: the right voltage for its STS3215 servos (check the label on the PSU).
- [ ] Workspace in front of the follower is clear. Nothing breakable within ~40 cm.
- [ ] **E-stop = pull the follower's power plug.** Agree on who holds it. Unplugging USB does *not* stop the motors.

## 1. Connect (order matters)

1. Plug in **power** to the follower's controller board, then (if it has one) the leader's.
2. Plug in the **USB** of both boards.
3. Find which COM port is which:
   ```powershell
   .venv\Scripts\python robot.py find-port
   ```
   It asks you to unplug one USB cable and reports the port that disappeared. Do it once per arm.
4. Put both ports into `config.toml`:
   ```toml
   [follower]
   port = "COM5"      # <- yours
   [leader]
   port = "COM6"      # <- yours
   ```
5. Check: `.venv\Scripts\python robot.py check` lists both COM ports.

> The arms come pre-assembled, so **don't run `setup-motors`**. It rewrites the motor ids.

## 2. Calibrate (once, then commit)

```powershell
.venv\Scripts\python robot.py calibrate follower
.venv\Scripts\python robot.py calibrate leader
```

For each arm:
1. Move every joint to the **middle** of its range (arm roughly upright, gripper half open) → **Enter**.
2. Move **every joint end to end**, slowly, through its full range → **Enter**.

This writes `calibration/follower.json` and `calibration/leader.json`. **Commit both**. Every laptop then
maps the joints the same way. If someone recalibrates later, recorded datasets and trained policies no longer match.

## 3. Test: follow the controller live (teleoperation)

```powershell
.venv\Scripts\python robot.py teleop --no-cameras
```

Move the leader slowly. The follower should mirror it. Run through this checklist:

| Check | How | If wrong |
|---|---|---|
| Each joint moves the **same direction** | move one leader joint at a time | recalibrate that arm |
| Follower doesn't jump on start | it should glide to the leader's pose | start with both arms in a similar pose |
| Gripper **fully opens and closes** | squeeze / release the leader trigger | recalibrate, move the gripper end to end |
| No buzzing or hot motors at rest | listen / touch the motor after 1 min | a joint is fighting a limit, so recalibrate |
| Smooth at full reach | stretch out slowly | lower the speed of your hand; check the base is clamped |

`Ctrl-C` stops. The follower's torque is released on exit, **so hold the arm**, or it drops.

Then do the same with the camera, to check the picture:
```powershell
.venv\Scripts\python robot.py find-cameras        # look at outputs\captured_images\*.png
.venv\Scripts\python robot.py teleop              # rerun viewer shows camera + joints
```
The PNG must show the workspace clearly. A black or dark image means a covered lens or the wrong camera index.

## 4. Make it follow a specific movement

There are two ways, and both use the leader as the controller.

**a) Live:** `robot.py teleop`. The follower mirrors whatever you do, in real time.

**b) Record once, replay exactly:** good for a scripted move on stage, like a wave or a bow.

```powershell
# record ONE episode of the movement (no task learning involved)
.venv\Scripts\python robot.py record --dataset moves_wave --episodes 1 --no-push
#   do the movement with the leader, press → (right arrow) when done
.venv\Scripts\python robot.py replay --dataset moves_wave --episode 0
```

The follower replays the exact joint trajectory. Record more moves into other datasets (`moves_bow`, …),
or add episodes with `--resume` and pick them by `--episode N`.
Replay is open-loop: it doesn't look at the camera, so start it from the same pose you recorded from.

## 5. Grab a pen

Grabbing a pen is a learned task: record demos with the leader, train ACT, then roll it out.

### Make it easy for the robot
- Use a **thick marker** (≥ 12 mm), not a thin ballpoint. Or wrap tape around the jaw tips for grip.
- Use a **dark, matte mat** under the pen for contrast. Keep the camera and the lamp fixed (tape them down).
- Pen lies **flat**, roughly **perpendicular to the arm's reach direction**, at the start of every demo.
- Approach from **above**, with the jaws opened across the pen. Close, lift ~10 cm, hold.

### Record (≈50 demos)
```powershell
.venv\Scripts\python robot.py record --dataset pick_pen --episodes 10 --no-push
.venv\Scripts\python robot.py record --dataset pick_pen --episodes 10 --resume --no-push   # repeat to ~50
```
- Use **5 start positions × ~10 demos each**. Move the pen a few cm and rotate it ±20° between demos.
- Use the same calm, smooth motion every time. Pause briefly before closing the gripper.
- A failed demo: press `←` to redo it. Don't keep bad demos.
- Set `[task] description = "Pick up the pen"` and `episode_time_s = 15` in `config.toml` first.

### Train (~2.7 h on the GTX 1650)
```powershell
.venv\Scripts\python robot.py train --dataset pick_pen --policy act_pick_pen --no-push
```

### First rollout: slow and supervised
```powershell
.venv\Scripts\python robot.py rollout --policy act_pick_pen --max-step 5 --duration 30
```
- `--max-step 5` caps every joint to a small move per control step, so the first run is slow and gentle.
  Once it behaves, drop the flag.
- Same camera position, same lighting, same mat as during recording.
- Hand on the follower's power plug the whole time.
- You can try intermediate checkpoints while training continues:
  `--policy outputs\train\act_pick_pen\checkpoints\005000\pretrained_model`

### If it doesn't work

| Symptom | Likely cause |
|---|---|
| Arm moves to roughly the right area but misses | too few demos at that pen position, or the camera moved |
| Arm freezes / jitters in place | camera key, resolution or calibration differs from recording |
| Grasps but drops the pen | pen too thin or slippery: use a marker, add tape to the jaws |
| Does something random | lighting changed a lot. Record ~10 demos under the stage light |

## 6. Stage checklist

- [ ] Calibration files are committed, and the laptop is the same one (or uses the same `calibration/`) as for recording.
- [ ] Camera taped in place. Run `find-cameras` once more on stage, because indexes change after a replug.
- [ ] Lamp on. Stage lighting differs from the hacking table.
- [ ] Fallback ready: `robot.py replay --dataset moves_wave --episode 0` always works.
- [ ] Someone on the power plug.
