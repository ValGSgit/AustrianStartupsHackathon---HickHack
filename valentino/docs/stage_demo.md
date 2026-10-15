# Stage demo: "the robot you can't trip up"

Team task: **"Pick up the pen"** (`[skills.pick_pen]`, dataset `pick_pen`, policy `act_pick_pen`).

A judge moves the pen while the arm reaches for it. The arm follows, grasps, lifts it, and the screen shows
**VERIFIED**. If it misses, it says so and tries again.

Pitch line: *"Robots that learn from demonstration usually break the moment the world changes. Ours adapts."*

Three layers, each one a fallback for the one above:

| Layer | What | Where |
|---|---|---|
| Learned recovery | ACT trained with disturbance demos, run with frequent re-planning | dataset + `[stage]` in `config.toml` |
| Self-check + retry | camera check "pen held in the air for 1 s?", automatic retry on a miss | `selfcheck.py`, `stage.py demo` |
| Replay | a recorded perfect episode | `robot.py replay --episode N` |

## 0. Before recording

* **Brightly coloured pen** (red, blue, green, orange) or wrap coloured tape around it. The self-check finds the pen
  by colour; it can't see a black or white pen against the table.
* Camera fixed and taped, seeing the pick area **and the air above it** (where the lifted pen will be).
* Pen lies flat, across the arm's reach, 10 to 25 cm in front of the base (see `docs/real_arm_guide.md` §8).

## 1. Recording protocol

One dataset, one task sentence, one operator. Total target: **~70 episodes** of 15 s.

| Block | Episodes | What happens |
|---|---|---|
| A. Clean | 50 | Pen starts on one of 5 taped marks (10 each), moved a few cm and rotated ±20° between demos. Above the pen → down → close → lift ~10 cm → hold. |
| B. Disturbance | 20 | Same start. When the gripper is ~5 cm above the pen, the **helper** slides the pen 5 to 10 cm with a stick. The operator re-targets smoothly, grasps, lifts, holds. |

Rules for block B:
* Record B **after** A is done and the pipeline smoke test passed. A is the core demo and comes first.
* The helper uses a stick or ruler, never a hand (the hand would be in the training images, and on stage it's unsafe).
* Move the pen to **another position inside the taped zone**, never off the table. 4 directions × 5 = 20 episodes.
* One move per episode. The operator never "freezes and waits": pause ≤ 0.5 s, then move toward the new position.
  The policy copies exactly what it sees, including hesitation.
* If the correction looks clumsy, press `←` and redo the episode. A bad recovery demo teaches a bad recovery.
* Same episode length (15 s); recoveries take longer, so move a bit faster in the first half.

Commands (same dataset, B appended with `--resume`):

```powershell
.venv\Scripts\python robot.py record --episodes 10            # pilot / block A part 1
.venv\Scripts\python robot.py record --episodes 10 --resume   # repeat until 50, then 20 more for block B
```

Write down which episode numbers are block B (e.g. 50 to 69) in the team notes.

## 2. Self-check setup (once the camera is taped)

```powershell
.venv\Scripts\python selfcheck.py setup     # click 4 corners of the AIR zone above the pick area, click the pen, press s
.venv\Scripts\python selfcheck.py live      # hold the pen up in the zone by hand, then put it down: verdict must flip
```

The zone is where the lifted pen appears in the camera image: above the table surface, below the top of the arm's
reach. The pen lying on the table must be **outside** it. Success = the pen stays in the zone for 1 s
(`--hold-s`). Saves `stage_setup.json` and a reference photo `stage_setup.jpg`. Commit both. Redo setup if the
camera or lighting changes.

## 3. Evaluation (after the first training, and after the overnight retrain)

```powershell
.venv\Scripts\python stage.py eval --label 10k --policy outputs\train\act_pick_pen\checkpoints\010000\pretrained_model
.venv\Scripts\python stage.py eval --label 10k --n-action-steps 100 --policy ...   # ACT default, for comparison
.venv\Scripts\python stage.py eval --label 10k --ensemble 0.01 --policy ...        # re-plan every step
.venv\Scripts\python stage.py report
```

`eval` runs one attempt per taped position (20), checks the result automatically and asks for the failure type
(keys 1 to 6) on a miss. Everything lands in `evals/eval_log.csv`. `report` ranks checkpoint × reactivity by success.

Also run **5 disturbance trials** per candidate (a helper moves the pen mid-reach) and note them in the log notes.
Pick the stage setting by: clean success ≥ 16/20 first, then disturbance success.

## 4. On stage

Rehearse without the arm (webcam stands in for the rollout, everything else is real):

```powershell
.venv\Scripts\python stage.py demo --mock --duration 8
```

During an attempt the screen shows the robot's camera live (frames come from the rollout process itself,
via `rollout_live.py`), with the zone outline turning green while the pen is lifted.

```powershell
.venv\Scripts\python stage.py demo --fullscreen
```

`SPACE` = start (the robot tries up to 3 times), `ESC` = emergency stop (arm freezes in place), `q` = quit.
The laptop speaks: "On it" → "Verified. pen lifted" (or "Missed. Trying again").

Script (under 30 s):
1. Presenter: "Robots that learn from demonstration usually break when the world changes. Watch."
2. Judge places the pen on any mark. Presenter presses SPACE.
3. As the gripper descends, the judge slides the pen with the stick. The arm follows, grasps and lifts it.
4. Screen: VERIFIED. If it missed: the robot announces the retry and goes again, which is the same story.

If disturbance success is < 50% at the first eval: the judge moves the pen **before** pressing SPACE instead
of mid-reach, and the retry loop carries the recovery story.
