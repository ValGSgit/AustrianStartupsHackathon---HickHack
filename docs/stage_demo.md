# Stage demo: "the robot you can't trip up"

A judge moves the cube while the arm reaches for it. The arm follows, grasps, places it in the box, and the screen shows
**VERIFIED**. If it misses, it says so and tries again.

Pitch line: *"Robots that learn from demonstration usually break the moment the world changes. Ours adapts."*

Three layers, each one a fallback for the one above:

| Layer | What | Where |
|---|---|---|
| Learned recovery | ACT trained with disturbance demos, run with frequent re-planning | dataset + `[stage]` in `config.toml` |
| Self-check + retry | camera check "cube in box?", automatic retry on a miss | `selfcheck.py`, `stage.py demo` |
| Replay | a recorded perfect episode | `robot.py replay --episode N` |

## 1. Recording protocol (operator reads this before 12:30)

One dataset, one task sentence, one operator. Total target: **~70 episodes**.

| Block | Episodes | What happens |
|---|---|---|
| A. Clean | 50 | Cube starts on one of the 5 taped marks (10 each). Calm grasp, place, return to rest. |
| B. Disturbance | 20 | Same start. When the gripper is ~5 cm above the cube, the **helper** slides the cube 5 to 10 cm with a stick. The operator re-targets smoothly and finishes. |

Rules for block B:
* Record B **after** A is done and the pipeline smoke test passed. A is the core demo and comes first.
* The helper uses a stick or ruler, never a hand (the hand would be in the training images, and on stage it's unsafe).
* Move the cube to **another taped zone position**, not off the table and not into the box. 4 directions × 5 = 20 episodes.
* One move per episode. The operator never "freezes and waits": pause ≤ 0.5 s, then move toward the new position.
  The policy copies exactly what it sees, including hesitation.
* If the correction looks clumsy, press `←` and redo the episode. A bad recovery demo teaches a bad recovery.
* Keep the same episode length (20 s); recoveries take longer, so move a bit faster in the first half.

Commands (same dataset, B appended with `--resume`):

```powershell
.venv\Scripts\python robot.py record --episodes 10            # pilot / block A part 1
.venv\Scripts\python robot.py record --episodes 10 --resume   # repeat until 50, then 20 more for block B
```

Write down which episode numbers are block B (e.g. 50 to 69) in the team notes.

## 2. Self-check setup (once the camera and box are taped)

```powershell
.venv\Scripts\python selfcheck.py setup     # click the 4 inner box corners, click the cube, press s
.venv\Scripts\python selfcheck.py live      # move the cube in/out of the box by hand: verdict must flip
```

Saves `stage_setup.json` and a reference photo `stage_setup.jpg`. Commit both. Redo setup if the camera,
box or lighting changes. Check with the live view: cube in box, on the rim, outside, and absent.

## 3. Evaluation (17:00 and after the overnight retrain)

```powershell
.venv\Scripts\python stage.py eval --label 10k --policy outputs\train\<run>\checkpoints\010000\pretrained_model
.venv\Scripts\python stage.py eval --label 10k --n-action-steps 100 --policy ...   # ACT default, for comparison
.venv\Scripts\python stage.py eval --label 10k --ensemble 0.01 --policy ...        # re-plan every step
.venv\Scripts\python stage.py report
```

`eval` runs one attempt per taped position (20), checks the result automatically and asks for the failure type
(keys 1 to 6) on a miss. Everything lands in `evals/eval_log.csv`. `report` ranks checkpoint × reactivity by success.

Also run **5 disturbance trials** per candidate (a helper moves the cube mid-reach) and note them in the log notes.
Pick the stage setting by: clean success ≥ 16/20 first, then disturbance success.

## 4. On stage

Rehearse without the arm (webcam stands in for the rollout, everything else is real):

```powershell
.venv\Scripts\python stage.py demo --mock --duration 8
```

During an attempt the screen shows the robot's camera live (frames come from the rollout process itself,
via `rollout_live.py`), with the box outline turning green the moment the cube is inside.

```powershell
.venv\Scripts\python stage.py demo --fullscreen
```

`SPACE` = start (the robot tries up to 3 times), `ESC` = emergency stop (arm freezes in place), `q` = quit.
The laptop speaks: "On it" → "Missed. Trying again" → "Verified. The cube is in the box".

Script (under 30 s):
1. Presenter: "Robots that learn from demonstration usually break when the world changes. Watch."
2. Judge places the cube on any mark. Presenter presses SPACE.
3. As the gripper descends, the judge slides the cube with the stick. The arm follows and places it.
4. Screen: VERIFIED. If it missed: the robot announces the retry and goes again, which is the same story.

If disturbance success is < 50% at the 17:00 eval: the judge moves the cube **before** pressing SPACE instead
of mid-reach, and the retry loop carries the recovery story.
