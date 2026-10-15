# Experiment: MolmoAct2 on the team dashboard, several objects per command, both cameras

Status: **experiment, not in use.** Built 2026-10-15 on top of the `molmo-skill` branch. After it was built, the
working setup was put back to `molmo-skill` and the original `so101-hackathon` code, so nothing here runs unless
you copy it in on purpose.

## Goal

Type "put the duck and the chess queen into the cup" on the dashboard
(https://robot-router.entinosusini.workers.dev) and watch the whole flow there: router decision, live picture
from both cameras (iPhone + Mac camera), which object and which step the robot is on, and the final result
("success: 2/2 objects") with the after-picture and the vision check.

No change to the Cloudflare Worker or the dashboard page: everything goes through the existing agent protocol
(`frame` messages with `label`, `jpeg`, `state`, `target`; `done` with the after-image).

## What changed

### Team repo (this branch, compared with `molmo-skill`)

| File | Change |
|---|---|
| `host/robot_agent.py` | `[cameras.front] name = ["iphone", "webcam"]`: the agent finds the camera by its macOS name (via ffmpeg's device list), because macOS renumbers cameras whenever the iPhone comes or goes (seen during testing: the iPhone moved from index 1 to 0 within minutes). After a run, if the camera gives no picture, the last live frame is used as the after-image. |
| `config.toml` | `molmo_pick` description now says it handles several objects, one after another; `duration_s` is per object. |

### Model side (`so101-hackathon/`, copies in `so101-hackathon/` here)

| File | What it does |
|---|---|
| `tasks.py` (new) | Splits one instruction into one MolmoAct2 sentence per object: "put the duck and the chess queen into the cup" -> "move the arm towards the duck, grasp it, lift it up, and drop it into the cup" + the same for the queen. Handles "X and Y into Z", "X in Z and Y in Z", "X into Z, then Y", lists with commas, already-model-worded sentences, "on the notebook" (put it on). Tested on 15 phrasings. |
| `live_view.py` (new) | When `ROBOT_LIVE=1` (the agent sets it), prints `@@LIVE {...}` lines ~6 per second, the team's existing live format. The picture: cam0 (iPhone) large, cam1 (Mac camera) small, and a band with the object, the step (model try, carry, reopen, home) and the time. Drawing runs on its own thread, so the control loop is never slowed; during carry/home it reads the cameras itself so the picture keeps moving. ~12 KB per frame. |
| `do_task.py` | Runs the objects one after another (`--seconds` per object), stops the rest on STOP/Ctrl+C, prints a final `RESULT: success: 2/2 objects (...)`. A missing camera no longer stops the run: the side camera is optional, and if the top camera is missing the side camera becomes the only one. Installs the live view. Full diff: `so101-hackathon/do_task.patch`. |
| `tests/fake_do_task.py` (new) | The real `do_task.py` with a fake model server and a simulated arm and drawn cameras: tests the whole flow without the GPU, robot or cameras. |
| `tests/offline_run_test.py` | One line: importable (used by `fake_do_task.py`). |
| `settings.env` (not copied, has the Studio address) | `FRONT_CAM=iphone`, `SIDE_CAM=webcam`, `MOLMO_TASK="put the duck and the chess queen into the cup"` (the dashboard's skill button uses MOLMO_TASK). |

`config.local.toml.example` here is the agent's local config used for the experiment (Mac ports, `[molmo] dir`,
camera by name).

## What was tested

* Task splitter: 15 phrasings, all correct.
* Real `do_task.py` with the fake model and simulated arm, 2 objects: both "success" (one missed grab + retry
  each), final `RESULT: success: 2/2 objects`, 130 live frames, all with picture, joints and targets, labels
  `duck (1/2) · model1 -> reopen -> model2 -> carry -> home -> chess queen (2/2) ... -> success: 2/2 objects`.
* The original offline scenarios (`retry_then_carry`, `old_behaviour`, `no_drop_pose`): same results as before.
* Team agent test `host/test_molmo_skill.py`: 5/5 pass.
* Against the deployed Worker: the agent connected and the dashboard listed `molmo_pick`. The typed-command
  round trip was **not finished** (the test script was blocked by Cloudflare's bot check on Python's default
  User-Agent, error 1010; fixed in the script, the rerun was stopped before it ran).

## Not tested / still needed

* Nothing on the real arm, real cameras or the Lightning GPU (Studio `arman1` was stopped).
* `above_cup` is not taught yet (`python teach_drop.py cup`). Without it there is no auto-drop, every object
  runs until its time is up, and the result can't say "success".
* Joint bars on the dashboard assume -100..100; MolmoAct2 runs in degrees, so large angles sit at the bar ends.

## Try it

```bash
# model side: copy the files into so101-hackathon (back up first), set the two cameras in settings.env
cp experiments/molmo-dashboard/so101-hackathon/{tasks.py,live_view.py,do_task.py} ~/Documents/workplace/so101-hackathon/
cp experiments/molmo-dashboard/so101-hackathon/tests/*.py ~/Documents/workplace/so101-hackathon/tests/
# offline flow test (no GPU, robot or camera)
cd ~/Documents/workplace/so101-hackathon/tests && ROBOT_LIVE=1 ../.venv/bin/python fake_do_task.py --seconds 25 "put the duck and the chess queen into the cup"
```
