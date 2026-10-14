# SO-101: chess piece into a cup

Two ways to do the same task. Build part 1 first (about 30 minutes), then part 2 if there is time.

| | Part 1: remembered poses | Part 2: learned model (ACT) |
|---|---|---|
| How | You show 4 poses, the arm replays them | You show the task 50 times, a neural net learns it |
| Piece position | Must be on the same taped spot | Anywhere in the area you trained on |
| Time | ~30 min | ~4 to 6 h (mostly recording + training) |
| Needs camera / GPU | No / No | Yes / Cloud GPU recommended |
| Use it as | Your guaranteed demo + fallback | The "AI" part judges care about |

## 0. Install (once)

```bash
scripts/setup_mac.sh
source .venv/bin/activate
hf auth login
```

`hf auth login` needs a **write** token from huggingface.co/settings/tokens.

Then fill in `settings.env`:
- `lerobot-find-port` gives you the two USB ports (unplug one arm when it asks).
- `lerobot-find-cameras opencv` gives you the camera indexes.
- `hf auth whoami` gives you your username.

Calibrate both arms once, then align them. Use these instead of `lerobot-calibrate`, which needs a
"middle pose" that is easy to get wrong:

```bash
python calibrate.py follower
```

```bash
python calibrate.py leader
```

```bash
python align.py
```

`calibrate.py`: move each joint end to end until all lines are green, no middle pose needed. Push the robot's joints to their **real** end stops (it is stiff): every end you miss makes that joint's range smaller and its zero point off. `align.py`: both arms in the same pose twice; saves the difference per joint to `offsets.json`, which joystick control subtracts.

Then save everything (restore later with `scripts/save_setup.sh restore`):

```bash
scripts/save_setup.sh
```

## Hardware notes (learned the hard way, 2026-10-14)

- **Power:** the robot (claw, follower) has **12V** motors and needs a **12V DC, 2A+** supply. On 5V it is too weak to lift its upper arm, and motors blink and report voltage errors. The joystick (handle, leader) stays on **5V, 4A**. Never put 12V on the joystick (one of its motors accepts max 8V).
- **Ports:** robot `/dev/tty.usbmodem5A680122691`, joystick `/dev/tty.usbmodem5AE60587711`. `python motors.py` shows which motors answer on each.
- **Camera:** use the iPhone (Continuity Camera, 640x360). On a Mac the RealSense only gives an infrared image covered in dots. Camera numbers change when cameras come and go: check with `python cam_test.py`.
- **Anything using the camera must run in the Terminal app**, not from Claude: macOS only lets Terminal use the camera.

## Tools

| Command | What it does |
|---|---|
| `python motors.py` | Which motors answer on each arm (read-only) |
| `python cam_test.py` | One photo per camera into `outputs/cam_test/` (run in the Terminal app) |
| `python calibrate.py follower` / `leader` | Guided calibration with live feedback |
| `python align.py` | Measure the difference between the arms into `offsets.json` |
| `python joystick.py` | Drive the robot with the joystick; type a name + Enter to save a pose |
| `python check.py` | Full pre-flight check |
| `scripts/save_setup.sh` / `restore` | Back up or restore calibrations, offsets, settings, poses |

## Part 1: remembered poses (no learning)

1. Tape 3 to 4 numbered spots for pieces and 1 spot for the cup, all within ~25 cm of the base.
2. Run `python poses.py teach` and move the **leader** arm. At each pose, type the name and press Enter:
   - `home`: arm folded and resting (when the script ends the arm goes limp, so this must be safe)
   - `above_cup`: above the cup opening
   - `above_1`: about 5 cm above spot 1, gripper **open**
   - `grasp_1`: jaws around the piece on spot 1, gripper **closed**, squeeze a little extra
   - same for `above_2` / `grasp_2`, `above_3` / `grasp_3`, ...
   - optional `safe`: a high pose between the spots and the cup. Every transfer then passes through it.
   - put the leader in its rest position, then `q` to quit
   - Teach every `above_N` **higher than the cup rim + 3 cm**, and grasp pieces **low** (near the base) so short and tall pieces both work.
3. Test one pose with `python poses.py go above_1`.
4. Run it: `python pick_place.py all` (or `python pick_place.py 2 1` for specific spots).

Test without the arm: `FAKE_ROBOT=1 python pick_place.py all`. Before the real arm, always run `python check.py`.

If it misses, re-teach just that pose (same name overwrites it). No leader arm? Use `python poses.py teach --hand` and move the follower by hand.

You can save any pose you want (`wave`, `bow`, `look_at_user`) and reuse `go_to()` from `poses.py` in your own scripts. This is also how you'd later expose skills to Claude over MCP: each function becomes one tool.

## Part 2: learned model

### Record 50 demonstrations

Recording rules that decide whether it works:
- Camera fixed (tape it), same lighting, plain table. Never move them after recording.
- 5 piece positions x 10 episodes each. Cup always at the same spot.
- Do the task the same way each time, smooth and not too fast. Bad episode: press `r` to redo it.
- Test: could you do the task looking **only** at the camera image? If not, move the camera.

```bash
scripts/record.sh 10
```

For positions 2 to 5, run the same command four more times. It adds to the existing dataset by itself.

Each episode is 20 s, with 10 s to reset after it. Press `n` when the piece is in the cup to end early. To start over completely, delete the `data/` folder. A failed run (wrong port, blocked camera) leaves no mess: the next run cleans up empty folders itself. `DISPLAY_DATA=false scripts/record.sh 10` turns off the live viewer window. Check your episodes in the browser at huggingface.co/spaces/lerobot/visualize_dataset.

### Train

```bash
scripts/train_act.sh cloud
```

This runs on a Hugging Face GPU (A10G). You pay per second, and your account (armannnnnnn) is prepaid with no payment method yet, so add credit at huggingface.co/settings/billing first or it will fail. Check prices with `hf jobs hardware`. Expect roughly 1 to 3 hours. Free alternative: the ACT Colab notebook from the LeRobot docs ("Train using Google Colab"). On the M2 Mac (`scripts/train_act.sh local`) it runs, but much slower. Use it only to check that training starts.

### Run it

```bash
scripts/run_policy.sh 30
```

Then count: 20 tries, how many succeed? Put that number on your slide.

## What you download from Hugging Face, and how big

| What | Needed for | Size |
|---|---|---|
| `lerobot` library + PyTorch + deps (pip, not HF Hub) | everything | ~1.5 to 2 GB on disk |
| `ffmpeg` (brew) | video in datasets | ~100 to 200 MB |
| ResNet-18 ImageNet weights (auto-downloaded by ACT) | ACT | ~45 MB |
| Your dataset (50 eps x 20 s, 640x480 video) | ACT training | ~100 to 300 MB (estimate) |
| Your trained ACT model | running the robot | ~200 MB |
| *Optional:* `lerobot/svla_so101_pickplace` example dataset | test training before you have data | 86 MB |
| *Optional:* `lerobot/smolvla_base` | SmolVLA instead of ACT | 907 MB + needs a real GPU |

Hugging Face accounts and pages you use (nothing to download):
- **HF account + write token:** stores your dataset and model.
- **Dataset visualizer Space:** watch your recorded episodes in the browser.
- **HF Jobs:** cloud GPU for training (paid). Optional if you use Colab.

**Why ACT and not SmolVLA:** for one task in one fixed setup, ACT is small, trains fast and works well. SmolVLA is the better choice only when you want one model that follows different text instructions ("put the white piece in the cup", "put the black one in the bowl"). It needs ~4 h on an A100 to fine-tune.

## Troubleshooting

- **Scripts end or you press Ctrl+C:** the arm goes slowly to `home`, then the motors go limp. Emergency stop: unplug the arm's power supply.
- **Arm jerks or stops at a pose:** `max_relative_target=15` in `config.py` limits jumps. If moves stop short, raise it a bit.
- **"Incorrect status packet":** check the 3-pin cables and the power supply.
- **Gripper drops the piece:** re-teach `grasp` with more squeeze, or add TPU or rubber tips to the jaws.
- **Model reaches the wrong spot:** the camera or lighting moved since recording, or there aren't enough episodes at that position.
