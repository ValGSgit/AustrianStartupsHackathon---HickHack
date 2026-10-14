# Hackathon plan: "Pick up the chess pieces and put them in the cup"

## Do I need to record a dataset with the leader arm ("joystick")?

| Version | Recording needed? | What you do with the leader arm |
|---|---|---|
| **Poses mode** (`pick_place.py`) | **No dataset** | Show each pose once (~5 min total) |
| **Learned mode** (ACT) | **Yes, ~50 demos** | Do the full task 50 times (~1.5 h) |

There is no model on Hugging Face that does this task on *your* table out of the box. Every learned policy needs demos from your own camera, lighting and arm. So the only thing downloaded "pretrained" is the image part of ACT. The rest learns from your demos.

## Status: what is already done on this Mac

| | Status |
|---|---|
| Python 3.12 env `.venv` with LeRobot 0.6.1, PyTorch 2.11 (Mac GPU works) | ✅ installed (1.4 GB) |
| ffmpeg, uv | ✅ installed |
| ACT image weights (ResNet-18) + HF example SO-101 dataset (86 MB) | ✅ downloaded |
| Training pipeline | ✅ tested: ACT 52M params trains, ~1.25 steps/s on the M2 (too slow for a full run, use cloud) |
| Running a model on the Mac | ✅ tested: 60 ms per plan of 100 moves (~3 s of motion). Easy. |
| Poses code + multi-spot pick and place | ✅ tested with a fake robot |
| `check.py` hardware check | ✅ ready, needs the robot plugged in |
| HF username in `settings.env` | ✅ `armannnnnnn` |
| USB ports in `settings.env` + `hf auth login` | ❌ you, when the arm is plugged in |
| Bugs found and fixed in review | ✅ leftover-folder crash on re-record, arm dropping on exit/Ctrl+C, follower snapping at teach start, pose-name typos, missing login/calibration checks |
| Real-arm test | ❌ not possible until you plug it in |

## Phases

### Phase 1: plug in and check (~1 h, do this BEFORE the hackathon if you can)
1. `source .venv/bin/activate` (every new terminal)
2. `lerobot-find-port` for both arms, then `lerobot-find-cameras opencv`, then fill in `settings.env`.
3. `hf auth login` (write token from huggingface.co/settings/tokens)
4. Calibrate both arms (commands in README section 0).
5. `python check.py` until everything says OK.
6. Teleop test: move the leader, the follower copies it.
   ```bash
   lerobot-teleoperate --robot.type=so101_follower --robot.port=PORT --robot.id=hack_follower --teleop.type=so101_leader --teleop.port=PORT --teleop.id=hack_leader
   ```

### Phase 2: guaranteed demo with poses (~30 min)
1. Tape 3 to 4 numbered spots plus the cup spot.
2. `python poses.py teach`: teach `home`, `above_cup`, and `above_N` / `grasp_N` for each spot.
3. `python pick_place.py all`, run it 10 times and write down how many pieces land in the cup.
4. From here on you **always have a working demo**. Everything else is upside.

### Phase 3: record demos for the learned model (~1.5 to 2 h)
- 1 to 3 pieces on the table, cup fixed. Each episode = pick **one** piece and drop it in the cup.
- **Same rule every time: always take the piece closest to the cup.** Consistency matters more than anything. Then running the model again and again clears the table.
- Vary piece positions across ~5 areas, ~10 episodes each.
- `scripts/record.sh 10`, five times (it adds to the dataset each time).

### Phase 4: train in the cloud (~1 to 3 h, the Mac stays cool)
- `scripts/train_act.sh cloud` (HF Jobs: your account has no credit yet, add some at huggingface.co/settings/billing) or Colab (free).
- While it trains: practise the pitch.

### Phase 5: test the learned model
- `scripts/run_policy.sh 30`. Count successes out of 20 attempts. That number goes on the slide.


## Bring to the hackathon
- Masking tape + marker (spots), 6+ chess pieces (tall ones grip best), a **wide** cup (easy drop)
- USB hub, both power supplies, a camera clamp or mount, a spare USB cable
- Optional: rubber or TPU gripper tips, a plain table cloth (better for the camera)

## Risks and fallbacks
| Risk | Fallback |
|---|---|
| Learned model is unreliable | Demo with poses mode, show ACT as "work in progress" with honest numbers |
| Cloud training slow or fails | Train a shorter run (10k steps) or show poses mode |
| Gripper drops pieces | Re-teach `grasp_N` with more squeeze; tall pieces; tips |
| Camera moved | Re-tape the camera; never touch it after recording |
| macOS blocks the camera | System Settings > Privacy > Camera: allow your terminal app |
