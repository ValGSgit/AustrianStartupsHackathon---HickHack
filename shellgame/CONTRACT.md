# Shell game: module contract

The robot plays the cup game: show the ball, shuffle 3 cups between 4 taped slots by **sliding** them (the ball
slides along inside), the judge guesses, the robot lifts the right cup, the camera confirms the ball.

Run everything with `.venv\Scripts\python shellgame\<script>.py` from the repo root. All scripts take `--fake`
(no arm: `arm.FakeArm`). Python 3.12, LeRobot 0.6.1, OpenCV GUI build, numpy. No new dependencies.

## Table layout

* Slots 1..4 on an arc around the robot base, same distance from it (~18-20 cm), left to right as the camera sees them.
* 3 identical upside-down cups, 1 slot always empty. A soft bright ball (pom-pom) under one cup.

## Poses (`shellgame/poses.json`, written by `teach.py`, read via `arm.load_poses()`)

Units are LeRobot degrees (`<joint>.pos`, gripper 0..100). For each slot `i` in 1..4:

| name | meaning | gripper |
|---|---|---|
| `above_i` | jaws open, ~4 cm above the top of a cup standing in slot i | open (this value = "open") |
| `grip_i`  | jaws around the LOWER part of the cup in slot i, touching the table side | closed on the cup (this value = "closed") |
| `in_i`    | same cup pulled straight toward the base by ~1.5 cup diameters, still on the table | closed |
| `lift_i`  | cup in slot i lifted straight up ~8 cm (reveal) | closed |

Plus `home` (resting, safe to go limp) and `ready` (high, out of the camera's view of the slots, between moves).

## Motion primitives (`shellgame/motions.py`)

```python
move_cup(arm, poses, src: int, dst: int) -> None
    # ready -> above_src(open) -> grip_src(open) -> grip_src(closed) -> in_src -> in_dst -> grip_dst(closed)
    # -> grip_dst(open) -> above_dst(open). Inner lane (in_*) passes inside the other cups.
reveal(arm, poses, slot: int, hold_s: float = 1.5) -> None   # above -> grip -> close -> lift, hold, put back, open, above
show_ball(arm, poses, slot: int) -> None                     # same as reveal, used at the start
nudge(arm, poses, slot: int) -> None                          # re-centre a cup: grip it and put it down in its slot
go_ready(arm, poses) -> None
```
Durations are base seconds; `arm.go_to` divides by `arm.SPEED`.

## Game logic (`shellgame/game.py`, pure, no hardware)

```python
@dataclass
class State:
    cups: list[int | None]   # index 0..3 = slot 1..4; value = cup id 0..2 or None (empty slot)
    ball_cup: int            # cup id hiding the ball
def new_state(ball_slot: int = 2) -> State   # cups in slots 1,2,3; slot 4 empty
def empty_slot(s) -> int                     # 1-based
def ball_slot(s) -> int                      # 1-based
def legal_moves(s) -> list[tuple[int, int]]  # (src, dst), dst is always the empty slot
def apply(s, move) -> State                  # returns a new State
def random_moves(s, n, rng) -> list[tuple[int, int]]  # never undoes the previous move; ball cup moves >= n//3 times
```

## Vision (`shellgame/vision.py`, deterministic HSV colour, no AI)

Setup file `shellgame/vision_setup.json`:
```json
{"slots": [[x, y, r], ...4], "cup_hsv": [[lo], [hi]], "ball_hsv": [[lo], [hi]],
 "min_cup_px": 400, "min_ball_px": 80, "max_offset_frac": 0.5, "frame_size": [w, h]}
```
```python
slot_states(frame, setup) -> list[dict]   # per slot: {"occupied": bool, "px": int, "offset": (dx, dy) px, "off_center": bool}
ball_visible(frame, setup, slot: int | None = None) -> tuple[bool, dict]   # near that slot (within ~2 r) or anywhere
check_layout(frame, setup, expected_empty: int) -> tuple[bool, list[str]]  # ok + human-readable problems
annotate(frame, setup, states, ball=None) -> np.ndarray
```
Camera: reuse `selfcheck.camera_cfg()`, `selfcheck.open_camera()`, `selfcheck.grab_frame()` (repo root).

## Show (`shellgame/show.py`)

Big status window (reuse `stage.screen`, `stage.show`, `stage.say` from the repo root) and the flow:
SPACE start -> show ball -> shuffle N random moves (camera check after each, nudge on a problem) -> "Which cup?"
(keys 1-4) -> reveal the ball's slot -> camera verifies the ball -> VERIFIED / MISSED screen. ESC = stop (arm holds), q = quit.
Flags: `--fake`, `--no-vision`, `--moves 7`, `--speed 1.0`, `--seed`, `--fullscreen`. Each game is logged to
`evals/shell_log.csv` (time, moves, speed, ball_slot, guess, verified, problems).

## Camera-guided grabbing (v2: "the robot sees")

The camera finds where each cup really is; the arm pose for a cup at image point (u, v) is predicted from taught samples.
Slots stay as table destinations (pixel centres clicked in `vision.py setup`); cups are grabbed where they are seen.

### Samples (`shellgame/reach_samples.json`, written by `teach.py sample`)
```json
[{"uv": [u, v], "frame_size": [w, h], "poses": {"above": {...6 joints}, "grip": {...}, "in": {...}, "lift": {...}}}, ...]
```
One sample = one cup alone on the table, its centroid seen by the camera (arm at `ready`, out of view), then the 4 poses
taught with the leader for that cup (same meaning as above_i/grip_i/in_i/lift_i). 6-8 samples spread over the cup area.

### `shellgame/reach.py`
```python
KINDS = ("above", "grip", "in", "lift")
class ReachModel:
    @classmethod
    def fit(cls, samples: list[dict]) -> "ReachModel"   # per kind, per joint: least squares on features of (u, v)
        # >= 6 samples: [1, u, v, uv, u^2, v^2] with small ridge; 3-5: [1, u, v]; < 3: ValueError. u, v normalised to 0..1.
        # gripper: not regressed, mean of the samples' value for that kind.
    def predict(self, u: float, v: float) -> dict[str, dict[str, float]]   # {"above": pose, "grip": pose, ...}
    def loo_errors(self) -> dict[str, float]           # leave-one-out mean abs error (deg) per kind, for `check`
def load_model(path=SAMPLES_FILE) -> ReachModel        # FileNotFoundError / ValueError with a clear message
```
CLI: `reach.py check` (samples count, LOO error per kind/joint, warns if any > 5 deg, prints the u,v coverage box),
`reach.py go U V [--kind above]` (moves slowly there, Enter between steps), `reach.py go-cup` (camera finds the single
cup, robot goes ready -> above -> grip(open), Enter between steps: the "can it find a cup" test).

### Vision additions (`shellgame/vision.py`)
```python
find_cups(frame, setup) -> list[dict]   # cup-colour blobs: {"uv": (u, v), "px": int, "bbox": (x, y, w, h)}, largest first,
                                        # area >= min_cup_px, split nothing (cups must not touch)
nearest_cup(cups, uv, max_dist) -> dict | None
```

### Seen motions (`shellgame/motions.py`)
```python
move_cup_seen(arm, model, src_uv, dst_uv) -> None    # same sequence as move_cup with model.predict(src) / predict(dst)
reveal_seen(arm, model, uv, hold_s=1.5) -> None
```
show.py / shuffle.py get `--seen`: before each move (arm at ready) grab a frame, find the cup nearest the source slot
centre (within 1.5 r) and grab it there; place it at the destination slot centre. The screen draws the detected cups and
the planned source -> destination arrow.
