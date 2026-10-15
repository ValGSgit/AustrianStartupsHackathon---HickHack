"""Shell game motion primitives built from taught poses (see CONTRACT.md for the pose names).

Gripper "open" = the gripper value of above_i, "closed" = the gripper value of grip_i. Durations below are base
seconds; arm.go_to divides them by arm.SPEED.

    poses = arm.load_poses()
    move_cup(a, poses, 1, 4)   # slide the cup in slot 1 into the empty slot 4 via the inner lane
"""

import time

import arm as A

SLOTS = (1, 2, 3, 4)
KINDS = ("above", "grip", "in", "lift")

# base durations (s)
T_TRAVEL = 1.0   # ready <-> above
T_DOWN = 1.2     # above -> grip, slow onto the cup
T_CLOSE = 0.7    # gripper close, slow so the cup is centred, not knocked
T_OPEN = 0.5
T_PULL = 1.0     # grip <-> in (sliding on the table)
T_LANE = 1.2     # in_src -> in_dst
T_UP = 0.8       # grip -> above
T_LIFT = 1.0     # grip <-> lift
SQUEEZE = 6.0    # gripper units past the taught grip: a just-touching grip lets the cup slip while sliding.
                 # Safe: LeRobot limits the SO-101 gripper torque/current.


def held(p) -> float:
    """Gripper value used while holding the cup: the taught grip, squeezed a bit more (0 = fully closed)."""
    return max(0.0, p["grip"]["gripper.pos"] - SQUEEZE)


def required_poses(slots=SLOTS) -> list[str]:
    return ["home", "ready"] + [f"{p}_{i}" for i in slots for p in ("above", "grip", "in", "lift")]


def missing_poses(poses: dict, slots=SLOTS) -> list[str]:
    return [n for n in required_poses(slots) if n not in poses]


def _need(poses: dict, names: list[str]) -> None:
    missing = [n for n in names if n not in poses]
    if missing:
        raise KeyError(f"poses not taught: {', '.join(missing)} (run shellgame/teach.py)")


def _hold(arm, seconds):
    if not isinstance(arm, A.FakeArm):
        time.sleep(seconds / A.SPEED)


def go_ready(arm, poses) -> None:
    _need(poses, ["ready"])
    A.go_to(arm, poses["ready"], T_TRAVEL)


def slot_poses(poses: dict, i: int) -> dict[str, dict]:
    """{"above": above_i, "grip": grip_i, ...}: the same shape as reach.ReachModel.predict()."""
    return {k: poses[f"{k}_{i}"] for k in KINDS if f"{k}_{i}" in poses}


def _grab(arm, p):
    """Current pose -> above -> grip (open) -> grip (closed). p = slot_poses() / model.predict()."""
    o, c = p["above"]["gripper.pos"], held(p)
    A.go_to(arm, A.with_gripper(p["above"], o), T_TRAVEL)
    A.go_to(arm, A.with_gripper(p["grip"], o), T_DOWN)
    A.go_to(arm, A.with_gripper(p["grip"], c), T_CLOSE)


def _let_go(arm, p):
    """grip (closed) -> grip (open) -> above (open)."""
    o = p["above"]["gripper.pos"]
    A.go_to(arm, A.with_gripper(p["grip"], o), T_OPEN)
    A.go_to(arm, A.with_gripper(p["above"], o), T_UP)


def _move(arm, src, dst, ready=None):
    """[ready ->] grab src -> in_src -> in_dst -> grip_dst -> let go -> above_dst. The cup slides on the table."""
    closed = held(src)  # keep the grip used on this cup for the whole slide
    if ready:
        A.go_to(arm, ready, T_TRAVEL)
    _grab(arm, src)
    A.go_to(arm, A.with_gripper(src["in"], closed), T_PULL)
    A.go_to(arm, A.with_gripper(dst["in"], closed), T_LANE)
    A.go_to(arm, A.with_gripper(dst["grip"], closed), T_PULL)
    _let_go(arm, dst)


def _reveal(arm, p, hold_s):
    """above -> grip -> close -> lift, hold, put back, open, above."""
    closed = held(p)
    _grab(arm, p)
    A.go_to(arm, A.with_gripper(p["lift"], closed), T_LIFT)
    _hold(arm, hold_s)
    A.go_to(arm, A.with_gripper(p["grip"], closed), T_LIFT)
    _let_go(arm, p)


def move_cup(arm, poses, src: int, dst: int) -> None:
    """ready -> grab src -> in_src -> in_dst -> grip_dst -> let go -> above_dst. The cup slides on the table."""
    _need(poses, ["ready"] + [f"{p}_{i}" for i in (src, dst) for p in ("above", "grip", "in")])
    _move(arm, slot_poses(poses, src), slot_poses(poses, dst), poses["ready"])


def move_cup_seen(arm, model, src_uv, dst_uv, frame_size=None) -> None:
    """Same slide as move_cup, poses predicted for the cup seen at src_uv and the target point dst_uv (pixels).
    Call it with the arm at ready (the camera frame is taken there)."""
    _move(arm, model.predict(*src_uv, frame_size=frame_size), model.predict(*dst_uv, frame_size=frame_size))


def reveal(arm, poses, slot: int, hold_s: float = 1.5) -> None:
    """above -> grip -> close -> lift, hold, put back, open, above."""
    _need(poses, [f"{p}_{slot}" for p in ("above", "grip", "lift")])
    _reveal(arm, slot_poses(poses, slot), hold_s)


def reveal_seen(arm, model, uv, hold_s: float = 1.5, frame_size=None) -> None:
    _reveal(arm, model.predict(*uv, frame_size=frame_size), hold_s)


def show_ball(arm, poses, slot: int) -> None:
    reveal(arm, poses, slot)


def nudge(arm, poses, slot: int) -> None:
    """Re-centre the cup in `slot`: closing the jaws at grip_i pulls it to the taught spot, then let go."""
    _need(poses, [f"{p}_{slot}" for p in ("above", "grip")])
    _grab(arm, slot_poses(poses, slot))
    _let_go(arm, slot_poses(poses, slot))


def fake_poses() -> dict[str, dict]:
    """Plausible poses for --fake runs without a poses.json (slots on an arc: pan -30..30 deg)."""
    def p(pan, lift, elbow, wrist, grip):
        return dict(zip(A.KEYS, (pan, lift, elbow, wrist, 0.0, grip)))

    poses = {"home": p(0, -100, 95, 60, 2), "ready": p(0, -60, 40, 70, 30)}
    for i, pan in zip(SLOTS, (-30, -10, 10, 30)):
        poses[f"above_{i}"] = p(pan, -15, 20, 75, 40)
        poses[f"grip_{i}"] = p(pan, 5, 15, 70, 8)
        poses[f"in_{i}"] = p(pan * 1.2, -10, 50, 60, 8)
        poses[f"lift_{i}"] = p(pan, -10, 15, 70, 8)
    return poses


class FakeModel:
    """--fake stand-in for reach.ReachModel: slot-1 fake poses, base turned by the image x (0..width px = -30..30 deg)."""

    def __init__(self, width: int = 640):
        self.width, self.base = width, fake_poses()

    def predict(self, u: float, v: float, frame_size=None) -> dict[str, dict]:
        turn = -30 + 60 * u / self.width - self.base["above_1"]["shoulder_pan.pos"]
        return {k: {**self.base[f"{k}_1"], "shoulder_pan.pos": self.base[f"{k}_1"]["shoulder_pan.pos"] + turn}
                for k in KINDS}
