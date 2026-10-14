"""Shell game motion primitives built from taught poses (see CONTRACT.md for the pose names).

Gripper "open" = the gripper value of above_i, "closed" = the gripper value of grip_i. Durations below are base
seconds; arm.go_to divides them by arm.SPEED.

    poses = arm.load_poses()
    move_cup(a, poses, 1, 4)   # slide the cup in slot 1 into the empty slot 4 via the inner lane
"""

import time

import arm as A

SLOTS = (1, 2, 3, 4)

# base durations (s)
T_TRAVEL = 1.0   # ready <-> above
T_DOWN = 1.2     # above -> grip, slow onto the cup
T_CLOSE = 0.7    # gripper close, slow so the cup is centred, not knocked
T_OPEN = 0.5
T_PULL = 1.0     # grip <-> in (sliding on the table)
T_LANE = 1.2     # in_src -> in_dst
T_UP = 0.8       # grip -> above
T_LIFT = 1.0     # grip <-> lift


def required_poses(slots=SLOTS) -> list[str]:
    return ["home", "ready"] + [f"{p}_{i}" for i in slots for p in ("above", "grip", "in", "lift")]


def missing_poses(poses: dict, slots=SLOTS) -> list[str]:
    return [n for n in required_poses(slots) if n not in poses]


def _need(poses: dict, names: list[str]) -> None:
    missing = [n for n in names if n not in poses]
    if missing:
        raise KeyError(f"poses not taught: {', '.join(missing)} (run shellgame/teach.py)")


def _open(poses, i):
    return poses[f"above_{i}"]["gripper.pos"]


def _closed(poses, i):
    return poses[f"grip_{i}"]["gripper.pos"]


def _hold(arm, seconds):
    if not isinstance(arm, A.FakeArm):
        time.sleep(seconds / A.SPEED)


def go_ready(arm, poses) -> None:
    _need(poses, ["ready"])
    A.go_to(arm, poses["ready"], T_TRAVEL)


def _grab(arm, poses, i):
    """Current pose -> above_i -> grip_i (open) -> grip_i (closed)."""
    A.go_to(arm, A.with_gripper(poses[f"above_{i}"], _open(poses, i)), T_TRAVEL)
    A.go_to(arm, A.with_gripper(poses[f"grip_{i}"], _open(poses, i)), T_DOWN)
    A.go_to(arm, A.with_gripper(poses[f"grip_{i}"], _closed(poses, i)), T_CLOSE)


def _let_go(arm, poses, i):
    """grip_i (closed) -> grip_i (open) -> above_i (open)."""
    A.go_to(arm, A.with_gripper(poses[f"grip_{i}"], _open(poses, i)), T_OPEN)
    A.go_to(arm, A.with_gripper(poses[f"above_{i}"], _open(poses, i)), T_UP)


def move_cup(arm, poses, src: int, dst: int) -> None:
    """ready -> grab src -> in_src -> in_dst -> grip_dst -> let go -> above_dst. The cup slides on the table."""
    _need(poses, ["ready"] + [f"{p}_{i}" for i in (src, dst) for p in ("above", "grip", "in")])
    closed = _closed(poses, src)  # keep the grip used on this cup for the whole slide
    A.go_to(arm, poses["ready"], T_TRAVEL)
    _grab(arm, poses, src)
    A.go_to(arm, A.with_gripper(poses[f"in_{src}"], closed), T_PULL)
    A.go_to(arm, A.with_gripper(poses[f"in_{dst}"], closed), T_LANE)
    A.go_to(arm, A.with_gripper(poses[f"grip_{dst}"], closed), T_PULL)
    _let_go(arm, poses, dst)


def reveal(arm, poses, slot: int, hold_s: float = 1.5) -> None:
    """above -> grip -> close -> lift, hold, put back, open, above."""
    _need(poses, [f"{p}_{slot}" for p in ("above", "grip", "lift")])
    closed = _closed(poses, slot)
    _grab(arm, poses, slot)
    A.go_to(arm, A.with_gripper(poses[f"lift_{slot}"], closed), T_LIFT)
    _hold(arm, hold_s)
    A.go_to(arm, A.with_gripper(poses[f"grip_{slot}"], closed), T_LIFT)
    _let_go(arm, poses, slot)


def show_ball(arm, poses, slot: int) -> None:
    reveal(arm, poses, slot)


def nudge(arm, poses, slot: int) -> None:
    """Re-centre the cup in `slot`: closing the jaws at grip_i pulls it to the taught spot, then let go."""
    _need(poses, [f"{p}_{slot}" for p in ("above", "grip")])
    _grab(arm, poses, slot)
    _let_go(arm, poses, slot)


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
