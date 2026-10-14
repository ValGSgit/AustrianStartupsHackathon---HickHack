"""Follower arm access for the shell game: connect (real or fake), smooth moves, named poses.

Uses the same port, id and calibration as robot.py (config.toml + config.local.toml, calibration/ in the repo),
so poses taught here stay valid for teleop/record and the other way round.

    arm = connect(fake=True)          # no hardware: prints nothing, tracks the commanded pose
    go_to(arm, poses["home"], 1.5)
    release(arm)                      # goes home slowly, then torque off
"""

import json
import logging
import sys
import time
from pathlib import Path

# LeRobot logs a multi-line warning on every step that max_relative_target slows down. Harmless, but it buries
# the instructions on screen.
logging.getLogger().addFilter(lambda r: "Relative goal position magnitude" not in str(r.msg))

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot import CALIB_DIR, CONFIG  # noqa: E402

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
KEYS = [f"{j}.pos" for j in JOINTS]
POSES_FILE = Path(__file__).resolve().parent / "poses.json"
FPS = 30
SPEED = 1.0  # global multiplier: 2.0 = twice as fast. Set by the CLIs (--speed).


class FakeArm:
    """Stands in for the follower: reports the last commanded pose as its position."""

    def __init__(self):
        self.state = {k: 0.0 for k in KEYS}
        self.commands = 0
        self.is_connected = True

    def get_observation(self) -> dict:
        return dict(self.state)

    def send_action(self, action: dict) -> dict:
        self.state.update({k: float(v) for k, v in action.items() if k in self.state})
        self.commands += 1
        return action

    def disconnect(self):
        self.is_connected = False


def connect(fake: bool = False, max_step: float | None = 10.0):
    """Real follower (torque on, holds its pose) or FakeArm. max_step = max degrees per command (safety)."""
    if fake:
        return FakeArm()
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    f = CONFIG["follower"]
    arm = SO101Follower(SO101FollowerConfig(
        port=f["port"], id=f["id"], calibration_dir=CALIB_DIR,
        max_relative_target=max_step, disable_torque_on_disconnect=True))
    arm.connect(calibrate=False)
    if not arm.is_calibrated:
        sys.exit(f"follower is not calibrated (no calibration/{f['id']}.json)")
    return arm


def read_pose(arm) -> dict[str, float]:
    obs = arm.get_observation()
    return {k: float(obs[k]) for k in KEYS}


def with_gripper(pose: dict, gripper: float) -> dict:
    return {**pose, "gripper.pos": gripper}


def go_to(arm, target: dict, seconds: float = 1.5) -> None:
    """Smooth (ease in/out) joint-space move from the current pose to `target`. Duration is divided by SPEED."""
    start = read_pose(arm)
    target = {k: float(target[k]) for k in KEYS if k in target}
    seconds = seconds / SPEED
    steps = max(1, int(seconds * FPS))
    fake = isinstance(arm, FakeArm)
    for i in range(1, steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)
        arm.send_action({k: start[k] + (target[k] - start[k]) * t for k in target})
        if not fake:
            time.sleep(1 / FPS)
    if not fake:
        time.sleep(0.15)  # let the servos settle


def load_poses() -> dict[str, dict]:
    return json.loads(POSES_FILE.read_text(encoding="utf-8")) if POSES_FILE.exists() else {}


def save_pose(name: str, pose: dict) -> None:
    poses = load_poses()
    poses[name] = {k: round(float(pose[k]), 2) for k in KEYS}
    POSES_FILE.write_text(json.dumps(poses, indent=2), encoding="utf-8")


def release(arm) -> None:
    """Go slowly to `home` (if taught), then torque off. Use instead of arm.disconnect()."""
    try:
        home = load_poses().get("home")
        if home:
            go_to(arm, home, seconds=2.0 * SPEED)  # always slow, whatever SPEED is
        elif not isinstance(arm, FakeArm):
            print("No 'home' pose taught: the arm goes limp NOW, hold it.", flush=True)
    finally:
        arm.disconnect()
