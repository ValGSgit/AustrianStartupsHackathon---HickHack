"""Teach the shell-game poses (shellgame/poses.json) and check them.

    .venv\\Scripts\\python shellgame\\teach.py teach                       # move the LEADER, the follower copies it
    .venv\\Scripts\\python shellgame\\teach.py teach --only above_2,grip_2  # re-teach just these
    .venv\\Scripts\\python shellgame\\teach.py teach --slots 1,2            # only slots 1 and 2 (plus home, ready)
    .venv\\Scripts\\python shellgame\\teach.py teach --hand                 # no leader: follower limp, move it by hand
    .venv\\Scripts\\python shellgame\\teach.py list                         # which poses are taught / missing
    .venv\\Scripts\\python shellgame\\teach.py go above_3 --seconds 2       # move there smoothly
    .venv\\Scripts\\python shellgame\\teach.py test-slot 2                  # above -> grip -> in -> grip -> above, Enter between

While teaching: Enter = save the FOLLOWER's pose for this step, b + Enter = back, s + Enter = skip, q + Enter = quit.
Every command ends by going slowly to `home` and switching the motors off (also on Ctrl+C).
All commands take --fake (no hardware).
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm as A  # noqa: E402

MAX_START_DIFF = 30.0   # degrees: refuse to start copying when leader and follower differ more than this
BLOCKED_DIFF = 25.0     # degrees between command and position that count as "not following"
BLOCKED_SECONDS = 1.0   # ...for this long: something blocks the arm, stop before a motor overloads
FAKE_POSES = A.POSES_FILE.with_name("poses_fake.json")

HINTS = {
    "home": "RESTING pose: arm folded down, safe to go limp.",
    "ready": "READY pose: arm high and OUT of the camera's view of the 4 slots (waits here between moves).",
    "above": "jaws OPEN, about 4 cm above the top of the cup in slot {i}.",
    "grip": "jaws around the LOW part of the cup in slot {i}, CLOSED, squeeze a little.",
    "in": "pull the cup in slot {i} about 1.5 cup widths toward the base, keep it ON the table (still closed).",
    "lift": "lift the cup in slot {i} straight up about 8 cm (still closed).",
}


def required(slots=(1, 2, 3, 4)) -> list[str]:
    return ["home", "ready"] + [f"{k}_{i}" for i in slots for k in ("above", "grip", "in", "lift")]


def hint(name: str) -> str:
    base, _, i = name.partition("_")
    return HINTS.get(base, HINTS.get(name, "")).format(i=i)


def gap(key: str, a: float, b: float) -> float:
    """Degrees between two readings; wrist_roll wraps around at +-180."""
    d = a - b
    return abs((d + 180) % 360 - 180) if key == "wrist_roll.pos" else abs(d)


class FakeLeader:
    """Returns a fixed pose (home if taught, else zeros)."""

    def __init__(self):
        self.pose = A.load_poses().get("home") or {k: 0.0 for k in A.KEYS}

    def get_action(self):
        return dict(self.pose)

    def disconnect(self):
        pass


def connect_leader(fake: bool):
    if fake:
        return FakeLeader()
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    l = A.CONFIG["leader"]
    leader = SO101Leader(SO101LeaderConfig(port=l["port"], id=l["id"], calibration_dir=A.CALIB_DIR))
    leader.connect(calibrate=False)
    if not leader.is_calibrated:
        sys.exit(f"leader is not calibrated (no calibration/{l['id']}.json)")
    return leader


def torque(arm, on: bool) -> None:
    if isinstance(arm, A.FakeArm):
        return
    if on:  # hold where the arm is now, not an old goal, then switch the motors on
        arm.bus.sync_write("Goal_Position", arm.bus.sync_read("Present_Position"))
        arm.bus.enable_torque()
    else:
        arm.bus.disable_torque()


def match_start_pose(arm, leader, typed: queue.Queue) -> bool:
    """Don't start copying while the arms are far apart: the follower would swing through the table."""
    while True:
        want, have = leader.get_action(), A.read_pose(arm)
        far = [k for k in A.KEYS if k != "gripper.pos" and gap(k, want[k], have[k]) > MAX_START_DIFF]
        if not far:
            return True
        print("\nLeader and follower are in different poses:")
        for k in far:
            print(f"  {k[:-4]:14s} leader {want[k]:7.1f}   follower {have[k]:7.1f}")
        torque(arm, False)
        print("Follower motors are OFF. Put both arms in the same pose (easiest: both resting), then Enter. (q = quit)")
        if typed.get() == "q":
            return False
        torque(arm, True)


def stdin_lines() -> queue.Queue:
    """Typed lines arrive here while the arm keeps following at 30 Hz. End of input counts as q."""
    q: queue.Queue = queue.Queue()

    def read():
        for line in sys.stdin:
            q.put(line.strip().lower())
        q.put("q")
    threading.Thread(target=read, daemon=True).start()
    return q


def cmd_teach(a):
    slots = [int(s) for s in a.slots.split(",")]
    steps = [s.strip() for s in a.only.split(",")] if a.only else required(slots)
    arm = A.connect(fake=a.fake)
    leader = None
    typed = stdin_lines()
    try:
        if a.hand:
            torque(arm, False)
            print("\nHAND mode: follower motors are OFF. Move the follower by hand, Enter saves its pose.")
        else:
            leader = connect_leader(a.fake)
            if not match_start_pose(arm, leader, typed):
                return
            A.go_to(arm, leader.get_action(), 2.0)  # ease to the leader instead of snapping there
            print("\nMove the LEADER and watch the FOLLOWER. Enter saves the follower's pose.")
        print("(b + Enter = back, s + Enter = skip, q + Enter = quit)")
        i, shown, blocked_since = 0, -1, None
        while i < len(steps):
            t0 = time.perf_counter()
            if shown != i:
                taught = " (already taught, Enter overwrites)" if steps[i] in A.load_poses() else ""
                print(f"\nSTEP {i + 1}/{len(steps)}  {steps[i]}: {hint(steps[i])}{taught}", flush=True)
                shown = i
            cmd = None
            if leader:
                cmd = leader.get_action()
                try:
                    arm.send_action({k: cmd[k] for k in A.KEYS})
                    have = A.read_pose(arm)
                except (RuntimeError, ConnectionError) as e:
                    print(f"\nA motor reported a problem ({e}). Stopping - something may block the arm.")
                    break
                stuck = [k for k in A.KEYS if k != "gripper.pos" and gap(k, cmd[k], have[k]) > BLOCKED_DIFF]
                blocked_since = None if not stuck else (blocked_since or time.time())
                if blocked_since and time.time() - blocked_since > BLOCKED_SECONDS:
                    print(f"\nThe follower cannot reach the leader's pose ({', '.join(k[:-4] for k in stuck)}): "
                          "something blocks it. Stopping to protect the motors.")
                    break
            try:
                line = typed.get_nowait()
            except queue.Empty:
                line = None
            if line == "q":
                break
            if line == "b":
                i = max(0, i - 1)
            elif line == "s":
                print(f"   skipped {steps[i]}")
                i += 1
            elif line is not None:
                pose = A.read_pose(arm)
                if cmd and steps[i].startswith("grip_"):
                    # keep the leader's squeeze: the follower's gripper is stopped by the cup, the command is not
                    pose["gripper.pos"] = cmd["gripper.pos"]
                A.save_pose(steps[i], pose)
                print(f"   saved {steps[i]}: " + " ".join(f"{k[:-4]}={v:.1f}" for k, v in pose.items()))
                i += 1
            time.sleep(max(0.0, 1 / A.FPS - (time.perf_counter() - t0)))
        missing = [n for n in required() if n not in A.load_poses()]
        print("\nDone." + (f" Still missing: {', '.join(missing)}" if missing else " All poses taught."))
    finally:
        if leader:
            leader.disconnect()
        if a.hand:
            print("Hand mode: motors stay OFF - hold the arm.")
            arm.disconnect()
        else:
            A.release(arm)


def cmd_list(a):
    poses = A.load_poses()
    print(f"{'pose':10s} " + " ".join(f"{k[:-4][:12]:>12}" for k in A.KEYS))
    for name in required() + [n for n in poses if n not in required()]:
        p = poses.get(name)
        print(f"{name:10s} " + (" ".join(f"{p[k]:12.1f}" for k in A.KEYS) if p else "   -- MISSING --"))
    missing = [n for n in required() if n not in poses]
    print(f"\n{len(required()) - len(missing)}/{len(required())} required poses taught"
          + (f". Missing: {', '.join(missing)}" if missing else "."))


def need(poses: dict, names: list[str]) -> None:
    missing = [n for n in names if n not in poses]
    if missing:
        sys.exit(f"Not taught yet: {', '.join(missing)}. Run: teach.py teach --only {','.join(missing)}")


def wait_enter(msg: str) -> None:
    print(msg, flush=True)
    try:
        if input().strip().lower() == "q":
            raise KeyboardInterrupt
    except EOFError:
        pass


def cmd_go(a):
    poses = A.load_poses()
    need(poses, [a.name])
    arm = A.connect(fake=a.fake)
    try:
        A.go_to(arm, poses[a.name], a.seconds)
        print(f"At {a.name}: " + " ".join(f"{k[:-4]}={v:.1f}" for k, v in A.read_pose(arm).items()))
        wait_enter("Holding. Enter = go home and release.")
    except KeyboardInterrupt:
        pass
    finally:
        A.release(arm)


def cmd_test_slot(a):
    i, poses = a.slot, A.load_poses()
    need(poses, [f"above_{i}", f"grip_{i}", f"in_{i}", "ready"])
    open_g = poses[f"above_{i}"]["gripper.pos"]
    closed = poses[f"grip_{i}"]["gripper.pos"]
    plan = [
        ("ready", poses["ready"]),
        (f"above_{i} (open)", poses[f"above_{i}"]),
        (f"grip_{i} (open)", A.with_gripper(poses[f"grip_{i}"], open_g)),
        (f"grip_{i} (close)", poses[f"grip_{i}"]),
        (f"in_{i} (closed)", A.with_gripper(poses[f"in_{i}"], closed)),
        (f"grip_{i} (back, closed)", poses[f"grip_{i}"]),
        (f"grip_{i} (open)", A.with_gripper(poses[f"grip_{i}"], open_g)),
        (f"above_{i} (open)", poses[f"above_{i}"]),
        ("ready", poses["ready"]),
    ]
    arm = A.connect(fake=a.fake)
    try:
        for n, (label, pose) in enumerate(plan, 1):
            wait_enter(f"[{n}/{len(plan)}] Enter = go to {label}   (q + Enter = stop)")
            A.go_to(arm, pose, a.seconds)
        print("Slot test done.")
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        A.release(arm)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fake", action="store_true", help="no hardware")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("teach", help="guided teaching with the leader (or --hand)")
    s.add_argument("--hand", action="store_true", help="no leader: follower torque off, move it by hand")
    s.add_argument("--only", help="comma-separated pose names to (re)teach, e.g. above_2,grip_2")
    s.add_argument("--slots", default="1,2,3,4")
    s.set_defaults(fn=cmd_teach)
    sub.add_parser("list", help="taught / missing poses").set_defaults(fn=cmd_list)
    s = sub.add_parser("go", help="move smoothly to a pose")
    s.add_argument("name")
    s.add_argument("--seconds", type=float, default=2.0)
    s.set_defaults(fn=cmd_go)
    s = sub.add_parser("test-slot", help="above -> grip -> in -> grip -> above, slowly, Enter between steps")
    s.add_argument("slot", type=int, choices=[1, 2, 3, 4])
    s.add_argument("--seconds", type=float, default=2.5)
    s.set_defaults(fn=cmd_test_slot)
    for s in sub.choices.values():  # --fake also accepted after the sub-command
        s.add_argument("--fake", action="store_true", default=argparse.SUPPRESS)
    a = p.parse_args()
    if a.fake and (a.cmd == "teach" or not A.POSES_FILE.exists()):
        A.POSES_FILE = FAKE_POSES  # fake teaching never overwrites the real poses.json
        print(f"(fake: using {FAKE_POSES.name})")
    try:
        a.fn(a)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
