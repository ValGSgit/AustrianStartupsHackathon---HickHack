"""Teach the shell-game poses (shellgame/poses.json) and check them.

    .venv\\Scripts\\python shellgame\\teach.py teach                       # move the LEADER, the follower copies it
    .venv\\Scripts\\python shellgame\\teach.py teach --only above_2,grip_2  # re-teach just these
    .venv\\Scripts\\python shellgame\\teach.py teach --slots 1,2            # only slots 1 and 2 (plus home, ready)
    .venv\\Scripts\\python shellgame\\teach.py teach --hand                 # no leader: follower limp, move it by hand
    .venv\\Scripts\\python shellgame\\teach.py list                         # which poses are taught / missing
    .venv\\Scripts\\python shellgame\\teach.py go above_3 --seconds 2       # move there smoothly
    .venv\\Scripts\\python shellgame\\teach.py test-slot 2                  # above -> grip -> in -> grip -> above, Enter between
    .venv\\Scripts\\python shellgame\\teach.py sample [--count 6]          # v2: camera sees ONE cup, teach its 4 poses
    .venv\\Scripts\\python shellgame\\teach.py samples [--drop 3]          # list / delete reach samples

While teaching: Enter = save the FOLLOWER's pose for this step, b + Enter = back, s + Enter = skip, q + Enter = quit.
Every command ends by going slowly to `home` and switching the motors off (also on Ctrl+C).
All commands take --fake (no hardware).
"""

import argparse
import json
import queue
import random
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arm as A  # noqa: E402

MAX_START_DIFF = 30.0   # degrees: refuse to start copying when leader and follower differ more than this
BLOCKED_DIFF = 25.0     # degrees between command and position that count as "not following"
BLOCKED_SECONDS = 1.0   # ...for this long: something blocks the arm, stop before a motor overloads
HAND_SQUEEZE = 5.0      # gripper units (0 = closed) added on top of a hand-taught grip so the jaws hold the cup
FAKE_POSES = A.POSES_FILE.with_name("poses_fake.json")

HINTS = {
    "home": "RESTING pose: arm folded down, safe to go limp.",
    "ready": "READY pose: arm high and OUT of the camera's view of the 4 slots (waits here between moves).",
    "above": "jaws OPEN, about 4 cm above the top of {cup}.",
    "grip": "jaws around the LOW part of {cup}, CLOSED, squeeze a little.",
    "in": "pull {cup} about 1.5 cup widths toward the base, keep it ON the table (still closed).",
    "lift": "lift {cup} straight up about 8 cm (still closed).",
}
KINDS = ("above", "grip", "in", "lift")
SAMPLES_FILE = Path(__file__).resolve().parent / "reach_samples.json"


def required(slots=(1, 2, 3, 4)) -> list[str]:
    return ["home", "ready"] + [f"{k}_{i}" for i in slots for k in KINDS]


def hint(name: str) -> str:
    base, _, i = name.partition("_")
    return HINTS.get(base, HINTS.get(name, "")).format(cup=f"the cup in slot {i}" if i else "this cup")


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


def match_start_pose(arm, leader, typed: queue.Queue, limp: bool = True) -> bool:
    """Don't start copying while the arms are far apart: the follower would swing through the table.
    limp=False: the follower keeps holding its pose (e.g. high at ready), only the leader is moved to match."""
    while True:
        want, have = leader.get_action(), A.read_pose(arm)
        far = [k for k in A.KEYS if k != "gripper.pos" and gap(k, want[k], have[k]) > MAX_START_DIFF]
        if not far:
            return True
        print("\nLeader and follower are in different poses:")
        for k in far:
            print(f"  {k[:-4]:14s} leader {want[k]:7.1f}   follower {have[k]:7.1f}")
        if limp:
            torque(arm, False)
            print("Follower motors are OFF. Put both arms in the same pose (easiest: both resting), then Enter. (q = quit)")
        else:
            print("The follower HOLDS its pose. Move the LEADER into the same pose, then Enter. (q = quit)")
        if typed.get() == "q":
            return False
        if limp:
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


def follow_steps(arm, leader, typed: queue.Queue, steps: list[str], get, put, hand: bool = False) -> bool:
    """Follower copies the leader (or hand mode); Enter stores the follower's pose for the current step via
    put(name, pose); get(name) = stored pose or None. b/s/q = back/skip/quit. True = went through all steps."""
    print("(b + Enter = back, s + Enter = skip, q + Enter = quit)")
    i, shown, blocked_since = 0, -1, None
    while i < len(steps):
        t0 = time.perf_counter()
        if shown != i:
            taught = " (already taught, Enter overwrites)" if get(steps[i]) is not None else ""
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
                return False
            stuck = [k for k in A.KEYS if k != "gripper.pos" and gap(k, cmd[k], have[k]) > BLOCKED_DIFF]
            blocked_since = None if not stuck else (blocked_since or time.time())
            if blocked_since and time.time() - blocked_since > BLOCKED_SECONDS:
                print(f"\nThe follower cannot reach the leader's pose ({', '.join(k[:-4] for k in stuck)}): "
                      "something blocks it. Stopping to protect the motors.")
                return False
        try:
            line = typed.get_nowait()
        except queue.Empty:
            line = None
        if line == "q":
            return False
        if line == "b":
            i = max(0, i - 1)
        elif line == "s":
            print(f"   skipped {steps[i]}")
            i += 1
        elif line is not None:
            pose = A.read_pose(arm)
            kind = steps[i].partition("_")[0]
            if cmd and kind == "grip":
                # keep the leader's squeeze: the follower's gripper is stopped by the cup, the command is not
                pose["gripper.pos"] = cmd["gripper.pos"]
            elif hand and kind == "grip":
                # by hand the jaws only touch the cup; command a bit further closed so they squeeze
                pose["gripper.pos"] = max(0.0, pose["gripper.pos"] - HAND_SQUEEZE)
            put(steps[i], pose)
            print(f"   saved {steps[i]}: " + " ".join(f"{k[:-4]}={v:.1f}" for k, v in pose.items()))
            above_name = steps[i].replace(kind, "above", 1)
            above = get(above_name)
            if kind in ("grip", "in", "lift") and above and abs(pose["shoulder_pan.pos"] - above["shoulder_pan.pos"]) > 12:
                print(f"   CHECK: base rotation differs {pose['shoulder_pan.pos'] - above['shoulder_pan.pos']:+.0f} deg "
                      f"from {above_name}. Same cup = almost the same rotation. Type b + Enter to redo.")
            i += 1
        time.sleep(max(0.0, 1 / A.FPS - (time.perf_counter() - t0)))
    return True


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
        follow_steps(arm, leader, typed, steps, lambda n: A.load_poses().get(n), A.save_pose, a.hand)
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


# ---------------------------------------------------------------- v2: reach samples (camera-guided grabbing)

def load_samples() -> list[dict]:
    return json.loads(SAMPLES_FILE.read_text(encoding="utf-8")) if SAMPLES_FILE.exists() else []


def save_samples(samples: list[dict]) -> None:
    SAMPLES_FILE.write_text(json.dumps(samples, indent=1), encoding="utf-8")


def see_one_cup(a, setup, typed: queue.Queue):
    """Camera (or --image) until exactly one cup is seen. Returns (uv, frame_size) or None (q)."""
    import vision as V
    while True:
        if a.fake and not a.image:  # no camera at all: pretend a cup somewhere on the table
            return (random.uniform(150, 490), random.uniform(250, 400)), [640, 480]
        if a.image:
            frame = cv2.imread(a.image)
            if frame is None:
                sys.exit(f"Cannot read image {a.image}")
        else:
            import selfcheck
            c = selfcheck.camera_cfg()
            frame = selfcheck.grab_frame(c["index"], c["width"], c["height"])
        cups = V.find_cups(frame, setup)
        if len(cups) == 1:
            break
        print(f"\nThe camera sees {len(cups)} cups" + "".join(
            f"\n  cup at ({c['uv'][0]:.0f}, {c['uv'][1]:.0f}), {c['px']} px" for c in cups))
        print("Put exactly ONE cup on the table (nothing else cup-coloured in view), then Enter = look again. (q = quit)")
        if typed.get() == "q":
            return None
    uv, size = tuple(round(x, 1) for x in cups[0]["uv"]), [frame.shape[1], frame.shape[0]]
    if not a.fake:
        win = "teach sample: the robot sees"
        cv2.imshow(win, V.draw_cups(frame, cups))
        cv2.waitKey(1000)
        cv2.destroyWindow(win)
    return uv, size


def cmd_sample(a):
    import motions as M
    import vision as V
    ready = A.load_poses().get("ready") or (M.fake_poses()["ready"] if a.fake else None)
    if not ready:
        sys.exit("No 'ready' pose yet. Run: teach.py teach --only ready")
    setup = None if a.fake and not a.image else V.load_setup(Path(a.setup) if a.setup else None)
    arm = A.connect(fake=a.fake)
    leader = None
    typed = stdin_lines()
    try:
        if not a.hand:
            leader = connect_leader(a.fake)
        for n in range(a.count):
            if n:
                print("\nMove the cup to a NEW place (spread samples over the whole cup area), then Enter. (q = quit)")
                if typed.get() == "q":
                    break
            torque(arm, True)
            A.go_to(arm, ready, 3.0)  # high and out of the camera's view
            seen = see_one_cup(a, setup, typed)
            if seen is None:
                break
            uv, size = seen
            print(f"\nCup seen at u={uv[0]:.0f}, v={uv[1]:.0f}. Now teach the 4 poses for THIS cup.")
            if leader:
                if a.fake:
                    leader.pose = A.read_pose(arm)
                if not match_start_pose(arm, leader, typed, limp=False):
                    break
                A.go_to(arm, leader.get_action(), 2.0)
                print("Move the LEADER and watch the FOLLOWER. Enter saves the follower's pose.")
            else:
                print("HAND mode: hold the follower, Enter = motors OFF. (q = quit)")
                if typed.get() == "q":
                    break
                torque(arm, False)
            got: dict[str, dict] = {}
            done = follow_steps(arm, leader, typed, list(KINDS), got.get,
                                lambda k, p: got.__setitem__(k, {j: round(float(p[j]), 2) for j in A.KEYS}), a.hand)
            torque(arm, True)
            if len(got) == len(KINDS):
                samples = load_samples() + [{"uv": list(uv), "frame_size": size, "poses": got}]
                save_samples(samples)
                print(f"\nsample {len(samples)} saved at {uv[0]:.0f},{uv[1]:.0f}; move the cup to a new place and run "
                      "sample again (6-8 samples spread over the whole cup area)")
            else:
                print(f"\nsample NOT saved (missing {', '.join(k for k in KINDS if k not in got)})")
            if "grip" in got and "above" in got:  # put the cup down, let go, back up
                o = got["above"]["gripper.pos"]
                A.go_to(arm, got["grip"], 1.5)
                A.go_to(arm, A.with_gripper(got["grip"], o), 0.7)
                A.go_to(arm, got["above"], 1.0)
            if not done:
                break
        print(f"\n{len(load_samples())} samples in {SAMPLES_FILE.name}. Next: reach.py check")
    finally:
        if leader:
            leader.disconnect()
        A.release(arm)


def cmd_samples(a):
    samples = load_samples()
    if a.drop is not None:
        if not 1 <= a.drop <= len(samples):
            sys.exit(f"no sample {a.drop} (have {len(samples)})")
        samples.pop(a.drop - 1)
        save_samples(samples)
        print(f"dropped sample {a.drop}")
    for n, smp in enumerate(samples, 1):
        print(f"{n:3d}  u={smp['uv'][0]:6.1f} v={smp['uv'][1]:6.1f}  frame {smp.get('frame_size')}  "
              f"poses {','.join(smp['poses'])}")
    print(f"{len(samples)} samples in {SAMPLES_FILE.name}")


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


def cmd_copy_slot(a):
    """Same reach, other direction: copy a good slot's 4 poses and only turn the base (shoulder_pan)."""
    poses = A.load_poses()
    src = [f"{k}_{a.src}" for k in ("above", "grip", "in", "lift")]
    missing = [n for n in src if n not in poses]
    if missing:
        sys.exit(f"slot {a.src} is not fully taught (missing {', '.join(missing)})")
    turn = a.pan - poses[f"above_{a.src}"]["shoulder_pan.pos"]
    for name in src:
        pose = dict(poses[name])
        pose["shoulder_pan.pos"] += turn
        A.save_pose(name.replace(f"_{a.src}", f"_{a.dst}"), pose)
    print(f"slot {a.dst} = slot {a.src} turned {turn:+.1f} deg (base at {a.pan:.1f}). "
          f"Next: teach.py go above_{a.dst}, tape the spot under the gripper, then test-slot {a.dst}.")


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
    s = sub.add_parser("copy-slot", help="copy slot SRC's poses to slot DST, base turned to --pan degrees")
    s.add_argument("src", type=int, choices=[1, 2, 3, 4])
    s.add_argument("dst", type=int, choices=[1, 2, 3, 4])
    s.add_argument("--pan", type=float, required=True, help="base rotation (shoulder_pan) for the new slot")
    s.set_defaults(fn=cmd_copy_slot)
    s = sub.add_parser("sample", help="v2: camera sees ONE cup, teach above/grip/in/lift for it (reach_samples.json)")
    s.add_argument("--count", type=int, default=1, help="samples in one run (move the cup between them)")
    s.add_argument("--hand", action="store_true", help="no leader: follower torque off, move it by hand")
    s.add_argument("--image", help="use a saved image instead of the camera (testing)")
    s.add_argument("--setup", help="vision setup file (default shellgame/vision_setup.json)")
    s.set_defaults(fn=cmd_sample)
    s = sub.add_parser("samples", help="list reach samples, --drop N deletes one")
    s.add_argument("--drop", type=int, help="1-based sample number to delete")
    s.set_defaults(fn=cmd_samples)
    for s in sub.choices.values():  # --fake also accepted after the sub-command
        s.add_argument("--fake", action="store_true", default=argparse.SUPPRESS)
    a = p.parse_args()
    if a.fake and (a.cmd in ("teach", "sample") or not A.POSES_FILE.exists()):
        A.POSES_FILE = FAKE_POSES  # fake teaching never overwrites the real poses.json
        print(f"(fake: using {FAKE_POSES.name})")
    if a.fake and a.cmd in ("sample", "samples"):
        global SAMPLES_FILE
        SAMPLES_FILE = SAMPLES_FILE.with_name("reach_samples_fake.json")  # nor the real reach samples
        print(f"(fake: using {SAMPLES_FILE.name})")
    try:
        a.fn(a)
    except KeyboardInterrupt:
        print("\nInterrupted.")


if __name__ == "__main__":
    main()
