"""Teach the arm named poses ("remember this") and play them back. No training needed.

    python poses.py teach          # move the LEADER arm, type a name + Enter to save the pose
    python poses.py teach --hand   # no leader: move the FOLLOWER by hand (torque off)
    python poses.py list           # show saved poses
    python poses.py go home        # move smoothly to a saved pose

Every script ends by moving slowly to `home` and only then releasing the motors (the arm goes limp
on release). Ctrl+C does the same. Emergency stop = unplug the arm's power supply.
"""

import json
import queue
import re
import sys
import threading
import time
from pathlib import Path

import config
from config import JOINTS

POSES_FILE = Path(__file__).parent / "poses.json"
FPS = 30
MAX_START_DIFF = 30.0  # degrees: refuse to start copying if the arms differ more than this
BLOCKED_DIFF = 25.0  # degrees between command and robot position that count as "not following"
BLOCKED_SECONDS = 1.0  # for this long: something blocks the robot, stop before the motor overloads


def load_poses() -> dict[str, dict[str, float]]:
    if not POSES_FILE.exists():
        return {}
    return json.loads(POSES_FILE.read_text())


def save_pose(name: str, pose: dict[str, float]) -> None:
    poses = load_poses()
    poses[name] = pose
    POSES_FILE.write_text(json.dumps(poses, indent=2))


def joints_only(values: dict) -> dict[str, float]:
    return {f"{j}.pos": float(values[f"{j}.pos"]) for j in JOINTS}


def with_gripper(pose: dict[str, float], gripper: float) -> dict[str, float]:
    return {**pose, "gripper.pos": gripper}


def go_to(robot, target: dict[str, float], seconds: float = 1.5) -> None:
    """Move smoothly from the current position to `target` (ease in, ease out)."""
    start = joints_only(robot.get_observation())
    steps = max(1, int(seconds * FPS))
    for i in range(1, steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)  # smoothstep
        robot.send_action({k: start[k] + (target[k] - start[k]) * t for k in target})
        if not config.FAKE:
            time.sleep(1 / FPS)
    if not config.FAKE:
        time.sleep(0.2)  # let the servos settle


def release(robot, fallback: dict | None = None) -> None:
    """Go home slowly (or to `fallback`, e.g. the pose the arm started in), then let the motors go limp.
    Use this instead of robot.disconnect()."""
    try:
        home = load_poses().get("home") or fallback
        if home:
            go_to(robot, home, seconds=2.0)
        else:
            print("No 'home' pose taught: the arm goes limp NOW, hold it.")
    except Exception as e:
        print(f"Could not go home ({e}). The arm goes limp NOW, hold it.")
    finally:
        try:
            robot.disconnect()
        except Exception as e:
            print(f"Could not switch the motors off cleanly ({e}).")
            print("Unplug the robot's power for 10 seconds, then plug it back in.")


def _gap(joint_key: str, a: float, b: float) -> float:
    """Degrees between two readings; the wrist rotation wraps around at +-180."""
    d = a - b
    return abs((d + 180) % 360 - 180) if joint_key == "wrist_roll.pos" else abs(d)


def _match_start_pose(robot, leader) -> None:
    """Don't start copying while the arms are in very different poses: the robot would swing straight
    through the table to get there (that overloaded the elbow motor once)."""
    while True:
        want = joints_only(config.to_robot(robot, leader.get_action()))
        have = joints_only(robot.get_observation())
        far = [k for k in want if k != "gripper.pos" and _gap(k, want[k], have[k]) > MAX_START_DIFF]
        if not far:
            return
        print("\nThe robot and the joystick are in different poses:")
        for k in far:
            print(f"  {k.split('.')[0]:14s} joystick {want[k]:7.1f}°   robot {have[k]:7.1f}°")
        robot.bus.disable_torque()
        print("The robot's motors are OFF. Put both arms in the same pose (easiest: both RESTING),")
        print("then press Enter.", flush=True)
        input()
        # Hold where the robot is now, not an old target, then switch the motors back on.
        robot.bus.sync_write("Goal_Position", robot.bus.sync_read("Present_Position"))
        robot.bus.enable_torque()


def clean_name(text: str) -> str:
    """'Grasp 1' -> 'grasp_1', so typos with spaces or capitals still match what the scripts expect."""
    return re.sub(r"[^a-z0-9_]", "", re.sub(r"\s+", "_", text.strip().lower()))


def teach(use_leader: bool = True, steps: list[tuple[str | None, str]] | None = None) -> queue.Queue:
    """Drive the robot with the joystick and save poses.

    Free mode: type a pose name + Enter to save it. Guided mode (`steps`): each step shows an instruction
    and Enter saves the pose under the step's name (None = just a pause, nothing saved); b + Enter = back.
    Returns the queue of typed lines, so a caller can keep reading the keyboard afterwards.
    """
    robot = config.make_robot()
    robot.connect()
    rest = joints_only(robot.get_observation())  # where the arm started: returned to on exit if no 'home'
    leader = None
    if use_leader:
        leader = config.make_leader()
        leader.connect()
        _match_start_pose(robot, leader)
        # Ease the follower to where the leader is, instead of snapping there at full speed.
        go_to(robot, joints_only(config.to_robot(robot, leader.get_action())), seconds=2.0)
    else:
        robot.bus.disable_torque()

    # Read typed names in the background so the arm keeps following the leader while you type.
    typed = queue.Queue()
    threading.Thread(target=lambda: [typed.put(line.strip()) for line in sys.stdin], daemon=True).start()

    step = 0
    if steps:
        print("\nMove the JOYSTICK and watch the ROBOT. Press Enter when the robot is in position.")
        print("(b + Enter = go back one step, q + Enter = stop)\n")
        print(f"STEP 1/{len(steps)}: {steps[0][1]}")
    else:
        print("\nMove the arm. Type a pose name and press Enter to save it.")
        print("To finish: put the arm in its rest position, then type q + Enter.")
        print("For pick_place.py teach: home, above_cup, then above_1 + grasp_1, above_2 + grasp_2, ...  (see README)\n")
    blocked_since = None
    try:
        while True:
            t0 = time.perf_counter()
            if leader:
                # Save what the leader COMMANDS (in robot terms), not where the follower ended up. For the
                # gripper this keeps the extra squeeze you apply on the piece, so the grasp holds when replayed.
                current = config.to_robot(robot, leader.get_action())
                try:
                    robot.send_action(current)
                    have = robot.get_observation()
                except (RuntimeError, ConnectionError) as e:
                    print(f"\nA motor reported a problem ({e}). Stopping. Something may be blocking the arm.")
                    break
                stuck = [k for k in current if k != "gripper.pos" and _gap(k, current[k], have[k]) > BLOCKED_DIFF]
                if not stuck:
                    blocked_since = None
                elif blocked_since is None:
                    blocked_since = time.time()
                elif time.time() - blocked_since > BLOCKED_SECONDS:
                    names = ", ".join(k.split(".")[0] for k in stuck)
                    print(f"\nThe robot cannot reach the joystick's pose ({names}): something blocks it.")
                    print("Stopping to protect the motors. Check the arm, then start again.")
                    break
            else:
                current = robot.get_observation()

            try:
                name = typed.get_nowait()
            except queue.Empty:
                name = None
            if name == "q":
                break
            if steps and name is not None:
                if name == "b":
                    step = max(0, step - 1)
                else:
                    if steps[step][0]:
                        save_pose(steps[step][0], joints_only(current))
                        print(f"   saved '{steps[step][0]}'")
                    step += 1
                    if step == len(steps):
                        break
                print(f"\nSTEP {step + 1}/{len(steps)}: {steps[step][1]}")
                name = None
            if name and clean_name(name) != name:
                print(f"(saving as '{clean_name(name)}')")
                name = clean_name(name)
            if name:
                pose = joints_only(current)
                save_pose(name, pose)
                print(f"saved '{name}': " + ", ".join(f"{k.split('.')[0]}={v:.1f}" for k, v in pose.items()))

            time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))
    finally:
        if leader:
            leader.disconnect()
            release(robot, fallback=rest)
        else:
            robot.disconnect()  # hand mode: torque is already off
    return typed


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in {"teach", "list", "go"}:
        print(__doc__)
        sys.exit(1)

    if args[0] == "teach":
        teach(use_leader="--hand" not in args)
    elif args[0] == "list":
        for name, pose in load_poses().items():
            print(f"{name:15s} " + "  ".join(f"{v:7.1f}" for v in pose.values()))
    elif args[0] == "go":
        poses = load_poses()
        if len(args) < 2 or args[1] not in poses:
            sys.exit(f"Unknown pose. Saved poses: {list(poses)}")
        robot = config.make_robot()
        robot.connect()
        try:
            go_to(robot, poses[args[1]])
            input("Holding pose. Press Enter to go home and release ")
        finally:
            release(robot)


if __name__ == "__main__":
    main()
