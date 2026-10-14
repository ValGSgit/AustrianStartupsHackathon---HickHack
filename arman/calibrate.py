"""Guided calibration with live feedback. Produces the same calibration file as lerobot-calibrate.

    python calibrate.py follower
    python calibrate.py leader
    python calibrate.py follower elbow_flex   # redo only some joints, keep the others as they are

Unlike lerobot-calibrate there is no "middle pose" step: you just move every joint end to end, and the
middle of each joint is computed from the movement. That avoids the most common mistake (holding the
arm in the wrong pose at the start), which makes the encoder roll over and ruins the calibration.
"""

import queue
import sys
import threading
import time

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import OperatingMode

import config

# Plain-language instruction and the minimum total movement (degrees) we expect for each joint.
JOINTS = {
    "shoulder_pan": ("BASE       turn fully LEFT, then RIGHT", 170),
    "shoulder_lift": ("UPPER ARM  tilt fully FORWARD, then BACK", 170),
    "elbow_flex": ("ELBOW      fold fully, then stretch fully", 170),
    "wrist_flex": ("WRIST      bend fully DOWN, then UP", 150),
    "gripper": ("GRIPPER    close fully, then open fully", 45),
}
RES = 4096
HALF = RES // 2 - 1  # 2047, the value the middle of each joint gets
STEPS_PER_DEG = RES / 360
GREEN, YELLOW, RESET = "\033[32m", "\033[33m", "\033[0m"


def ask(prompt: str) -> None:
    """input() writes its prompt to stderr on some setups; print it to stdout so it always shows."""
    print(prompt, end="", flush=True)
    input()


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else ""
    only = sys.argv[2:]
    if which not in {"follower", "leader"} or any(j not in JOINTS for j in only):
        sys.exit(__doc__ + f"\nJoint names: {', '.join(JOINTS)}")
    dev = config.make_robot() if which == "follower" else config.make_leader()
    dev.connect(calibrate=False)
    bus = dev.bus
    names = only or list(JOINTS)
    keep = {m: c for m, c in (dev.calibration or {}).items() if only and m not in only}
    if only and len(keep) != len(bus.motors) - len(only):
        sys.exit("No complete saved calibration to keep. Run without joint names first.")

    try:
        label = "the robot, WITHOUT the handle" if which == "follower" else "the joystick, WITH the handle"
        print(f"\n=== Calibrating the {which.upper()} ({label}) ===\n")
        print("Hold the arm: the motors switch OFF now and it goes limp.")
        ask("Press Enter when you are holding it... ")
        bus.disable_torque()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        bus.reset_calibration(names)  # raw readings from here on (homing offset 0) for the joints we redo

        print("\nMove each joint to BOTH ends, slowly, one at a time, until it STOPS. Any starting pose is fine.")
        print("The elbow must fold as far as in the resting pose. Push firmly, the robot is stiff.")
        print("Each line turns green when done. Press Enter when all lines are green.\n")
        start = bus.sync_read("Present_Position", names, normalize=False, num_retry=5)
        prev = dict(start)
        travel = {n: 0 for n in names}  # unwrapped movement since start, in steps
        lo, hi = dict(travel), dict(travel)

        typed = queue.Queue()
        threading.Thread(target=lambda: [typed.put(line) for line in sys.stdin], daemon=True).start()
        warning, drawn = "", 0
        while True:
            pos = bus.sync_read("Present_Position", names, normalize=False, num_retry=5)
            for n in names:
                step = (pos[n] - prev[n] + RES // 2) % RES - RES // 2  # shortest way round, handles rollover
                travel[n] += step
                lo[n], hi[n] = min(lo[n], travel[n]), max(hi[n], travel[n])
                prev[n] = pos[n]

            lines = []
            for n, (text, need) in JOINTS.items():
                if n not in names:
                    continue
                total = (hi[n] - lo[n]) / STEPS_PER_DEG
                mark = f"{GREEN}✓ done{RESET}" if total >= need else f"{YELLOW}need {need}°{RESET}"
                lines.append(f"{text:42s}{total:4.0f}°  {mark}")
            lines.append(warning)
            if drawn:
                sys.stdout.write(f"\033[{drawn}A")
            sys.stdout.write("".join(f"\033[2K{line}\n" for line in lines))
            sys.stdout.flush()
            drawn = len(lines)

            if not typed.empty():
                typed.get()
                missing = [n for n in names if (hi[n] - lo[n]) / STEPS_PER_DEG < JOINTS[n][1]]
                if not missing or warning:
                    break
                warning = f"{YELLOW}Not done: {', '.join(missing)}. Enter again = save anyway{RESET}"
            time.sleep(0.05)

        # Centre each joint's movement on HALF: the middle of what you moved becomes the joint's zero.
        homing, mins, maxes = {}, {}, {}
        for n in names:
            span = hi[n] - lo[n]
            if span >= RES - 50:
                sys.exit(f"{n} moved a full turn, which a real joint cannot do. Nothing saved, run again.")
            centre_raw = (start[n] + (lo[n] + hi[n]) // 2) % RES
            homing[n] = max(-HALF, min(HALF, centre_raw - HALF))
            mins[n], maxes[n] = HALF - span // 2, HALF + span // 2
        if not only:
            # Wrist roll is never limited: its zero is the angle you set now, so set it the same on both arms.
            print()
            ask("Last step: turn the WRIST ROTATION so the claw/handle is LEVEL, like in the picture. Press Enter... ")
            roll_raw = bus.read("Present_Position", "wrist_roll", normalize=False)
            homing["wrist_roll"] = max(-HALF, min(HALF, roll_raw - HALF))
            mins["wrist_roll"], maxes["wrist_roll"] = 0, RES - 1

        dev.calibration = {
            motor: MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=int(homing[motor]),
                range_min=int(mins[motor]),
                range_max=int(maxes[motor]),
            )
            for motor, m in bus.motors.items()
            if motor not in keep
        } | keep
        bus.write_calibration(dev.calibration)
        dev._save_calibration()

        # Check: every joint's current reading must now fall inside its new range.
        now = bus.sync_read("Present_Position", names, normalize=False, num_retry=5)
        bad = [n for n in names if not (mins[n] - 40 <= now[n] <= maxes[n] + 40)]
        if bad:
            print(f"\n{YELLOW}Saved, but check failed for {bad}. Tell Claude.{RESET}")
        else:
            print(f"\n{GREEN}Saved and checked.{RESET} Put the arm back in its resting pose.")
    finally:
        dev.disconnect()


if __name__ == "__main__":
    main()
