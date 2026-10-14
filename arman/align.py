"""Make the robot (follower) agree with the joystick (leader): same pose on both arms = same numbers.

    python align.py

You put both arms in the same pose twice; the difference per joint is saved to offsets.json, which
joystick control subtracts. Also reports any joint that moves in the opposite direction. Run after
calibrate.py.
"""

import json

import config

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]  # gripper uses 0-100 %
PLAIN = {"shoulder_pan": "base", "shoulder_lift": "upper arm", "elbow_flex": "elbow",
         "wrist_flex": "wrist bend", "wrist_roll": "wrist turn"}
RES = 4095
K = 360 / RES  # degrees per step, as LeRobot normalizes
GREEN, YELLOW, RED, RESET = "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def ask(prompt: str) -> None:
    print(prompt, end="", flush=True)
    input()


def read_both(robot, leader) -> tuple[dict, dict]:
    f = robot.get_observation()
    l = leader.get_action()
    return ({j: f[f"{j}.pos"] for j in JOINTS}, {j: l[f"{j}.pos"] for j in JOINTS})


def show(title: str, f: dict, l: dict) -> None:
    print(f"\n{title}\n  {'joint':11s} {'joystick':>9s} {'robot':>8s} {'diff':>7s}")
    for j in JOINTS:
        d = l[j] - f[j]
        col = GREEN if abs(d) < 8 else YELLOW if abs(d) < 20 else RED
        print(f"  {PLAIN[j]:11s} {l[j]:8.0f}° {f[j]:7.0f}° {col}{d:6.0f}°{RESET}")


def main() -> None:
    robot, leader = config.make_robot(), config.make_leader()
    robot.connect()
    leader.connect()
    try:
        robot.bus.disable_torque()  # both arms limp so you can pose them
        print("\nBoth arms are now limp. The robot stays where you put it; the joystick may sag, so hold it.")

        print("\nPOSE 1: both arms FOLDED in their resting position, both bases facing straight FORWARD.")
        ask("Press Enter when both are folded the same way... ")
        f1, l1 = read_both(robot, leader)
        show("Pose 1 (folded):", f1, l1)

        print("\nPOSE 2: put the ROBOT in this pose (it stays): upper arm straight UP, forearm straight FORWARD,")
        print("        wrist straight, claw pointing forward. Then HOLD the joystick in the exact same pose next to it.")
        ask("Press Enter while both are in pose 2... ")
        f2, l2 = read_both(robot, leader)
        show("Pose 2 (L shape):", f2, l2)

        problems, shifts = [], {}
        for j in JOINTS:
            df, dl = f2[j] - f1[j], l2[j] - l1[j]
            if abs(dl) > 25 and abs(df) > 25 and (df > 0) != (dl > 0):
                problems.append(f"{PLAIN[j]} moves in the OPPOSITE direction on the robot")
            d1, d2 = l1[j] - f1[j], l2[j] - f2[j]
            if abs(d1 - d2) > 20:
                print(f"{YELLOW}Note: {PLAIN[j]} differs by {abs(d1 - d2):.0f}° between the two poses "
                      f"(arms not posed the same?). Using the average.{RESET}")
            shifts[j] = (d1 + d2) / 2

        if problems:
            print(f"\n{RED}Not changing anything, because:{RESET}")
            for p in problems:
                print(f"  - {p}")
            print("Tell Claude what this says.")
            return

        # Store the differences instead of changing the robot's calibration: its limits stay exactly at
        # its real end stops, and joystick control (config.to_robot) subtracts these offsets.
        config.OFFSETS_FILE.write_text(json.dumps({j: round(d, 1) for j, d in shifts.items()}, indent=2))
        print(f"\nSaved offsets.json. Remaining difference after the offsets (pose 2):")
        for j in JOINTS:
            print(f"  {PLAIN[j]:11s} {(l2[j] - f2[j]) - shifts[j]:5.1f}°")
        print("Put both arms back in their resting position.")
    finally:
        leader.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
