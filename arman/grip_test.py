"""Measure the gripper and feel the firm grip. No cameras, no model; only the gripper moves.

    python grip_test.py              # the arm stays where it is (resting pose is fine)
    python grip_test.py above_cup    # first move to a taught pose, so the gripper is up in the air

1. Closes on nothing: measures the 'empty' reading.
2. You hold an object between the jaws, it closes: measures it, you pull to feel how firmly it holds.
Saves grip.json. do_task.py uses it to tell 'holding something' from 'closed on nothing'.
"""

import sys
import time

import assist
import config
import poses


def ask(text: str) -> str:
    print(text, end="", flush=True)
    return input().strip()


def reading(robot, seconds: float = 1.0) -> float:
    values = []
    t0 = time.time()
    while time.time() - t0 < seconds:
        values.append(robot.get_observation()["gripper.pos"])
        time.sleep(0.05)
    return sum(values[-5:]) / len(values[-5:])


def set_gripper(robot, value: float, seconds: float = 0.6) -> None:
    here = poses.joints_only(robot.get_observation())
    assist.move(robot, {**here, "gripper.pos": value}, seconds)


def main() -> None:
    robot = config.make_robot()
    robot.connect()
    try:
        bus = robot.bus
        regs = {k: bus.read(k, "gripper", normalize=False, num_retry=3) for k in
                ("Max_Torque_Limit", "Overload_Torque", "Protective_Torque", "Protection_Time", "Present_Temperature")}
        print(f"Gripper motor now: {regs}")
        print(f"Grip strength: {assist.set_grip_strength(robot)}")
        if len(sys.argv) > 1:
            target = poses.load_poses().get(sys.argv[1])
            if target is None:
                sys.exit(f"No pose '{sys.argv[1]}'. Taught: {list(poses.load_poses())}")
            poses.go_to(robot, target, seconds=2.0)
        grip = assist.load_grip()

        set_gripper(robot, 35)
        ask("\n1. Nothing between the jaws. Enter = close: ")
        set_gripper(robot, 0)
        grip["empty_closed"] = round(reading(robot), 1)
        print(f"   closed on nothing reads {grip['empty_closed']}")

        while True:
            set_gripper(robot, 35)
            name = ask("\n2. Type the object's name (Enter = finish), then hold it between the jaws: ")
            if not name:
                break
            ask("   Holding it between the jaws? Enter = close: ")
            set_gripper(robot, 0)
            value = round(reading(robot, 1.5), 1)
            grip.setdefault("objects", {})[poses.clean_name(name)] = value
            verdict = "OK, clearly more than empty" if value >= grip["empty_closed"] + assist.EMPTY_MARGIN + 1 else \
                "TOO THIN: looks like 'closed on nothing' (thicker object or rubber on the jaws)"
            print(f"   {name} reads {value} ({verdict})")
            ask("   Pull it gently: does it hold? Enter = open: ")
        assist.save_grip(grip)
        print(f"\nSaved grip.json: {grip}")
        print(f"Gripper temperature: {bus.read('Present_Temperature', 'gripper', normalize=False)} degrees C")
        set_gripper(robot, 35)
    finally:
        poses.release(robot)


if __name__ == "__main__":
    main()
