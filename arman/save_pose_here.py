"""Save a pose by moving the robot BY HAND (no joystick).   python save_pose_here.py above_cup

The motors go loose: hold the arm, move the gripper where you want it, press Enter. Then put the arm back
into its resting pose and press Enter again.
"""

import sys

import config
from poses import clean_name, joints_only, save_pose

name = clean_name(sys.argv[1]) if len(sys.argv) > 1 else "above_cup"
import time

for attempt in range(4):  # a weak 5V supply can dip when connecting ("Input voltage error"): try again
    robot = config.make_robot()
    try:
        robot.connect()
        break
    except Exception as e:  # noqa: BLE001
        if attempt == 3:
            raise
        print(f"(connect failed: {str(e)[:60]}; trying again)", flush=True)
        try:
            robot.bus.disconnect(disable_torque=False)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.5)
try:
    print("\nHOLD THE ARM: the motors go loose now.", flush=True)
    robot.bus.disable_torque()
    print(f"Move the gripper about 5 cm ABOVE THE CUP OPENING, then press Enter to save '{name}'... ", end="", flush=True)
    input()
    pose = joints_only(robot.get_observation())
    save_pose(name, pose)
    print(f"Saved '{name}': " + ", ".join(f"{k.split('.')[0]}={v:.1f}" for k, v in pose.items()))
    print("Now put the arm back into its RESTING pose and press Enter... ", end="", flush=True)
    input()
finally:
    robot.disconnect()
print("DONE")
