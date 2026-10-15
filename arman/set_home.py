"""Save the robot's resting pose as 'home' (where every run starts and ends). Nothing moves.

    python set_home.py      # first fold the robot into its resting pose by hand

Use after calibrating a robot: the old 'home' angles belong to the old calibration.
"""

import config
from poses import joints_only, save_pose

robot = config.make_robot()
robot.connect()
try:
    robot.bus.disable_torque()  # limp: it stays exactly where you put it
    pose = joints_only(robot.get_observation())
    save_pose("home", pose)
    print("Saved 'home': " + ", ".join(f"{k.split('.')[0]}={v:.1f}" for k, v in pose.items()))
finally:
    robot.disconnect()
