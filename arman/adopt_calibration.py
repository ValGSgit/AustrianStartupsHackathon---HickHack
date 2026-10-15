"""Use the calibration that is already stored in a robot's motors (for example set by its previous owners):
read it out and save it as this robot's calibration file. Writes nothing to the motors, moves nothing.

    python adopt_calibration.py
"""

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

import config

bus = FeetechMotorsBus(port=config.S["FOLLOWER_PORT"],
                       motors={j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, j in enumerate(config.JOINTS)})
bus.connect()
try:
    cal = bus.read_calibration()
finally:
    bus.disconnect(disable_torque=False)

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig  # noqa: E402

robot = SO101Follower(SO101FollowerConfig(port=config.S["FOLLOWER_PORT"], id=config.S["FOLLOWER_ID"]))
robot.calibration = cal
robot._save_calibration()
for name, c in cal.items():
    span = (c.range_max - c.range_min) * 360 / 4096
    print(f"  {name:14s} offset {c.homing_offset:6d}  range {c.range_min:5d}-{c.range_max:5d}  ({span:5.0f} degrees)")
print(f"Saved as {robot.calibration_fpath}")
