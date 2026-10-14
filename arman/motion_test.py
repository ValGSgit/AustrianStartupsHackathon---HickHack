"""Health check with movement: each robot joint moves a little toward the middle of its range and back.

    python motion_test.py

Checks that every joint reaches its target, and shows load, voltage and error flags. Small, slow moves only.
"""

import time

import config
from poses import go_to, joints_only

MOVES = {"shoulder_pan": 15, "shoulder_lift": 30, "elbow_flex": 25, "wrist_flex": 25, "wrist_roll": 20, "gripper": 20}
FLAGS = {0: "VOLTAGE", 1: "SENSOR", 2: "OVERHEAT", 3: "OVERCURRENT", 5: "OVERLOAD"}

robot = config.make_robot()
robot.connect()
start = joints_only(robot.get_observation())
results = []
try:
    for joint, size in MOVES.items():
        key = f"{joint}.pos"
        middle = 50 if joint == "gripper" else 0
        target = {**start, key: start[key] + (size if start[key] < middle else -size)}
        go_to(robot, target, seconds=1.5)
        time.sleep(0.5)
        reached = robot.get_observation()[key]
        r = lambda reg: robot.bus.read(reg, joint, normalize=False, num_retry=3)
        load, volts, status = abs(r("Present_Load")) / 10, r("Present_Voltage") / 10, r("Status")
        errors = ",".join(n for b, n in FLAGS.items() if status >> b & 1) or "none"
        ok = abs(reached - target[key]) < 5
        results.append(ok)
        print(f"{joint:14s} asked {target[key]:7.1f}  reached {reached:7.1f}  {'OK  ' if ok else 'MISS'}  "
              f"load {load:5.1f}%  {volts:4.1f}V  errors: {errors}", flush=True)
        go_to(robot, start, seconds=1.5)
finally:
    go_to(robot, start, seconds=1.5)
    robot.disconnect()
print(f"\n{sum(results)}/{len(results)} joints OK")
