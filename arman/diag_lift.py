"""Measure the robot's upper-arm motor while you drive it with the joystick. Writes outputs/diag_lift.log.

    python diag_lift.py   (then lift the joystick's upper arm up and down for 25 s)
"""

import time

import config
from poses import go_to, joints_only

J, SECONDS = "shoulder_lift", 25
robot, leader = config.make_robot(), config.make_leader()
robot.connect()
leader.connect()
log = open("outputs/diag_lift.log", "w")


def out(line: str) -> None:
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


rest = joints_only(robot.get_observation())
try:
    r = lambda reg: robot.bus.read(reg, J, normalize=False, num_retry=3)
    out("settings: " + ", ".join(f"{k}={r(k)}" for k in ("Max_Torque_Limit", "Torque_Limit", "Protection_Current",
                                                        "Overload_Torque", "Protective_Torque", "Protection_Time")))
    go_to(robot, joints_only(leader.get_action()), seconds=2.0)
    out("t     joystick  sent   robot  load  current  volts  temp  status")
    t0, last = time.time(), 0.0
    while time.time() - t0 < SECONDS:
        a = leader.get_action()
        sent = robot.send_action(a)
        if time.time() - last > 0.25:
            last = time.time()
            now = robot.bus.read("Present_Position", J, num_retry=3)
            out(f"{time.time() - t0:4.1f}  {a[J + '.pos']:8.1f} {sent[J + '.pos']:6.1f} {now:6.1f} {r('Present_Load'):5d} "
                f"{r('Present_Current'):8d} {r('Present_Voltage') / 10:6.1f} {r('Present_Temperature'):5d} {r('Status'):6d}")
        time.sleep(1 / 30)
finally:
    try:
        go_to(robot, rest, seconds=2.5)
    finally:
        leader.disconnect()
        robot.disconnect()
