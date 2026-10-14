"""Which motors answer on each plugged-in arm? Read-only, never moves anything.

    python motors.py

Good result: 6 motors (ids 1-6) on each port. "NO MOTORS" = board is connected but the motors are
silent: no motor power, loose 3-pin cable, wrong jumpers, or motors never set up (see README).
"""

import glob

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

import config

NAMES = {f"{j}": i + 1 for i, j in enumerate(config.JOINTS)}
ROLE = {config.S.get("FOLLOWER_PORT"): "follower", config.S.get("LEADER_PORT"): "leader"}

ports = sorted(glob.glob("/dev/tty.usbmodem*"))
if not ports:
    print("No arm boards found. Plug in the USB cables.")

for port in ports:
    print(f"\n{port}  ({ROLE.get(port, 'not in settings.env')})")
    bus = FeetechMotorsBus(port=port, motors={n: Motor(i, "sts3215", MotorNormMode.RANGE_M100_100) for n, i in NAMES.items()})
    try:
        bus.connect(handshake=False)
        found = bus.broadcast_ping() or {}
        if not found:
            print("  NO MOTORS answer. Check: power supply in the board + wall, light on the board,")
            print("  3-pin cable board -> base motor, Waveshare jumpers on B, motors set up.")
            continue
        for name, mid in NAMES.items():
            if mid not in found:
                print(f"  id {mid} {name:14s} MISSING  -> check the cable going into this motor")
                continue
            volts = bus.read("Present_Voltage", name, normalize=False) / 10
            print(f"  id {mid} {name:14s} OK  {volts:.1f}V")
        extra = sorted(set(found) - set(NAMES.values()))
        if extra:
            print(f"  unexpected ids {extra}: motors not set up yet (run lerobot-setup-motors)")
    except Exception as e:
        print(f"  ERROR: {e}")
    finally:
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            pass
