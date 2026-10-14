"""Stream the real leader arm (or a test motion) from the host into the Docker sim over UDP.

USB passthrough into Docker on Windows is painful, so the leader is read here with LeRobot and
each reading is sent as one JSON datagram to the container's teleop_udp_node (port 5005).

    .venv\\Scripts\\python host\\leader_to_sim.py              # real leader arm (port from config.toml)
    .venv\\Scripts\\python host\\leader_to_sim.py --wave       # no hardware: scripted wave motion

Units: the real robot's LeRobot units (arm -100..100, gripper 0..100) - exactly what lerobot-record
stores, so the sim moves the way the real follower would. Directions were checked against real
SO-101 data; if a joint still looks mirrored on your arm, flip it in SIGN below.
"""

import argparse
import json
import math
import socket
import sys
import time
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
# Per-joint direction between the real leader and the MuJoCo model (+1 = matches LeRobot's URDF convention).
SIGN = {n: 1.0 for n in NAMES}
OFFSET = {n: 0.0 for n in NAMES}


def wave(t: float) -> dict:
    """Smooth demo motion: pan left/right, nod, open/close the gripper."""
    return {
        "shoulder_pan.pos": 35 * math.sin(0.6 * t),
        "shoulder_lift.pos": -30 + 20 * math.sin(0.9 * t),
        "elbow_flex.pos": 40 + 20 * math.sin(0.9 * t + 1.0),
        "wrist_flex.pos": 50 + 15 * math.sin(1.3 * t),
        "wrist_roll.pos": 45 * math.sin(0.4 * t),
        "gripper.pos": 50 + 45 * math.sin(1.5 * t),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="where the container's udp/5005 is published")
    p.add_argument("--port", type=int, default=5005)
    p.add_argument("--hz", type=float, default=30.0)
    p.add_argument("--wave", action="store_true", help="send a test motion instead of reading the leader arm")
    p.add_argument("--seconds", type=float, default=0, help="stop after N seconds (0 = run until Ctrl-C)")
    a = p.parse_args()

    leader = None
    if not a.wave:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        cfg = tomllib.loads((ROOT / "config.toml").read_text(encoding="utf-8"))["leader"]
        leader = SO101Leader(SO101LeaderConfig(
            port=cfg["port"], id=cfg["id"], calibration_dir=ROOT / "calibration",
        ))
        leader.connect()  # uses calibration/leader.json; asks to calibrate if it's missing

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"sending {'wave motion' if a.wave else 'leader arm'} to udp://{a.host}:{a.port} at {a.hz:.0f} Hz - Ctrl-C to stop")
    t0 = time.perf_counter()
    n = 0
    try:
        while True:
            t = time.perf_counter() - t0
            if a.seconds and t > a.seconds:
                break
            action = wave(t) if a.wave else leader.get_action()
            out = {k: SIGN[k] * float(action[k]) + OFFSET[k] for k in NAMES}
            sock.sendto(json.dumps(out).encode(), (a.host, a.port))
            n += 1
            if n % int(a.hz) == 0:
                print("  " + "  ".join(f"{k.split('.')[0]}={v:6.1f}" for k, v in out.items()), end="\r", file=sys.stderr)
            time.sleep(max(0.0, n / a.hz - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if leader is not None:
            leader.disconnect()
    print(f"\nsent {n} packets")


if __name__ == "__main__":
    main()
