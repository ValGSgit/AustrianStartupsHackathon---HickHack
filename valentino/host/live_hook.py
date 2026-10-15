"""Run a lerobot CLI tool (rollout / replay) and mirror what the follower sees and does to stdout.

Only one process can hold a webcam (and a COM port) on Windows, so the dashboard can't watch the arm from outside
while lerobot drives it. This wrapper patches the follower's own get_observation / send_action and prints, at most
8 times a second, one line:

    @@LIVE {"state": [6 joint positions] | null, "target": [6 commanded positions] | null, "jpeg": base64 | null}

host/robot_agent.py reads those lines from the subprocess and forwards them to the dashboard's Live panel.
robot.py routes rollout/replay through here when ROBOT_LIVE=1 (the agent sets it); otherwise nothing changes.

    python host/live_hook.py lerobot-rollout <the usual lerobot-rollout arguments>
"""

import base64
import json
import sys
import time
from importlib.metadata import entry_points

import cv2
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HZ = 8
live = {"next": 0.0, "state": None, "target": None, "img": None}


def emit() -> None:
    now = time.perf_counter()
    if now < live["next"]:
        return
    live["next"] = now + 1 / HZ
    jpeg = None
    if live["img"] is not None:
        small = cv2.resize(live["img"], (320, 240), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 70])  # lerobot frames are RGB
        jpeg = base64.b64encode(buf.tobytes()).decode() if ok else None
    msg = {"state": live["state"], "target": live["target"], "jpeg": jpeg}
    print("@@LIVE " + json.dumps(msg), flush=True)


def joints(d: dict) -> list[float] | None:
    vals = [d.get(f"{j}.pos") for j in JOINTS]
    return None if any(v is None for v in vals) else [round(float(v), 1) for v in vals]


def install() -> None:
    from lerobot.robots.so_follower.so_follower import SOFollower

    get_observation, send_action = SOFollower.get_observation, SOFollower.send_action

    def observed(self):
        obs = get_observation(self)
        try:  # the live view must never break the control loop
            live["state"] = joints(obs) or live["state"]
            img = next((v for v in obs.values() if isinstance(v, np.ndarray) and v.ndim == 3), None)
            if img is not None:
                live["img"] = img
            emit()
        except Exception as e:
            print(f"live view: {e}", file=sys.stderr)
        return obs

    def sent(self, action):
        out = send_action(self, action)
        try:
            live["target"] = joints(out) or live["target"]
            emit()
        except Exception as e:
            print(f"live view: {e}", file=sys.stderr)
        return out

    SOFollower.get_observation, SOFollower.send_action = observed, sent


def main() -> None:
    tool, args = sys.argv[1], sys.argv[2:]
    (ep,) = entry_points(group="console_scripts", name=tool)
    install()
    sys.argv = [tool, *args]
    ep.load()()


if __name__ == "__main__":
    main()
