"""`lerobot-rollout` that also publishes the policy's camera view, so stage.py can show it live.

Only one process can hold a webcam on Windows, so stage.py can't open the camera during a rollout. This wrapper
hooks the rollout's own observation read and writes every 3rd front frame to $LIVE_FRAME (atomic replace).
All arguments are passed to lerobot-rollout unchanged. Without $LIVE_FRAME it behaves exactly like lerobot-rollout.

    python rollout_live.py --mock --duration=10    # no robot: stream the webcam for N seconds (rehearsal/testing)
"""

import os
import sys
import time

import cv2
import numpy as np

LIVE_FRAME = os.environ.get("LIVE_FRAME")
CAMERA_KEY = os.environ.get("LIVE_CAMERA", "front")
EVERY = 3


def publish(img_rgb: np.ndarray) -> None:
    tmp = LIVE_FRAME + ".tmp.jpg"
    cv2.imwrite(tmp, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    try:
        os.replace(tmp, LIVE_FRAME)
    except OSError:  # reader has the file open right now; skip this frame
        pass


def mock(duration: float) -> None:
    import selfcheck

    cap = selfcheck.open_camera()
    t0, n = time.time(), 0
    try:
        while time.time() - t0 < duration:
            ok, f = cap.read()
            if ok and LIVE_FRAME and n % EVERY == 0:
                publish(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            n += 1
    finally:
        cap.release()


def main() -> None:
    if "--mock" in sys.argv:
        durs = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--duration=")]
        mock(float(durs[0]) if durs else 10.0)
        return

    from lerobot.rollout import robot_wrapper
    from lerobot.scripts import lerobot_rollout

    if LIVE_FRAME:
        original = robot_wrapper.ThreadSafeRobot.get_observation
        count = {"n": 0}

        def get_observation(self):
            obs = original(self)
            count["n"] += 1
            img = obs.get(CAMERA_KEY)
            if count["n"] % EVERY == 0 and isinstance(img, np.ndarray):
                try:
                    publish(img)
                except Exception as e:  # never let the live view break the control loop
                    print(f"live view: {e}", file=sys.stderr)
            return obs

        robot_wrapper.ThreadSafeRobot.get_observation = get_observation

    lerobot_rollout.main()


if __name__ == "__main__":
    main()
