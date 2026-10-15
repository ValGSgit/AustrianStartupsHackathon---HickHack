"""Reads settings.env and builds the robot objects.

Set FAKE_ROBOT=1 to test scripts without hardware (prints moves instead of sending them).
"""

import functools
import json
import logging
import os
from pathlib import Path

# LeRobot logs two lines on EVERY step where the robot moves slower than the joystick (it limits each step
# to 15 degrees for safety). Harmless, but it floods the screen and hides real messages.
logging.getLogger().addFilter(
    lambda r: not any(t in str(r.msg) for t in ("Relative goal position magnitude", "original goal_pos"))
)

SETTINGS_FILE = Path(__file__).parent / "settings.env"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_settings() -> dict[str, str]:
    settings = {}
    for line in SETTINGS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        settings[key.strip()] = value.strip().strip('"').strip("'")
    return settings


S = load_settings()
FAKE = os.environ.get("FAKE_ROBOT") == "1"
OFFSETS_FILE = Path(__file__).parent / "offsets.json"
DEG_PER_STEP = 360 / 4095  # how LeRobot converts motor steps to degrees
_offsets = None
_last_roll = None


def _roll_short_way(value: float) -> float:
    """The wrist rotation has no end stop, but its reading jumps from +180 to -180. Always take the short
    way to the new angle and stop 5 degrees before the jump, so the wrist never spins a full turn."""
    global _last_roll
    value = (value + 180) % 360 - 180
    if _last_roll is not None:
        value = min((value - 360, value, value + 360), key=lambda v: abs(v - _last_roll))
    value = max(-175.0, min(175.0, value))
    _last_roll = value
    return value


def to_robot(robot, action: dict) -> dict:
    """Joystick reading -> robot command.

    Removes the measured difference between the two arms (offsets.json, written by align.py) and keeps
    each limited joint 3 degrees inside the robot's own calibrated range, so it never pushes against an
    end stop when the joystick can bend further than the robot.
    """
    global _offsets
    if _offsets is None:
        _offsets = json.loads(OFFSETS_FILE.read_text()) if OFFSETS_FILE.exists() else {}
    calibration = getattr(robot, "calibration", None) or {}
    command = {}
    for key, value in action.items():
        joint = key.removesuffix(".pos")
        value -= _offsets.get(joint, 0.0)
        if joint == "wrist_roll":
            value = _roll_short_way(value)
        cal = calibration.get(joint)
        if cal is not None and joint not in ("gripper", "wrist_roll"):
            limit = (cal.range_max - cal.range_min) / 2 * DEG_PER_STEP - 3
            value = max(-limit, min(limit, value))
        command[key] = value
    return command


class FakeRobot:
    """Stands in for the arm: remembers the last command and reports it as its position.
    With cameras=True it also returns a grey 'front' image of the configured size."""

    name = "so_follower"

    def __init__(self, cameras: bool = False):
        self.state = {f"{j}.pos": 0.0 for j in JOINTS}
        self.commands = 0
        self.cameras = {"front": None} if cameras else {}
        self.calibration = {}
        self.shape = (int(S.get("FRONT_HEIGHT") or 480), int(S.get("FRONT_WIDTH") or 640), 3)

    @property
    def action_features(self):
        return {f"{j}.pos": float for j in JOINTS}

    @property
    def observation_features(self):
        return {**self.action_features, **({"front": self.shape} if self.cameras else {})}

    def connect(self, calibrate: bool = True):
        print("[fake] robot connected")

    def disconnect(self):
        print(f"[fake] robot disconnected after {self.commands} commands")

    def get_observation(self):
        obs = dict(self.state)
        if self.cameras:
            import numpy as np

            obs["front"] = np.full(self.shape, 128, dtype=np.uint8)
        return obs

    def send_action(self, action):
        self.state.update(action)
        self.commands += 1
        return action


def _retry_torque_writes() -> None:
    """Switching all motors on at once can garble one reply ("Incorrect status packet"), and LeRobot does
    not retry those writes, so a whole connect fails. Retry them a few times instead."""
    from functools import partialmethod

    from lerobot.motors.feetech import FeetechMotorsBus

    if not getattr(FeetechMotorsBus, "_retry_patched", False):
        FeetechMotorsBus.enable_torque = partialmethod(FeetechMotorsBus.enable_torque, num_retry=3)
        FeetechMotorsBus.disable_torque = partialmethod(FeetechMotorsBus.disable_torque, num_retry=3)
        FeetechMotorsBus._retry_patched = True


def _sync_calibration_on_connect(dev):
    """Motors can lose their stored calibration (seen after a power change), and LeRobot then stops at an
    input() prompt on every connect. Instead, write the saved calibration file into the motors (with the
    motors off, so nothing moves) just before connecting."""
    original_connect = dev.connect

    def connect(calibrate: bool = True) -> None:
        if dev.calibration:
            bus = dev.bus
            bus.connect()
            try:
                if not bus.is_calibrated:
                    bus.disable_torque()
                    bus.write_calibration(dev.calibration)
                    print(f"(wrote the saved calibration into the {dev} motors)")
            finally:
                bus.disconnect(disable_torque=False)
        original_connect(calibrate=calibrate)

    dev.connect = connect
    return dev


def _cameras_by_name() -> dict[str, int]:
    """Camera numbers by their macOS names (ffmpeg lists them in the same order OpenCV numbers them).
    More reliable than the picture shape: the iPhone over a cable switches between 4:3 and 16:9."""
    import re
    import shutil
    import subprocess

    ffmpeg = shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg" if Path("/opt/homebrew/bin/ffmpeg").exists() else None)
    if not ffmpeg:
        return {}
    out = subprocess.run([ffmpeg, "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True, timeout=20).stderr
    found = {}
    video = out.split("video devices:")[-1].split("audio devices:")[0]
    for idx, name in re.findall(r"\[(\d+)\] (.+)", video):
        n = name.lower()
        if "capture screen" in n or "desk view" in n:
            continue
        if "iphone" in n:
            found.setdefault("iphone", int(idx))
        elif "realsense" in n:
            found.setdefault("infrared", int(idx))
        else:
            found.setdefault("webcam", int(idx))
    return found


@functools.lru_cache(maxsize=1)
def identify_cameras() -> dict[str, int]:
    """Find the cameras, because macOS renumbers them whenever one comes or goes. First by their macOS names;
    if that is not possible, by what they deliver: iPhone = widescreen 16:9, MacBook webcam = 4:3 in colour,
    RealSense = grey infrared."""
    try:
        by_name = _cameras_by_name()
        if by_name:
            return by_name
    except Exception:  # noqa: BLE001
        pass
    import cv2
    import numpy as np

    found = {}
    for idx in range(5):
        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame, t0 = None, __import__("time").time()
        while __import__("time").time() - t0 < 4:  # the iPhone over Wi-Fi needs a few seconds to deliver
            got, f = cap.read()
            if got:
                frame = f
                if __import__("time").time() - t0 > 1:  # let auto-exposure settle a moment
                    break
        cap.release()
        if frame is None:
            continue
        h, w = frame.shape[:2]
        colour = float(np.abs(frame[..., 0].astype(int) - frame[..., 2].astype(int)).mean())
        if abs(w / h - 16 / 9) < 0.05:
            found.setdefault("iphone", idx)
        elif colour > 3:
            found.setdefault("webcam", idx)
        else:
            found.setdefault("infrared", idx)
    return found


def camera_index(prefix: str) -> int | str | None:
    """FRONT_CAM / SIDE_CAM / WRIST_CAM from settings.env as a camera number. Accepts a number or a name
    ('iphone', 'webcam') that is looked up by what the camera delivers. None if not set or not found.
    'realsense' stays 'realsense': its colour image comes from realsense_bridge.py, see camera_config()."""
    value = S.get(f"{prefix}_CAM", "").strip().lower()
    if value.isdigit():
        return int(value)
    if value == "realsense":
        return value
    return identify_cameras().get(value) if value else None


REALSENSE_PORT = 5555


def _tolerant_camera_sizes() -> None:
    """The iPhone (and some webcams) switch picture size by themselves, e.g. 640x480 -> 1920x1080. LeRobot then
    stops with "frame width ... do not match". Resize such frames to the configured size instead."""
    from lerobot.cameras.opencv import camera_opencv

    cls = camera_opencv.OpenCVCamera
    if getattr(cls, "_tolerant", False):
        return
    original = cls._postprocess_image

    def postprocess(self, image):
        import cv2

        if image is not None and image.ndim == 3 and (image.shape[0], image.shape[1]) != (self.capture_height, self.capture_width):
            image = cv2.resize(image, (self.capture_width, self.capture_height), interpolation=cv2.INTER_AREA)
        return original(self, image)

    cls._postprocess_image = postprocess
    cls._tolerant = True


def camera_config(index: int | str, width: int, height: int):
    """LeRobot camera settings for a camera number, or for 'realsense' (the D435i's colour camera, passed on
    by realsense_bridge.py, which runs with sudo because macOS gives normal apps only its infrared image)."""
    _tolerant_camera_sizes()
    if index == "realsense":
        from lerobot.cameras.zmq import ZMQCameraConfig

        return ZMQCameraConfig(server_address="127.0.0.1", port=REALSENSE_PORT, camera_name="realsense",
                               width=640, height=480, fps=30, timeout_ms=3000)
    from lerobot.cameras.opencv import OpenCVCameraConfig

    return OpenCVCameraConfig(index_or_path=index, width=width, height=height, fps=30)


def make_robot(cameras: bool = False):
    """cameras=False (default) for poses and driving: faster to connect. cameras=True for recording and
    running a trained model; uses FRONT_CAM, SIDE_CAM and WRIST_CAM (if set) from settings.env. Camera access only
    works when the script runs in the Terminal app (macOS camera permission)."""
    if FAKE:
        return FakeRobot(cameras)
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    _retry_torque_writes()
    cams = {}
    if cameras:
        cams["front"] = camera_config(camera_index("FRONT"), int(S.get("FRONT_WIDTH") or 640),
                                      int(S.get("FRONT_HEIGHT") or 480))
        if camera_index("SIDE") is not None:
            cams["side"] = camera_config(camera_index("SIDE"), int(S.get("SIDE_WIDTH") or 640),
                                         int(S.get("SIDE_HEIGHT") or 480))
        if S.get("WRIST_CAM"):
            cams["wrist"] = camera_config(camera_index("WRIST"), 640, 480)
    return _sync_calibration_on_connect(SO101Follower(
        SO101FollowerConfig(
            port=S["FOLLOWER_PORT"],
            id=S["FOLLOWER_ID"],
            cameras=cams,
            # Safety: never jump more than 15 degrees in a single command.
            max_relative_target=15.0,
        )
    ))


class FakeLeader:
    """Stands in for the joystick: slowly swings the elbow so a fake recording has some motion."""

    def __init__(self):
        import time

        self.t0 = time.time()

    def connect(self, calibrate: bool = True):
        print("[fake] joystick connected")

    def disconnect(self):
        print("[fake] joystick disconnected")

    def get_action(self):
        import math
        import time

        action = {f"{j}.pos": 0.0 for j in JOINTS}
        action["elbow_flex.pos"] = 10 * math.sin(time.time() - self.t0)
        return action


def make_leader():
    if FAKE:
        return FakeLeader()
    from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

    _retry_torque_writes()

    return _sync_calibration_on_connect(SO101Leader(SO101LeaderConfig(port=S["LEADER_PORT"], id=S["LEADER_ID"])))
