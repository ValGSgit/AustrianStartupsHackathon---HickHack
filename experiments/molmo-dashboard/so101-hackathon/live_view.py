"""Live view for the team dashboard (robot-router), on while do_task.py runs under the team's robot agent.

hickhack/host/robot_agent.py starts do_task.py with ROBOT_LIVE=1 and forwards every stdout line

    @@LIVE {"state": [6 joints], "target": [6 joints], "jpeg": base64, "label": "duck (1/2) · model1 · 12 s"}

to the dashboard's Live panel (the same format as the team's host/live_hook.py). The picture shows what the model
sees: cam0 large, cam1 small in the band below, plus which object and which step this is. Without ROBOT_LIVE
nothing changes.

The control loop only hands over references; drawing and printing happen on a separate thread, so the live view
can't slow the robot down. When the loop stops reading the cameras (carrying, going home), this thread reads the
newest camera frame itself so the picture keeps moving.
"""

import base64
import json
import os
import sys
import threading
import time

import cv2
import numpy as np

ON = os.environ.get("ROBOT_LIVE") == "1"
HZ = 6  # ~25 KB per frame; the model's own uploads to Lightning share the same Wi-Fi
W, H, TOP = 448, 336, 252  # 4:3 like the dashboard's picture; the top 448x252 is 16:9 for cam0
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

_live = {"imgs": {}, "img_at": 0.0, "state": None, "target": None, "robot": None, "sup": None,
         "label": "", "names": {}, "t0": time.time(), "dirty": False}
_thread: threading.Thread | None = None


def _joints(d: dict) -> list[float] | None:
    vals = [d.get(f"{j}.pos") for j in JOINTS]
    return None if any(v is None for v in vals) else [round(float(v), 1) for v in vals]


def set_label(text: str, camera_names: dict[str, str] | None = None) -> None:
    """What the dashboard shows on the picture, e.g. 'duck (1/2)'. camera_names: {'cam0': 'iphone', ...}."""
    _live["label"] = text
    if camera_names is not None:
        _live["names"] = camera_names
    _live["dirty"] = True


def install(robot, sup=None) -> None:
    """Mirror this robot's camera frames, joint positions and commanded targets. Call before shrink_images()."""
    if not ON:
        return
    observe, act = robot.get_observation, robot.send_action

    def get_observation():
        obs = observe()
        try:  # the live view must never break the control loop
            _live["state"] = _joints(obs) or _live["state"]
            imgs = {k: v for k, v in obs.items() if isinstance(v, np.ndarray) and v.ndim == 3}
            if imgs:
                _live["imgs"], _live["img_at"] = imgs, time.time()
            _live["dirty"] = True
        except Exception:  # noqa: BLE001
            pass
        return obs

    def send_action(action):
        out = act(action)
        try:
            _live["target"] = _joints(out) or _live["target"]
            _live["dirty"] = True
        except Exception:  # noqa: BLE001
            pass
        return out

    robot.get_observation, robot.send_action = get_observation, send_action
    _live["robot"] = robot
    _start()


def watch(sup) -> None:
    """The assist Supervisor: its phase (model1, carry, reopen, home) goes into the label."""
    _live["sup"] = sup


def final(text: str) -> None:
    """One last frame with the outcome, so the dashboard ends on the result."""
    if ON:
        _live["sup"] = None  # the run is over: no phase any more
        set_label(text)
        _emit(force=True)


def _start() -> None:
    global _thread
    if _thread is None:
        _thread = threading.Thread(target=_loop, daemon=True)
        _thread.start()


def _loop() -> None:
    while True:
        time.sleep(1 / HZ)
        try:
            _refresh_cameras()
            _emit()
        except Exception as e:  # noqa: BLE001
            print(f"live view: {e}", file=sys.stderr, flush=True)


def _refresh_cameras() -> None:
    """While the control loop isn't reading the cameras (carry, home), read their newest frame here."""
    robot = _live["robot"]
    if robot is None or time.time() - _live["img_at"] < 0.5 or not getattr(robot, "is_connected", False):
        return
    imgs = {}
    for name, cam in (getattr(robot, "cameras", None) or {}).items():
        try:
            imgs[name] = cam.async_read(timeout_ms=100)
        except Exception:  # noqa: BLE001
            pass
    if imgs:
        _live["imgs"], _live["img_at"], _live["dirty"] = imgs, time.time(), True


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - nh) // 2, (w - nw) // 2
    out[y:y + nh, x:x + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return out


def _text(img: np.ndarray, s: str, x: int, y: int, scale: float = 0.45, bold: int = 1) -> None:
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), bold + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), bold, cv2.LINE_AA)


def compose(imgs: dict[str, np.ndarray], label: str, names: dict[str, str]) -> np.ndarray | None:
    """cam0 on top, a band below with what is happening and cam1 small. RGB in, RGB out."""
    keys = sorted(imgs)
    if not keys:
        return None
    canvas = np.full((H, W, 3), 28, np.uint8)
    canvas[:TOP] = _fit(np.asarray(imgs[keys[0]]), W, TOP)
    _text(canvas, f"{keys[0]}: {names.get(keys[0], keys[0])}", 8, 18)
    band_w = W
    if len(keys) > 1:
        bw, bh = (H - TOP) * 4 // 3, H - TOP
        canvas[TOP:, W - bw:] = _fit(np.asarray(imgs[keys[1]]), bw, bh)
        _text(canvas, names.get(keys[1], keys[1]), W - bw + 4, TOP + 14, 0.38)
        band_w = W - bw
    # OpenCV's font is ASCII only ("·" would print as "??"); the dashboard badge keeps the original label
    label = label.replace(" · ", " | ").encode("ascii", "replace").decode()
    words, lines, line = label.split(" "), [], ""
    max_chars = band_w // 10
    for w in words:  # wrap the label into the band
        if len(line) + len(w) + 1 > max_chars and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    for i, ln in enumerate(lines[:4]):
        _text(canvas, ln, 8, TOP + 22 + i * 20, 0.5)
    return canvas


def _emit(force: bool = False) -> None:
    if not (_live["dirty"] or force):
        return
    _live["dirty"] = False
    sup = _live["sup"]
    phase = getattr(sup, "phase", "") if sup is not None else ""
    label = " · ".join(p for p in (_live["label"], phase, f"{time.time() - _live['t0']:.0f} s") if p)
    img = compose(_live["imgs"], label, _live["names"])
    jpeg = None
    if img is not None:
        ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 65])
        jpeg = base64.b64encode(buf.tobytes()).decode() if ok else None
    msg = {"state": _live["state"], "target": _live["target"], "jpeg": jpeg, "label": label}
    # one write starting with a newline: a print from another thread can't glue itself to the front of this line
    sys.stdout.write("\n@@LIVE " + json.dumps(msg) + "\n")
    sys.stdout.flush()
