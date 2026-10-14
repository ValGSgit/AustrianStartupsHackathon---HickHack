"""Deterministic "is the object in the target zone?" check from one camera frame. No AI, works offline.

Pen task: the target zone is the AIR above the pick area (as the camera sees it), so "pen in zone for 1 s" = lifted.
Use a brightly coloured pen (or wrap coloured tape around it): colour detection cannot find a black or white pen.

Setup (once per stage layout, camera must be fixed and taped):
    .venv\\Scripts\\python selfcheck.py setup     # click 4 corners of the target zone, click the pen, press s
    .venv\\Scripts\\python selfcheck.py live      # live view: mask + verdict, to test lighting

The setup is saved to stage_setup.json in the repo. Commit it together with the photos of the tape marks,
so every laptop checks the same box region.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SETUP_FILE = ROOT / "stage_setup.json"


def camera_cfg() -> dict:
    # Same camera as the policy: [cameras.front] from config.toml / config.local.toml.
    from robot import load_config
    return load_config()["cameras"]["front"]

GREEN, RED, YELLOW, WHITE = (60, 200, 60), (40, 40, 230), (0, 210, 255), (255, 255, 255)


# ---------------------------------------------------------------- camera

def open_camera(index: int | None = None, width: int = 640, height: int = 480) -> cv2.VideoCapture:
    if index is None:
        c = camera_cfg()
        index, width, height = c["index"], c["width"], c["height"]
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        sys.exit(f"Camera {index} did not open. Is another process (rollout, teleop) still holding it?")
    return cap


def grab_frame(index: int | None = None, width: int = 640, height: int = 480, warmup: int = 10) -> np.ndarray:
    """Open, let auto-exposure settle, take one frame, release. Only one process can hold a webcam on Windows."""
    cap = open_camera(index, width, height)
    try:
        frame = None
        for _ in range(warmup):
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            sys.exit("Camera opened but returned no frames.")
        return frame
    finally:
        cap.release()


# ---------------------------------------------------------------- check

def load_setup() -> dict:
    if not SETUP_FILE.exists():
        sys.exit(f"{SETUP_FILE.name} missing. Run: python selfcheck.py setup")
    return json.loads(SETUP_FILE.read_text(encoding="utf-8"))


def cube_mask(frame: np.ndarray, hsv_lo, hsv_hi) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lo, hi = np.array(hsv_lo), np.array(hsv_hi)
    if lo[0] <= hi[0]:
        mask = cv2.inRange(hsv, lo, hi)
    else:  # hue wraps around 180 (red)
        mask = cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]])) | cv2.inRange(hsv, np.array([0, lo[1], lo[2]]), hi)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def check(frame: np.ndarray, setup: dict) -> tuple[bool, dict, np.ndarray]:
    """Returns (cube_in_box, stats, annotated frame)."""
    mask = cube_mask(frame, setup["hsv_lo"], setup["hsv_hi"])
    box = np.array(setup["box"], np.int32)
    box_mask = np.zeros(mask.shape, np.uint8)
    cv2.fillPoly(box_mask, [box], 255)
    in_box = int(cv2.countNonZero(mask & box_mask))
    total = int(cv2.countNonZero(mask))
    # Enough cube inside the box, and most of the visible cube: a cube on the rim does not count.
    ok = in_box >= setup["min_pixels"] and in_box >= setup.get("min_fraction", 0.8) * total

    out = frame.copy()
    tint = np.zeros_like(out)
    tint[mask > 0] = YELLOW
    out = cv2.addWeighted(out, 1.0, tint, 0.6, 0)
    cv2.polylines(out, [box], True, GREEN if ok else RED, 3)
    stats = {"in_box_px": in_box, "cube_px": total, "in_box_frac": round(in_box / total, 2) if total else 0.0}
    return ok, stats, out


# ---------------------------------------------------------------- setup UI

def sample_hsv(frame: np.ndarray, x: int, y: int, r: int = 6) -> tuple[list, list]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    patch = hsv[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
    h, s, v = np.median(patch, axis=0).astype(int)
    dh = 10
    lo = [int((h - dh) % 180), max(60, int(s) - 70), max(40, int(v) - 90)]
    hi = [int((h + dh) % 180), 255, 255]
    return lo, hi


def cmd_setup(a):
    cap = open_camera(a.camera)
    pts: list[tuple[int, int]] = []
    state = {"frame": None, "hsv": None}

    def on_click(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN or state["frame"] is None:
            return
        if len(pts) < 4:
            pts.append((x, y))
        else:
            state["hsv"] = sample_hsv(state["frame"], x, y)
            print(f"cube colour: lo={state['hsv'][0]} hi={state['hsv'][1]}")

    win = "selfcheck setup"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    print("Click 4 corners of the target zone (pen task: the air above the pick area), then click the pen. "
          "r = reset, s = save, q = quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        state["frame"] = frame
        view = frame.copy()
        for p in pts:
            cv2.circle(view, p, 5, GREEN, -1)
        if len(pts) >= 2:
            cv2.polylines(view, [np.array(pts, np.int32)], len(pts) == 4, GREEN, 2)
        if len(pts) == 4 and state["hsv"]:
            setup = {"box": pts, "goal": a.goal, "hold_s": a.hold_s, "hsv_lo": state["hsv"][0], "hsv_hi": state["hsv"][1],
                     "min_pixels": a.min_pixels,
                     "min_fraction": a.min_fraction}
            ok_, stats, view = check(frame, setup)
            msg = f"IN ZONE  {stats}" if ok_ else f"not in zone  {stats}"
            cv2.putText(view, msg, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if ok_ else RED, 2)
        else:
            step = "click zone corner %d/4" % (len(pts) + 1) if len(pts) < 4 else "click the pen"
            cv2.putText(view, step, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"):
            pts.clear()
            state["hsv"] = None
        elif k == ord("s") and len(pts) == 4 and state["hsv"]:
            SETUP_FILE.write_text(json.dumps(setup, indent=2), encoding="utf-8")
            cv2.imwrite(str(ROOT / "stage_setup.jpg"), frame)
            print(f"saved {SETUP_FILE.name} + stage_setup.jpg (reference photo of the layout)")
        elif k in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_live(a):
    setup = load_setup()
    cap = open_camera()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        t = time.perf_counter()
        ok_, stats, view = check(frame, setup)
        ms = (time.perf_counter() - t) * 1000
        cv2.putText(view, f"{'IN ZONE' if ok_ else 'not in zone'}  {stats}  {ms:.1f} ms", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN if ok_ else RED, 2)
        cv2.imshow("selfcheck live", view)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def cmd_once(a):
    setup = load_setup()
    ok, stats, view = check(grab_frame(), setup)
    cv2.imwrite(str(ROOT / "selfcheck_last.jpg"), view)
    print(("IN ZONE" if ok else "NOT IN ZONE"), stats)
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="mark the target zone and sample the object colour")
    s.add_argument("--camera", type=int, help="override the [cameras.front] index from config")
    s.add_argument("--min-pixels", type=int, default=400, help="object pixels inside the zone that count as success")
    s.add_argument("--goal", default="pen lifted", help="what success is called on screen and in speech")
    s.add_argument("--hold-s", type=float, default=1.0, help="seconds the object must stay in the zone")
    s.add_argument("--min-fraction", type=float, default=0.6, help="share of the visible object that must be inside (pen: the gripper hides part of it)")
    s.set_defaults(fn=cmd_setup)
    sub.add_parser("live", help="live view of the check").set_defaults(fn=cmd_live)
    sub.add_parser("once", help="one check, exit code 0 = in box").set_defaults(fn=cmd_once)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
