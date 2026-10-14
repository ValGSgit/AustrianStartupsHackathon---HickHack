"""Shell game vision: which of the 4 taped slots hold a cup, is a cup off its mark, is the ball visible.
Deterministic HSV colour masks, no AI (same approach as selfcheck.py).

    .venv\\Scripts\\python shellgame\\vision.py setup [--camera N]   # click slot centres 1-4, edge of cup 1, a cup, the ball; s = save
    .venv\\Scripts\\python shellgame\\vision.py live                 # annotated live view, q quits
    .venv\\Scripts\\python shellgame\\vision.py once [--image PATH]  # one frame, prints slot states
    .venv\\Scripts\\python shellgame\\vision.py cups [--image PATH]  # live detected cups with their u,v, q quits

Saves shellgame/vision_setup.json + shellgame/vision_setup.jpg (reference photo of the layout).
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from selfcheck import cube_mask, grab_frame, open_camera  # noqa: E402  (camera: selfcheck.camera_cfg)
from selfcheck import sample_hsv as _sample_colour  # noqa: E402


def sample_hsv(frame: np.ndarray, x: int, y: int, r: int = 6) -> tuple[list, list]:
    """Colour range around a clicked point. White/grey objects (low saturation) get an any-hue, low-saturation,
    bright range: white cups on a dark board."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    patch = hsv[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1].reshape(-1, 3)
    _, s, v = np.median(patch, axis=0).astype(int)
    if s < 50:
        return [0, 0, max(90, int(v) - 70)], [179, min(110, int(s) + 50), 255]
    return _sample_colour(frame, x, y, r)

SETUP_FILE = HERE / "vision_setup.json"
GREEN, RED, ORANGE, GREY, YELLOW, WHITE = (60, 200, 60), (40, 40, 230), (0, 140, 255), (150, 150, 150), (0, 210, 255), (255, 255, 255)
SEARCH = 1.3  # cup search radius, in slot radii
BALL_NEAR = 2.0  # ball counts as "at slot i" within this many slot radii


def load_setup(path: Path | None = None) -> dict:
    path = path or SETUP_FILE
    if not path.exists():
        sys.exit(f"{path.name} missing. Run: .venv\\Scripts\\python shellgame\\vision.py setup")
    return json.loads(path.read_text(encoding="utf-8"))


def mask(frame: np.ndarray, hsv) -> np.ndarray:
    m = cube_mask(frame, hsv[0], hsv[1])  # opened 5x5, handles red hue wraparound
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def scaled(frame: np.ndarray, setup: dict) -> tuple[list[tuple[float, float, float]], float]:
    """Slots in this frame's pixels (setup may come from a different resolution) + pixel-area scale."""
    h, w = frame.shape[:2]
    fw, fh = setup.get("frame_size", [w, h])
    sx, sy = w / fw, h / fh
    return [(x * sx, y * sy, r * (sx + sy) / 2) for x, y, r in setup["slots"]], sx * sy


def disc(shape, x, y, r) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.circle(m, (round(x), round(y)), max(1, round(r)), 255, -1)
    return m


def slot_states(frame: np.ndarray, setup: dict) -> list[dict]:
    cup = mask(frame, setup["cup_hsv"])
    slots, area = scaled(frame, setup)
    out = []
    for x, y, r in slots:
        m = cup & disc(frame.shape, x, y, SEARCH * r)
        px = int(cv2.countNonZero(m))
        occ = px >= setup.get("min_cup_px", 400) * area
        dx = dy = 0
        if occ:
            mo = cv2.moments(m, binaryImage=True)
            dx, dy = round(mo["m10"] / mo["m00"] - x), round(mo["m01"] / mo["m00"] - y)
        off = occ and (dx * dx + dy * dy) ** 0.5 > setup.get("max_offset_frac", 0.5) * r
        out.append({"occupied": bool(occ), "px": px, "offset": (dx, dy), "off_center": bool(off)})
    return out


def ball_visible(frame: np.ndarray, setup: dict, slot: int | None = None) -> tuple[bool, dict]:
    """Ball pixels near slot (1-based, within BALL_NEAR * r) or anywhere in the frame."""
    m = mask(frame, setup["ball_hsv"])
    slots, area = scaled(frame, setup)
    if slot is not None:
        x, y, r = slots[slot - 1]
        m &= disc(frame.shape, x, y, BALL_NEAR * r)
    px = int(cv2.countNonZero(m))
    ok = px >= setup.get("min_ball_px", 80) * area
    pos = None
    if px:
        mo = cv2.moments(m, binaryImage=True)
        pos = (round(mo["m10"] / mo["m00"]), round(mo["m01"] / mo["m00"]))
    return bool(ok), {"px": px, "pos": pos, "slot": slot}


def check_layout(frame: np.ndarray, setup: dict, expected_empty: int) -> tuple[bool, list[str]]:
    problems = []
    for i, s in enumerate(slot_states(frame, setup), 1):
        if i == expected_empty:
            if s["occupied"]:
                problems.append(f"slot {i}: should be empty but a cup is there")
        elif not s["occupied"]:
            problems.append(f"slot {i}: cup missing")
        elif s["off_center"]:
            problems.append(f"slot {i}: cup off its mark (dx={s['offset'][0]:+d} px, dy={s['offset'][1]:+d} px)")
    return not problems, problems


def annotate(frame: np.ndarray, setup: dict, states: list[dict], ball=None) -> np.ndarray:
    """ball = result of ball_visible() (the (ok, info) tuple), or None."""
    out = frame.copy()
    slots, _ = scaled(frame, setup)
    for i, ((x, y, r), s) in enumerate(zip(slots, states), 1):
        c = (round(x), round(y))
        col = ORANGE if s["off_center"] else GREEN if s["occupied"] else GREY
        cv2.circle(out, c, round(r), col, 2)
        cv2.circle(out, c, 3, col, -1)
        if s["occupied"]:
            cv2.arrowedLine(out, c, (c[0] + s["offset"][0], c[1] + s["offset"][1]), col, 2, tipLength=0.3)
        label = f"{i}: {'OFF' if s['off_center'] else 'cup' if s['occupied'] else 'empty'}"
        cv2.putText(out, label, (c[0] - round(r), c[1] - round(r) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)
    if ball is not None:
        ok, info = ball
        if info["pos"] is not None:
            cv2.drawMarker(out, info["pos"], YELLOW if ok else GREY, cv2.MARKER_CROSS, 24, 3)
        txt = f"ball {'SEEN' if ok else 'not seen'} ({info['px']} px)"
        cv2.putText(out, txt, (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW if ok else GREY, 2)
    return out


def board_hull(frame: np.ndarray, min_frac: float = 0.08):
    """Outline of the dark board the cups stand on (largest dark area), or None. Cups are only searched inside it,
    so the white table, cables and the robot's wires around it never count as cups."""
    dark = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (0, 0, 0), (179, 255, 70))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, 4]))
    if stats[i][4] < min_frac * frame.shape[0] * frame.shape[1]:
        return None
    pts = cv2.findNonZero((labels == i).astype(np.uint8))
    return cv2.convexHull(pts)


def find_cups(frame: np.ndarray, setup: dict) -> list[dict]:
    """Cup-colour blobs anywhere in the frame: {"uv": (u, v), "px", "bbox": (x, y, w, h)}, largest first.
    Blobs touching the border are kept. Touching cups merge into one blob (cups must not touch)."""
    if not setup.get("cup_hsv"):
        raise ValueError("vision setup has no cup_hsv. Run: .venv\\Scripts\\python shellgame\\vision.py setup")
    _, area = scaled(frame, {**setup, "slots": []})
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask(frame, setup["cup_hsv"]))
    cups = [{"uv": (float(cents[i][0]), float(cents[i][1])), "px": int(stats[i][4]),
             "bbox": tuple(int(x) for x in stats[i][:4])}
            for i in range(1, n) if setup.get("min_cup_px", 400) * area <= stats[i][4] <= setup.get("max_cup_px", 15000) * area]
    hull = board_hull(frame) if setup.get("board_only", True) else None
    if hull is not None:
        cups = [c for c in cups if cv2.pointPolygonTest(hull, c["uv"], False) >= 0]
    return sorted(cups, key=lambda c: -c["px"])


def nearest_cup(cups: list[dict], uv, max_dist: float) -> dict | None:
    """Cup whose centroid is closest to uv (pixels), or None if none within max_dist."""
    best = min(cups, key=lambda c: np.hypot(c["uv"][0] - uv[0], c["uv"][1] - uv[1]), default=None)
    if best is None or np.hypot(best["uv"][0] - uv[0], best["uv"][1] - uv[1]) > max_dist:
        return None
    return best


def draw_cups(frame: np.ndarray, cups: list[dict]) -> np.ndarray:
    out = frame.copy()
    for i, c in enumerate(cups, 1):
        x, y, w, h = c["bbox"]
        u, v = round(c["uv"][0]), round(c["uv"][1])
        cv2.rectangle(out, (x, y), (x + w, y + h), GREEN, 2)
        cv2.drawMarker(out, (u, v), YELLOW, cv2.MARKER_CROSS, 18, 2)
        cv2.putText(out, f"{i}: ({u},{v}) {c['px']}px", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
    return out


# ---------------------------------------------------------------- CLI

def frames(a):
    """Yields frames from --image (repeated) or the camera (--camera or selfcheck.camera_cfg)."""
    if a.image:
        img = cv2.imread(a.image)
        if img is None:
            sys.exit(f"Cannot read image {a.image}")
        while True:
            yield img
    cap = open_camera(a.camera)
    try:
        while True:
            ok, f = cap.read()
            if ok:
                yield f
    finally:
        cap.release()


def cmd_setup(a):
    pts: list[tuple[int, int]] = []  # 4 slot centres, then a point on the edge of cup 1
    st = {"frame": None, "cup": None, "ball": None}

    def on_click(event, x, y, *_):
        if event != cv2.EVENT_LBUTTONDOWN or st["frame"] is None:
            return
        if len(pts) < 5:
            pts.append((x, y))
        elif st["cup"] is None:
            st["cup"] = sample_hsv(st["frame"], x, y)
            print(f"cup colour: lo={st['cup'][0]} hi={st['cup'][1]}")
        else:
            st["ball"] = sample_hsv(st["frame"], x, y)
            print(f"ball colour: lo={st['ball'][0]} hi={st['ball'][1]}")

    win = "vision setup"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_click)
    print("Click the centre of slots 1-4 (left to right), then the edge of cup 1, then a cup (colour), "
          "then the ball (colour). r = reset, s = save, q = quit.")
    setup = None
    for frame in frames(a):
        st["frame"] = frame
        view = frame.copy()
        r = int(np.hypot(pts[4][0] - pts[0][0], pts[4][1] - pts[0][1])) if len(pts) == 5 else 0
        if len(pts) == 5 and st["cup"] and st["ball"]:
            h, w = frame.shape[:2]
            setup = {"slots": [[x, y, r] for x, y in pts[:4]], "cup_hsv": st["cup"], "ball_hsv": st["ball"],
                     "min_cup_px": a.min_cup_px, "max_cup_px": a.max_cup_px, "min_ball_px": a.min_ball_px,
                     "max_offset_frac": a.max_offset_frac, "frame_size": [w, h]}
            states = slot_states(frame, setup)
            view = annotate(frame, setup, states, ball_visible(frame, setup))
            cv2.putText(view, "s = save, r = reset, q = quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
        else:
            for i, p in enumerate(pts[:4], 1):
                cv2.circle(view, p, 5, GREEN, -1)
                cv2.putText(view, str(i), (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
                if r:
                    cv2.circle(view, p, r, GREEN, 1)
            step = (f"click centre of slot {len(pts) + 1}/4" if len(pts) < 4 else
                    "click the EDGE of cup 1 (sets radius)" if len(pts) == 4 else
                    "click a cup (colour)" if st["cup"] is None else "click the ball (colour)")
            cv2.putText(view, step, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        cv2.imshow(win, view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"):
            pts.clear()
            st["cup"] = st["ball"] = None
            setup = None
        elif k == ord("s") and setup:
            SETUP_FILE.write_text(json.dumps(setup, indent=2), encoding="utf-8")
            cv2.imwrite(str(HERE / "vision_setup.jpg"), frame)
            print(f"saved {SETUP_FILE.name} + vision_setup.jpg (reference photo of the layout)")
        elif k in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def report(frame, setup):
    states = slot_states(frame, setup)
    ball = ball_visible(frame, setup)
    return states, ball, annotate(frame, setup, states, ball)


def cmd_live(a):
    setup = load_setup()
    for frame in frames(a):
        _, _, view = report(frame, setup)
        cv2.imshow("vision live", view)
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def cmd_once(a):
    setup = load_setup()
    frame = next(frames(a)) if a.image else grab_frame(a.camera)
    states, ball, view = report(frame, setup)
    for i, s in enumerate(states, 1):
        print(f"slot {i}: {'OFF-CENTRE' if s['off_center'] else 'cup' if s['occupied'] else 'empty'}  "
              f"px={s['px']} offset={s['offset']}")
    print(f"ball: {'seen' if ball[0] else 'not seen'} {ball[1]}")
    cv2.imwrite(str(HERE / "vision_last.jpg"), view)
    print("annotated frame: shellgame/vision_last.jpg")


def cmd_cups(a):
    setup = load_setup()
    last = None
    for frame in frames(a):
        cups = find_cups(frame, setup)
        msg = "  ".join(f"({c['uv'][0]:.0f},{c['uv'][1]:.0f})" for c in cups)
        if msg != last:
            print(f"{len(cups)} cup(s): {msg}", flush=True)
            last = msg
        cv2.imshow("vision cups", draw_cups(frame, cups))
        if cv2.waitKey(30) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [("setup", cmd_setup, "click slots, radius, cup and ball colours"),
                            ("live", cmd_live, "annotated live view"), ("once", cmd_once, "one frame, print states"),
                            ("cups", cmd_cups, "live detected cups with their u,v")]:
        s = sub.add_parser(name, help=help_)
        s.add_argument("--camera", type=int, help="override the [cameras.front] index from config")
        s.add_argument("--image", help="use a saved image instead of the camera")
        s.set_defaults(fn=fn)
        if name == "setup":
            s.add_argument("--min-cup-px", type=int, default=1500)
            s.add_argument("--max-cup-px", type=int, default=15000, help="bigger white areas (table edge) are not cups")
            s.add_argument("--min-ball-px", type=int, default=80)
            s.add_argument("--max-offset-frac", type=float, default=0.5)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
