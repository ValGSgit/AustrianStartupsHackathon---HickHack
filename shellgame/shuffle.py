"""Run a shell-game shuffle on the arm without the stage screen (motion test).

    .venv\\Scripts\\python shellgame\\shuffle.py --fake
    .venv\\Scripts\\python shellgame\\shuffle.py --slow-test          # speed 0.4, Enter between moves
    .venv\\Scripts\\python shellgame\\shuffle.py --moves 7 --speed 1.0 --seed 3 --ball 2
    .venv\\Scripts\\python shellgame\\shuffle.py --seen --slow-test   # camera finds each cup, grab it where it is

Ctrl+C at any time: the arm goes home slowly and torque is switched off.
"""

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

import arm as A
import game as G
import motions as M

FAKE_SETUP = {"slots": [[140, 300, 45], [260, 330, 45], [380, 330, 45], [500, 300, 45]], "frame_size": [640, 480],
              "cup_hsv": [[100, 120, 60], [130, 255, 255]], "ball_hsv": [[5, 150, 120], [20, 255, 255]]}
SEEN_DIST = 1.5  # a detected cup belongs to a slot within this many slot radii


# ---------------------------------------------------------------- camera-guided (--seen), shared with show.py

def seen_setup(path: str | None, fake: bool) -> dict:
    import vision as V
    p = Path(path) if path else V.SETUP_FILE
    if fake and not p.exists():
        print("fake run: generated vision setup (no vision_setup.json)")
        return FAKE_SETUP
    return V.load_setup(p)


def seen_model(fake: bool):
    try:
        import reach as R
        return R.load_model()
    except (ImportError, FileNotFoundError, ValueError) as e:
        if not fake:
            sys.exit(f"--seen needs the reach model: {e}")
        print(f"fake run: fake reach model ({e})")
        return M.FakeModel(FAKE_SETUP["frame_size"][0])


def slot_centres(setup: dict, size) -> list[tuple[float, float, float]]:
    """Slot (x, y, r) in pixels of a frame of this (w, h)."""
    fw, fh = setup.get("frame_size", size)
    sx, sy = size[0] / fw, size[1] / fh
    return [(x * sx, y * sy, r * (sx + sy) / 2) for x, y, r in setup["slots"]]


def fake_cups(setup: dict, s: G.State, rng: random.Random) -> list[dict]:
    """Detections a camera would give: one per occupied slot, a little off the mark."""
    out = []
    for i, (x, y, r) in enumerate(slot_centres(setup, setup["frame_size"])):
        if s.cups[i] is not None:
            u, v = x + rng.uniform(-0.4, 0.4) * r, y + rng.uniform(-0.4, 0.4) * r
            out.append({"uv": (u, v), "px": int(3.14 * r * r), "bbox": (round(u - r), round(v - r), round(2 * r), round(2 * r))})
    return out


def plan(cups: list[dict], setup: dict, size, src: int, dst: int | None = None):
    """(src_uv, dst_uv, found): the detected cup nearest slot src (else its slot centre), dst = slot centre."""
    import vision as V
    # a detected cup's centre (whole visible cup) sits a fixed offset from the clicked spot centre (cup top);
    # samples were taught in detected-cup coordinates, so spots are converted to them too
    ox, oy = setup.get("cup_uv_offset", (0, 0))
    slots = [(x + ox, y + oy, r) for x, y, r in slot_centres(setup, size)]
    x, y, r = slots[src - 1]
    cup = V.nearest_cup(cups, (x, y), SEEN_DIST * r)
    dst_uv = slots[dst - 1][:2] if dst else None
    return (tuple(cup["uv"]) if cup else (x, y)), dst_uv, cup is not None


def draw_plan(img: np.ndarray, cups: list[dict], src_uv, dst_uv, found: bool, size) -> np.ndarray:
    """Detected cups + the planned src -> dst arrow ("the robot sees"), in img's pixels."""
    sx, sy = img.shape[1] / size[0], img.shape[0] / size[1]
    pt = lambda uv: (round(uv[0] * sx), round(uv[1] * sy))  # noqa: E731
    out = img.copy()
    for c in cups:
        x, y, w, h = c["bbox"]
        cv2.circle(out, pt(c["uv"]), max(8, round(max(w * sx, h * sy) / 2)), (60, 200, 60), 2)
    col = (0, 210, 255) if found else (40, 40, 230)
    cv2.drawMarker(out, pt(src_uv), col, cv2.MARKER_CROSS, 26, 3)
    if dst_uv:
        cv2.arrowedLine(out, pt(src_uv), pt(dst_uv), col, 3, tipLength=0.15)
        cv2.circle(out, pt(dst_uv), 14, col, 2)
    cv2.putText(out, "robot sees" if found else "cup NOT seen: using the mark", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
    return out


def camera_frame(image: str | None):
    import selfcheck
    if image:
        f = cv2.imread(image)
        if f is None:
            sys.exit(f"Cannot read image {image}")
        return f
    c = selfcheck.camera_cfg()
    return selfcheck.grab_frame(c["index"], c["width"], c["height"])


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="no arm (FakeArm); fake poses if poses.json is missing")
    ap.add_argument("--moves", type=int, default=7)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ball", type=int, default=2, choices=(1, 2, 3), help="start slot of the ball")
    ap.add_argument("--slow-test", action="store_true", help="speed 0.4 and wait for Enter between moves")
    ap.add_argument("--seen", action="store_true", help="camera-guided: grab each cup where the camera sees it")
    ap.add_argument("--setup", help="vision setup file (default shellgame/vision_setup.json)")
    ap.add_argument("--image", help="--seen: use this still image instead of the camera (testing)")
    args = ap.parse_args()

    A.SPEED = 0.4 if args.slow_test else args.speed
    poses, fake_poses = A.load_poses(), False
    need = lambda p: [n for n in ("home", "ready") if n not in p] if args.seen else M.missing_poses(p)  # noqa: E731
    if args.fake and need(poses):
        poses, fake_poses = M.fake_poses(), True
        print("fake run: using generated poses (poses.json missing or incomplete)")
    missing = need(poses)
    if missing:
        sys.exit(f"poses not taught: {', '.join(missing)}\nrun shellgame/teach.py first")

    seed = args.seed if args.seed is not None else random.randrange(10**6)
    rng = random.Random(seed)
    s = G.new_state(args.ball)
    moves = G.random_moves(s, args.moves, random.Random(seed))
    print(f"seed {seed}, speed {A.SPEED}, {len(moves)} moves, ball starts in slot {G.ball_slot(s)}")
    if args.seen:
        import vision as V
        setup, model = seen_setup(args.setup, args.fake), seen_model(args.fake)

    def target(src, dst=None):
        """--seen: arm at ready, look, return (src_uv, dst_uv, frame size)."""
        M.go_ready(arm, poses)
        if args.fake and not args.image:
            size, cups = setup["frame_size"], fake_cups(setup, s, rng)
        else:
            frame = camera_frame(args.image)
            size, cups = (frame.shape[1], frame.shape[0]), V.find_cups(frame, setup)
        src_uv, dst_uv, found = plan(cups, setup, size, src, dst)
        print(f"  camera: {len(cups)} cups; slot {src} cup at ({src_uv[0]:.0f}, {src_uv[1]:.0f})"
              + ("" if found else "  WARNING: no cup seen near the slot, using the slot centre"), flush=True)
        return src_uv, dst_uv, size

    arm = A.connect(fake=args.fake)
    try:
        M.go_ready(arm, poses)
        print(f"showing the ball in slot {G.ball_slot(s)}", flush=True)
        if args.seen:
            uv, _, size = target(G.ball_slot(s))
            M.reveal_seen(arm, model, uv, frame_size=size)
        else:
            M.show_ball(arm, poses, G.ball_slot(s))
        for k, move in enumerate(moves, 1):
            if args.slow_test:
                input(f"Enter = move {k}: {move[0]} -> {move[1]} ")
            ball_moves = move[0] == G.ball_slot(s)
            if args.seen:
                src_uv, dst_uv, size = target(*move)
                M.move_cup_seen(arm, model, src_uv, dst_uv, frame_size=size)
            else:
                M.move_cup(arm, poses, *move)
            s = G.apply(s, move)
            print(f"move {k}: {move[0]} -> {move[1]}{'  (ball cup)' if ball_moves else ''}"
                  f"   ball in slot {G.ball_slot(s)}, empty {G.empty_slot(s)}", flush=True)
        M.go_ready(arm, poses)
        print(f"done: ball is in slot {G.ball_slot(s)}")
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C)")
    finally:
        if fake_poses:  # release() would look for home in poses.json
            A.go_to(arm, poses["home"], 2.0 * A.SPEED)
            arm.disconnect()
        else:
            A.release(arm)
        if args.fake:
            print(f"fake arm: {arm.commands} commands sent")


if __name__ == "__main__":
    main()
