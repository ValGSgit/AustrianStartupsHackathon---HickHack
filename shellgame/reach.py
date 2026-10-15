"""Camera-guided reach: predict the arm poses for a cup seen at image point (u, v) from taught samples.

Per kind (above/grip/in/lift) and per joint: ridge least squares on [1, u, v, uv, u^2, v^2] (>= 6 samples) or
[1, u, v] (3-5 samples), u, v normalised by the frame size. Gripper: mean of the samples (not regressed).

    .venv\\Scripts\\python shellgame\\reach.py check                    # LOO error per kind/joint, coverage box
    .venv\\Scripts\\python shellgame\\reach.py go U V [--kind grip]     # ready -> above -> kind, Enter between steps
    .venv\\Scripts\\python shellgame\\reach.py go-cup [--image PATH]    # camera finds the one cup, arm goes there
All take --fake (no arm) and --speed.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

import arm as A

HERE = Path(__file__).resolve().parent
SAMPLES_FILE = HERE / "reach_samples.json"
KINDS = ("above", "grip", "in", "lift")
REG_KEYS = [k for k in A.KEYS if k != "gripper.pos"]
RIDGE = 1e-3
WARN_DEG = 5.0


def n_terms(n_samples: int) -> int:
    """3 = linear; 4 = + u^2 (a row of cups: the arm bends differently in the middle than at the ends); 6 = full."""
    return 6 if n_samples >= 6 else 4 if n_samples >= 4 else 3


def features(uv: np.ndarray, terms) -> np.ndarray:
    terms = 6 if terms is True else 3 if terms is False else terms
    u, v = uv[:, 0], uv[:, 1]
    cols = [np.ones_like(u), u, v, u * u, u * v, v * v][:terms]
    return np.stack(cols, axis=1)


def solve(uv: np.ndarray, y: np.ndarray) -> tuple:
    """Ridge on the non-constant terms. uv (n, 2) normalised, y (n, joints) -> (centre, scale, coefficients).
    uv is standardised over the samples first, so the ridge shrinks every term about equally."""
    if len(uv) < 3:
        raise ValueError(f"need at least 3 samples, got {len(uv)}")
    # one common scale for u and v: cups in a row barely differ in v, and a per-axis scale would blow that noise up
    c, sc = uv.mean(axis=0), max(float(uv.std(axis=0).max()), 1e-3)
    X = features((uv - c) / sc, n_terms(len(uv)))
    D = np.eye(X.shape[1]) * RIDGE
    D[0, 0] = 0.0
    coef = np.linalg.solve(X.T @ X + D, X.T @ y)
    # + thin-plate spline on what the polynomial misses: the model then hits every taught sample exactly
    # (a cup on a taught spot gets exactly the taught pose) and bends smoothly in between
    pts = (uv - c) / sc
    resid = y - X @ coef
    K = tps(pts, pts) + np.eye(len(pts)) * TPS_SMOOTH
    P = np.c_[np.ones(len(pts)), pts]
    A = np.block([[K, P], [P.T, np.zeros((3, 3))]])
    sol = np.linalg.solve(A, np.r_[resid, np.zeros((3, resid.shape[1]))])
    return c, sc, coef, pts, sol


TPS_SMOOTH = 1e-4


def tps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    r2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.where(r2 > 0, 0.5 * r2 * np.log(np.maximum(r2, 1e-12)), 0.0)


def apply(fit: tuple, uv: np.ndarray) -> np.ndarray:
    c, sc, coef = fit[:3]
    q = (uv - c) / sc
    out = features(q, coef.shape[0]) @ coef
    if len(fit) == 5:
        pts, sol = fit[3], fit[4]
        out = out + tps(q, pts) @ sol[:len(pts)] + np.c_[np.ones(len(q)), q] @ sol[len(pts):]
    return out


class ReachModel:
    def __init__(self, samples: list[dict]):
        self.samples = samples
        self.frame_size = tuple(samples[0].get("frame_size", [640, 480])) if samples else (640, 480)
        self.uv = {}       # kind -> (n, 2) normalised uv of the samples that have this kind
        self.y = {}        # kind -> (n, joints)
        self.coef = {}     # kind -> (centre, scale, coefficients (features, joints))
        self.gripper = {}  # kind -> mean gripper

    @classmethod
    def fit(cls, samples: list[dict]) -> "ReachModel":
        m = cls(samples)
        for kind in KINDS:
            have = [s for s in samples if kind in s.get("poses", {})]
            if len(have) < 3:
                raise ValueError(f"'{kind}': need at least 3 samples with this pose, got {len(have)} "
                                 f"(teach more with teach.py sample)")
            m.uv[kind] = np.array([[s["uv"][0] / s.get("frame_size", m.frame_size)[0],
                                    s["uv"][1] / s.get("frame_size", m.frame_size)[1]] for s in have])
            m.y[kind] = np.array([[s["poses"][kind][k] for k in REG_KEYS] for s in have], float)
            m.coef[kind] = solve(m.uv[kind], m.y[kind])
            m.gripper[kind] = float(np.mean([s["poses"][kind]["gripper.pos"] for s in have]))
        return m

    def norm(self, u: float, v: float, frame_size=None) -> np.ndarray:
        w, h = frame_size or self.frame_size
        return np.array([[u / w, v / h]])

    def predict(self, u: float, v: float, frame_size=None) -> dict[str, dict[str, float]]:
        """Poses for a cup at pixel (u, v) (in frame_size, default: the samples' frame size). Warns outside coverage."""
        if not self.inside(u, v, frame_size=frame_size):
            warnings.warn(f"reach: ({u:.0f}, {v:.0f}) is outside the taught samples' coverage, extrapolating",
                          stacklevel=2)
        uv = self.norm(u, v, frame_size)
        out = {}
        for kind in KINDS:
            q = apply(self.coef[kind], uv)[0]
            out[kind] = {**{k: float(x) for k, x in zip(REG_KEYS, q)}, "gripper.pos": self.gripper[kind]}
        return out

    def coverage(self) -> dict[str, tuple[float, float]]:
        """Min/max u and v of the samples, in pixels of self.frame_size."""
        uv = np.vstack(list(self.uv.values())) * np.array(self.frame_size)
        return {"u": (float(uv[:, 0].min()), float(uv[:, 0].max())),
                "v": (float(uv[:, 1].min()), float(uv[:, 1].max()))}

    def inside(self, u: float, v: float, margin: float = 0.1, frame_size=None) -> bool:
        """(u, v) pixels within the samples' bounding box, grown by margin * box size on each side."""
        p = self.norm(u, v, frame_size)[0] * np.array(self.frame_size)
        cov = self.coverage()
        for x, (lo, hi) in zip(p, (cov["u"], cov["v"])):
            pad = margin * (hi - lo)
            if not lo - pad <= x <= hi + pad:
                return False
        return True

    def loo_joint_errors(self) -> dict[str, dict[str, float]]:
        """Leave-one-out mean abs error (deg) per kind per joint. Each fold refits with n-1 samples."""
        out = {}
        for kind in KINDS:
            uv, y = self.uv[kind], self.y[kind]
            if len(uv) < 4:  # a fold would have < 3 samples
                out[kind] = {k: float("nan") for k in REG_KEYS}
                continue
            errs = []
            for i in range(len(uv)):
                keep = np.arange(len(uv)) != i
                errs.append(np.abs(apply(solve(uv[keep], y[keep]), uv[i:i + 1])[0] - y[i]))
            out[kind] = dict(zip(REG_KEYS, (float(e) for e in np.mean(errs, axis=0))))
        return out

    def loo_errors(self) -> dict[str, float]:
        """Leave-one-out mean abs error (deg) per kind, averaged over the regressed joints."""
        return {kind: float(np.mean(list(j.values()))) for kind, j in self.loo_joint_errors().items()}


def load_model(path: Path = SAMPLES_FILE) -> ReachModel:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing. Teach samples: .venv\\Scripts\\python shellgame\\teach.py sample")
    try:
        samples = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path.name} is not valid JSON: {e}") from e
    return ReachModel.fit(samples)


# ---------------------------------------------------------------- CLI

def step(arm, name: str, pose: dict, seconds: float = 2.5) -> bool:
    """Wait for Enter, then move. False = user typed q (stop)."""
    if input(f"Enter = go to {name}, q = stop: ").strip().lower() == "q":
        return False
    A.go_to(arm, pose, seconds)
    print(f"  at {name}: " + " ".join(f"{k[:-4]}={pose[k]:.1f}" for k in A.KEYS), flush=True)
    return True


def run_steps(arm, steps: list[tuple[str, dict]]) -> None:
    try:
        for name, pose in steps:
            if not step(arm, name, pose):
                break
    finally:
        A.release(arm)


def get_model(a) -> ReachModel:
    try:
        return load_model(a.samples or SAMPLES_FILE)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))


def need_ready() -> dict:
    ready = A.load_poses().get("ready")
    if not ready:
        sys.exit("no 'ready' pose in poses.json. Teach it: .venv\\Scripts\\python shellgame\\teach.py")
    return ready


def cmd_check(a):
    m = get_model(a)
    print(f"{len(m.samples)} samples, frame {m.frame_size[0]}x{m.frame_size[1]}")
    joints = m.loo_joint_errors()
    print(f"LOO mean abs error (deg)   {'  '.join(f'{k[:-4]:>13}' for k in REG_KEYS)}     mean")
    bad = []
    for kind, j in joints.items():
        n = len(m.uv[kind])
        model = {6: "quadratic", 4: "row (u^2)", 3: "linear"}[n_terms(n - 1)] if n - 1 >= 3 else "too few"
        print(f"  {kind:<6} ({n} smp, {model:<9}) " + "  ".join(f"{e:13.2f}" for e in j.values())
              + f"  {np.mean(list(j.values())):7.2f}")
        bad += [f"{kind}.{k[:-4]} {e:.1f}" for k, e in j.items() if e > WARN_DEG]
    if bad:
        print(f"WARNING: LOO error > {WARN_DEG:.0f} deg: {', '.join(bad)}. Add samples there or re-teach outliers.")
    cov = m.coverage()
    print(f"coverage: u {cov['u'][0]:.0f}..{cov['u'][1]:.0f}  v {cov['v'][0]:.0f}..{cov['v'][1]:.0f} px")


def cmd_go(a):
    m = get_model(a)
    if not m.inside(a.u, a.v):
        print(f"WARNING: ({a.u:.0f}, {a.v:.0f}) is outside the samples' coverage {m.coverage()}: extrapolating")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = m.predict(a.u, a.v)
    ready = need_ready()
    above = p["above"]
    target = A.with_gripper(p[a.kind], above["gripper.pos"]) if a.kind == "grip" else p[a.kind]
    steps = [("ready", ready), ("above", above)]
    if a.kind != "above":
        steps += [(a.kind, target), ("above", above)]
    run_steps(A.connect(fake=a.fake), steps + [("ready", ready)])


def cmd_go_cup(a):
    import vision as V
    m = get_model(a)
    ready = need_ready()
    setup = V.load_setup(Path(a.setup) if a.setup else None)
    arm = A.connect(fake=a.fake)
    try:
        print("moving to ready (out of the camera's view) ...", flush=True)
        A.go_to(arm, ready, 2.5)
        if a.image:
            import cv2
            frame = cv2.imread(a.image)
            if frame is None:
                sys.exit(f"Cannot read image {a.image}")
        else:
            if not a.fake:
                import time
                time.sleep(0.5)  # arm settled, no motion blur / shadow change
            frame = V.grab_frame(a.camera)
        try:
            cups = V.find_cups(frame, setup)
        except ValueError as e:
            sys.exit(str(e))
        if len(cups) != 1:
            sys.exit(f"need exactly one cup on the table, found {len(cups)}: "
                     + ", ".join(f"({c['uv'][0]:.0f},{c['uv'][1]:.0f}) {c['px']}px" for c in cups))
        u, v = cups[0]["uv"]
        fs = (frame.shape[1], frame.shape[0])
        inside = m.inside(u, v, frame_size=fs)
        print(f"cup at u={u:.0f} v={v:.0f} ({cups[0]['px']} px), {'inside' if inside else 'OUTSIDE'} the coverage "
              f"{ {k: tuple(round(x) for x in r) for k, r in m.coverage().items()} }")
        if not inside:
            print("WARNING: extrapolating outside the taught samples, poses may be wrong. Hand on the arm, q stops.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p = m.predict(u, v, frame_size=fs)
    except BaseException:
        A.release(arm)
        raise
    above, opened, closed = p["above"], p["above"]["gripper.pos"], p["grip"]["gripper.pos"]
    steps = [("above", above), ("grip (open)", A.with_gripper(p["grip"], opened))]
    if a.lift:  # the full reveal: close on the cup, lift it, put it back, let go
        steps += [("grip (CLOSE)", p["grip"]), ("lift", A.with_gripper(p["lift"], closed)),
                  ("put back", p["grip"]), ("open", A.with_gripper(p["grip"], opened))]
    run_steps(arm, steps + [("above", above), ("ready", ready)])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("check", help="sample count, LOO errors, coverage")
    s.set_defaults(fn=cmd_check)
    s = sub.add_parser("go", help="go to the predicted pose for pixel U V")
    s.add_argument("u", type=float)
    s.add_argument("v", type=float)
    s.add_argument("--kind", choices=KINDS, default="above")
    s.set_defaults(fn=cmd_go)
    s = sub.add_parser("go-cup", help="camera finds the one cup, arm goes above it and to grip (open)")
    s.add_argument("--image", help="use a saved image instead of the camera")
    s.add_argument("--camera", type=int, help="override the [cameras.front] index from config")
    s.add_argument("--setup", help="vision setup JSON (default shellgame/vision_setup.json)")
    s.add_argument("--lift", action="store_true", help="also close the gripper, lift the cup, put it back")
    s.set_defaults(fn=cmd_go_cup)
    for s in sub.choices.values():
        s.add_argument("--fake", action="store_true", help="no arm (FakeArm)")
        s.add_argument("--speed", type=float, default=1.0)
        s.add_argument("--samples", help="samples JSON (default shellgame/reach_samples.json)")
    a = p.parse_args()
    A.SPEED = a.speed
    a.fn(a)


if __name__ == "__main__":
    main()
