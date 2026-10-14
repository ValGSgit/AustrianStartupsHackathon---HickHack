"""Synthetic-IK tests for reach.py (no hardware). Run: .venv\\Scripts\\python shellgame\\test_reach.py"""

import json
import math
import random
import tempfile
import warnings
from pathlib import Path

import arm as A
import reach as R

FRAME = [640, 480]
BASE = (320, 600)  # robot base, below the bottom edge of the image


def true_poses(u: float, v: float) -> dict:
    """Smooth fake IK: pan from the bearing to the base, the rest smooth in distance / v."""
    du, dv = u - BASE[0], BASE[1] - v
    pan = -math.degrees(math.atan2(du, dv))
    d = math.hypot(du, dv) / 100  # ~1.8 .. 4.5
    base = {"shoulder_pan.pos": pan, "wrist_roll.pos": 20 + 0.3 * pan}
    return {
        "above": {**base, "shoulder_lift.pos": -60 + 12 * d, "elbow_flex.pos": 70 - 18 * d + 1.5 * d * d,
                  "wrist_flex.pos": 90 - 4 * d, "gripper.pos": 35.0},
        "grip": {**base, "shoulder_lift.pos": -45 + 12 * d, "elbow_flex.pos": 75 - 16 * d + 1.2 * d * d,
                 "wrist_flex.pos": 70 - 3 * d, "gripper.pos": 19.0},
        "in": {**base, "shoulder_lift.pos": -80 + 5 * d, "elbow_flex.pos": 62 - 2 * d,
               "wrist_flex.pos": 92 - d, "gripper.pos": 21.0},
        "lift": {**base, "shoulder_lift.pos": -55 + 10 * d, "elbow_flex.pos": 70 - 10 * d + d * d,
                 "wrist_flex.pos": 65, "gripper.pos": 21.0},
    }


def make_samples(points, noise=0.3, seed=0) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for u, v in points:
        poses = {kind: {k: round(x + (rng.gauss(0, noise) if k != "gripper.pos" else rng.gauss(0, 1)), 2)
                        for k, x in p.items()} for kind, p in true_poses(u, v).items()}
        out.append({"uv": [u, v], "frame_size": FRAME, "poses": poses})
    return out


GRID8 = [(180, 240), (320, 230), (460, 240), (170, 330), (470, 330), (180, 420), (320, 430), (460, 420)]
HELD = [(250, 280), (390, 300), (320, 330), (230, 390), (420, 380), (320, 400)]


def max_err(model, points, frame_size=None) -> float:
    worst = 0.0
    for u, v in points:
        s = 1 if frame_size is None else frame_size[0] / FRAME[0]
        p, t = model.predict(u * s, v * s, frame_size=frame_size), true_poses(u, v)
        for kind in R.KINDS:
            worst = max(worst, *(abs(p[kind][k] - t[kind][k]) for k in R.REG_KEYS))
    return worst


def test_quadratic_fit():
    m = R.ReachModel.fit(make_samples(GRID8))
    assert all(c[2].shape[0] == 6 for c in m.coef.values())
    err = max_err(m, HELD)
    assert err < 2.0, err
    p = m.predict(320, 330)
    assert set(p) == set(R.KINDS) and all(set(q) == set(A.KEYS) for q in p.values()), p
    assert abs(p["above"]["gripper.pos"] - 35) < 1.5 and abs(p["grip"]["gripper.pos"] - 19) < 1.5, p


def test_loo():
    m = R.ReachModel.fit(make_samples(GRID8))
    loo = m.loo_errors()
    assert set(loo) == set(R.KINDS) and all(0 < e < 5 for e in loo.values()), loo
    joints = m.loo_joint_errors()
    assert all(set(j) == set(R.REG_KEYS) for j in joints.values()), joints


def test_few_samples():
    for n in (0, 1, 2):
        try:
            R.ReachModel.fit(make_samples(GRID8[:n]))
            raise AssertionError(f"{n} samples should raise")
        except ValueError:
            pass
    m = R.ReachModel.fit(make_samples([GRID8[i] for i in (0, 2, 5, 7)]))  # 4 corners -> linear
    assert all(c[2].shape[0] == 3 for c in m.coef.values())
    assert max_err(m, [(320, 330)]) < 6.0  # linear is coarse but sane in the middle


def test_inside_and_warning():
    m = R.ReachModel.fit(make_samples(GRID8))
    cov = m.coverage()
    assert cov["u"] == (170, 470) and cov["v"] == (230, 430), cov
    assert m.inside(320, 330) and m.inside(170, 230) and m.inside(160, 330)  # 160: within 10% margin (30 px)
    assert not m.inside(100, 330) and not m.inside(320, 100) and not m.inside(160, 330, margin=0.0)
    assert m.inside(640, 660, frame_size=(1280, 960))  # same point at double resolution
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m.predict(320, 330)
        assert not w
        m.predict(20, 20)
        assert len(w) == 1 and "outside" in str(w[0].message)


def test_frame_scaling():
    m = R.ReachModel.fit(make_samples(GRID8))
    assert max_err(m, HELD, frame_size=(1280, 960)) < 2.0


def test_load_model():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "reach_samples.json"
        try:
            R.load_model(p)
            raise AssertionError("missing file should raise")
        except FileNotFoundError:
            pass
        p.write_text("{nope", encoding="utf-8")
        try:
            R.load_model(p)
            raise AssertionError("bad JSON should raise")
        except ValueError:
            pass
        p.write_text(json.dumps(make_samples(GRID8)), encoding="utf-8")
        assert max_err(R.load_model(p), HELD) < 2.0


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
