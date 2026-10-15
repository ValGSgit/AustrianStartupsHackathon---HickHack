"""Predict the arm's movements for "put object N into box L" and measure whether they fulfil the task.
Nothing moves on the real arm.

1. look:     camera frame -> markers -> objects + the printed boxes, in robot coordinates (host/workspace.py)
2. intent:   optional --instruction: the Workers AI vision model picks the box letter and the object
3. plan:     the same skill code the arm runs (host/skills.py), on a DryArm that records every 30 Hz joint target
4. measure:  forward kinematics of every target -> where the gripper will be; checks against the task
5. physics:  (--sim) the recorded targets replayed in MuJoCo: does the object really end up in the box?
6. picture:  predicted gripper path drawn over the camera image -> outputs/validate_move.jpg

    python host/validate_move.py --sim --seed 3 --box B                       # sim scene, object #1 -> box B
    python host/validate_move.py --sim --obj cube --instruction "put the red cube in box Z"
    python host/validate_move.py --camera --instruction "move the orange disc to box B"   # real camera, sheet in view
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "host"), str(ROOT / "sim"), str(ROOT)]

import skills  # noqa: E402
import workspace  # noqa: E402
from so101_sim.env import lerobot_to_rad  # noqa: E402

GRASP_TOL = 0.005  # m: grasps succeeded with up to ~5 mm error in the sim tolerance test
MIN_CARRY_Z = 0.03  # m: gripper height while carrying (clears the other objects / box lines)
COLORS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "white", "grey", "gray"}
SHAPES = {"pen": "elongated", "marker": "elongated", "pencil": "elongated", "stick": "elongated",
          "cube": "square", "block": "square", "box": "square", "disc": "round", "ball": "round", "eraser": "blob"}


# ---------------------------------------------------------------- object choice from the vision model's words

def match_object(name: str, objects: list[workspace.Obj]) -> tuple[workspace.Obj | None, str]:
    """'orange disc' -> the detected object whose colour/shape fit best. Returns (object, why)."""
    if not objects:
        return None, "no objects detected on the sheet"
    if len(objects) == 1:
        return objects[0], "only object on the sheet"
    words = set(name.lower().replace("grey", "gray").split())
    def score(o):
        s = 2 * (o.color.replace("grey", "gray") in words)
        s += any(SHAPES.get(w) == o.shape for w in words)
        return s
    ranked = sorted(objects, key=score, reverse=True)
    if score(ranked[0]) == 0:
        return None, f"no detected object matches '{name}'"
    if len(ranked) > 1 and score(ranked[1]) == score(ranked[0]):
        return None, f"'{name}' fits objects #{ranked[0].id} and #{ranked[1].id} equally"
    return ranked[0], f"colour/shape match for '{name}'"


# ---------------------------------------------------------------- measurement

def measure(arm: skills.DryArm, obj: workspace.Obj, letter: str) -> dict:
    """Forward kinematics of every recorded target -> numbers that say whether the plan does the task."""
    kin = arm.kin
    trace = np.array(arm.trace)
    pos = np.zeros((len(trace), 3))
    jaw = np.zeros((len(trace), 3))  # jaw opening direction (site z axis)
    for i, q in enumerate(trace):
        p, R = kin.fk(q)
        pos[i], jaw[i] = p, R[:, 2]
    ev = dict((label, i) for i, label in arm.events)
    g, r = ev.get("grasp"), ev.get("release")
    out: dict = {"ticks": len(trace), "seconds": round(len(trace) / 30, 1)}

    # where the object will sit between the jaws: the planner puts the fixed jaw (site) half a width + 3 mm away
    half_w = float(np.clip(obj.width / 2, 0.004, 0.02)) + 0.003  # same offset as skills._grasp_plan
    pinch = pos + jaw * half_w
    if g is not None:
        gi = g - 1
        out["grasp_error_mm"] = round(float(np.hypot(*(pinch[gi, :2] - [obj.x, obj.y]))) * 1000, 1)
        out["grasp_height_mm"] = round(float(pos[gi, 2]) * 1000, 1)
        long_axis = np.array([math.cos(obj.yaw), math.sin(obj.yaw), 0.0])
        out["jaw_vs_object_deg"] = round(math.degrees(math.acos(min(1.0, abs(float(jaw[gi] @ long_axis))))), 1)
    if r is not None:
        ri = r - 1
        poly = workspace.box_polygon(letter)
        c = pinch[ri, :2]
        inside = cv2.pointPolygonTest(poly.astype(np.float32), (float(c[0]), float(c[1])), True)  # + inside, m
        out["release_point_cm"] = [round(float(c[0]) * 100, 1), round(float(c[1]) * 100, 1)]
        out["box_centre_cm"] = [round(v * 100, 1) for v in workspace.box_centre(letter)]
        out["release_inside_box"] = inside > 0
        out["release_margin_mm"] = round(float(inside) * 1000, 1)
        # how much of the object's footprint (a length x width rectangle, turned with the wrist) lands in the box
        L = obj.length
        yaw_turn = math.atan2(jaw[ri, 1], jaw[ri, 0]) - math.atan2(jaw[g - 1, 1], jaw[g - 1, 0]) if g else 0.0
        ax = np.array([math.cos(obj.yaw + yaw_turn), math.sin(obj.yaw + yaw_turn)])
        samples = [c + ax * t for t in np.linspace(-L / 2, L / 2, 21)]
        out["object_length_in_box"] = round(sum(
            cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), False) >= 0 for p in samples) / 21, 2)
        if g is not None:
            # carrying = after close + hold + lift (~42 ticks) until the set-down starts (~24 ticks before release)
            carry = pos[g + 42: r - 24, 2] if r - g > 70 else pos[g:r, 2]
            out["min_carry_height_mm"] = round(float(carry.min()) * 1000, 1) if len(carry) else None
    # joint limits and per-tick steps (RealArm caps each 30 Hz step at max_relative_target = 8 units)
    q_rad = np.array([lerobot_to_rad(q) for q in trace])
    lo, hi = kin.m.jnt_range[:5, 0], kin.m.jnt_range[:5, 1]
    out["joint_limit_margin_deg"] = round(float(np.degrees(np.minimum(q_rad[:, :5] - lo, hi - q_rad[:, :5]).min())), 1)
    out["max_step_units"] = round(float(np.abs(np.diff(trace, axis=0)).max()), 2) if len(trace) > 1 else 0.0
    out["min_height_mm"] = round(float(pos[:, 2].min()) * 1000, 1)
    out["_pos"], out["_events"] = pos, ev
    return out


def checks(m: dict) -> list[tuple[str, bool, str]]:
    c = [
        ("plan exists", True, f"{m['ticks']} targets, {m['seconds']} s"),
        ("grasp on the object", m.get("grasp_error_mm", 1e9) <= GRASP_TOL * 1000,
         f"pinch point {m.get('grasp_error_mm')} mm from the object centre (tolerance {GRASP_TOL * 1000:.0f} mm)"),
        ("jaws across the object", abs(m.get("jaw_vs_object_deg", 0) - 90) <= 15,
         f"jaw axis {m.get('jaw_vs_object_deg')} deg to the long side (want ~90)"),
        ("released inside the box", bool(m.get("release_inside_box")),
         f"release at {m.get('release_point_cm')} cm, box centre {m.get('box_centre_cm')} cm, "
         f"{m.get('release_margin_mm')} mm inside the edge; {m.get('object_length_in_box', 0) * 100:.0f}% of the object's length in the box"),
        ("carried above the table", (m.get("min_carry_height_mm") or 0) >= MIN_CARRY_Z * 1000,
         f"lowest point while carrying {m.get('min_carry_height_mm')} mm (want >= {MIN_CARRY_Z * 1000:.0f})"),
        ("never below the table", m["min_height_mm"] > 0, f"lowest gripper point {m['min_height_mm']} mm"),
        ("within joint limits", m["joint_limit_margin_deg"] >= 0, f"closest to a limit: {m['joint_limit_margin_deg']} deg"),
        ("smooth enough for the real arm", m["max_step_units"] <= 8,
         f"largest step per 30 Hz tick {m['max_step_units']} units (RealArm caps at 8)"),
    ]
    return c


# ---------------------------------------------------------------- picture

def draw(scene: workspace.Scene, m: dict, letter: str, obj: workspace.Obj, rows, sim_result: str | None) -> np.ndarray:
    img = scene.annotated.copy()
    pos, ev = m["_pos"], m["_events"]
    px = scene.project(pos)
    if px is not None:
        g, r = ev.get("grasp", 0), ev.get("release", len(pos))
        for i in range(1, len(px)):
            col = (60, 200, 60) if g <= i <= r else (255, 255, 255)  # green = carrying the object
            cv2.line(img, tuple(px[i - 1].astype(int)), tuple(px[i].astype(int)), col, 2)
        for key, col in (("grasp", (0, 0, 255)), ("release", (255, 0, 255))):
            if key in ev:
                p = px[ev[key] - 1].astype(int)
                cv2.circle(img, tuple(p), 7, col, -1)
                cv2.putText(img, key, (p[0] + 8, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
    poly = scene.project(np.column_stack([workspace.box_polygon(letter), np.zeros(4)]))
    if poly is not None:
        cv2.polylines(img, [poly.astype(np.int32)], True, (255, 0, 255), 3)
    # check list below the image
    h = 22 * (len(rows) + 2)
    panel = np.full((h, img.shape[1], 3), 255, np.uint8)
    title = f"object #{obj.id} ({obj.color} {obj.shape}) -> box {letter}"
    if sim_result:
        title += f"   | physics: {sim_result}"
    cv2.putText(panel, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    for k, (name, ok, _) in enumerate(rows):
        cv2.putText(panel, f"{'PASS' if ok else 'FAIL'}  {name}", (8, 40 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 140, 0) if ok else (0, 0, 220), 1)
    return np.vstack([img, panel])


# ---------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--sim", action="store_true", help="MuJoCo scene with the printed sheet")
    src.add_argument("--camera", action="store_true", help="real camera ([cameras.front] index)")
    src.add_argument("--image", help="a saved photo")
    p.add_argument("--seed", type=int, default=1, help="(sim) object placement")
    p.add_argument("--obj", default="pen", choices=["pen", "cube", "disc"], help="(sim) object")
    p.add_argument("--box", help="target box letter (skip the vision model)")
    p.add_argument("--object", type=int, help="object number from the look (default: from --instruction / only one)")
    p.add_argument("--instruction", help="ask the Workers AI vision model which object and box")
    a = p.parse_args()

    env = None
    if a.sim:
        from so101_sim.env import SO101PickEnv

        env = SO101PickEnv(a.obj, sheet=True)
        env.reset(seed=a.seed, home=skills.LOOK)
        frame = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
    elif a.camera:
        import intent_test

        frame = intent_test.grab(intent_test.CONFIG["cameras"]["front"]["index"])
    else:
        frame = cv2.imread(a.image)

    scene = workspace.look(frame)
    if not scene.ok:
        sys.exit(f"look failed: {scene.detail}\n(print outputs/workspace_sheet.png - `python host/workspace.py sheet` - "
                 f"and put the objects on it)")
    print(f"look: {scene.detail}")
    for o in scene.objects:
        print("  " + o.describe())

    letter, obj, why = (a.box or "").upper() or None, None, ""
    if a.object:
        obj = next((o for o in scene.objects if o.id == a.object), None)
        why = "given with --object"
    if a.instruction:
        import intent_test

        token, account = intent_test.auth()
        r = intent_test.evaluate(token, account, intent_test.MODELS["mistral"], frame, a.instruction, 5)
        ans = r["answer"]
        print(f"vision model: {ans.get('object')} -> box {r['box']} | verdict {r['verdict']}, "
              f"agreement {r['pair_agreement'] * 100:.0f}%" + ("" if not r["reasons"] else f" ({'; '.join(r['reasons'])})"))
        letter = letter or r["box"]
        if obj is None:
            obj, why = match_object(str(ans.get("object", "")), scene.objects)
    if obj is None and len(scene.objects) == 1:
        obj, why = scene.objects[0], "only object on the sheet"
    if obj is None:
        sys.exit(f"which object? {why or 'pass --object N or --instruction'}")
    if letter not in workspace.BOXES:
        sys.exit(f"which box? got '{letter}' - pass --box A|B|C|D|X|Z or an --instruction that names one")
    print(f"task: object #{obj.id} ({obj.color} {obj.shape}, {why}) -> box {letter}")

    arm = skills.DryArm(frame)
    sk = skills.Skills(arm, log=lambda t: None)
    sk.look()
    ok, detail = skills.run_plan(sk, [{"skill": "put", "object": obj.id, "box": letter}])
    if not ok:
        sys.exit(f"planning failed: {detail}")
    m = measure(arm, obj, letter)
    rows = checks(m)

    sim_result = None
    if env is not None:  # replay the predicted targets with physics, from the same starting scene
        env.reset(seed=a.seed, home=skills.LOOK)
        for t in arm.trace:
            env.step(t, render=False)
        for _ in range(30):
            env.step(arm.trace[-1], render=False)
        pos, _ = env.object_pose()
        inside = cv2.pointPolygonTest(workspace.box_polygon(letter).astype(np.float32), (float(pos[0]), float(pos[1])), True)
        sim_result = (f"object centre ended at ({pos[0] * 100:.1f}, {pos[1] * 100:.1f}) cm, "
                      + (f"{inside * 1000:.0f} mm inside box {letter}" if inside > 0 else f"OUTSIDE box {letter} by {-inside * 1000:.0f} mm"))
        rows.append(("physics: object ends in the box", inside > 0, sim_result))

    print()
    for name, good, info in rows:
        print(f"  {'PASS' if good else 'FAIL'}  {name:34} {info}")
    passed = all(g for _, g, _ in rows)
    print(f"\n{'VALID' if passed else 'NOT VALID'}: the predicted moves {'do' if passed else 'do not'} put object "
          f"#{obj.id} into box {letter}")
    out = ROOT / "outputs" / "validate_move.jpg"
    cv2.imwrite(str(out), draw(scene, m, letter, obj, rows, sim_result))
    print(f"picture -> {out}")
    report = {k: v for k, v in m.items() if not k.startswith("_")}
    (ROOT / "outputs" / "validate_move.json").write_text(json.dumps(
        {"object": obj.as_dict(), "box": letter, "measurements": report,
         "checks": [{"check": n, "pass": bool(g), "detail": i} for n, g, i in rows], "valid": passed}, indent=1))


if __name__ == "__main__":
    main()
