"""Hand-drawn page: 8 red circles (corners + side middles) and a + in the middle = 9 named spots.

"move the yellow duck from the top to the bottom"
  1. language  (Workers AI vision model, one call): -> {object, colour, from, to}, spot names from a fixed list
  2. spots     (code): find the red circles, name them, fit a page grid (any camera angle; 4+ circles suffice)
  3. object    (code): find objects of that colour, where each stands on the page, which spot it is nearest
  4. check     (code): every claim the instruction makes is verified; the answer is either a concrete move or
                       the list of reasons why not (object missing / off the page / not at 'from' / target taken...)

"top" = the side of the page farthest from the camera (the top of the camera picture), as the dashboard shows it.

    .venv\\Scripts\\python host\\page_grid.py "move the yellow duck from the top to the bottom"
    .venv\\Scripts\\python host\\page_grid.py --image photo.jpg "put the duck in the centre"
    .venv\\Scripts\\python host\\page_grid.py --spots-only          # just find + name the circles (no AI call)
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
sys.path[:0] = [str(ROOT / "host"), str(ROOT)]

# page coordinates: u = 0 (left) .. 1 (right), v = 0 (top, far from the camera) .. 1 (bottom)
SPOTS = {"top-left": (0, 0), "top": (0.5, 0), "top-right": (1, 0),
         "left": (0, 0.5), "centre": (0.5, 0.5), "right": (1, 0.5),
         "bottom-left": (0, 1), "bottom": (0.5, 1), "bottom-right": (1, 1)}
RING = [n for n in SPOTS if n != "centre"]  # the 8 drawn circles
AT_SPOT = 0.15  # page units: an object closer than this to a spot is "at" it (spots are 0.5 apart)
COLOURS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "black", "white", "grey", "brown"]


# ---------------------------------------------------------------- 1. red circles -> named spots -> page grid

def red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (((h < 10) | (h > 165)) & (s > 60) & (v > 80)).astype(np.uint8) * 255


def find_spots(bgr: np.ndarray) -> dict:
    """Red ink -> circles (grouped strokes) -> 8 ring circles named by position -> homography image -> page."""
    m = red_mask(bgr)
    # a hand-drawn circle is several strokes: group pieces that are close, measure on the original ink
    groups = cv2.dilate(m, np.ones((25, 25), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(groups)
    blobs = []
    for k in range(1, n):
        ink = (lab == k) & (m > 0)
        area = int(ink.sum())
        if area < 120:  # specks / the thin + strokes
            continue
        ys, xs = np.nonzero(ink)
        blobs.append({"px": (float(xs.mean()), float(ys.mean())), "area": area,
                      "size": (int(xs.max() - xs.min()), int(ys.max() - ys.min()))})
    out = {"blobs": len(blobs), "named": {}, "H": None, "problems": [], "fit_error": None}
    if len(blobs) < 4:
        out["problems"].append(f"only {len(blobs)} red circles visible (need at least 4 of the 8 to place the page)")
        return out
    pts = np.array([b["px"] for b in blobs], np.float32)

    # corners = the 4-gon of maximum area spanned by the blobs (the page outline seen in perspective)
    hull = cv2.convexHull(pts).reshape(-1, 2)
    best, quad = -1.0, None
    import itertools
    for c in itertools.combinations(range(len(hull)), 4):
        q = hull[list(c)]
        a = cv2.contourArea(q.astype(np.float32))
        if a > best:
            best, quad = a, q
    # order corners: top two = smallest image y; left/right by x
    quad = sorted(quad.tolist(), key=lambda p: p[1])
    tl, tr = sorted(quad[:2], key=lambda p: p[0])
    bl, br = sorted(quad[2:], key=lambda p: p[0])
    named = {"top-left": tl, "top-right": tr, "bottom-left": bl, "bottom-right": br}
    corner_page = np.array([SPOTS[k] for k in named], np.float32)
    H0 = cv2.getPerspectiveTransform(np.array(list(named.values()), np.float32), corner_page)
    # every other blob: map with the corner-only grid, snap to the nearest side middle if close enough
    for b in blobs:
        if any(np.allclose(b["px"], v) for v in named.values()):
            continue
        u, v = cv2.perspectiveTransform(np.array([[b["px"]]], np.float32), H0)[0, 0]
        name, d = min(((k, math.hypot(u - p[0], v - p[1])) for k, p in SPOTS.items()
                       if k not in named and k != "centre"), key=lambda t: t[1], default=(None, 9))
        if name and d < 0.2:
            named[name] = list(b["px"])
        else:
            out.setdefault("extra", []).append([round(u, 2), round(v, 2)])
    missing = [k for k in RING if k not in named]
    src = np.array([named[k] for k in named], np.float32)
    dst = np.array([SPOTS[k] for k in named], np.float32)
    H, _ = cv2.findHomography(src, dst)
    back = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    out["fit_error"] = round(float(np.abs(back - dst).max()), 3)  # page units; > 0.08 = circles not on a grid
    out.update(named={k: [round(c, 1) for c in v] for k, v in named.items()}, H=H, missing=missing)
    centre = cv2.perspectiveTransform(np.array([[SPOTS["centre"]]], np.float32), np.linalg.inv(H))[0, 0]
    out["named"]["centre"] = [round(float(centre[0]), 1), round(float(centre[1]), 1)]
    if missing:
        out["problems"].append(f"circle(s) not found: {', '.join(missing)} (covered, or outside the picture)")
    if out["fit_error"] > 0.08:
        out["problems"].append(f"the circles don't sit on a regular 3x3 grid (fit error {out['fit_error']:.2f}) - "
                               f"is the page bent, or a red object mistaken for a circle?")
    return out


def to_page(H: np.ndarray, px) -> tuple[float, float]:
    u, v = cv2.perspectiveTransform(np.array([[px]], np.float32), H)[0, 0]
    return float(u), float(v)


def to_image(H: np.ndarray, uv) -> tuple[int, int]:
    x, y = cv2.perspectiveTransform(np.array([[uv]], np.float32), np.linalg.inv(H))[0, 0]
    return int(x), int(y)


# ---------------------------------------------------------------- 2. objects of a colour, where they stand

HUES = {"red": [(0, 10), (165, 180)], "orange": [(10, 22)], "yellow": [(22, 38)], "green": [(38, 85)],
        "blue": [(85, 130)], "purple": [(130, 150)], "pink": [(150, 165)]}


def find_objects(bgr: np.ndarray, H: np.ndarray, colour: str) -> list[dict]:
    """Blobs of `colour` near the page (up to half a page outside it). Position = where the object touches the
    table: the bottom of its blob in the picture (a duck is tall; its centre would sit 'behind' its foot)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if colour in HUES:
        m = np.zeros(h.shape, bool)
        for lo, hi in HUES[colour]:
            m |= (h >= lo) & (h < hi)
        # saturation 150+: measured duck 253, beige chess piece 128 (also "yellow" by hue), paper 21
        m &= (s > 150) & (v > 70)
    elif colour == "black":
        m = v < 50
    elif colour == "white":
        m = (s < 30) & (v > 200)
    else:
        m = (s < 40) & (v > 60) & (v < 160)
    m = m.astype(np.uint8) * 255
    if colour == "red":
        m &= ~cv2.dilate(red_mask(bgr), np.ones((7, 7), np.uint8)) | 0  # the drawn circles are red too
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    objs = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 150:
            continue
        ys, xs = np.nonzero(lab == k)
        foot_y = ys.max()
        low = ys >= foot_y - max(3, (foot_y - ys.min()) * 0.15)
        foot = (float(xs[low].mean()), float(foot_y))
        u, v_ = to_page(H, foot)
        if not (-1.0 <= u <= 2.0 and -1.5 <= v_ <= 2.0):
            continue  # far from the page: background clutter (beyond the far edge, perspective stretches a lot)
        name, d = min(((n_, math.hypot(u - p[0], v_ - p[1])) for n_, p in SPOTS.items()), key=lambda t: t[1])
        objs.append({"px": [int(xs.mean()), int(ys.mean())], "foot_px": [int(foot[0]), int(foot[1])],
                     "page": [round(u, 2), round(v_, 2)], "on_page": -0.05 <= u <= 1.05 and -0.05 <= v_ <= 1.05,
                     "nearest": name, "distance": round(d, 2), "at": name if d <= AT_SPOT else None,
                     "area": int(st[k, cv2.CC_STAT_AREA]),
                     "cut_off": bool(ys.min() == 0 or xs.min() == 0 or ys.max() == m.shape[0] - 1 or xs.max() == m.shape[1] - 1),
                     "bbox": [int(st[k, 0]), int(st[k, 1]), int(st[k, 2]), int(st[k, 3])]})
    return sorted(objs, key=lambda o: -o["area"])


def where(o: dict) -> str:
    """'at the TOP spot' / 'off the page, 22% of a page beyond the top edge' / 'between top and centre'."""
    u, v = o["page"]
    if not o["on_page"]:
        sides = [f"{-v:.0%} of the page beyond the top edge" if v < -0.05 else "",
                 f"{v - 1:.0%} beyond the bottom edge" if v > 1.05 else "",
                 f"{-u:.0%} beyond the left edge" if u < -0.05 else "",
                 f"{u - 1:.0%} beyond the right edge" if u > 1.05 else ""]
        return ("off the page, " + " and ".join(x for x in sides if x) + f" (nearest spot: {o['nearest']})"
                + (", partly outside the camera picture" if o.get("cut_off") else ""))
    if o["at"]:
        return f"at the {o['at'].upper()} spot"
    return f"on the page near {o['nearest'].upper()} ({o['distance']:.2f} page from it)"


# ---------------------------------------------------------------- 3. language -> structured request

PARSE_PROMPT = """You turn a spoken instruction for a robot arm into JSON. The robot looks at a sheet of paper with
9 named spots: top-left, top, top-right, left, centre, right, bottom-left, bottom, bottom-right
("top" = the far side of the page, at the top of the camera picture). The image is the robot's camera view.

Instruction: "{instruction}"

Answer with ONLY this JSON:
{{"action": "move" if the user wants something moved, else "none",
 "object": "the thing to move, as the user calls it (e.g. 'yellow duck')",
 "colour": one of {colours} - the object's main colour; if the instruction doesn't say, look at the image,
 "from": one of the 9 spot names, or "unspecified" if the instruction doesn't say where it is now,
 "to": one of the 9 spot names, or "unclear",
 "note": "anything ambiguous, or empty"}}"""


def parse(instruction: str, frame: np.ndarray) -> dict:
    import base64

    import intent_test

    token, account = intent_test.auth()
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    text = PARSE_PROMPT.format(instruction=instruction.replace('"', "'"), colours=json.dumps(COLOURS))
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": text}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        "max_tokens": 200, "temperature": 0}
    out = intent_test._api("POST", f"/accounts/{account}/ai/run/{intent_test.MODELS['mistral']}", token, body)
    raw = (out.get("result") or {}).get("response", "")
    if isinstance(raw, dict):
        return raw
    import re

    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else {"action": "none", "note": f"no JSON: {raw[:100]}"}
    except json.JSONDecodeError:
        return {"action": "none", "note": f"bad JSON: {raw[:100]}"}


def norm_spot(s) -> str | None:
    s = str(s or "").lower().strip().replace("_", "-").replace(" ", "-").replace("center", "centre")
    s = {"middle": "centre", "upper": "top", "lower": "bottom", "upper-left": "top-left", "upper-right": "top-right",
         "lower-left": "bottom-left", "lower-right": "bottom-right", "left-top": "top-left", "right-top": "top-right",
         "left-bottom": "bottom-left", "right-bottom": "bottom-right"}.get(s, s)
    return s if s in SPOTS else None


# ---------------------------------------------------------------- 4. check everything -> answer or reasons

def check(frame: np.ndarray, instruction: str, req: dict | None = None) -> dict:
    res = {"instruction": instruction, "ok": False, "problems": [], "warnings": [], "answer": ""}
    page = find_spots(frame)
    res["page"] = {"circles_found": sorted(k for k in page["named"] if k != "centre"),
                   "missing": page.get("missing", RING), "fit_error": page["fit_error"]}
    if page["H"] is None:
        res["problems"] += page["problems"]
        res["answer"] = "I can't see the page: " + "; ".join(page["problems"])
        return res | {"_page": page}
    res["warnings"] += page["problems"]  # missing circles are fine as long as the grid could be fitted

    req = req if req is not None else parse(instruction, frame)
    res["understood"] = req
    if str(req.get("action", "")).lower() != "move":
        res["problems"].append("the instruction doesn't ask to move anything" + (f" ({req['note']})" if req.get("note") else ""))
    to, frm = norm_spot(req.get("to")), norm_spot(req.get("from"))
    colour = str(req.get("colour", "")).lower().replace("gray", "grey")
    obj_name = req.get("object") or "object"
    if not to:
        res["problems"].append(f"no target spot understood (got '{req.get('to')}'); spots are: {', '.join(SPOTS)}")
    if colour not in COLOURS:
        res["problems"].append(f"don't know which colour to look for (got '{req.get('colour')}')")
        objs = []
    else:
        objs = find_objects(frame, page["H"], colour)
    res["candidates"] = [{"page": o["page"], "where": where(o)} for o in objs]

    obj = None
    if colour in COLOURS and not objs:
        res["problems"].append(f"no {colour} object anywhere near the page - is the {obj_name} in view?")
    elif len(objs) > 1 and len(on := [o for o in objs if o["on_page"]]) == 1:
        obj = on[0]  # the arm only works on the page: things beyond it can't be meant
        res["warnings"].append(f"ignored {len(objs) - 1} other {colour} thing(s) off the page: "
                               + "; ".join(where(o) for o in objs if o is not obj))
    elif len(objs) > 1:
        at_from = [o for o in objs if frm and o["at"] == frm]
        if len(at_from) == 1:
            obj = at_from[0]
            res["warnings"].append(f"{len(objs)} {colour} things in view; took the one at {frm.upper()}")
        else:
            res["problems"].append(f"{len(objs)} {colour} things in view: " + "; ".join(where(o) for o in objs)
                                   + " - say which one (e.g. 'the one at the top')")
    elif objs:
        obj = objs[0]
    if obj:
        res["object"] = {"name": obj_name, "colour": colour, "page": obj["page"], "where": where(obj),
                         "image_px": obj["foot_px"]}
        if not obj["on_page"]:
            res["problems"].append(f"the {obj_name} is {where(obj)} - put it on the page first (the arm only works on the page)")
        elif frm and obj["at"] != frm:
            res["problems"].append(f"you said it's at {frm.upper()}, but the {obj_name} is {where(obj)}")
        elif not frm:
            res["warnings"].append(f"start not given; the {obj_name} is {where(obj)}")
        if to and obj["on_page"] and obj["at"] == to:
            res["problems"].append(f"the {obj_name} is already at {to.upper()}")
    if to and page["H"] is not None:
        # target must be free: anything clearly coloured sitting on that spot (other colours than our object)
        for c in COLOURS[:7]:
            if c == colour:
                continue
            for o in find_objects(frame, page["H"], c):
                if o["at"] == to:
                    res["problems"].append(f"{to.upper()} is taken by a {c} object")
    res["ok"] = not res["problems"]
    if res["ok"]:
        u0, v0 = obj["page"]
        u1, v1 = SPOTS[to]
        res["move"] = {"from_page": [u0, v0], "to_page": [u1, v1],
                       "distance_page": round(math.hypot(u1 - u0, v1 - v0), 2),
                       "from_spot": obj["at"] or obj["nearest"], "to_spot": to}
        res["answer"] = (f"OK: move the {obj_name} ({colour}) from {(obj['at'] or 'near ' + obj['nearest']).upper()} "
                         f"to {to.upper()} - {res['move']['distance_page']:.2f} page lengths "
                         f"{'down' if v1 > v0 + 0.1 else 'up' if v1 < v0 - 0.1 else 'across'} the page.")
    else:
        res["answer"] = "Can't do it: " + " | ".join(res["problems"])
    return res | {"_page": page, "_obj": obj}


def draw(frame: np.ndarray, res: dict) -> np.ndarray:
    img = frame.copy()
    page = res["_page"]
    for k, p in page["named"].items():
        c = tuple(int(x) for x in p)
        cv2.circle(img, c, 6, (255, 255, 255), -1)
        cv2.circle(img, c, 6, (0, 0, 0), 1)
        cv2.putText(img, k, (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
        cv2.putText(img, k, (c[0] + 8, c[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    if page["H"] is not None:  # the fitted page grid
        for t in (0, 0.5, 1):
            cv2.line(img, to_image(page["H"], (t, 0)), to_image(page["H"], (t, 1)), (200, 160, 0), 1)
            cv2.line(img, to_image(page["H"], (0, t)), to_image(page["H"], (1, t)), (200, 160, 0), 1)
    obj = res.get("_obj")
    if obj:
        x, y, w, h = obj["bbox"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 255), 2)
        cv2.circle(img, tuple(obj["foot_px"]), 5, (0, 0, 255), -1)
    to = norm_spot((res.get("understood") or {}).get("to"))
    if obj and to and page["H"] is not None:
        cv2.arrowedLine(img, tuple(obj["foot_px"]), to_image(page["H"], SPOTS[to]),
                        (0, 180, 0) if res["ok"] else (0, 0, 255), 3, tipLength=0.06)
    lines = [("OK" if res["ok"] else "CAN'T") + ": " + res["instruction"]] + \
            [p for p in res["problems"]] + [w for w in res["warnings"]]
    panel = np.full((22 * len(lines) + 10, img.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(lines):
        col = (0, 140, 0) if (i == 0 and res["ok"]) else (0, 0, 200) if i <= len(res["problems"]) else (0, 120, 200)
        cv2.putText(panel, t[:95], (8, 18 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    return np.vstack([img, panel])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("instruction", nargs="?")
    p.add_argument("--image")
    p.add_argument("--camera", type=int)
    p.add_argument("--spots-only", action="store_true")
    a = p.parse_args()
    if a.image:
        frame = cv2.imread(a.image)
    else:
        import intent_test

        frame = intent_test.grab(a.camera if a.camera is not None else intent_test.CONFIG["cameras"]["front"]["index"])
    if a.spots_only or not a.instruction:
        page = find_spots(frame)
        print(json.dumps({k: v for k, v in page.items() if k != "H"}, indent=1))
        res = {"_page": page, "instruction": "(spots only)", "ok": not page["problems"], "problems": page["problems"],
               "warnings": []}
    else:
        res = check(frame, a.instruction)
        print(json.dumps({k: v for k, v in res.items() if not k.startswith("_")}, indent=1))
        print("\n" + res["answer"])
    out = ROOT / "outputs" / "page_grid.jpg"
    cv2.imwrite(str(out), draw(frame, res))
    print(f"picture -> {out}")


if __name__ == "__main__":
    main()
