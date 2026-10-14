"""Does the vision model understand "move X to box B"? Test it on the live camera, no arm needed.

Takes a camera frame, sends it + your instruction to a Workers AI vision model, and prints which object it
thinks you mean, which lettered box is the target, and a confidence:
  agreement  = how often N independent answers (temperature 0.7) pick the same object + box. The honest number.
  self       = the model's own 0-100 estimate (LLMs are usually over-confident; shown for comparison).

    .venv\\Scripts\\python host\\intent_test.py                       # interactive: type instructions, fresh frame each time
    .venv\\Scripts\\python host\\intent_test.py "move the orange disc to box B" --expect B
    .venv\\Scripts\\python host\\intent_test.py --image photo.jpg "put the pen in C"

Auth: CLOUDFLARE_API_TOKEN (+ CLOUDFLARE_ACCOUNT_ID), else the token of `wrangler login`.
Each answer is one Workers AI call (billed to the Cloudflare account; Mistral Small 3.1 is a few cents per 100).
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robot import CONFIG  # noqa: E402

MODELS = {
    "mistral": "@cf/mistralai/mistral-small-3.1-24b-instruct",  # the router's vision model (Apache-2.0)
    "gemma": "@cf/google/gemma-3-12b-it",  # second opinion (Gemma terms, no EU restriction)
}
BOXES = ["A", "B", "C", "D", "X", "Z"]

PROMPT = """You are the eyes of a robot arm. The image is from its camera, looking at a table.
On the table there is paper with boxes (rectangles) drawn on it; each box has one letter inside: {boxes}.
Some boxes may be partly outside the picture or seen at a steep angle.

The user says: "{instruction}"

Work out what the user wants moved and into which lettered box. Answer with ONLY this JSON, no other text:
{{"object": "short name of the thing to move, with its colour",
 "object_where": "where that object is now, in words (e.g. 'just above box D')",
 "target_box": "one of {boxes_json} or 'none' if the instruction names no box or that box isn't visible",
 "target_where": "where that box is in the picture, in words",
 "boxes_seen": [letters you can actually read in the image],
 "confidence": 0-100 how sure you are that you understood both the object and the box,
 "problem": "anything unclear (object not visible, two candidates, box not found) or empty"}}"""


# ---------------------------------------------------------------- Cloudflare auth

def _wrangler_token() -> str | None:
    """OAuth token of `wrangler login` (has the ai:write scope). Refreshed via `wrangler whoami` when expired."""
    cfg = Path(os.environ.get("APPDATA", Path.home())) / "xdg.config" / ".wrangler" / "config" / "default.toml"
    if not cfg.exists():
        cfg = Path.home() / ".wrangler" / "config" / "default.toml"
    if not cfg.exists():
        return None
    t = tomllib.loads(cfg.read_text(encoding="utf-8"))
    exp = str(t.get("expiration_time", ""))
    if exp and exp < time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + 120)):
        # wrangler refreshes its own token on any authenticated command
        subprocess.run("npx wrangler whoami", shell=True, cwd=ROOT / "cloudflare" / "robot-router",
                       capture_output=True, timeout=120)
        t = tomllib.loads(cfg.read_text(encoding="utf-8"))
    return t.get("oauth_token")


def _api(method: str, path: str, token: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Cloudflare API {e.code}: {e.read().decode(errors='replace')[:300]}") from None


def auth() -> tuple[str, str]:
    token = os.environ.get("CLOUDFLARE_API_TOKEN") or _wrangler_token()
    if not token:
        sys.exit("No Cloudflare credentials: run `npx wrangler login` (in cloudflare/robot-router) "
                 "or set CLOUDFLARE_API_TOKEN")
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID") or CONFIG.get("cloudflare", {}).get("account_id")
    if not account:
        accounts = _api("GET", "/accounts", token).get("result") or []
        if not accounts:
            sys.exit("token sees no Cloudflare account - set CLOUDFLARE_ACCOUNT_ID")
        account = accounts[0]["id"]
    return token, account


# ---------------------------------------------------------------- model call

def ask(token: str, account: str, model: str, jpeg_b64: str, instruction: str, temperature: float) -> dict:
    text = PROMPT.format(boxes=", ".join(BOXES), boxes_json=json.dumps(BOXES), instruction=instruction)
    body = {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{jpeg_b64}"}},
        ]}],
        "max_tokens": 300,
        "temperature": temperature,
    }
    out = _api("POST", f"/accounts/{account}/ai/run/{model}", token, body)
    raw = (out.get("result") or {}).get("response", "")
    if isinstance(raw, dict):
        return raw
    m = re.search(r"\{.*\}", raw, re.S)  # models sometimes wrap the JSON in ``` or add a sentence
    try:
        return json.loads(m.group(0)) if m else {"problem": f"no JSON in answer: {raw[:120]}"}
    except json.JSONDecodeError:
        return {"problem": f"bad JSON: {raw[:120]}"}


def _norm_obj(s: str) -> str:
    """'the orange round disc' ~ 'orange disc': compare on colour + noun words, ignoring filler."""
    stop = {"the", "a", "an", "small", "little", "round", "flat", "object", "piece", "of", "thing", "big", "large"}
    return " ".join(sorted(w for w in re.findall(r"[a-z]+", s.lower()) if w not in stop))


def evaluate(token, account, model, frame, instruction, runs: int) -> dict:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf).decode()
    t0 = time.time()
    # first answer at temperature 0 (what the robot would actually use), the rest sampled to measure stability
    temps = [0.0] + [0.7] * (runs - 1)
    with ThreadPoolExecutor(max_workers=runs) as ex:
        answers = list(ex.map(lambda t: _safe(ask, token, account, model, b64, instruction, t), temps))
    box_of = lambda a: str(a.get("target_box", "none")).strip().upper().replace("BOX ", "")  # noqa: E731
    best = answers[0]
    box = box_of(best)
    ref = set(_norm_obj(str(best.get("object", ""))).split())
    # same object = names share a word ("orange eraser" ~ "orange disc"); "red pen" vs "blue pen" don't
    same_obj = [bool(ref) and (ref <= (w := set(_norm_obj(str(a.get("object", ""))).split())) or w <= ref)
                and bool(w) for a in answers]
    same_box = [box_of(a) == box for a in answers]
    box_votes = collections.Counter(box_of(a) for a in answers)
    obj_votes = collections.Counter(_norm_obj(str(a.get("object", ""))) for a in answers)
    selfc = [float(a["confidence"]) for a in answers if isinstance(a.get("confidence"), (int, float))]
    seen = [str(b).strip().upper() for b in best.get("boxes_seen", []) if isinstance(b, str)]
    agree = sum(o and b for o, b in zip(same_obj, same_box)) / len(answers)
    # the model guesses even when unsure, but writes the doubt into 'problem': gate on facts, not its score
    reasons = [r for r, bad in [
        (f"answers disagree ({agree * 100:.0f}% < 80%)", agree < 0.8),
        (f"box {box} is not one it could read {seen}", box not in seen),
        (f"model reports: {best.get('problem')}", bool(str(best.get("problem") or "").strip())),
    ] if bad]
    return {
        "answer": best, "box": box,
        "verdict": "GO" if not reasons else "ASK", "reasons": reasons,
        "box_agreement": sum(same_box) / len(answers),
        "object_agreement": sum(same_obj) / len(answers),
        "pair_agreement": agree,
        "self_confidence": sum(selfc) / len(selfc) if selfc else None,
        "box_votes": dict(box_votes), "object_votes": dict(obj_votes),
        "seconds": time.time() - t0, "answers": answers,
    }


def _safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:  # one failed call shouldn't hide the others
        return {"problem": f"call failed: {e}"}


# ---------------------------------------------------------------- camera + CLI

def grab(index: int) -> "cv2.Mat":
    cam = CONFIG["cameras"]["front"]
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
    frame = None
    for _ in range(15):  # first frames are dark while auto-exposure settles
        ok, f = cap.read()
        frame = f if ok else frame
    cap.release()
    if frame is None or frame.std() < 6:
        sys.exit(f"camera {index} gives no usable image - try --camera 0/1/2 (robot.py find-cameras)")
    return frame


def report(r: dict, expect: str | None, frame, model_name: str) -> None:
    a = r["answer"]
    pct = lambda v: f"{v * 100:.0f}%"  # noqa: E731
    print(f"\n  object : {a.get('object', '?')}   ({a.get('object_where', '')})")
    print(f"  target : box {r['box']}   ({a.get('target_where', '')})")
    print(f"  boxes the model can read: {a.get('boxes_seen', '?')}")
    if a.get("problem"):
        print(f"  problem: {a['problem']}")
    print(f"  CONFIDENCE  agreement {pct(r['pair_agreement'])} (box {pct(r['box_agreement'])}, "
          f"object {pct(r['object_agreement'])} over {len(r['answers'])} answers)"
          + (f"   self-reported {r['self_confidence']:.0f}%" if r["self_confidence"] is not None else ""))
    if len(r["box_votes"]) > 1 or len(r["object_votes"]) > 1:
        print(f"  votes  box {r['box_votes']}   object {r['object_votes']}")
    print(f"  VERDICT {r['verdict']}" + ("  - robot would act" if r["verdict"] == "GO" else
                                          "  - robot would ask a human: " + "; ".join(r["reasons"])))
    if expect:
        e = expect.upper()
        ok = (r["verdict"] == "ASK") if e == "NONE" else (r["box"] == e and r["verdict"] == "GO")
        print(f"  expected {'no action' if e == 'NONE' else 'box ' + e}: {'CORRECT' if ok else 'WRONG'}")
    print(f"  ({model_name}, {r['seconds']:.1f} s)")
    out = frame.copy()
    color = (0, 170, 0) if r["verdict"] == "GO" else (0, 140, 255)
    cv2.rectangle(out, (0, 0), (out.shape[1], 54), (255, 255, 255), -1)
    cv2.putText(out, f"{r['verdict']}: {a.get('object', '?')} -> box {r['box']}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    cv2.putText(out, f"agreement {pct(r['pair_agreement'])}  self {r['self_confidence'] or 0:.0f}%", (10, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    path = ROOT / "outputs" / "intent_test.jpg"
    path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(path), out)
    print(f"  image  -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("instruction", nargs="?", help="e.g. 'move the orange disc to box B' (omit = interactive)")
    p.add_argument("--camera", type=int, default=CONFIG["cameras"]["front"]["index"])
    p.add_argument("--image", help="use a saved photo instead of the camera")
    p.add_argument("--expect", help="the correct box letter, to score the answer")
    p.add_argument("--runs", type=int, default=5, help="answers per instruction (agreement = confidence)")
    p.add_argument("--model", choices=list(MODELS), default="mistral")
    a = p.parse_args()

    token, account = auth()
    model = MODELS[a.model]
    print(f"model {model}, {a.runs} answers per instruction")

    def once(instruction: str, expect: str | None) -> None:
        frame = cv2.imread(a.image) if a.image else grab(a.camera)
        report(evaluate(token, account, model, frame, instruction, a.runs), expect, frame, a.model)

    if a.instruction:
        return once(a.instruction, a.expect)
    print("Type an instruction (fresh camera frame each time). Add '= B' to score it, e.g. "
          "'move the orange disc to box B = B'. Empty line quits.")
    while True:
        try:
            line = input("\n> ").strip()
        except EOFError:
            break
        if not line:
            break
        instruction, _, expect = line.partition("=")
        once(instruction.strip(), expect.strip() or None)


if __name__ == "__main__":
    main()
