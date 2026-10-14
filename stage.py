"""Stage runner: rollout -> self-check -> retry, with a big status screen. Plus the 20-position evaluation log.

    .venv\\Scripts\\python stage.py demo                      # SPACE = start attempt, ESC = emergency stop, q = quit
    .venv\\Scripts\\python stage.py eval --label 10k --policy outputs\\train\\act_x\\checkpoints\\010000\\pretrained_model
    .venv\\Scripts\\python stage.py report                    # success rate per checkpoint / reactivity setting

Every attempt is a separate `lerobot-rollout` process (same flags as robot.py rollout). Rollout returns the arm
to its start pose when it ends, then this script grabs one frame from the SAME camera and runs selfcheck.py.
The camera is only opened after the rollout process has exited (Windows lets one process hold a webcam).

Reactivity: ACT by default plans 100 steps (3.3 s) and executes them blind, so it ignores a cube that gets moved.
[stage] n_action_steps re-plans more often; temporal_ensemble_coeff (ACT paper: 0.01) re-plans every step and
blends overlapping plans. Both are rollout-time overrides, no retraining needed.
"""

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

import robot
import selfcheck

ROOT = Path(__file__).resolve().parent
EVAL_LOG = ROOT / "evals" / "eval_log.csv"
LIVE_FRAME = Path(tempfile.gettempdir()) / "stage_live.jpg"
STAGE = {"duration_s": 25, "attempts": 3, "n_action_steps": 100, "temporal_ensemble_coeff": 0.0,
         **robot.CONFIG.get("stage", {})}

FAILURE_TYPES = ["missed_grasp", "dropped", "wrong_place", "froze", "collision", "other"]
W, H = 1280, 720
BG, FG, GREEN, RED, AMBER, GREY = (20, 20, 20), (240, 240, 240), (80, 210, 90), (60, 60, 235), (0, 190, 255), (150, 150, 150)
WIN = "robot status"


# ---------------------------------------------------------------- rollout

def reactivity_args(n_action_steps: int, ensemble: float) -> list[str]:
    if ensemble > 0:
        return [f"--policy.temporal_ensemble_coeff={ensemble}", "--policy.n_action_steps=1"]
    return [f"--policy.n_action_steps={n_action_steps}"]


def reactivity_label(n_action_steps: int, ensemble: float) -> str:
    return f"ensemble={ensemble}" if ensemble > 0 else f"replan={n_action_steps}"


def resolve_policy(policy: str | None) -> str:
    policy = policy or robot.CONFIG["hf"]["policy"]
    if Path(policy).exists():
        return str(Path(policy).resolve())
    local = robot.local_policy_dir(policy)
    return str(local) if local.exists() else robot.hf_repo(policy)


def rollout_cmd(policy_path: str, duration: int, n_action_steps: int, ensemble: float, mock: bool = False) -> list[str]:
    # rollout_live.py = lerobot-rollout + live camera frames for the status screen.
    if mock:  # no robot: stream the webcam for `duration` seconds, to rehearse the stage flow
        return [sys.executable, str(ROOT / "rollout_live.py"), "--mock", f"--duration={duration}"]
    return [
        sys.executable, str(ROOT / "rollout_live.py"),
        "--strategy.type=base",
        f"--policy.path={policy_path}",
        *reactivity_args(n_action_steps, ensemble),
        *robot.robot_args(),
        f"--task={robot.CONFIG['task']['description']}",
        f"--duration={duration}",
        "--display_data=false",
        "--play_sounds=false",
    ]


# ---------------------------------------------------------------- status screen

def screen(title: str, color, sub: str = "", image: np.ndarray | None = None) -> np.ndarray:
    canvas = np.full((H, W, 3), BG, np.uint8)
    if image is not None:  # camera view on the right half
        img = cv2.resize(image, (640, 480))
        canvas[120:600, 620:1260] = img
    cv2.putText(canvas, title, (40, 330), cv2.FONT_HERSHEY_DUPLEX, 2.4 if image is None else 1.8, color, 4)
    for i, line in enumerate(sub.split("\n")):
        cv2.putText(canvas, line, (44, 410 + 45 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GREY, 2)
    return canvas


def show(canvas: np.ndarray, wait_ms: int = 1) -> int:
    cv2.imshow(WIN, canvas)
    return cv2.waitKey(wait_ms) & 0xFF


def say(text: str) -> None:
    from lerobot.utils.utils import say as lerobot_say
    lerobot_say(text, blocking=False)


def live_view(setup: dict, cache: dict):
    """Latest frame the rollout published, annotated by the self-check. None until the first frame arrives."""
    try:
        mtime = LIVE_FRAME.stat().st_mtime
    except OSError:
        return cache.get("view"), cache.get("ok", False)
    if mtime != cache.get("mtime"):
        frame = cv2.imread(str(LIVE_FRAME))
        if frame is not None:
            cache["ok"], _, cache["view"] = selfcheck.check(frame, setup)
            cache["mtime"] = mtime
    return cache.get("view"), cache.get("ok", False)


def run_attempt(cmd: list[str], title: str, sub: str, last_img, setup: dict):
    """Runs one rollout while keeping the status window alive.

    Returns (status, live_ok, proof): status 'done' or 'estop'; live_ok = the object stayed in the target zone for
    hold_s during the attempt (the pen-lift check: by the end the arm has returned to rest); proof = that frame.
    """
    print(">", " ".join(cmd), flush=True)
    LIVE_FRAME.unlink(missing_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "LIVE_FRAME": str(LIVE_FRAME)}
    proc = subprocess.Popen(cmd, env=env)
    t0 = time.time()
    cache: dict = {}
    hold_s = setup.get("hold_s", 1.0)
    since, live_ok, proof = None, False, None
    while proc.poll() is None:
        view, in_box = live_view(setup, cache)
        since = (since or time.time()) if in_box else None
        if since and time.time() - since >= hold_s and not live_ok:
            live_ok, proof = True, view
            say(f"Verified. {goal(setup)}")
        live = f"live: {goal(setup) if in_box else 'working'}" if view is not None else "starting..."
        k = show(screen(title, GREEN if in_box else AMBER,
                        f"{sub}\n{live}\n{time.time() - t0:4.0f} s   (ESC = emergency stop)",
                        view if view is not None else last_img), 50)
        if k == 27:
            # Killing the process skips the torque-off on disconnect: the arm freezes in its pose.
            proc.kill()
            proc.wait()
            show(screen("EMERGENCY STOP", RED, "arm frozen in place - cut follower power if needed"), 1)
            say("Emergency stop")
            return "estop", False, None
    if proc.returncode != 0:
        print(f"rollout exited with code {proc.returncode}", flush=True)
    return "done", live_ok, proof


def goal(setup: dict) -> str:
    return setup.get("goal", "pen lifted")


def verdict(live_ok: bool, proof, setup: dict):
    """Live success counts first; otherwise one more look after the arm returned (the object-in-box case)."""
    if live_ok:
        return True, {"live": True}, proof
    ok, stats, img = check_now(setup)
    return ok, {"live": False, **stats}, img


def check_now(setup: dict):
    frame = selfcheck.grab_frame()
    return selfcheck.check(frame, setup)


# ---------------------------------------------------------------- commands

def cmd_demo(a):
    setup = selfcheck.load_setup()
    policy = "mock" if a.mock else resolve_policy(a.policy)
    cmd = rollout_cmd(policy, a.duration, a.n_action_steps, a.ensemble, a.mock)
    react = reactivity_label(a.n_action_steps, a.ensemble)
    print(f"policy {policy}\nreactivity {react}")
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    if a.fullscreen:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    last_img = None

    while True:
        k = show(screen("Ready", FG, "place the pen - SPACE to start, q to quit", last_img), 50)
        if k == ord("q"):
            break
        if k != ord(" "):
            continue
        for attempt in range(1, a.attempts + 1):
            say("On it" if attempt == 1 else "Missed. Trying again")
            status, live_ok, proof = run_attempt(cmd, f"Attempt {attempt}/{a.attempts}", "robot working", last_img, setup)
            if status == "estop":
                while show(screen("EMERGENCY STOP", RED, "SPACE to reset"), 50) != ord(" "):
                    pass
                break
            show(screen("Checking...", AMBER, "", last_img), 1)
            ok, stats, last_img = verdict(live_ok, proof, setup)
            log_row(a.label or "demo", policy, react, "-", ok, "" if ok else "auto", f"demo attempt {attempt} {stats}")
            if ok:
                show(screen("VERIFIED", GREEN, f"{goal(setup)}  (attempt {attempt})", last_img), 1)
                break
            show(screen("MISSED", RED, "not done" + (" - retrying" if attempt < a.attempts else ""), last_img), 1500)
        else:
            say("I could not do it this time")
    cv2.destroyAllWindows()


def cmd_eval(a):
    setup = selfcheck.load_setup()
    policy = "mock" if a.mock else resolve_policy(a.policy)
    cmd = rollout_cmd(policy, a.duration, a.n_action_steps, a.ensemble, a.mock)
    react = reactivity_label(a.n_action_steps, a.ensemble)
    label = a.label
    results = []
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    last_img = None
    pos = a.start
    while pos <= a.positions:
        k = show(screen(f"Position {pos}/{a.positions}", FG,
                        f"{label}  {react}\nput the cube on mark {pos} - SPACE start, s skip, q quit", last_img), 50)
        if k == ord("q"):
            break
        if k == ord("s"):
            pos += 1
            continue
        if k != ord(" "):
            continue
        status, live_ok, proof = run_attempt(cmd, f"Position {pos}", f"{label}  {react}", last_img, setup)
        if status == "estop":
            log_row(label, policy, react, pos, False, "collision", "emergency stop")
            results.append(False)
            pos += 1
            continue
        ok, stats, last_img = verdict(live_ok, proof, setup)
        failure, notes = "", str(stats)
        if not ok:
            keys = "  ".join(f"{i + 1}={t}" for i, t in enumerate(FAILURE_TYPES))
            while True:
                k = show(screen("MISSED - why?", RED, keys[:60] + "\n" + keys[60:] + "\nc = actually a success", last_img), 50)
                if k == ord("c"):
                    ok, notes = True, notes + " (manual override)"
                    break
                if ord("1") <= k < ord("1") + len(FAILURE_TYPES):
                    failure = FAILURE_TYPES[k - ord("1")]
                    break
        else:
            show(screen("SUCCESS", GREEN, str(stats), last_img), 800)
        log_row(label, policy, react, pos, ok, failure, notes)
        results.append(ok)
        pos += 1
    cv2.destroyAllWindows()
    if results:
        print(f"\n{label} {react}: {sum(results)}/{len(results)} successful")


def log_row(label, policy, react, position, ok, failure, notes):
    EVAL_LOG.parent.mkdir(exist_ok=True)
    new = not EVAL_LOG.exists()
    with EVAL_LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "checkpoint", "reactivity", "start_position", "success", "failure_type", "notes", "policy_path"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), label, react, position, int(ok), failure, notes, policy])


def cmd_report(_):
    if not EVAL_LOG.exists():
        sys.exit("no evals yet")
    rows = list(csv.DictReader(EVAL_LOG.open(encoding="utf-8")))
    groups: dict[tuple, list] = {}
    for r in rows:
        if r["start_position"] == "-":  # demo attempts are logged but not part of the 20-position protocol
            continue
        groups.setdefault((r["checkpoint"], r["reactivity"]), []).append(r)
    print(f"{'checkpoint':<16}{'reactivity':<18}{'success':>10}   failures")
    for (ckpt, react), rs in sorted(groups.items(), key=lambda kv: -sum(int(r["success"]) for r in kv[1]) / len(kv[1])):
        n_ok = sum(int(r["success"]) for r in rs)
        fails: dict[str, int] = {}
        for r in rs:
            if r["failure_type"]:
                fails[r["failure_type"]] = fails.get(r["failure_type"], 0) + 1
        print(f"{ckpt:<16}{react:<18}{n_ok:>4}/{len(rs):<5}   " + ", ".join(f"{k} {v}" for k, v in fails.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--policy", help="policy name, checkpoint folder or Hub repo (default: [hf] policy)")
        s.add_argument("--duration", type=int, default=STAGE["duration_s"], help="seconds per attempt")
        s.add_argument("--n-action-steps", type=int, default=STAGE["n_action_steps"], help="re-plan every N steps")
        s.add_argument("--ensemble", type=float, default=STAGE["temporal_ensemble_coeff"],
                       help="temporal ensembling coeff, e.g. 0.01 (0 = off)")
        s.add_argument("--mock", action="store_true", help="no robot: stream the webcam instead of a rollout")

    s = sub.add_parser("demo", help="stage loop: attempt, self-check, retry")
    common(s)
    s.add_argument("--attempts", type=int, default=STAGE["attempts"])
    s.add_argument("--label", help="checkpoint label for the log")
    s.add_argument("--fullscreen", action="store_true")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("eval", help="20-position protocol, one attempt per position, logged to evals/eval_log.csv")
    common(s)
    s.add_argument("--label", required=True, help="checkpoint label, e.g. 10k")
    s.add_argument("--positions", type=int, default=20)
    s.add_argument("--start", type=int, default=1, help="continue from this position")
    s.set_defaults(fn=cmd_eval)

    sub.add_parser("report", help="success rate per checkpoint and reactivity setting").set_defaults(fn=cmd_report)

    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
