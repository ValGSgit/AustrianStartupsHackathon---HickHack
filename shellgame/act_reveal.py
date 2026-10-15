"""Learned reveal: the ACT policy "pick_cup" lifts the cup (one `lerobot-rollout` run, same command as stage.py).

    .venv\\Scripts\\python shellgame\\act_reveal.py                         # newest checkpoint of act_pick_cup
    .venv\\Scripts\\python shellgame\\act_reveal.py --policy outputs\\train\\act_pick_cup\\checkpoints\\006000\\pretrained_model
    .venv\\Scripts\\python shellgame\\act_reveal.py --mock --duration 5     # no robot: stream the webcam for 5 s
    .venv\\Scripts\\python shellgame\\act_reveal.py --mock --no-camera      # no robot, no camera: just wait

The rollout process needs the follower's COM port and the camera for itself: nothing else may hold them while it runs.
Start with the arm at rest (home) and the cup on the board, like in the recorded episodes.
"""

import argparse
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "syed"  # robot.py, stage.py, outputs/
sys.path.insert(0, str(ROOT))
import robot  # noqa: E402
import stage  # noqa: E402

POLICY_NAME = "act_pick_cup"
CHECKPOINTS = robot.OUTPUTS_DIR / POLICY_NAME / "checkpoints"


def latest_checkpoint() -> str | None:
    """Newest outputs/train/act_pick_cup/checkpoints/<step>/pretrained_model (highest step), else checkpoints/last."""
    if not CHECKPOINTS.is_dir():
        return None
    steps = [d for d in CHECKPOINTS.iterdir() if d.name.isdigit() and (d / "pretrained_model").is_dir()]
    if steps:
        return str(max(steps, key=lambda d: int(d.name)) / "pretrained_model")
    last = CHECKPOINTS / "last" / "pretrained_model"
    return str(last) if last.is_dir() else None


def run_act_reveal(policy_path: str, duration_s: int = 14, n_action_steps: int = 20, mock: bool = False,
                   stop: threading.Event | None = None, camera: bool = True) -> int:
    """Run the policy for duration_s seconds and return the exit code.

    stop: when set, the rollout process is killed (the arm freezes in its pose, torque stays on) -> returns -1.
    mock + camera: rollout_live.py --mock (streams the webcam, no robot). mock without camera: just waits.
    Live camera frames are written to stage.LIVE_FRAME while the rollout runs.
    """
    if mock and not camera:
        print(f"act reveal (mock, no camera): waiting {duration_s} s", flush=True)
        return _wait(duration_s, stop)
    try:  # checkpoint folder, outputs/train/<name>, or a Hub repo; never exit the caller (show.py's worker)
        policy = "mock" if mock else stage.resolve_policy(policy_path)
    except SystemExit as e:
        print(f"act reveal: policy {policy_path!r} not found ({e})", flush=True)
        return 2
    cmd = stage.rollout_cmd(policy, int(duration_s), n_action_steps, 0.0, mock)
    print(">", " ".join(cmd), flush=True)
    stage.LIVE_FRAME.unlink(missing_ok=True)
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "LIVE_FRAME": str(stage.LIVE_FRAME)}
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, cwd=str(ROOT))
    while proc.poll() is None:
        if stop is not None and stop.is_set():
            proc.kill()  # skips the rollout's torque-off: the arm holds where it is
            proc.wait()
            print("act reveal: stopped (rollout killed)", flush=True)
            return -1
        time.sleep(0.05)
    code = proc.returncode
    if code != 0:
        print(f"act reveal: rollout exited with code {code}", flush=True)
        if mock:  # camera busy or missing: keep the timing of a real run
            print("act reveal (mock): camera not available, waiting instead", flush=True)
            return _wait(duration_s - (time.time() - t0), stop)
    return code


def _wait(seconds: float, stop: threading.Event | None) -> int:
    t_end = time.time() + max(0.0, seconds)
    while time.time() < t_end:
        if stop is not None and stop.is_set():
            return -1
        time.sleep(0.05)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", help=f"checkpoint folder or policy name (default: newest {POLICY_NAME} checkpoint)")
    p.add_argument("--duration", type=int, default=14, help="seconds (episodes were 12 s)")
    p.add_argument("--n-action-steps", type=int, default=20, help="ACT re-plans every N steps (100 = blind 3.3 s)")
    p.add_argument("--mock", action="store_true", help="no robot: stream the webcam instead of a rollout")
    p.add_argument("--no-camera", action="store_true", help="with --mock: don't open the camera, just wait")
    a = p.parse_args()
    policy = a.policy or latest_checkpoint()
    if policy is None and not a.mock:
        sys.exit(f"no checkpoint in {CHECKPOINTS} - train first (robot.py train) or pass --policy")
    print(f"policy {policy or 'mock'}   task \"{robot.CONFIG['task']['description']}\"   "
          f"{a.duration} s   n_action_steps {a.n_action_steps}")
    code = run_act_reveal(policy or "mock", a.duration, a.n_action_steps, a.mock, camera=not a.no_camera)
    print(f"exit code {code}")
    sys.exit(code)


if __name__ == "__main__":
    main()
