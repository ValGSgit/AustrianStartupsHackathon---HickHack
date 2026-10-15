"""Generate scripted pick demos in MuJoCo and save them as a LeRobot dataset.

The dataset uses the real robot's feature names (shoulder_pan.pos ... gripper.pos,
observation.images.front at 640x480, 30 fps), so `robot.py train --dataset <name>` trains on
it unchanged. Only successful episodes are kept.

    python sim/gen_demos.py --obj pen --episodes 50            -> data/sim_pick_pen
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_sim.env import CAM_H, CAM_W, FPS, MOTOR_NAMES, SO101PickEnv  # noqa: E402
from so101_sim.expert import plan_pick  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TASKS = {"pen": "Pick up the pen", "cube": "Pick up the red cube"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--obj", choices=list(TASKS), default="pen")
    p.add_argument("--episodes", type=int, default=50, help="successful episodes to keep")
    p.add_argument("--name", help="dataset name (default: sim_pick_<obj>)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    name = a.name or f"sim_pick_{a.obj}"
    root = ROOT / "data" / name
    if root.exists():
        if not a.overwrite:
            sys.exit(f"{root} exists - pass --overwrite to replace it")
        shutil.rmtree(root)

    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": MOTOR_NAMES},
        "action": {"dtype": "float32", "shape": (6,), "names": MOTOR_NAMES},
        "observation.images.front": {
            "dtype": "video", "shape": (CAM_H, CAM_W, 3), "names": ["height", "width", "channels"],
        },
    }
    ds = LeRobotDataset.create(
        repo_id=f"local/{name}", fps=FPS, features=features, robot_type="so101_follower_sim",
        root=root, use_videos=True,
    )
    env = SO101PickEnv(a.obj)
    kept = tried = 0
    t0 = time.time()
    while kept < a.episodes:
        seed = a.seed + tried
        tried += 1
        obs = env.reset(seed=seed)
        actions = plan_pick(env, np.random.default_rng(seed))
        frames = []
        for act in actions:
            # Same pairing as lerobot-record: observation at time t, action commanded at time t.
            frames.append((obs.state, act, obs.image))
            obs = env.step(act)
        if not env.is_success():
            print(f"  seed {seed}: failed, skipped")
            continue
        for state, act, img in frames:
            ds.add_frame({
                "observation.state": state, "action": np.asarray(act, np.float32),
                "observation.images.front": img, "task": TASKS[a.obj],
            })
        ds.save_episode()
        kept += 1
        print(f"episode {kept}/{a.episodes} (seed {seed}, {len(frames)} frames)")
    ds.finalize()
    env.close()
    print(f"\nSaved {kept} episodes ({ds.num_frames} frames) to {root}  "
          f"[{kept}/{tried} scripted successes, {time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
