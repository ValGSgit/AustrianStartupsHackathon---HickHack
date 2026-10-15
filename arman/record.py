"""Record demonstrations for ACT / SmolVLA, driving the robot with the protected joystick control.

    python record.py 10      # record 10 episodes (adds to the dataset if it already exists)

Same dataset format as lerobot-record (same LeRobot functions), but the robot is driven through our
corrections (offsets.json, end-stop limits) with the start-pose check and the blocked-joint guard.
Run it in the Terminal app: macOS only lets Terminal use the camera.

Before each episode: pawn on a spot, arm at rest, press Enter to start.
During an episode:   Enter = done (pawn is in the cup)   r + Enter = redo it   q + Enter = stop
Test without hardware: FAKE_ROBOT=1 python record.py 1
"""

import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path

import config
from poses import BLOCKED_DIFF, BLOCKED_SECONDS, FPS, _gap, _match_start_pose, go_to, joints_only, release

MAX_EPISODE_S = 30


def follow(robot, leader, typed, dataset=None, task=None, max_s=None) -> str:
    """Robot copies the joystick until a key is pressed. Records frames if a dataset is given.
    Returns "enter", "r", "q", "timeout" or "blocked"."""
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import build_dataset_frame

    start, blocked_since = time.time(), None
    while True:
        t0 = time.perf_counter()
        command = config.to_robot(robot, leader.get_action())
        try:
            robot.send_action(command)
            obs = robot.get_observation()
        except (RuntimeError, ConnectionError) as e:
            print(f"\nA motor reported a problem ({e}). Stopping. Something may be blocking the arm.")
            return "blocked"

        stuck = [k for k in command if k != "gripper.pos" and _gap(k, command[k], obs[k]) > BLOCKED_DIFF]
        if not stuck:
            blocked_since = None
        elif blocked_since is None:
            blocked_since = time.time()
        elif time.time() - blocked_since > BLOCKED_SECONDS:
            print(f"\nThe robot cannot reach the joystick's pose ({', '.join(k.split('.')[0] for k in stuck)}).")
            print("Stopping to protect the motors. Check what blocks the arm.")
            return "blocked"

        if dataset is not None:
            # Record what we command (in robot terms), like lerobot-record does with the teleop action.
            dataset.add_frame({**build_dataset_frame(dataset.features, obs, prefix=OBS_STR),
                               **build_dataset_frame(dataset.features, command, prefix=ACTION),
                               "task": task})
        try:
            key = typed.get_nowait()
        except queue.Empty:
            key = None
        if key is not None:
            return key if key in ("r", "q") else "enter"
        if max_s and time.time() - start > max_s:
            print(f"\n{max_s} s are up, episode ends.")
            return "timeout"
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - t0)))


def main() -> None:
    episodes = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 10
    from lerobot.datasets import LeRobotDataset
    from lerobot.utils.constants import ACTION, OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    S = config.S
    name = os.environ.get("DATASET_NAME") or S["DATASET_NAME"]  # env override, used for tests
    repo_id = f"{S['HF_USER']}/{name}"
    root = Path(__file__).parent / "data" / name
    task = S["TASK"]
    if episodes == 0:  # upload only
        LeRobotDataset(repo_id, root=root).push_to_hub()
        print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")
        return
    robot, leader = config.make_robot(cameras=True), config.make_leader()

    if root.exists() and not any(root.glob("data/**/*.parquet")):
        shutil.rmtree(root)  # empty leftover from a run that failed before saving anything
    if root.exists():
        dataset = LeRobotDataset.resume(repo_id, root=root, image_writer_threads=4)
        print(f"Adding to the existing dataset ({dataset.num_episodes} episodes so far).")
    else:
        features = {**hw_to_dataset_features(robot.action_features, ACTION),
                    **hw_to_dataset_features(robot.observation_features, OBS_STR)}
        dataset = LeRobotDataset.create(repo_id, FPS, features, root=root, robot_type=robot.name,
                                        use_videos=True, image_writer_threads=4)
        print(f"New dataset: {root}")

    typed = queue.Queue()
    threading.Thread(target=lambda: [typed.put(line.strip().lower()) for line in sys.stdin], daemon=True).start()
    robot.connect()
    leader.connect()
    rest = joints_only(robot.get_observation())
    saved = 0
    try:
        if not config.FAKE:
            _match_start_pose(robot, leader)
        go_to(robot, joints_only(config.to_robot(robot, leader.get_action())), seconds=2.0)
        print(f"\nTask text: \"{task}\"   (Enter = start/done, r = redo, q = stop)")
        while saved < episodes:
            print(f"\nEPISODE {saved + 1}/{episodes}: pawn on a spot, arm at rest. Press Enter to START.", flush=True)
            if follow(robot, leader, typed) in ("q", "blocked"):
                break
            print("RECORDING... press Enter when the pawn is in the cup.", flush=True)
            result = follow(robot, leader, typed, dataset, task, MAX_EPISODE_S)
            if result in ("enter", "timeout"):
                dataset.save_episode()
                saved += 1
                print(f"   saved ({dataset.num_episodes} episodes in the dataset)")
                continue
            dataset.clear_episode_buffer()
            print("   not saved")
            if result != "r":
                break
    finally:
        dataset.finalize()
        leader.disconnect()
        release(robot, fallback=rest)
        print(f"\nRecorded {saved} new episodes, {dataset.num_episodes} in total, saved in {root}")
        if saved and not config.FAKE and not os.environ.get("DATASET_NAME"):
            try:
                dataset.push_to_hub()
                print(f"Uploaded: https://huggingface.co/datasets/{repo_id}")
            except Exception as e:
                print(f"Not uploaded ({str(e)[:120]}). The recordings are safe on this Mac.")
                print("Log in with `hf auth login`, then upload with: python record.py 0")


if __name__ == "__main__":
    main()
