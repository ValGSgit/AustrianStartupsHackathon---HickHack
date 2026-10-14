"""Roll out a trained LeRobot policy in the MuJoCo sim and report its success rate.

Uses LeRobot's own `predict_action` with the policy's saved pre/post-processors, i.e. the same
inference path as `lerobot-rollout` on the real arm.

    python sim/eval_policy.py --policy outputs/train/act_sim_pick_pen/checkpoints/last/pretrained_model --obj pen
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from so101_sim.env import SO101PickEnv  # noqa: E402

TASKS = {"pen": "Pick up the pen", "cube": "Pick up the red cube"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", required=True, help="checkpoint folder (â€¦/pretrained_model) or Hub repo id")
    p.add_argument("--obj", choices=list(TASKS), default="pen")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--steps", type=int, default=210, help="control steps per episode (30 Hz)")
    p.add_argument("--seed", type=int, default=10_000, help="first seed; demos use 0.., so these layouts are unseen")
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    p.add_argument("--video", help="save the first episode as this .mp4")
    a = p.parse_args()

    import torch
    from lerobot.common.control_utils import predict_action
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = PreTrainedConfig.from_pretrained(a.policy)
    cfg.device = device.type
    policy = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=a.policy,
        preprocessor_overrides={"device_processor": {"device": device.type}},
    )

    env = SO101PickEnv(a.obj)
    successes, frames = 0, []
    for ep in range(a.episodes):
        obs = env.reset(seed=a.seed + ep)
        policy.reset()
        pre.reset()
        post.reset()
        lifted = False
        for _ in range(a.steps):
            action = predict_action(
                {"observation.state": obs.state, "observation.images.front": obs.image},
                policy, device, pre, post, use_amp=False, task=TASKS[a.obj], robot_type="so101_follower_sim",
            )
            obs = env.step(action.cpu().numpy().reshape(-1))  # predict_action keeps the batch dim
            lifted |= env.is_success()
            if ep == 0 and a.video:
                frames.append(obs.image)
        successes += lifted
        h = env.object_height()
        end = "knocked off the table" if h < -0.05 else f"object {h * 100:.1f} cm above the table at the end"
        print(f"episode {ep + 1}/{a.episodes}: {'success' if lifted else 'fail'}  ({end})")
    env.close()
    print(f"\nSuccess rate: {successes}/{a.episodes} = {100 * successes / a.episodes:.0f}%")

    if a.video and frames:
        import av

        with av.open(a.video, "w") as out:
            stream = out.add_stream("libx264", rate=30)
            stream.width, stream.height, stream.pix_fmt = frames[0].shape[1], frames[0].shape[0], "yuv420p"
            for f in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                    out.mux(packet)
            for packet in stream.encode():
                out.mux(packet)
        print(f"video: {a.video}")


if __name__ == "__main__":
    main()
