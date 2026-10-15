"""Can this laptop run SmolVLA? Loads a SmolVLA checkpoint and times action-chunk inference on the GPU.

    .venv\\Scripts\\python sim\\bench_smolvla.py                         # lerobot/smolvla_base (~900 MB download)
    .venv\\Scripts\\python sim\\bench_smolvla.py --policy outputs\\train\\smolvla_x\\checkpoints\\last\\pretrained_model

Feeds random camera frames + joint state in the checkpoint's own feature names, so it measures speed and
memory only (the actions are meaningless). One inference = one chunk of `chunk_size` future actions.
"""

import argparse
import time

import numpy as np
import torch
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--policy", default="lerobot/smolvla_base")
    p.add_argument("--device", default="cuda")
    p.add_argument("--runs", type=int, default=5)
    a = p.parse_args()

    dev = torch.device(a.device)
    cfg = PreTrainedConfig.from_pretrained(a.policy)
    cfg.device = dev.type
    t = time.perf_counter()
    policy = get_policy_class(cfg.type).from_pretrained(a.policy, config=cfg).to(dev).eval()
    pre, post = make_pre_post_processors(policy_cfg=cfg, pretrained_path=a.policy,
                                         preprocessor_overrides={"device_processor": {"device": dev.type}})
    print(f"loaded {cfg.type} in {time.perf_counter() - t:.1f} s, "
          f"{sum(p.numel() for p in policy.parameters()) / 1e6:.0f} M params")
    print("inputs:", {k: tuple(v.shape) for k, v in cfg.input_features.items()})
    print(f"chunk_size={cfg.chunk_size} n_action_steps={cfg.n_action_steps}")

    obs = {}
    for k, f in cfg.input_features.items():
        if k.startswith("observation.images."):
            c, h, w = f.shape
            obs[k] = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        else:
            obs[k] = np.random.uniform(-50, 50, f.shape).astype(np.float32)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(a.runs):
        policy.reset(); pre.reset(); post.reset()  # empty action queue -> every call runs the model
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        act = predict_action(obs, policy, dev, pre, post, use_amp=False, task="Pick up the pen",
                             robot_type="so101_follower")
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
        print(f"run {i + 1}: {times[-1] * 1000:.0f} ms -> action {np.round(act.cpu().numpy().reshape(-1), 1)}")
    steady = times[1:] or times
    print(f"\nchunk inference: {np.mean(steady) * 1000:.0f} ms (first {times[0] * 1000:.0f} ms)")
    print(f"at 30 Hz a chunk of {cfg.n_action_steps} steps lasts {cfg.n_action_steps / 30:.2f} s")
    if dev.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
