"""Test MolmoAct2 on the GPU without a robot: made-up camera image + joint angles in, movements out.

    python test_molmo.py 1      # one camera (copied into both slots by molmo_server)
    python test_molmo.py 2      # two cameras

Runs exactly the steps the model server runs for every request, and prints how long they take.
"""

import sys
import time

import numpy as np
import torch

import molmo_server  # noqa: F401  (applies the MolmoAct2 + one-camera patches)
from lerobot.async_inference import helpers
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features

CKPT = "/teamspace/studios/this_studio/molmo/MolmoAct2-SO100_101-LeRobot"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
REST = [0.0, -97.0, 95.0, 75.0, 0.0, 2.0]  # roughly our arm's resting pose, in degrees / gripper %

n_cams = int(sys.argv[1]) if len(sys.argv) > 1 else 1
hw = {f"{j}.pos": float for j in JOINTS} | {f"cam{i}": (360, 640, 3) for i in range(n_cams)}
lerobot_features = hw_to_dataset_features(hw, OBS_STR, use_video=False)

t = time.perf_counter()
policy = get_policy_class("molmoact2").from_pretrained(CKPT).to("cuda")
pre, post = make_pre_post_processors(
    policy.config,
    pretrained_path=CKPT,
    preprocessor_overrides={"device_processor": {"device": "cuda"}},
    postprocessor_overrides={"device_processor": {"device": "cuda"}},
)
print(f"model loaded in {time.perf_counter() - t:.0f} s, GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print("model camera slots:", list(policy.config.image_features))

rng = np.random.default_rng(0)
raw = {f"{j}.pos": v for j, v in zip(JOINTS, REST)}
raw |= {f"cam{i}": rng.integers(0, 255, (360, 640, 3), dtype=np.uint8) for i in range(n_cams)}
raw["task"] = "move the arm towards the white pawn, grasp it, lift it up, and drop it into the cup"

for attempt in range(3):
    t = time.perf_counter()
    obs = pre(helpers.raw_observation_to_observation(raw, lerobot_features, policy.config.image_features))
    with torch.no_grad():
        chunk = policy.predict_action_chunk(obs)
    if chunk.ndim == 2:
        chunk = chunk.unsqueeze(0)
    actions = torch.stack([post(chunk[:, i, :]) for i in range(chunk.shape[1])], dim=1).squeeze(0)
    torch.cuda.synchronize()
    print(f"attempt {attempt + 1}: {actions.shape[0]} movements in {(time.perf_counter() - t) * 1000:.0f} ms")

np.set_printoptions(precision=1, suppress=True)
print("first movement:", actions[0].float().cpu().numpy())
print("last movement: ", actions[-1].float().cpu().numpy())
print("joint order:   ", JOINTS)
