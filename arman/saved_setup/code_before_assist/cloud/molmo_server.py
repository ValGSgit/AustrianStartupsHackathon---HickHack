"""MolmoAct2 model server for the SO-101, runs on the Lightning AI GPU (LeRobot async inference).

    python molmo_server.py          # listens on 127.0.0.1:8080 (reach it through an SSH tunnel)

Additions to LeRobot 0.6.1's policy server:
1. MolmoAct2 is not on its list of supported policies yet, although loading it works the generic way.
2. The checkpoint has two camera slots (cam0, cam1). With one camera, its image is copied into the
   missing slot, so a single phone camera works.
3. The model is loaded and warmed up ONCE at start and reused for every robot connection. Stock LeRobot
   reloads it on every connection (~100 s of waiting before each task).
4. Planning on every request (stock LeRobot skips requests when the joints barely moved).
5. Images stay at 378x378, the size MolmoAct2's own image processor uses. Stock LeRobot squeezes every
   camera image to the 224x224 listed in the checkpoint config, so the model saw a blurry upscaled copy.
Prints "MODEL READY ... checkpoint=<path>" when it can take tasks.

    python molmo_server.py --checkpoint=/path/to/finetuned/pretrained_model   # a fine-tuned model
"""

import logging
import pickle
import sys
import time
from concurrent import futures

import grpc
import numpy as np
import torch
from lerobot.async_inference import constants, helpers
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import hw_to_dataset_features

CHECKPOINT = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--checkpoint=")),
                  "/teamspace/studios/this_studio/molmo/MolmoAct2-SO100_101-LeRobot")
MODEL_IMAGE_SIZE = 378  # MolmoAct2 image processor: crop_mode "resize" to 378x378 (processor_config.json)
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

if "molmoact2" not in constants.SUPPORTED_POLICIES:
    constants.SUPPORTED_POLICIES.append("molmoact2")

_prepare = helpers.prepare_raw_observation


def _decode_jpegs(robot_obs: dict) -> dict:
    """The Mac sends camera images as small JPEGs (224x224, ~15 KB instead of ~0.7 MB raw): unpack them."""
    import cv2

    out = dict(robot_obs)
    for k, v in robot_obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.dtype == np.uint8:
            out[k] = cv2.cvtColor(cv2.imdecode(v, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return out


def prepare_with_camera_copy(robot_obs, lerobot_features, policy_image_features):
    obs = _prepare(_decode_jpegs(robot_obs), lerobot_features, policy_image_features)
    present = [k for k in policy_image_features if k in obs]
    for key in policy_image_features:
        if key not in obs and present:
            obs[key] = obs[present[0]].clone() if isinstance(obs[present[0]], torch.Tensor) else obs[present[0]]
    return obs


helpers.prepare_raw_observation = prepare_with_camera_copy


def resize_for_molmo(image: torch.Tensor, resize_dims) -> torch.Tensor:
    """(H, W, C) -> (C, 378, 378). Replaces LeRobot's resize to the config's 224x224."""
    image = image.permute(2, 0, 1)
    if tuple(image.shape[1:]) != (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE):
        image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
                                                mode="bilinear", align_corners=False).squeeze(0)
    return image


helpers.resize_robot_observation_image = resize_for_molmo

from lerobot.async_inference import policy_server as _ps  # noqa: E402  (after the patches)
from lerobot.async_inference.policy_server import PolicyServer  # noqa: E402

# 4. Stock LeRobot skips a request when the joints moved < ~1 degree since the last plan. With one robot and a
#    fast GPU that only causes pauses (the robot waits while gripping or holding still), so plan every time.
_ps.observations_similar = lambda *args, **kwargs: False

_CACHE: dict = {}
_load_instructions = PolicyServer.SendPolicyInstructions


def send_policy_instructions(self, request, context):
    """Reuse the loaded model when a robot connects with the same model; only take its camera/joint layout."""
    specs = pickle.loads(request.data)  # nosec  (same as LeRobot; only our own Mac connects via the tunnel)
    key = (specs.policy_type, specs.pretrained_name_or_path, specs.device)
    if _CACHE.get("key") == key:
        self.device, self.policy_type = specs.device, specs.policy_type
        self.lerobot_features, self.actions_per_chunk = specs.lerobot_features, specs.actions_per_chunk
        self.policy, self.preprocessor, self.postprocessor = _CACHE["policy"], _CACHE["pre"], _CACHE["post"]
        self.policy.reset()
        self.logger.info(f"Robot connected, reusing the loaded model ({specs.actions_per_chunk} actions per chunk)")
        return services_pb2.Empty()
    result = _load_instructions(self, request, context)
    _CACHE.update(key=key, policy=self.policy, pre=self.preprocessor, post=self.postprocessor)
    return result


PolicyServer.SendPolicyInstructions = send_policy_instructions


class _Context:
    def peer(self):
        return "preload"


def choose_precision() -> None:
    """float32 (as the authors tested) needs ~26 GB; on smaller GPUs (L4, A10G: 24 GB) use bfloat16 (<16 GB)."""
    import json
    from pathlib import Path

    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    dtype = "float32" if gb > 40 else "bfloat16"
    cfg_file = Path(CHECKPOINT) / "config.json"
    cfg = json.loads(cfg_file.read_text())
    if cfg.get("model_dtype") != dtype:
        cfg["model_dtype"] = dtype
        cfg_file.write_text(json.dumps(cfg, indent=2))
    logging.getLogger("molmo").info(f"GPU {torch.cuda.get_device_name(0)} ({gb:.0f} GB): using {dtype}")


def preload(server: PolicyServer) -> None:
    """Load the model and run one inference so CUDA is warmed up before the first real task."""
    hw = {f"{j}.pos": float for j in JOINTS} | {"cam0": (MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3)}
    features = hw_to_dataset_features(hw, OBS_STR, use_video=False)
    specs = RemotePolicyConfig("molmoact2", CHECKPOINT, features, 30, "cuda")
    t = time.perf_counter()
    server.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(specs)), _Context())
    raw = {f"{j}.pos": 0.0 for j in JOINTS} | {"cam0": np.zeros((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE, 3), dtype=np.uint8),
                                              "task": "warm up"}
    obs = server.preprocessor(helpers.raw_observation_to_observation(raw, features, server.policy_image_features))
    with torch.no_grad():
        server.policy.predict_action_chunk(obs)
    server.policy.reset()
    server.logger.info(f"MODEL READY (loaded and warmed up in {time.perf_counter() - t:.0f} s) checkpoint={CHECKPOINT}")


def main() -> None:
    port = next((int(a.split("=", 1)[1]) for a in sys.argv[1:] if a.startswith("--port=")), 8080)
    cfg = PolicyServerConfig(host="127.0.0.1", port=port)
    server = PolicyServer(cfg)
    choose_precision()
    preload(server)
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server, grpc_server)
    grpc_server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    grpc_server.start()
    server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    grpc_server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
