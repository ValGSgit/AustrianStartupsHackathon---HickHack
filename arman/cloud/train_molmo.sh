#!/usr/bin/env bash
# Fine-tune MolmoAct2-SO100_101 on our own demonstrations. Runs ON THE LIGHTNING STUDIO (started by
# scripts/train_molmo.sh on the Mac). Follows the official LeRobot MolmoAct2 docs:
#   - start from the LeRobot checkpoint (keeps its SO-100/101 joint-frame correction and normalisation)
#   - small real dataset (<200 demos): "action expert only" or LoRA, keep the action expert trainable
#   - bf16, gradient checkpointing, 8 flow timesteps, image augmentation
#
#   bash train_molmo.sh <dataset_name> [expert|lora] [steps]
#     expert: trains only the action expert (fits an L4 at batch 16, ~18 GB)      <- start here
#     lora:   also adapts the vision-language part via LoRA (batch 8, ~20 GB)    <- if vision is the problem
set -euo pipefail
cd ~/molmo && source .venv/bin/activate

DATA="${1:?dataset name, e.g. so101_chess_to_cup}"
MODE="${2:-expert}"
STEPS="${3:-3000}"
ROOT="$HOME/molmo/data/${DATA}"
BASE="$HOME/molmo/MolmoAct2-SO100_101-LeRobot"
OUT="$HOME/molmo/outputs/ft_${DATA}_${MODE}"

if [ "${MODE}" = "lora" ]; then
  MODE_ARGS=(--policy.enable_lora_vlm=true --policy.action_mode=continuous --batch_size=8)
else
  MODE_ARGS=(--policy.train_action_expert_only=true --policy.action_mode=continuous --batch_size=16)
fi

# MolmoAct2 normalises with quantile statistics; fills them in only if the dataset lacks them.
python -m lerobot.scripts.augment_dataset_quantile_stats --repo-id "local/${DATA}" --root "${ROOT}" || true

rm -rf "${OUT}"
lerobot-train \
  --policy.path="${BASE}" \
  --dataset.repo_id="local/${DATA}" \
  --dataset.root="${ROOT}" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --rename_map='{"observation.images.front": "observation.images.cam0", "observation.images.side": "observation.images.cam1"}' \
  --policy.device=cuda \
  --policy.model_dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.num_flow_timesteps=8 \
  --policy.push_to_hub=false \
  "${MODE_ARGS[@]}" \
  --steps="${STEPS}" \
  --save_freq=500 \
  --log_freq=20 \
  --num_workers=4 \
  --wandb.enable=false \
  --output_dir="${OUT}" \
  --job_name="ft_${DATA}_${MODE}"

echo "TRAINING_DONE checkpoints in ${OUT}/checkpoints (use .../<step>/pretrained_model)"
