#!/usr/bin/env bash
# Train an ACT policy on your dataset.
#   scripts/train_act.sh cloud   on a Hugging Face GPU (paid per second, fastest)
#   scripts/train_act.sh local   on this Mac (Apple GPU, slow: use only to test that it starts)
# Checkpoints are saved every 5000 steps, so you can test an early one while training continues.
source "$(dirname "$0")/_common.sh"

WHERE="${1:-cloud}"
ARGS=(
  --dataset.repo_id="${DATASET}"
  --policy.type=act
  --policy.repo_id="${POLICY_REPO}"
  --output_dir="outputs/train/act_${DATASET_NAME}"
  --job_name="act_${DATASET_NAME}"
  --steps=30000
  --save_freq=5000
  --batch_size=8
  --wandb.enable=false
)

if [ "${WHERE}" = "cloud" ]; then
  lerobot-train "${ARGS[@]}" --policy.device=cuda --job.target=a10g-small --job.timeout=4h
else
  lerobot-train "${ARGS[@]}" --policy.device=mps
fi
