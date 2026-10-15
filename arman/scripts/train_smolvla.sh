#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base, pretrained on community SO-100/SO-101 data) on your dataset.
# Uses the SAME recordings as ACT (scripts/record.sh), so record once and train both.
# Works with one camera (front, top-down) or two (front + wrist): see SMOLVLA_RENAME in _common.sh.
#   scripts/train_smolvla.sh           on a Hugging Face A100 (paid per second; needs credit on your HF account)
#   STEPS=5000 scripts/train_smolvla.sh   shorter run to try it first (the full recipe is 20000 steps, ~4 h)
# Never train this on the MacBook Air: far too slow, and it gets very hot.
source "$(dirname "$0")/_common.sh"

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.repo_id="${HF_USER}/smolvla_${DATASET_NAME}" \
  --dataset.repo_id="${DATASET}" \
  --output_dir="outputs/train/smolvla_${DATASET_NAME}" \
  --job_name="smolvla_${DATASET_NAME}" \
  --batch_size=64 \
  --steps="${STEPS:-20000}" \
  --save_freq=5000 \
  --policy.device=cuda \
  --rename_map="${SMOLVLA_RENAME}" \
  --policy.empty_cameras="${SMOLVLA_EMPTY}" \
  --wandb.enable=false \
  --job.target=a100-large \
  --job.timeout=6h
