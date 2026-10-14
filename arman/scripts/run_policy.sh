#!/usr/bin/env bash
# Let the trained model control the arm.
#   scripts/run_policy.sh            run for 30 seconds with the model on the Hub
#   scripts/run_policy.sh 60         run for 60 seconds
#   POLICY=outputs/train/act_so101_chess_to_cup/checkpoints/last/pretrained_model scripts/run_policy.sh
# Keep a hand near the arm the first few times.
source "$(dirname "$0")/_common.sh"

lerobot-rollout \
  --strategy.type=base \
  --policy.path="${POLICY:-${POLICY_REPO}}" \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.cameras="${CAMERAS}" \
  --task="${TASK}" \
  --duration="${1:-30}"
