#!/usr/bin/env bash
# Let the trained model control the arm.
#   scripts/run_policy.sh            run for 30 seconds with the model on the Hub
#   scripts/run_policy.sh 60         run for 60 seconds
#   POLICY=outputs/train/act_so101_chess_to_cup/checkpoints/last/pretrained_model scripts/run_policy.sh
# Keep a hand near the arm the first few times.
source "$(dirname "$0")/_common.sh"
# For a SmolVLA model: POLICY=<your hf user>/smolvla_so101_chess_to_cup scripts/run_policy.sh
EXTRA=()
if [[ "${POLICY:-}" == *smolvla* ]]; then
  EXTRA+=(--rename_map="${SMOLVLA_RENAME}")
fi

lerobot-rollout \
  --strategy.type=base \
  --policy.path="${POLICY:-${POLICY_REPO}}" \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.max_relative_target=15 \
  --robot.cameras="${CAMERAS}" \
  --task="${TASK}" \
  --duration="${1:-30}" \
  ${EXTRA[@]+"${EXTRA[@]}"}
