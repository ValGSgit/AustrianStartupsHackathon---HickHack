#!/usr/bin/env bash
# Record demonstrations with the leader arm.
#   scripts/record.sh 10    record 10 episodes (creates the dataset, or adds to it if it exists)
# Keys while recording: n / right arrow = episode done, r / left arrow = redo, q / Esc = stop + save
# Never change camera names, camera positions or resolution between batches.
source "$(dirname "$0")/_common.sh"

EPISODES="${1:-10}"
EXTRA=()

# A failed run (wrong port, camera blocked) leaves a dataset folder with no episodes in it,
# which makes every later run fail with "File exists". Remove it only if it holds no recordings.
if [ -d "${DATASET_ROOT}" ]; then
  if find "${DATASET_ROOT}" -name "*.parquet" -path "*data*" | grep -q .; then
    echo "Adding ${EPISODES} episodes to the existing dataset in ${DATASET_ROOT}"
    EXTRA+=(--resume=true)
  else
    echo "Removing empty leftover folder ${DATASET_ROOT} from a failed run"
    rm -rf "${DATASET_ROOT}"
  fi
fi

lerobot-record \
  --robot.type=so101_follower \
  --robot.port="${FOLLOWER_PORT}" \
  --robot.id="${FOLLOWER_ID}" \
  --robot.cameras="${CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port="${LEADER_PORT}" \
  --teleop.id="${LEADER_ID}" \
  --display_data="${DISPLAY_DATA:-true}" \
  --dataset.repo_id="${DATASET}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.num_episodes="${EPISODES}" \
  --dataset.single_task="${TASK}" \
  --dataset.episode_time_s=20 \
  --dataset.reset_time_s=10 \
  --dataset.push_to_hub=true \
  ${EXTRA[@]+"${EXTRA[@]}"}
