# Shared setup for the scripts. Not meant to be run directly.
set -euo pipefail
cd "$(dirname "$0")/.."
source settings.env

if ! command -v lerobot-record >/dev/null 2>&1; then
  echo "LeRobot not found. Run first:  source .venv/bin/activate" >&2
  exit 1
fi
if grep -q "CHANGE_ME" settings.env; then
  echo "settings.env still has CHANGE_ME values. Fill them in first (see README section 0)." >&2
  exit 1
fi

CAMS="front: {type: opencv, index_or_path: ${FRONT_CAM}, width: ${FRONT_WIDTH:-640}, height: ${FRONT_HEIGHT:-480}, fps: 30}"
if [ -n "${WRIST_CAM:-}" ]; then
  CAMS="${CAMS}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30}"
fi
CAMERAS="{ ${CAMS} }"

DATASET="${HF_USER}/${DATASET_NAME}"
DATASET_ROOT="data/${DATASET_NAME}"
POLICY_REPO="${HF_USER}/act_${DATASET_NAME}"
