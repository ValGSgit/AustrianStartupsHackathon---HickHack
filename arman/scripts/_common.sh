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

# Camera names (iphone, webcam) are turned into numbers by config.py; only run this in the Terminal app.
FRONT_CAM=$(python -c "import config; print(config.camera_index('FRONT'))" 2>/dev/null | tail -1)
if [ -n "${WRIST_CAM:-}" ]; then WRIST_CAM=$(python -c "import config; print(config.camera_index('WRIST'))" 2>/dev/null | tail -1); fi
if [ "${FRONT_CAM}" = "realsense" ]; then  # colour image from realsense_bridge.py (runs with sudo)
  CAMS="front: {type: zmq, server_address: 127.0.0.1, port: 5555, camera_name: realsense, width: 640, height: 480, fps: 30}"
else
  CAMS="front: {type: opencv, index_or_path: ${FRONT_CAM}, width: ${FRONT_WIDTH:-640}, height: ${FRONT_HEIGHT:-480}, fps: 30}"
fi
if [ -n "${WRIST_CAM:-}" ]; then
  CAMS="${CAMS}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30}"
fi
CAMERAS="{ ${CAMS} }"

# SmolVLA was pretrained with cameras named camera1 (top-down), camera2 (wrist), camera3. Map ours onto
# those names; missing cameras are filled with blank images (works with just one camera).
SMOLVLA_RENAME='"observation.images.front": "observation.images.camera1"'
SMOLVLA_EMPTY=2
if [ -n "${WRIST_CAM:-}" ]; then
  SMOLVLA_RENAME="${SMOLVLA_RENAME}, \"observation.images.wrist\": \"observation.images.camera2\""
  SMOLVLA_EMPTY=1
fi
SMOLVLA_RENAME="{${SMOLVLA_RENAME}}"

DATASET="${HF_USER}/${DATASET_NAME}"
DATASET_ROOT="data/${DATASET_NAME}"
POLICY_REPO="${HF_USER}/act_${DATASET_NAME}"
