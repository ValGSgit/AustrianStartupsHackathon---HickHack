#!/usr/bin/env bash
# Fine-tune MolmoAct2 on the Lightning Studio with the demos recorded on this Mac (record.py).
#   scripts/train_molmo.sh               action-expert-only, 3000 steps (start here)
#   scripts/train_molmo.sh lora 3000     LoRA on the vision-language part too
# Copies data/<DATASET_NAME> to the Studio (no Hugging Face account needed), starts training there in the
# background and shows the first progress lines. Watch it later with:
#   ssh $LIGHTNING_SSH 'tail -f ~/molmo/train.log'
# Then use a checkpoint: set MOLMO_CHECKPOINT=/teamspace/studios/this_studio/molmo/outputs/ft_<data>_<mode>/checkpoints/<step>/pretrained_model
# in settings.env and run do_task.py as usual.
# IMPORTANT: turn off auto-sleep for the Studio in Lightning while it trains (it can pause when idle).
set -euo pipefail
cd "$(dirname "$0")/.."
source settings.env
MODE="${1:-expert}"
STEPS="${2:-3000}"

[ -d "data/${DATASET_NAME}" ] || { echo "No recordings in data/${DATASET_NAME}. Record first: python record.py 10" >&2; exit 1; }
echo "Copying data/${DATASET_NAME} ($(du -sh "data/${DATASET_NAME}" | cut -f1)) to the Studio..."
ssh -o BatchMode=yes "${LIGHTNING_SSH}" "mkdir -p ~/molmo/data"
rsync -az --delete -e "ssh -o BatchMode=yes" "data/${DATASET_NAME}/" "${LIGHTNING_SSH}:molmo/data/${DATASET_NAME}/"
scp -q -o BatchMode=yes cloud/train_molmo.sh "${LIGHTNING_SSH}:molmo/"

# The model server and training both need the GPU: stop the server while training.
ssh -o BatchMode=yes "${LIGHTNING_SSH}" "pkill -f '[m]olmo_server' || true"
ssh -o BatchMode=yes "${LIGHTNING_SSH}" "cd ~/molmo && (nohup bash train_molmo.sh '${DATASET_NAME}' '${MODE}' '${STEPS}' > train.log 2>&1 &)"
echo "Training started on the Studio (${MODE}, ${STEPS} steps). First lines:"
sleep 90
ssh -o BatchMode=yes "${LIGHTNING_SSH}" "grep -E 'step:|Error|Traceback|num_learnable' ~/molmo/train.log | tail -5"
