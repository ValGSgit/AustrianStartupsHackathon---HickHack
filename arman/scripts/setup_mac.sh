#!/usr/bin/env bash
# One-time install on a Mac (Apple Silicon). About 1.5 to 2 GB of downloads in total.
set -euo pipefail
cd "$(dirname "$0")/.."

brew install ffmpeg uv
uv python install 3.12
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install 'lerobot[core_scripts,training,feetech]'

echo
echo "Done. In every new terminal run:  source .venv/bin/activate"
echo "Then log in to Hugging Face once:  hf auth login"
