#!/usr/bin/env bash
# Let MolmoAct2 (on the Lightning AI GPU) do a task. Same as: python do_task.py "..."  (see do_task.py)
cd "$(dirname "$0")/.." && source .venv/bin/activate && python -u do_task.py "$@"
