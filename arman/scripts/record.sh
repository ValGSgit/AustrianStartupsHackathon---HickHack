#!/usr/bin/env bash
# Record demonstrations with the protected joystick control (see record.py). Run in the Terminal app.
#   scripts/record.sh 10    record 10 episodes; run again to add more to the same dataset
source "$(dirname "$0")/_common.sh"
python -u record.py "${1:-10}"
