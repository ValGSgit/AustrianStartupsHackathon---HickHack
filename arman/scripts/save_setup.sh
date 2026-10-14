#!/usr/bin/env bash
# Save or restore everything that took effort to set up: both calibrations, the arm offsets,
# settings and taught poses.
#   scripts/save_setup.sh           copy the current setup into saved_setup/
#   scripts/save_setup.sh restore   put saved_setup/ back (new laptop, cleared cache, broken calibration)
set -euo pipefail
cd "$(dirname "$0")/.."
source settings.env

CAL="${HOME}/.cache/huggingface/lerobot/calibration"
ROBOT_CAL="${CAL}/robots/so_follower/${FOLLOWER_ID}.json"
JOY_CAL="${CAL}/teleoperators/so_leader/${LEADER_ID}.json"

if [ "${1:-}" = "restore" ]; then
  mkdir -p "$(dirname "${ROBOT_CAL}")" "$(dirname "${JOY_CAL}")"
  cp saved_setup/robot_calibration.json "${ROBOT_CAL}"
  cp saved_setup/joystick_calibration.json "${JOY_CAL}"
  cp saved_setup/offsets.json offsets.json
  [ -f saved_setup/poses.json ] && cp saved_setup/poses.json poses.json
  echo "Restored. Settings were not touched (see saved_setup/settings.env if you need them)."
  echo "Next connect asks once to write the calibration to the motors: press Enter to accept it."
else
  mkdir -p saved_setup
  cp "${ROBOT_CAL}" saved_setup/robot_calibration.json
  cp "${JOY_CAL}" saved_setup/joystick_calibration.json
  cp offsets.json settings.env saved_setup/
  [ -f poses.json ] && cp poses.json saved_setup/
  echo "Saved to saved_setup/ ($(date '+%Y-%m-%d %H:%M'))"
fi
