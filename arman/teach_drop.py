"""Teach where to drop things. do_task.py then carries a grabbed object there by itself and opens (AUTO_DROP).

    python teach_drop.py cup       # saves 'above_cup'
    python teach_drop.py bowl      # saves 'above_bowl'
    python teach_drop.py box       # saves 'above_box'

Put the container where it will stay and mark its spot with tape: the robot always drops at this exact place.
Uses the joystick. Also teaches the resting pose 'home' if it is missing.
"""

import sys

from poses import clean_name, load_poses, teach

if len(sys.argv) != 2:
    sys.exit(__doc__)
name = clean_name(sys.argv[1])
steps = []
if "home" not in load_poses():
    steps.append(("home", "Fold the robot into its RESTING pose."))
steps += [
    (f"above_{name}", f"Hold the gripper about 5 cm ABOVE the middle of the {name}, higher than its rim, "
                      "gripper closed. Then press Enter."),
    (None, "Done. Now bring the robot back to its RESTING pose and press Enter."),
]
teach(use_leader=True, steps=steps)
print(f"\nSaved 'above_{name}'. do_task.py uses it when the task mentions '{name}' "
      f"(or with --into {name} / --drop {name}).")
