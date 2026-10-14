"""Teach the robot to put a pawn in the cup, step by step. You only press Enter.

    python teach_pawn.py       # pawn on spot 1 (also teaches resting pose and cup)
    python teach_pawn.py 2     # another spot; resting pose and cup stay as taught

At the end it offers to run it right away: the robot then picks up the pawn and drops it in the cup.
"""

import subprocess
import sys

from poses import load_poses, teach

spot = sys.argv[1] if len(sys.argv) > 1 else "1"
if not spot.isdigit():
    sys.exit(__doc__)
known = load_poses()
steps = []
if spot == "1" or "home" not in known:
    steps.append(("home", "Fold the robot into its RESTING pose."))
if spot == "1" or "above_cup" not in known:
    steps.append(("above_cup", "Move the gripper right ABOVE the cup opening."))
steps += [
    (f"above_{spot}", f"Release the trigger (gripper OPEN). Hold the gripper about 10 cm ABOVE pawn {spot}, higher than the cup."),
    (f"grasp_{spot}", "Go straight DOWN until the jaws are around the BOTTOM of the pawn, then SQUEEZE the trigger."),
    (None, "Now RELEASE the trigger (leave the pawn standing) and lift the gripper straight up."),
]
typed = teach(use_leader=True, steps=steps)

print("\nRun it now? The robot picks up the pawn and drops it in the cup by itself.")
print("Put the pawn back on its spot if it moved. Enter = run, n + Enter = skip: ", end="", flush=True)
if typed.get().strip().lower() != "n":
    subprocess.run([sys.executable, "-u", "pick_place.py", spot], check=False)
