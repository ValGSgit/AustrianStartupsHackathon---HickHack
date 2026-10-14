"""Scripted pick and place from taught poses: chess pieces on taped spots go into the cup.

    python pick_place.py all        # every taught spot, in order
    python pick_place.py 2 1        # only spots 2 then 1
    python pick_place.py all --ask  # wait for Enter between pieces

Teach these with `python poses.py teach`:
    home        resting pose, safe to go limp in (the arm goes limp when the script ends)
    above_cup   above the cup opening
    above_N     above spot N, gripper OPEN (N = 1, 2, 3, ...). HIGHER than the cup rim + 3 cm,
                so the hanging piece clears the rim on the way to the cup.
    grasp_N     jaws around the LOW part of the piece on spot N, gripper CLOSED (squeeze a little).
                Low grasp = works for short pawns and tall kings alike.
    safe        (optional) a high pose between the spots and the cup. If taught, every
                transfer passes through it, which avoids knocking over the cup or other pieces.

"""

import sys

from poses import go_to, load_poses, release, with_gripper
import config


def taught_spots(poses: dict) -> list[int]:
    return sorted(
        int(name.split("_")[1])
        for name in poses
        if name.startswith("above_") and name.split("_")[1].isdigit() and f"grasp_{name.split('_')[1]}" in poses
    )


def pick_place(robot, poses: dict, spot: int) -> None:
    above, grasp = poses[f"above_{spot}"], poses[f"grasp_{spot}"]
    opened, closed = above["gripper.pos"], grasp["gripper.pos"]

    go_to(robot, with_gripper(above, opened), seconds=1.5)
    go_to(robot, with_gripper(grasp, opened), seconds=1.0)  # go down slowly
    go_to(robot, grasp, seconds=0.6)  # close on the piece
    go_to(robot, with_gripper(above, closed), seconds=1.0)  # lift straight up
    if "safe" in poses:
        go_to(robot, with_gripper(poses["safe"], closed), seconds=1.2)
    go_to(robot, with_gripper(poses["above_cup"], closed), seconds=2.0)
    go_to(robot, with_gripper(poses["above_cup"], opened), seconds=0.5)  # drop
    if "safe" in poses:
        go_to(robot, with_gripper(poses["safe"], opened), seconds=1.2)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    poses = load_poses()
    missing = [p for p in ("home", "above_cup") if p not in poses]
    spots = taught_spots(poses)
    if missing or not spots:
        sys.exit(f"Teach poses first (python poses.py teach). Missing: {missing or 'above_N + grasp_N for at least one spot'}")
    if not args:
        sys.exit(f"Which spots? e.g. `python pick_place.py all` (taught spots: {spots})")

    if args != ["all"] and not all(a.isdigit() for a in args):
        sys.exit(f"Spots must be numbers or 'all', e.g. `python pick_place.py 1 2` (taught spots: {spots})")
    order = spots if args == ["all"] else [int(a) for a in args]
    unknown = [s for s in order if s not in spots]
    if unknown:
        sys.exit(f"Spots {unknown} are not taught. Taught spots: {spots}")

    robot = config.make_robot()
    robot.connect()
    try:
        go_to(robot, poses["home"], seconds=1.5)
        for i, spot in enumerate(order):
            if i > 0 and "--ask" in sys.argv:
                input(f"Next: spot {spot}. Press Enter ")
            print(f"picking spot {spot}")
            pick_place(robot, poses, spot)
    finally:
        release(robot)


if __name__ == "__main__":
    main()
