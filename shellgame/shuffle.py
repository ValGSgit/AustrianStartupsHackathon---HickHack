"""Run a shell-game shuffle on the arm without the stage screen (motion test).

    .venv\\Scripts\\python shellgame\\shuffle.py --fake
    .venv\\Scripts\\python shellgame\\shuffle.py --slow-test          # speed 0.4, Enter between moves
    .venv\\Scripts\\python shellgame\\shuffle.py --moves 7 --speed 1.0 --seed 3 --ball 2

Ctrl+C at any time: the arm goes home slowly and torque is switched off.
"""

import argparse
import random
import sys

import arm as A
import game as G
import motions as M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", action="store_true", help="no arm (FakeArm); fake poses if poses.json is missing")
    ap.add_argument("--moves", type=int, default=7)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--ball", type=int, default=2, choices=(1, 2, 3), help="start slot of the ball")
    ap.add_argument("--slow-test", action="store_true", help="speed 0.4 and wait for Enter between moves")
    args = ap.parse_args()

    A.SPEED = 0.4 if args.slow_test else args.speed
    poses, fake_poses = A.load_poses(), False
    if args.fake and M.missing_poses(poses):
        poses, fake_poses = M.fake_poses(), True
        print("fake run: using generated poses (poses.json missing or incomplete)")
    missing = M.missing_poses(poses)
    if missing:
        sys.exit(f"poses not taught: {', '.join(missing)}\nrun shellgame/teach.py first")

    seed = args.seed if args.seed is not None else random.randrange(10**6)
    s = G.new_state(args.ball)
    moves = G.random_moves(s, args.moves, random.Random(seed))
    print(f"seed {seed}, speed {A.SPEED}, {len(moves)} moves, ball starts in slot {G.ball_slot(s)}")

    arm = A.connect(fake=args.fake)
    try:
        M.go_ready(arm, poses)
        print(f"showing the ball in slot {G.ball_slot(s)}", flush=True)
        M.show_ball(arm, poses, G.ball_slot(s))
        for k, move in enumerate(moves, 1):
            if args.slow_test:
                input(f"Enter = move {k}: {move[0]} -> {move[1]} ")
            ball_moves = move[0] == G.ball_slot(s)
            M.move_cup(arm, poses, *move)
            s = G.apply(s, move)
            print(f"move {k}: {move[0]} -> {move[1]}{'  (ball cup)' if ball_moves else ''}"
                  f"   ball in slot {G.ball_slot(s)}, empty {G.empty_slot(s)}", flush=True)
        M.go_ready(arm, poses)
        print(f"done: ball is in slot {G.ball_slot(s)}")
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C)")
    finally:
        if fake_poses:  # release() would look for home in poses.json
            A.go_to(arm, poses["home"], 2.0 * A.SPEED)
            arm.disconnect()
        else:
            A.release(arm)
        if args.fake:
            print(f"fake arm: {arm.commands} commands sent")


if __name__ == "__main__":
    main()
