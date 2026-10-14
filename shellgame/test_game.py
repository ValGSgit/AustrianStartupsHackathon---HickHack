"""Plain-assert tests for game.py and motions.py (no pytest, no hardware).

    .venv\\Scripts\\python shellgame\\test_game.py
"""

import random

import arm as A
import game as G
import motions as M


def check_invariants(s: G.State) -> None:
    assert s.cups.count(None) == 1, s
    assert sorted(c for c in s.cups if c is not None) == [0, 1, 2], s
    assert 1 <= G.empty_slot(s) <= 4 and G.ball_slot(s) != G.empty_slot(s)


def same(a: dict, b: dict) -> bool:
    return all(abs(a[k] - b[k]) < 1e-6 for k in A.KEYS)


def test_new_state():
    for b in (1, 2, 3):
        s = G.new_state(b)
        check_invariants(s)
        assert G.empty_slot(s) == 4 and G.ball_slot(s) == b
    for bad in (0, 4):
        try:
            G.new_state(bad)
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_apply_tracks_ball():
    s = G.new_state(2)
    assert G.legal_moves(s) == [(1, 4), (2, 4), (3, 4)]
    s2 = G.apply(s, (2, 4))
    assert G.ball_slot(s2) == 4 and G.empty_slot(s2) == 2
    assert G.ball_slot(s) == 2, "apply must not mutate its input"
    s3 = G.apply(s2, (1, 2))
    assert G.ball_slot(s3) == 4 and G.empty_slot(s3) == 1
    try:
        G.apply(s3, (2, 3))
        raise AssertionError("expected ValueError for a move into an occupied slot")
    except ValueError:
        pass


def test_random_moves():
    for seed in range(500):
        for n in (1, 3, 5, 7, 10, 15):
            s = G.new_state(1 + seed % 3)
            moves = G.random_moves(s, n, random.Random(seed))
            assert len(moves) == n
            assert moves == G.random_moves(s, n, random.Random(seed)), "not deterministic"
            ball_moves, last = 0, None
            for m in moves:
                assert m in G.legal_moves(s) and m[1] == G.empty_slot(s)
                assert last is None or m != (last[1], last[0]), f"undo {last} -> {m}"
                if m[0] == G.ball_slot(s):
                    ball_moves += 1
                s, last = G.apply(s, m), m
                check_invariants(s)
            assert ball_moves >= n // 3, (seed, n, ball_moves)


def test_poses_lists():
    full = M.fake_poses()
    assert M.missing_poses(full) == []
    assert len(M.required_poses()) == 2 + 4 * 4
    partial = {k: v for k, v in full.items() if k != "in_3"}
    assert M.missing_poses(partial) == ["in_3"]
    try:
        M.move_cup(A.FakeArm(), partial, 3, 4)
        raise AssertionError("expected KeyError for missing pose")
    except KeyError as e:
        assert "in_3" in str(e)


def test_fake_arm_motions():
    poses = M.fake_poses()
    arm = A.FakeArm()
    M.go_ready(arm, poses)
    assert same(A.read_pose(arm), poses["ready"])
    M.move_cup(arm, poses, 1, 4)
    assert same(A.read_pose(arm), poses["above_4"])
    for slot in (2, 4):
        M.reveal(arm, poses, slot, hold_s=0)
        assert same(A.read_pose(arm), poses[f"above_{slot}"])
    M.nudge(arm, poses, 3)
    assert same(A.read_pose(arm), poses["above_3"])
    # gripper never closes tighter than grip_src's value and never opens wider than above_src's; ends open
    arm = A.FakeArm()
    arm.state = dict(poses["ready"])
    sent = []
    arm.send_action = lambda a, f=arm.send_action: (sent.append(dict(a)), f(a))[1]
    M.move_cup(arm, poses, 2, 4)
    grips = [a["gripper.pos"] for a in sent]
    assert min(grips) == poses["grip_2"]["gripper.pos"] and max(grips) <= poses["above_2"]["gripper.pos"]
    assert grips[-1] == poses["above_4"]["gripper.pos"]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"all {len(tests)} tests passed")
