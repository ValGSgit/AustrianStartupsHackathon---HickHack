"""Shell game logic: 3 cups on 4 slots, one slot always empty. Pure, no hardware.

    s = new_state(ball_slot=2)
    for move in random_moves(s, 7, random.Random(1)):
        s = apply(s, move)
    ball_slot(s)   # where the ball is now (1-based)
"""

import random
from dataclasses import dataclass

SLOTS = 4


@dataclass
class State:
    cups: list[int | None]  # index 0..3 = slot 1..4; value = cup id 0..2 or None (empty slot)
    ball_cup: int           # cup id hiding the ball


def new_state(ball_slot: int = 2) -> State:
    """Cups 0,1,2 in slots 1,2,3; slot 4 empty; ball under the cup in `ball_slot` (1..3)."""
    if ball_slot not in (1, 2, 3):
        raise ValueError(f"ball_slot must be 1, 2 or 3 at the start, got {ball_slot}")
    return State(cups=[0, 1, 2, None], ball_cup=ball_slot - 1)


def empty_slot(s: State) -> int:
    return s.cups.index(None) + 1


def ball_slot(s: State) -> int:
    return s.cups.index(s.ball_cup) + 1


def legal_moves(s: State) -> list[tuple[int, int]]:
    dst = empty_slot(s)
    return [(i + 1, dst) for i, c in enumerate(s.cups) if c is not None]


def apply(s: State, move: tuple[int, int]) -> State:
    src, dst = move
    if move not in legal_moves(s):
        raise ValueError(f"illegal move {move}: empty slot is {empty_slot(s)}")
    cups = list(s.cups)
    cups[dst - 1], cups[src - 1] = cups[src - 1], None
    return State(cups=cups, ball_cup=s.ball_cup)


def random_moves(s: State, n: int, rng: random.Random) -> list[tuple[int, int]]:
    """n moves into the empty slot, never undoing the previous one, ball cup moved >= n//3 times.

    The ball cup can never move twice in a row (that would be an undo), so with r moves left and the ball
    movable, at most ceil(r/2) ball moves are still possible: force a ball move when that limit is reached.
    """
    need, moves, last = n // 3, [], None
    for i in range(n):
        options = [m for m in legal_moves(s) if last is None or m != (last[1], last[0])]
        ball_move = next((m for m in options if m[0] == ball_slot(s)), None)
        left_after = n - i - 1
        if ball_move and need > 0 and (left_after + 1) // 2 < need:
            move = ball_move
        else:
            move = rng.choice(options)
        if move[0] == ball_slot(s):
            need -= 1
        s, last = apply(s, move), move
        moves.append(move)
    return moves
