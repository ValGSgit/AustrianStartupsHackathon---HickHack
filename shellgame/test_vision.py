"""Synthetic-frame tests for vision.py. Run: .venv\\Scripts\\python shellgame\\test_vision.py"""

import cv2
import numpy as np

from vision import annotate, ball_visible, check_layout, find_cups, nearest_cup, sample_hsv, slot_states

SLOTS = [(120, 300), (250, 300), (390, 300), (520, 300)]
R = 40
BLUE, ORANGE, RED, RED2 = (200, 80, 30), (0, 140, 255), (30, 30, 220), (70, 20, 220)  # BGR; RED2 hue ~ 172


def scene(cups: dict[int, tuple], ball=None, size=(640, 480), cup_col=BLUE) -> np.ndarray:
    """cups: slot -> (dx, dy) shift; ball: (x, y). Coordinates are for 640x480 and scaled to size."""
    w, h = size
    s = w / 640
    img = np.full((h, w, 3), 70, np.uint8)
    for slot, (dx, dy) in cups.items():
        x, y = SLOTS[slot - 1]
        cv2.circle(img, (round((x + dx) * s), round((y + dy) * s)), round(38 * s), cup_col, -1)
    if ball:
        cv2.circle(img, (round(ball[0] * s), round(ball[1] * s)), round(9 * s), ORANGE, -1)
    return img


def make_setup(cup_col=BLUE) -> dict:
    ref = scene({1: (0, 0)}, ball=(320, 120), cup_col=cup_col)
    return {"slots": [[x, y, R] for x, y in SLOTS], "cup_hsv": list(sample_hsv(ref, *SLOTS[0])),
            "ball_hsv": list(sample_hsv(ref, 320, 120)), "min_cup_px": 400, "min_ball_px": 80,
            "max_offset_frac": 0.5, "frame_size": [640, 480]}


def occ(states):
    return [s["occupied"] for s in states]


def test_occupied_empty():
    setup = make_setup()
    st = slot_states(scene({1: (0, 0), 2: (0, 0), 3: (0, 0)}), setup)
    assert occ(st) == [True, True, True, False], st
    assert not any(s["off_center"] for s in st), st
    ok, probs = check_layout(scene({1: (0, 0), 2: (0, 0), 3: (0, 0)}), setup, expected_empty=4)
    assert ok and probs == [], probs
    ok, probs = check_layout(scene({1: (0, 0), 3: (0, 0), 4: (0, 0)}), setup, expected_empty=4)
    assert not ok and "slot 2: cup missing" in probs and any(p.startswith("slot 4: should be empty") for p in probs), probs


def test_off_center():
    setup = make_setup()
    st = slot_states(scene({1: (0, 0), 2: (8, 0), 3: (0, 0), 4: (32, 0)}), setup)  # 0.2 r ok, 0.8 r off
    assert occ(st) == [True] * 4
    assert [s["off_center"] for s in st] == [False, False, False, True], st
    assert st[3]["offset"][0] > 20 and abs(st[3]["offset"][1]) <= 1, st[3]
    ok, probs = check_layout(scene({1: (0, 0), 2: (0, 0), 4: (32, 0)}), setup, expected_empty=3)
    assert not ok and len(probs) == 1 and probs[0].startswith("slot 4: cup off its mark (dx=+"), probs


def test_ball():
    setup = make_setup()
    f = scene({1: (0, 0), 3: (0, 0), 4: (0, 0)}, ball=(255, 305))  # cup 2 lifted: ball on slot 2
    assert ball_visible(f, setup)[0]
    assert ball_visible(f, setup, slot=2)[0]
    assert not ball_visible(f, setup, slot=4)[0]
    ok, info = ball_visible(scene({1: (0, 0), 2: (0, 0), 3: (0, 0)}), setup)
    assert not ok and info["px"] == 0, info
    annotate(f, setup, slot_states(f, setup), ball_visible(f, setup, 2))  # must not crash


def test_red_wraparound():
    for col in (RED, RED2):
        setup = make_setup(RED)
        lo, hi = setup["cup_hsv"]
        assert lo[0] > hi[0], setup["cup_hsv"]  # range wraps around hue 180
        st = slot_states(scene({1: (0, 0), 2: (0, 0), 4: (0, 0)}, cup_col=col), setup)
        assert occ(st) == [True, True, False, True], (col, st)


def test_frame_scaling():
    setup = make_setup()  # recorded at 640x480
    for size in ((1280, 960), (320, 240)):
        f = scene({2: (0, 0), 3: (0, 0), 4: (32, 0)}, ball=(120, 300), size=size)
        st = slot_states(f, setup)
        assert occ(st) == [False, True, True, True], (size, st)
        assert [s["off_center"] for s in st] == [False, False, False, True], (size, st)
        assert ball_visible(f, setup, slot=1)[0] and not ball_visible(f, setup, slot=3)[0], size


def test_find_cups():
    setup = make_setup()
    img = np.full((480, 640, 3), 70, np.uint8)
    cv2.circle(img, (150, 200), 45, BLUE, -1)      # biggest
    cv2.circle(img, (400, 330), 35, BLUE, -1)
    cv2.circle(img, (630, 100), 30, BLUE, -1)      # touches the right border: kept
    cv2.circle(img, (300, 420), 25, ORANGE, -1)    # ball colour: not a cup
    cv2.circle(img, (500, 420), 6, BLUE, -1)       # speck below min_cup_px
    cups = find_cups(img, setup)
    assert len(cups) == 3, cups
    assert [c["px"] for c in cups] == sorted((c["px"] for c in cups), reverse=True), cups
    (u, v), x, y = cups[0]["uv"], *cups[0]["bbox"][:2]
    assert abs(u - 150) < 1 and abs(v - 200) < 1 and abs(x - 105) <= 1 and abs(y - 155) <= 1, cups[0]
    assert abs(cups[1]["uv"][0] - 400) < 1 and abs(cups[1]["uv"][1] - 330) < 1, cups[1]
    assert nearest_cup(cups, (410, 320), 40) is cups[1]
    assert nearest_cup(cups, (300, 420), 40) is None and nearest_cup([], (0, 0), 1e9) is None
    big = cv2.resize(img, (1280, 960), interpolation=cv2.INTER_NEAREST)  # min_cup_px scales with area
    assert [round(c["uv"][0] / 2) for c in find_cups(big, setup)] == [round(c["uv"][0]) for c in cups]
    try:
        find_cups(img, {k: v for k, v in setup.items() if k != "cup_hsv"})
        raise AssertionError("missing cup_hsv should raise")
    except ValueError as e:
        assert "cup_hsv" in str(e)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
