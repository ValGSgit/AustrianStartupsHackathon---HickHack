"""Help for the learned model with what a weak gripper and a fixed bowl make easy to do in code.
Used by do_task.py. Every part can be switched off in settings.env (AUTO_DROP=0, RETRIES=0, GRIP_STRENGTH=normal).

- Firmer grip (GRIP_STRENGTH=strong): the gripper motor has an overload protection. LeRobot makes it start at 25%
  power: squeezing anything for 2 s then drops the gripper to the motor's 20% holding power, and objects slip.
  'strong' starts it at 45% and holds with 35% afterwards. The maximum (50%, set by LeRobot) stays the same, and the
  run stops if a motor gets hot.
- Grab watch: compares where the model tells the gripper to go with where the gripper really is.
    holding   the jaws stopped part way while told to close further: something is between them
    missed    the jaws closed all the way: nothing is between them
    dropped   it was holding and the jaws suddenly closed all the way: the object slipped out
    released  it was holding and the gripper opened again (the model let go by itself)
- Carry (AUTO_DROP=1): once the model holds the object, the code squeezes, lifts, moves to the taught pose
  'above_<container>' (python teach_drop.py <container>) and opens. The container never moves, so this part
  works every time, and slow smooth moves keep a weak grip from shaking the object loose.
- Retry (RETRIES=n): after a missed grab, open, lift a little, and let the model try again.
- Run log: outputs/runs/<time>/ with every command and position (log.csv), what happened (events.txt) and
  what the cameras saw at the start of each try (camN_tryM.jpg).
"""

import csv
import json
import time
from collections import deque
from pathlib import Path

import config
import poses

HERE = Path(__file__).parent
S = config.S
GRIP_FILE = HERE / "grip.json"
KEYS = [f"{j}.pos" for j in config.JOINTS]

AUTO_DROP = S.get("AUTO_DROP", "1").strip() == "1"
RETRIES = int(S.get("RETRIES") or 2)
GRIP_STRENGTH = (S.get("GRIP_STRENGTH") or "strong").strip().lower()
ROLL_OFFSET = float(__import__("os").environ.get("ROLL_OFFSET") or S.get("ROLL_OFFSET") or 0)

# Gripper readings go from 0 (closed) to 100 (open). Closed on nothing reads about 0-2; anything between the
# jaws reads more. grip_test.py measures it for this gripper and saves it in grip.json.
EMPTY_MARGIN = 1.0  # readings below empty_closed + this = nothing between the jaws
STEADY_SECONDS = 0.3  # the jaws must stand still this long before we decide anything
STEADY_RANGE = 1.0  # "standing still" = the reading changes less than this
SQUEEZE_GAP = 3.5  # "told to close further": the command is at least this much more closed than the reading
FREE_GAP = 2.0  # the reading is this close to the command: the jaws moved freely
CLOSED_SHARE = 0.35  # a command below this share of how wide it was open counts as "closing"
OPEN_ABOVE = 10.0  # "open" = at least this much above the empty reading
LIFT_DEG = 20.0  # carry: tilt the upper arm this far towards its resting angle = about 8-10 cm up
RETRY_LIFT_DEG = 12.0
BLOCKED_DIFF, BLOCKED_SECONDS = 25.0, 1.5  # a joint this far from its command for this long: something blocks it
MIN_HOLD_SECONDS = 2.0  # holding shorter than this, then opening = a re-grab, not "carried and let go"
IDLE_SECONDS = 6.0  # the model has not moved the arm out of its resting pose for this long: wake it up
UNFOLD = {"shoulder_lift.pos": 35.0, "elbow_flex.pos": -30.0}  # degrees from rest into a "ready" pose
REACHED_DEG = 30.0  # the upper arm is at least this far from its resting angle: the arm is reaching out
HOT_C = 60  # stop if any motor gets this hot (the motors switch themselves off at about 70)
STRONG = {"Overload_Torque": 45, "Protective_Torque": 35}  # 'normal' = LeRobot's 25 + the motor's own value


def load_grip() -> dict:
    return json.loads(GRIP_FILE.read_text()) if GRIP_FILE.exists() else {}


def save_grip(data: dict) -> None:
    GRIP_FILE.write_text(json.dumps(data, indent=2))


def empty_below() -> float:
    return float(load_grip().get("empty_closed", 2.0)) + EMPTY_MARGIN


def set_grip_strength(robot, mode: str = GRIP_STRENGTH) -> str:
    """Write the gripper's overload settings. Needs the gripper motor off for a split second (it is empty then)."""
    bus = robot.bus
    grip = load_grip()
    now = {k: int(bus.read(k, "gripper", normalize=False, num_retry=3)) for k in STRONG}
    if mode == "strong":
        if "motor_defaults" not in grip and now["Protective_Torque"] != STRONG["Protective_Torque"]:
            grip["motor_defaults"] = {"Protective_Torque": now["Protective_Torque"]}
            save_grip(grip)
        want = dict(STRONG)
    else:
        want = {"Overload_Torque": 25,
                "Protective_Torque": grip.get("motor_defaults", {}).get("Protective_Torque", now["Protective_Torque"])}
    if want != now:
        present = bus.read("Present_Position", "gripper", num_retry=3)
        bus.disable_torque("gripper", num_retry=3)
        for k, v in want.items():
            bus.write(k, "gripper", v, normalize=False, num_retry=3)
        bus.write("Goal_Position", "gripper", present, num_retry=3)  # hold where it is when switched back on
        bus.enable_torque("gripper", num_retry=3)
    return f"{mode} (squeezes at up to 50%, holds with {want['Protective_Torque']}% after 2 s of squeezing)"


class GripWatch:
    """Feed it (time, commanded gripper, measured gripper) on every move; returns an event name or None.
    It compares the two instead of using fixed numbers, because the model may not command 'fully closed'."""

    def __init__(self, empty: float):
        self.empty = empty  # below this reading the jaws touch each other
        self.hist = deque(maxlen=200)
        self.opened = False  # the gripper was open since the last event
        self.held_at = None  # gripper reading when it closed on something (None = not holding)
        self.held_t = 0.0
        self.done = False  # the model let go by itself: stop judging
        self.open_level = empty + 25  # how far the model opens the gripper (used to open it again)

    def update(self, t: float, cmd: float, pos: float) -> str | None:
        self.hist.append((t, cmd, pos))
        if self.done:
            return None
        recent = [p for (tt, _, p) in self.hist if t - tt <= STEADY_SECONDS]
        steady = len(recent) >= 4 and t - self.hist[0][0] >= STEADY_SECONDS and max(recent) - min(recent) < STEADY_RANGE
        free = pos - cmd < FREE_GAP  # the jaws got where they were told: nothing in between
        if self.held_at is not None:
            if pos > self.held_at + 5:  # the jaws opened wider than the object
                if t - self.held_t < MIN_HOLD_SECONDS:  # a quick close-and-open: the model is re-grabbing, keep going
                    self.held_at, self.opened = None, True
                    return None
                self.held_at, self.done = None, True  # held it for a while, then let go: carried and dropped
                return "released"
            if steady and free and pos < self.held_at - 3:
                self.held_at = None
                return "dropped"
            return None
        if pos >= self.empty + OPEN_ABOVE:
            self.opened = True
            self.open_level = max(self.open_level, min(60.0, pos))
        if not (steady and self.opened):
            return None
        if pos >= self.empty and pos - cmd >= SQUEEZE_GAP:
            self.opened, self.held_at, self.held_t = False, pos, t
            return "holding"
        if free and cmd <= max(self.empty, CLOSED_SHARE * self.open_level):
            self.opened = False
            return "missed"
        return None


MESSAGES = {
    "holding": "the gripper is holding something",
    "missed": "the gripper closed on nothing (missed the object)",
    "dropped": "the object slipped out of the gripper",
    "released": "the model opened the gripper again (let go)",
}


class Supervisor:
    """Wraps the robot once for the whole run: safety stop, grab watch, motor temperature and the run log.
    The model's control loop and the scripted moves both go through it."""

    def __init__(self, robot):
        self.robot = robot
        self.client = None
        self.watch: GripWatch | None = None  # set while the model drives
        self.stop_on: set[str] = set()
        self.outcome = None
        self.phase = "start"
        self.present: dict[str, float] = {}
        self.last_cmd: dict[str, float] = {}
        self.blocked_since = None
        self.next_temp_check = 0.0
        self.rows, self.events = [], []
        self.t0 = time.time()
        self.dir = HERE / "outputs/runs" / time.strftime("%Y%m%d_%H%M%S")
        self.save_frames = None
        self.open_level = 30.0  # how wide the model opened the gripper last time (to open it as wide again)
        home = poses.load_poses().get("home")
        self.rest_lift = home["shoulder_lift.pos"] if home else None
        self._send, self._observe = robot.send_action, robot.get_observation
        robot.send_action, robot.get_observation = self.send_action, self.get_observation
        bus = getattr(robot, "bus", None)
        if bus is not None:
            # LeRobot reads every joint before each move (for its 15-degree safety limit): keep that reading,
            # so the watch knows where the gripper really is without extra reads.
            read = bus.sync_read

            def sync_read(data_name, *args, **kwargs):
                out = read(data_name, *args, **kwargs)
                if data_name == "Present_Position" and out:
                    self.present = {f"{m}.pos": float(v) for m, v in out.items()}
                return out

            bus.sync_read = sync_read

    # ------------------------------------------------------------ the model's turn
    def begin(self, client, attempt: int, stop_on: set[str]) -> None:
        self.client, self.outcome, self.stop_on = client, None, stop_on
        self.watch = GripWatch(empty_below())
        self.phase, self.blocked_since, self.save_frames = f"model{attempt}", None, f"try{attempt}"
        self.turn_start, self.moved_out = time.time(), False

    def finish(self) -> None:
        if self.watch is not None:
            self.open_level = self.watch.open_level
        self.watch, self.client = None, None

    def end(self, reason: str) -> None:
        if self.outcome is None:
            self.outcome = reason
        if self.client is not None:
            self.client.shutdown_event.set()

    def event(self, name: str, msg: str) -> None:
        t = time.time() - self.t0
        self.events.append((round(t, 2), self.phase, name, msg))
        print(f"\n  [{t:5.1f} s] {msg}", flush=True)

    def measured(self) -> dict[str, float]:
        fake = getattr(self.robot, "measured", None)
        if fake:
            return fake()
        return self.present or dict(getattr(self.robot, "state", {}))

    def send_action(self, action):
        if self.watch is not None and ROLL_OFFSET and "wrist_roll.pos" in action:
            # Turn the gripper by a fixed angle on top of the model's command (e.g. 90: jaws across a pen, not along it).
            action = {**action, "wrist_roll.pos": action["wrist_roll.pos"] + ROLL_OFFSET}
        try:
            sent = self._send(action)
        except Exception as e:  # noqa: BLE001
            self.event("motor-error", f"STOP: a motor reported a problem ({e}). Something may be blocking the arm.")
            self.end("motor-error")
            raise
        self.last_cmd.update(sent)
        have = self.measured()
        t = time.time() - self.t0
        self.rows.append([round(t, 3), self.phase] + [round(float(sent.get(k, float("nan"))), 2) for k in KEYS]
                         + [round(float(have.get(k, float("nan"))), 2) for k in KEYS])
        if self.watch is not None and "gripper.pos" in sent and "gripper.pos" in have:
            # Only judge grabs while the arm is reached out: near its resting pose the model often just closes
            # the gripper while folding back, which is not a missed grab.
            lift = sent.get("shoulder_lift.pos", have.get("shoulder_lift.pos", 0.0))
            reached = self.rest_lift is None or abs(lift - self.rest_lift) > REACHED_DEG
            if self.rest_lift is not None and abs(lift - self.rest_lift) > 15:
                self.moved_out = True
            if ("idle" in self.stop_on and not self.moved_out and self.rest_lift is not None
                    and time.time() - self.turn_start > IDLE_SECONDS):
                self.event("idle", f"the model has not left the resting pose for {IDLE_SECONDS:.0f} s: unfolding the arm")
                self.end("idle")
            ev = self.watch.update(t, sent["gripper.pos"], have["gripper.pos"]) if reached else None
            if not reached:
                self.watch.opened = False  # an opening near the rest pose does not count either
            if ev:
                self.event(ev, MESSAGES[ev])
                if ev in self.stop_on:
                    self.end(ev)
        return sent

    def get_observation(self):
        obs = self._observe()
        if self.watch is None:
            return obs
        stuck = [k for k, v in self.last_cmd.items() if k not in ("gripper.pos", "wrist_roll.pos")
                 and k in obs and abs(v - obs[k]) > BLOCKED_DIFF]
        if not stuck:
            self.blocked_since = None
        elif self.blocked_since is None:
            self.blocked_since = time.time()
        elif time.time() - self.blocked_since > BLOCKED_SECONDS:
            self.event("blocked", f"STOP: the robot cannot reach its target ({', '.join(k.split('.')[0] for k in stuck)}).")
            self.end("blocked")
        bus = getattr(self.robot, "bus", None)
        if bus is not None and time.time() > self.next_temp_check:
            self.next_temp_check = time.time() + 2.0
            try:
                temps = bus.sync_read("Present_Temperature")
                hot = {m: v for m, v in temps.items() if v >= HOT_C}
                if hot:
                    self.event("hot", f"STOP: motor too hot ({hot} degrees C). Let it cool for 10 minutes.")
                    self.end("hot")
            except Exception:  # noqa: BLE001  (a missed temperature reading is not a reason to stop)
                pass
        if self.save_frames:
            self.dir.mkdir(parents=True, exist_ok=True)
            for k, v in obs.items():
                if hasattr(v, "ndim") and v.ndim == 1 and str(v.dtype) == "uint8":  # JPEG from do_task.shrink_images
                    (self.dir / f"{k}_{self.save_frames}.jpg").write_bytes(v.tobytes())
            self.save_frames = None
        return obs

    def save(self, summary: str) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / "log.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "phase"] + [f"cmd_{k[:-4]}" for k in KEYS] + [f"pos_{k[:-4]}" for k in KEYS])
            w.writerows(self.rows)
        lines = [f"{t:6.2f} s  {phase:8s} {name:10s} {msg}" for t, phase, name, msg in self.events]
        (self.dir / "events.txt").write_text("\n".join(lines + [f"RESULT: {summary}"]) + "\n")
        return self.dir


# ---------------------------------------------------------------- scripted moves

def move(robot, target: dict[str, float], seconds: float, start: dict[str, float] | None = None) -> None:
    """Smooth move like poses.go_to, but from a given start: go_to starts from the measured gripper, which
    for a gripper squeezing an object would loosen the grip for a moment."""
    start = start or poses.joints_only(robot.get_observation())
    steps = max(1, int(seconds * poses.FPS))
    for i in range(1, steps + 1):
        t = i / steps
        t = t * t * (3 - 2 * t)
        robot.send_action({k: start[k] + (target[k] - start[k]) * t for k in target})
        time.sleep(1 / poses.FPS)


def lifted(pose: dict[str, float], home: dict[str, float], degrees: float) -> dict[str, float]:
    """Tilt the upper arm back towards its resting angle: the gripper goes up (and a little back)."""
    out = dict(pose)
    d = home["shoulder_lift.pos"] - pose["shoulder_lift.pos"]
    out["shoulder_lift.pos"] += max(-degrees, min(degrees, d))
    return out


def _check(robot, sup: Supervisor, target: dict[str, float]) -> str | None:
    """After a scripted move: is something blocking the arm, and is the object still in the gripper?"""
    time.sleep(0.15)
    have = poses.joints_only(robot.get_observation())
    stuck = [k for k in target if k not in ("gripper.pos", "wrist_roll.pos") and abs(target[k] - have[k]) > BLOCKED_DIFF]
    if stuck:
        sup.event("blocked", f"STOP: the arm cannot reach the carry pose ({', '.join(k.split('.')[0] for k in stuck)}).")
        return "blocked"
    # No "slipped out" check here any more: a thin object (chess piece neck) reads almost like an empty gripper,
    # and stopping the carry opened the gripper and dropped it. Once carrying, always go to the cup and open there.
    return None


def carry(robot, sup: Supervisor, drop: dict[str, float], home: dict[str, float], open_to: float) -> str:
    """The model holds the object: squeeze, lift, move above the container, open. Returns 'done' or a problem."""
    sup.phase = "carry"
    sup.event("carry", "taking over: lifting the object and carrying it to the drop pose")
    here = poses.joints_only(robot.get_observation())
    closed = {"gripper.pos": 0.0}
    squeeze = {**here, **closed}
    move(robot, squeeze, 0.3, start=squeeze)  # squeeze fully before lifting
    time.sleep(0.4)  # this gripper closes slowly on 5 V
    if poses.joints_only(robot.get_observation())["gripper.pos"] < empty_below():
        sup.event("missed", "checked before carrying: the gripper closed all the way, nothing in it (missed)")
        return "missed"
    up = {**lifted(here, home, LIFT_DEG), **closed}
    move(robot, up, 1.0, start=squeeze)
    problem = _check(robot, sup, up)
    if problem:
        return problem
    above = {**drop, **closed}
    move(robot, above, 2.0, start=up)
    problem = _check(robot, sup, above)
    if problem:
        return problem
    move(robot, {**above, "gripper.pos": open_to}, 0.4, start=above)
    time.sleep(0.6)
    sup.event("dropped-in", "opened the gripper above the container")
    return "done"


def reopen(robot, sup: Supervisor, home: dict[str, float], open_to: float) -> None:
    """After a missed grab: open and lift a little, so the model gets a clean second try."""
    sup.phase = "reopen"
    here = poses.joints_only(robot.get_observation())
    move(robot, {**lifted(here, home, RETRY_LIFT_DEG), "gripper.pos": open_to}, 0.8, start=here)


def unfold(robot, sup: Supervisor, home: dict[str, float]) -> None:
    """Out of the folded resting pose into a 'ready' pose above the table, so the model starts moving."""
    sup.phase = "unfold"
    here = poses.joints_only(robot.get_observation())
    target = dict(here)
    for k, d in UNFOLD.items():
        target[k] = home[k] + d
    move(robot, target, 1.2, start=here)


def drop_pose_for(task: str, name: str | None) -> tuple[str | None, dict | None]:
    """The taught drop pose for this task: 'above_<name>' if given, else the first taught 'above_<x>' whose
    <x> appears in the task text (e.g. 'cup' in '... drop it into the cup')."""
    taught = poses.load_poses()
    if name:
        key = f"above_{poses.clean_name(name)}"
        return key, taught.get(key)
    words = set(poses.clean_name(w) for w in task.replace(",", " ").split())
    for key, pose in taught.items():
        if key.startswith("above_") and not key[6:].isdigit() and key[6:] in words:
            return key, pose
    return None, None
