"""Arm skills an LLM can combine: look, pick, place, flip, rotate, move, stack, point, wave, nod, shake.

No training: every skill is planned on the spot with inverse kinematics (the MuJoCo SO-101 model) from where
host/workspace.py sees the objects. The planner (Workers AI, see cloudflare/robot-router) only writes a list of
steps such as [{"skill": "flip", "object": 1}, ...]; run_plan() validates and executes them, re-locating the
object before every step, and stops at the first step that fails.

The same code drives the MuJoCo sim (SimArm) and the real follower (RealArm), both in LeRobot units.

    python host/skills.py sim  '[{"skill":"flip","object":1,"times":3}]' --show   # try a plan in the sim
    python host/skills.py teach-sheet          # real arm: measure where the printed sheet lies (once per setup)
    python host/skills.py look                 # real arm: park, look, list objects (annotated image in outputs/)
    python host/skills.py real '[{"skill":"point","object":1}]'
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "sim"), str(ROOT / "host"), str(ROOT)]

import workspace  # noqa: E402
from so101_sim.env import FPS, MODEL_DIR, ROBOT_XML, lerobot_to_rad, rad_to_lerobot  # noqa: E402

# Poses in LeRobot units: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
LOOK = np.array([70.0, -90.0, 60.0, 90.0, 0.0, 2.0], np.float32)  # folded up and turned aside: sheet in full view
REST = np.array([0.0, -99.0, 99.0, 74.0, 0.0, 2.0], np.float32)  # lying on its stops (power-off safe)
REACH = {"x": (0.10, 0.30), "y": (-0.17, 0.17), "r": (0.10, 0.31)}  # where top-down grasps are allowed (m)
GRIP_CLOSED_CMD = 0.0
HOLD_MIN = 4.0  # gripper reading above this after closing = something is between the jaws
MAX_STEPS = 15
LIFT = 0.06  # m above the grasp height
MAX_STEP_UNITS = 6.0  # per 30 Hz tick; go() stretches faster moves (RealArm hard-caps at 8)


def release_cmd(width_m: float) -> float:
    """Gripper command whose jaw gap is the object's width + ~2 cm: wide enough to let go (a cube stuck in a 15
    opening and was carried away again), small enough that the swinging jaw doesn't shove it (a pen at 55 did).
    Jaw gap at the fingertips ~ 0.6 cm + 0.155 cm per unit (measured in the MuJoCo model, commands 0..40)."""
    return float(np.clip((width_m + 0.02 - 0.006) / 0.00155, 12, 40))
ROLL_SIGN = 1.0  # +wrist_roll turns the object counter-clockwise seen from above (checked in the sim)

# Catalogue the planner sees. Keep it in sync with SKILLS below (run_plan checks every field).
CATALOG = [
    {"skill": "look", "args": {}, "does": "park the arm aside and refresh the list of objects"},
    {"skill": "pick", "args": {"object": "id"}, "does": "grab the object and lift it ~10 cm (keeps holding it)"},
    {"skill": "place", "args": {"x": "cm", "y": "cm"}, "does": "put the held object down at x,y (cm) and let go"},
    {"skill": "flip", "args": {"object": "id", "times": "1-5"},
     "does": "pick the object up, turn it end over end (180 deg) in the air, put it back on the same spot"},
    {"skill": "rotate", "args": {"object": "id", "degrees": "-180..180"},
     "does": "pick it up, turn it by that many degrees (positive = counter-clockwise seen from above), put it back"},
    {"skill": "move", "args": {"object": "id", "x": "cm", "y": "cm"}, "does": "carry the object to x,y (cm)"},
    {"skill": "put", "args": {"object": "id", "box": "letter"},
     "does": "put the object into one of the boxes printed on the sheet (A, B, C far row; D, X, Z near row)"},
    {"skill": "stack", "args": {"object": "id", "onto": "id"}, "does": "put the object on top of another object"},
    {"skill": "point", "args": {"object": "id"}, "does": "point at the object with the closed gripper for 2 s"},
    {"skill": "wave", "args": {}, "does": "wave hello"},
    {"skill": "nod", "args": {}, "does": "nod yes"},
    {"skill": "shake", "args": {}, "does": "shake 'no'"},
    {"skill": "rest", "args": {}, "does": "lay the arm down on its rest pose"},
]
WORKSPACE_TEXT = (
    "Coordinates: x = cm forward from the robot base, y = cm to the robot's LEFT (negative = right). "
    f"Reachable: x {REACH['x'][0] * 100:.0f}..{REACH['x'][1] * 100:.0f}, y {REACH['y'][0] * 100:.0f}.."
    f"{REACH['y'][1] * 100:.0f}. The paper sheet is centred at about x=20, y=0."
)


class SkillError(RuntimeError):
    """A step can't be done (unreachable, object gone, grasp missed). The message is shown to the audience."""


# ---------------------------------------------------------------- kinematics (MuJoCo model, no physics)

class Kinematics:
    def __init__(self, model: mujoco.MjModel | None = None):
        self.m = model or mujoco.MjModel.from_xml_path(str(MODEL_DIR / ROBOT_XML))
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("gripperframe").id
        self.lo, self.hi = self.m.jnt_range[:5, 0].copy(), self.m.jnt_range[:5, 1].copy()

    def fk(self, q_lerobot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gripper site position (m) and rotation matrix for a LeRobot joint vector."""
        self.d.qpos[:6] = lerobot_to_rad(q_lerobot)
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def ik(self, target: np.ndarray, close_axis: np.ndarray, q0_rad: np.ndarray, iters: int = 300) -> np.ndarray:
        """Arm joints (rad, 5): site at target, approach straight down, jaws opening along close_axis.
        Same damped least squares as sim/so101_sim/expert.py, plus a convergence check."""
        q = np.array(q0_rad[:5], np.float64)
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        down = np.array([0.0, 0.0, -1.0])
        for _ in range(iters):
            self.d.qpos[:5] = q
            mujoco.mj_kinematics(self.m, self.d)
            mujoco.mj_comPos(self.m, self.d)
            R = self.d.site_xmat[self.site].reshape(3, 3)
            e_pos = target - self.d.site_xpos[self.site]
            e_rot = 0.5 * (np.cross(R[:, 0], down) + np.cross(R[:, 2], close_axis))
            if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_rot) < 1e-2:
                break
            mujoco.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, :5], 0.3 * jacr[:, :5]])
            err = np.concatenate([e_pos, 0.3 * e_rot])
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err), self.lo, self.hi)
        if np.linalg.norm(e_pos) > 0.006 or np.linalg.norm(e_rot) > 0.15:
            raise SkillError(f"can't reach ({target[0] * 100:.0f}, {target[1] * 100:.0f}, {target[2] * 100:.0f}) cm "
                             f"from above (off by {np.linalg.norm(e_pos) * 1000:.0f} mm)")
        return q


def check_reach(x: float, y: float) -> None:
    r = math.hypot(x, y)
    if not (REACH["x"][0] <= x <= REACH["x"][1] and REACH["y"][0] <= y <= REACH["y"][1]
            and REACH["r"][0] <= r <= REACH["r"][1]):
        raise SkillError(f"({x * 100:.0f}, {y * 100:.0f}) cm is outside the area the arm can reach from above")


# ---------------------------------------------------------------- arm I/O

class Arm:
    """30 Hz joint-target interface in LeRobot units. Subclasses: SimArm, RealArm."""

    def __init__(self, kin: Kinematics, on_frame=None, stop=None):
        self.kin = kin
        self.on_frame = on_frame  # callback(rgb | None, state, target) for live views; may be None
        self.stop = stop  # threading.Event-like; set = abort the current skill
        self.cmd = None  # last commanded target

    # subclasses
    def state(self) -> np.ndarray: ...
    def tick(self, target: np.ndarray) -> None: ...
    def image(self) -> np.ndarray: ...  # BGR
    def close(self) -> None: ...

    def go(self, target, seconds: float) -> None:
        """Minimum-jerk move from the last commanded target to `target` (6 values)."""
        target = np.asarray(target, np.float32)
        start = self.cmd if self.cmd is not None else self.state()
        # minimum-jerk peaks at 1.875x the average speed: size n so the fastest tick stays under MAX_STEP_UNITS
        n = max(1, round(seconds * FPS), math.ceil(1.875 * float(np.abs(target - start).max()) / MAX_STEP_UNITS))
        for i in range(1, n + 1):
            if self.stop is not None and self.stop.is_set():
                raise SkillError("stopped")
            s = i / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5
            self.cmd = (start + (target - start) * s).astype(np.float32)
            self.tick(self.cmd)

    def is_holding(self) -> bool:
        """After closing + lifting: jaws stopped by something (real arm: the gripper reading stays open a bit)."""
        return float(self.state()[5]) > HOLD_MIN

    def hold(self, seconds: float) -> None:
        self.go(self.cmd if self.cmd is not None else self.state(), seconds)


class SimArm(Arm):
    def __init__(self, env, on_frame=None, stop=None, frame_every: int = 3):
        super().__init__(Kinematics(env.model), on_frame, stop)
        self.env = env
        self.n = 0
        self.frame_every = frame_every

    def state(self):
        return rad_to_lerobot(self.env.data.qpos[:6])

    def tick(self, target):
        self.n += 1
        render = self.on_frame is not None and self.n % self.frame_every == 0
        obs = self.env.step(target, render=render)
        if render:
            self.on_frame(obs.image, obs.state, target)

    def image(self):
        return cv2.cvtColor(self.env.render(), cv2.COLOR_RGB2BGR)

    def is_holding(self) -> bool:
        # the sim's finger pads are separate boxes, so its gripper joint closes fully even on a pen: use physics
        return self.env.object_height() > 0.02


class DryArm(Arm):
    """No physics, no hardware: every commanded target is recorded and 'reached' at once. For predicting and
    measuring a plan before the arm moves (host/validate_move.py)."""

    def __init__(self, frame_bgr: np.ndarray, start: np.ndarray | None = None):
        super().__init__(Kinematics())
        self.frame = frame_bgr
        self.cmd = np.array(LOOK if start is None else start, np.float32)
        self.trace: list[np.ndarray] = []
        self.events: list[tuple[int, str]] = []  # (tick, label) e.g. grasp / release, filled in by Skills

    def state(self):
        return self.cmd.copy()

    def tick(self, target):
        self.trace.append(np.array(target, np.float32))

    def image(self):
        return self.frame

    def is_holding(self) -> bool:
        return True  # assume the grasp works; the physics check is the sim run

    def close(self):
        pass


class RealArm(Arm):
    """Holds the follower's port AND the camera (Windows: one process per device). max_relative_target caps every
    30 Hz step, so even a wrong target can't make the arm jump."""

    def __init__(self, on_frame=None, stop=None, max_step: float = 8.0, camera: bool = True):
        from robot import CALIB_DIR, CONFIG
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        super().__init__(Kinematics(), on_frame, stop)
        f = CONFIG["follower"]
        self.robot = SO101Follower(SO101FollowerConfig(port=f["port"], id=f["id"], calibration_dir=CALIB_DIR,
                                                       max_relative_target=max_step))
        self.robot.connect(calibrate=False)
        if not self.robot.is_calibrated:
            self.robot.disconnect()
            raise SkillError("motors hold another calibration - run: robot.py restore-calibration follower")
        self.names = [f"{j}.pos" for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                             "wrist_roll", "gripper")]
        self.cam = None
        if camera:
            from robot_agent import CameraReader

            self.cam = CameraReader(CONFIG["cameras"]["front"])
        self.next_t = time.perf_counter()
        self.n = 0

    def state(self):
        obs = self.robot.bus.sync_read("Present_Position")
        return np.array([obs[n[:-4]] for n in self.names], np.float32)

    def tick(self, target):
        self.robot.send_action({n: float(v) for n, v in zip(self.names, target)})
        self.n += 1
        if self.on_frame is not None and self.n % 3 == 0:
            self.on_frame(self.cam.rgb() if self.cam else None, None, target)
        self.next_t += 1 / FPS
        delay = self.next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_t = time.perf_counter()

    def image(self):
        if self.cam is None:
            raise SkillError("no camera")
        time.sleep(0.3)  # let the image settle after the arm stopped moving
        return self.cam.frame.copy()

    def close(self):
        if self.cam:
            self.cam.close()
        self.robot.bus.disconnect(disable_torque=False)  # keep holding the pose; REST first if you want it down


# ---------------------------------------------------------------- skills

class Skills:
    def __init__(self, arm: Arm, log=print):
        self.arm = arm
        self.log = log
        self.scene: workspace.Scene | None = None
        self.objects: dict[int, workspace.Obj] = {}
        self.holding: workspace.Obj | None = None

    # ---- perception
    def look(self) -> workspace.Scene:
        if self.holding is None:
            self.travel(LOOK, 1.2)
        self.arm.hold(0.3)
        s = workspace.look(self.arm.image())
        self.scene = s
        if not s.ok:
            raise SkillError(s.detail)
        self.objects = {o.id: o for o in s.objects}
        self.log(f"look: {s.detail}, {len(s.objects)} object(s)" + "".join(f"\n  {o.describe()}" for o in s.objects))
        return s

    def track(self, obj: workspace.Obj) -> workspace.Obj:
        """Find `obj` again after things moved: nearest object of the same colour within 8 cm."""
        self.look()
        cands = [o for o in self.scene.objects if o.color == obj.color] or self.scene.objects
        best = min(cands, key=lambda o: math.hypot(o.x - obj.x, o.y - obj.y), default=None)
        if best is None or math.hypot(best.x - obj.x, best.y - obj.y) > 0.08:
            raise SkillError(f"lost sight of the {obj.color} {obj.shape} object")
        return best

    def _mark(self, label: str) -> None:
        if hasattr(self.arm, "events"):
            self.arm.events.append((len(self.arm.trace), label))

    # ---- motion helpers
    def travel(self, target, seconds: float = 1.5) -> None:
        """Move without sweeping over the table: fold up at the current heading, turn, then unfold to target.
        (A straight joint-space move from a low pose can drag the gripper through the objects.)"""
        target = np.asarray(target, np.float32)
        cur = self.arm.cmd if self.arm.cmd is not None else self.arm.state()
        z = self.arm.kin.fk(cur)[0][2]
        if z < 0.12 and abs(float(target[0] - cur[0])) > 3:
            up = cur.copy()
            up[1:4] = LOOK[1:4]
            self.arm.go(up, 0.9)
            mid = up.copy()
            mid[0] = target[0]
            mid[4] = target[4]
            self.arm.go(mid, 0.9)
        self.arm.go(target, seconds)

    def _arm_q(self, x, y, z, close_axis, seed=None) -> np.ndarray:
        q0 = lerobot_to_rad(seed if seed is not None else np.array([5.8, -7.3, 40.2, 55.4, 0.0, 7.0]))
        return self.arm.kin.ik(np.array([x, y, z]), close_axis, q0)

    def _target(self, q5, grip) -> np.ndarray:
        t = rad_to_lerobot(np.append(q5, 0.0))
        t[5] = grip
        return t

    def _roll_after(self, q4: float, turn_deg: float) -> float | None:
        """wrist_roll (rad) after turning the object by turn_deg, or None if the joint can't get there."""
        lo, hi = self.arm.kin.lo[4], self.arm.kin.hi[4]
        r = q4 + ROLL_SIGN * math.radians(turn_deg)
        for cand in (r, r - 2 * math.pi, r + 2 * math.pi):  # +-2pi: same final orientation, other way round
            if lo <= cand <= hi:
                return cand
        return None

    def _grasp_plan(self, o: workspace.Obj, turn_deg: float = 0.0):
        """Jaw axis across the object's short side. Of the two jaw orientations, the one with less wrist roll
        that still leaves the wrist room to turn the object by turn_deg."""
        check_reach(o.x, o.y)
        # Fixed jaw (at the site) right beside the object, moving jaw opened wide (55: its tip is above the
        # fingertips and closes down-and-in). In the MuJoCo model this lifts a pen 6/6 with exact positions and
        # tolerates ~5 mm of error; centring the object in a narrower gap failed (the angled jaw pushes it under).
        # Full blob width, not shrunk: tall objects look ~1.5x too wide (side view), flat ones (disc) do not, and
        # a fixed jaw that lands too far out still sweeps the object in, one that lands too close lands ON it.
        w = o.width
        half_w = float(np.clip(w / 2, 0.004, 0.02))
        open_cmd = 55.0 if w < 0.02 else 75.0
        # Site height at the grasp: fingertips ~2 mm above the table. One camera can't tell an object's height (a
        # flat 8 mm disc and a 3 cm cube look alike from above), and the 4 cm pads hold tall objects low too.
        z = 0.008
        best = None
        for flip in (1.0, -1.0):
            a = o.yaw + math.pi / 2
            c = flip * np.array([math.cos(a), math.sin(a), 0.0])
            gx, gy = np.array([o.x, o.y]) - c[:2] * (half_w + 0.003)
            try:
                q = self._arm_q(gx, gy, z, c)
            except SkillError:
                continue
            if self._roll_after(q[4], turn_deg) is None:
                continue
            if best is None or abs(q[4]) < abs(best[1][4]):
                best = (c, q, (gx, gy, z))
        if best is None:
            raise SkillError(f"can't find a grasp for object #{o.id}" + (f" that allows a {turn_deg:.0f} deg turn" if turn_deg else ""))
        return best[0], best[1], best[2], open_cmd

    # ---- skills
    def pick(self, o: workspace.Obj, turn_deg: float = 0.0) -> None:
        if self.holding:
            raise SkillError("already holding something - place it first")
        c, q_g, (gx, gy, z), open_cmd = self._grasp_plan(o, turn_deg)
        # a vertical gripper can't go much higher than ~8 cm (wrist_flex limit); 6 cm clears a pen or a cube
        q_pre = self._arm_q(gx, gy, z + LIFT - 0.01, c, rad_to_lerobot(np.append(q_g, 0)))
        q_up = self._arm_q(gx, gy, z + LIFT, c, rad_to_lerobot(np.append(q_g, 0)))
        self.log(f"pick #{o.id}: grasp at ({gx * 100:.1f}, {gy * 100:.1f}) cm, jaws at {math.degrees(math.atan2(c[1], c[0])):.0f} deg")
        self.travel(self._target(q_pre, open_cmd), 1.2)
        self.arm.go(self._target(q_g, open_cmd), 0.8)
        self.arm.go(self._target(q_g, GRIP_CLOSED_CMD), 0.5)
        self._mark("grasp")
        self.arm.hold(0.3)
        self.arm.go(self._target(q_up, GRIP_CLOSED_CMD), 0.9)
        self.arm.hold(0.2)
        if not self.arm.is_holding():
            self.arm.go(self._target(q_up, open_cmd), 0.4)
            raise SkillError(f"missed object #{o.id} (gripper closed on nothing)")
        self.holding = o
        self.grasp = (c, q_g, z, open_cmd, float(np.hypot(o.x - gx, o.y - gy)))  # off = object centre - site

    def place(self, x: float, y: float, z_extra: float = 0.0, roll_deg: float = 0.0) -> None:
        if not self.holding:
            raise SkillError("not holding anything")
        check_reach(x, y)
        c, q_g, z, open_cmd, off = self.grasp
        # The object turns with the wrist (about the vertical) and sits `off` beside the site along the jaw axis:
        # put the OBJECT centre on (x, y), not the fingertip (that left a cube 2 cm off the box centre).
        a = math.radians(roll_deg)
        c_now = np.array([c[0] * math.cos(a) - c[1] * math.sin(a), c[0] * math.sin(a) + c[1] * math.cos(a), 0.0])
        sx, sy = x - c_now[0] * off, y - c_now[1] * off
        roll = self._roll_after(q_g[4], roll_deg)
        if roll is None:
            raise SkillError(f"wrist can't turn {roll_deg:.0f} deg from here")
        q0 = q_g.copy()
        q0[4] = roll  # seed IK with the wrist already turned: a 180 deg jaw-axis error would otherwise stall IK
        seed = rad_to_lerobot(np.append(q0, 0))
        q_up = self._arm_q(sx, sy, z + z_extra + LIFT, c_now, seed)
        # set it down, don't drop it. (Pressing 3 mm into the table flung pens away in the sim; a perfectly round
        # sim pen can still roll off after release - flat objects like the eraser don't.)
        q_down = self._arm_q(sx, sy, z + z_extra + 0.001, c_now, seed)
        if roll_deg:
            self.arm.go(self._target(q_up, GRIP_CLOSED_CMD), 2.0)  # visible turn in the air
        else:
            self.arm.go(self._target(q_up, GRIP_CLOSED_CMD), 1.4)
        self.arm.go(self._target(q_down, GRIP_CLOSED_CMD), 0.8)
        # release with a small opening: opened wide, the moving jaw's tip swings down-and-out on its arc and
        # shoves the object several cm (seen in the sim)
        self._mark("release")
        rel = release_cmd(self.holding.width)
        self.arm.go(self._target(q_down, rel), 0.8)
        self.arm.hold(0.3)
        self.arm.go(self._target(q_up, rel), 0.8)
        self.holding = None

    def turn_in_place(self, o: workspace.Obj, degrees: float) -> workspace.Obj:
        """Pick, turn in the air, put down on the same spot. If the wrist can't turn that far in one go
        (its roll range is ~320 deg), turn in two halves with a re-grasp in between."""
        try:
            self._grasp_plan(o, degrees)
        except SkillError:
            if abs(degrees) < 20:
                raise
            self.log(f"wrist range: turning {degrees:.0f} deg in two halves")
            o = self.turn_in_place(o, degrees / 2)
            return self.turn_in_place(self.track(o), degrees / 2)
        self.pick(o, degrees)
        self.place(o.x, o.y, roll_deg=degrees)
        return o

    def point(self, o: workspace.Obj) -> None:
        check_reach(o.x, o.y)
        q = self._arm_q(o.x, o.y, 0.06, np.array([0.0, 1.0, 0.0]))
        self.travel(self._target(q, GRIP_CLOSED_CMD), 1.2)
        self.arm.hold(2.0)

    def gesture(self, kind: str) -> None:
        base = np.array([0.0, -40.0, 20.0, 0.0, 0.0, 30.0], np.float32)
        self.travel(base, 1.5)
        joint, amp, n = {"wave": (4, 25, 3), "nod": (3, 20, 2), "shake": (0, 15, 3)}[kind]
        for _ in range(n):
            for sgn in (1, -1):
                t = base.copy()
                t[joint] += sgn * amp
                self.arm.go(t, 0.45)
        self.arm.go(base, 0.4)

    def rest(self) -> None:
        if self.holding:
            raise SkillError("holding an object - place it first")
        self.travel(LOOK, 1.5)
        self.arm.go(REST, 1.5)


# ---------------------------------------------------------------- plans from the LLM

def _num(step: dict, key: str, lo: float, hi: float) -> float:
    try:
        v = float(step[key])
    except (KeyError, TypeError, ValueError):
        raise SkillError(f"'{step.get('skill')}' needs a number '{key}'") from None
    if not lo <= v <= hi:
        raise SkillError(f"'{step.get('skill')}': {key}={v:g} is outside {lo:g}..{hi:g}")
    return v


def validate(plan: list[dict], object_ids: set[int]) -> list[dict]:
    """Check every step BEFORE the arm moves; expand `times`. Raises SkillError with a readable reason."""
    if not isinstance(plan, list) or not plan:
        raise SkillError("empty plan")
    names = {c["skill"]: c for c in CATALOG}
    out = []
    for step in plan:
        if not isinstance(step, dict) or step.get("skill") not in names:
            raise SkillError(f"unknown skill {step.get('skill') if isinstance(step, dict) else step!r}")
        for key, kind in names[step["skill"]]["args"].items():
            if key == "times":
                continue
            if kind == "id":
                try:
                    oid = int(step[key])
                except (KeyError, TypeError, ValueError):
                    raise SkillError(f"'{step['skill']}' needs an object number") from None
                if oid not in object_ids:
                    raise SkillError(f"there is no object #{oid}")
                step[key] = oid
            elif key in ("x", "y"):
                step[key] = _num(step, key, -40, 40)
            elif key == "degrees":
                step[key] = _num(step, key, -180, 180)
            elif kind == "letter":
                letter = str(step.get(key, "")).strip().upper().replace("BOX ", "")
                if letter not in workspace.BOXES:
                    raise SkillError(f"there is no box '{step.get(key)}' (boxes: {', '.join(workspace.BOXES)})")
                step[key] = letter
        times = int(_num(step, "times", 1, 5)) if "times" in step else 1
        out += [{k: v for k, v in step.items() if k != "times"} for _ in range(times)]
    if len(out) > MAX_STEPS:
        raise SkillError(f"plan too long ({len(out)} steps, max {MAX_STEPS})")
    return out


def run_plan(sk: Skills, plan: list[dict], log=print) -> tuple[bool, str]:
    """Executes a validated plan. Objects are referred to by the ids of the LAST look the planner saw;
    before each step the object is re-located (it may have moved, e.g. after a flip)."""
    if sk.scene is None:
        sk.look()
    try:
        steps = validate(plan, set(sk.objects))
    except SkillError as e:
        return False, f"plan rejected: {e}"
    known = dict(sk.objects)  # planner's ids -> last known Obj
    done = 0
    try:
        for i, st in enumerate(steps, 1):
            name = st["skill"]
            log(f"step {i}/{len(steps)}: {json.dumps(st)}")
            obj = None
            if "object" in st:
                obj = known[st["object"]] = sk.track(known[st["object"]])
            if name == "look":
                sk.look()
            elif name == "pick":
                sk.pick(obj)
            elif name == "place":
                sk.place(st["x"] / 100, st["y"] / 100)
            elif name == "flip":
                sk.turn_in_place(obj, 180)
            elif name == "rotate":
                sk.turn_in_place(obj, st["degrees"])
            elif name == "move":
                sk.pick(obj)
                sk.place(st["x"] / 100, st["y"] / 100)
            elif name == "put":
                sk.pick(obj)
                sk.place(*workspace.box_centre(st["box"]))
            elif name == "stack":
                onto = known[st["onto"]] = sk.track(known[st["onto"]]) if not sk.holding else known[st["onto"]]
                sk.pick(obj)
                sk.place(onto.x, onto.y, z_extra=max(onto.width, 0.012) + 0.005)
            elif name == "point":
                sk.point(obj)
            elif name in ("wave", "nod", "shake"):
                sk.gesture(name)
            elif name == "rest":
                sk.rest()
            done += 1
    except SkillError as e:
        log(f"step {done + 1} failed: {e}")
        if sk.holding and str(e) != "stopped":
            try:  # never leave the object dangling: put it back where it came from
                sk.place(sk.holding.x, sk.holding.y)
            except SkillError:
                pass
        return False, f"{done}/{len(steps)} steps done - {e}"
    if not sk.holding:
        sk.travel(LOOK, 1.2)
    return True, f"all {len(steps)} steps done"


# ---------------------------------------------------------------- CLI

def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["sim", "real", "look", "teach-sheet"])
    p.add_argument("plan", nargs="?", default="[]", help="JSON list of steps")
    p.add_argument("--show", action="store_true", help="(sim) show the camera in a window")
    p.add_argument("--seed", type=int, default=1, help="(sim) object placement")
    p.add_argument("--obj", default="pen", help="(sim) pen | cube")
    p.add_argument("--video", help="(sim) save the run as an mp4")
    a = p.parse_args()

    if a.mode == "teach-sheet":
        return teach_sheet()

    frames = []

    def on_frame(rgb, state, target):
        if rgb is None:
            return
        if a.show:
            cv2.imshow("skills", rgb[:, :, ::-1])
            cv2.waitKey(1)
        if a.video:
            frames.append(rgb.copy())

    if a.mode == "sim":
        from so101_sim.env import SO101PickEnv

        env = SO101PickEnv(a.obj, sheet=True)
        env.reset(seed=a.seed, home=LOOK)
        arm = SimArm(env, on_frame=on_frame)
    else:
        arm = RealArm(on_frame=on_frame)
    sk = Skills(arm)
    try:
        sk.look()
        out = ROOT / "outputs" / "skills_look.jpg"
        cv2.imwrite(str(out), sk.scene.annotated)
        print(f"annotated view -> {out}")
        plan = json.loads(a.plan)
        if plan:
            print(run_plan(sk, plan))
            if a.mode == "sim":
                pos, yaw = env.object_pose()
                print(f"sim truth: object at ({pos[0] * 100:.1f}, {pos[1] * 100:.1f}) cm, yaw {math.degrees(yaw):.0f} deg")
    finally:
        arm.close()
        if a.video and frames:
            import imageio.v3 as iio

            iio.imwrite(a.video, np.stack(frames), fps=FPS // 3)
            print(f"video -> {a.video}")


def teach_sheet() -> None:
    """Torque off; put the fixed jaw's tip on two marker centres; the sheet pose goes to calibration/workspace.json."""
    from robot import CALIB_DIR, CONFIG
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    f = CONFIG["follower"]
    robot = SO101Follower(SO101FollowerConfig(port=f["port"], id=f["id"], calibration_dir=CALIB_DIR))
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        sys.exit("motors hold another calibration - run: robot.py restore-calibration follower")
    kin = Kinematics()
    input("The follower goes LIMP now - hold it. ENTER...")
    robot.bus.disable_torque()
    touched = {}
    try:
        for mid in (0, 1, 2, 3):
            ans = input(f"Gripper closed, pointing down: put the FIXED jaw's tip on the centre of marker {mid}, "
                        f"ENTER (s = skip, at least 2): ").strip().lower()
            if ans == "s":
                continue
            obs = robot.bus.sync_read("Present_Position")
            q = np.array([obs[j] for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                             "wrist_roll", "gripper")], np.float32)
            pos, _ = kin.fk(q)
            touched[mid] = (float(pos[0]), float(pos[1]))
            print(f"  marker {mid}: x={pos[0] * 100:.1f} y={pos[1] * 100:.1f} z={pos[2] * 100:.1f} cm")
    finally:
        input("Put the arm on its rest pose, ENTER (it stays limp)...")
        robot.bus.disconnect(disable_torque=True)
    pose = workspace.pose_from_touches(touched)
    workspace.SHEET_JSON.write_text(json.dumps(pose, indent=2), encoding="utf-8")
    print(f"sheet pose {pose} -> {workspace.SHEET_JSON}")
    if pose["touch_error_mm"] > 8:
        print("WARNING: the touched points don't fit the printed layout well (> 8 mm). Re-check the print scale "
              "(markers 40 mm) and touch again.")


if __name__ == "__main__":
    _cli()
