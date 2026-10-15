"""SO-101 pick task in MuJoCo, pure Python (no ROS). The ROS nodes and scripts wrap this.

Units at the interface are the REAL robot's LeRobot units (SO101Follower default, use_degrees=False),
so sim datasets/policies and real recordings mean the same thing:
  arm joints -100..100 = the calibrated joint range (0 = middle, the "new calibration" zero),
  wrist_roll -100..100 = -180..180 deg (LeRobot calibrates it as a full turn),
  gripper 0..100 (0 = closed).
Mapping derived from LeRobot's motor normalization + kinematics code and checked against real Hugging Face
datasets: youliangtan/so101-table-cleanup (robot_type so101_follower) and lerobot/svla_so101_pickplace
(recorded as so100_follower): the gripper reaches table height exactly when it closes on an object.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
# SO101_SIM_LOWPOLY=1 uses decimated meshes (sim/fetch_model.py --lowpoly): ~4x faster software rendering
# for the Docker container's live view. Images differ slightly from the full-detail meshes, so keep
# dataset generation and policy evaluation on the same setting.
ROBOT_XML = "so101_new_calib_lowpoly.xml" if os.environ.get("SO101_SIM_LOWPOLY") == "1" else "so101_new_calib.xml"
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
MOTOR_NAMES = [f"{j}.pos" for j in JOINTS]  # LeRobot feature names
FPS = 30
CAM_W, CAM_H = 640, 480

# Gripper joint range in the MJCF (rad): closed .. open
GRIP_CLOSED, GRIP_OPEN = -0.17453, 1.74533

SCENE_XML = """
<mujoco model="so101_pick">
  <include file="{robot_xml}"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual>
    <headlight diffuse="0.5 0.5 0.5" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global offwidth="{w}" offheight="{h}"/>
  </visual>
  <asset>
    <texture type="2d" name="wood" builtin="flat" rgb1="0.62 0.5 0.38" width="64" height="64"/>
    <material name="table" texture="wood" rgba="1 1 1 1"/>
    <material name="mat" rgba="0.15 0.15 0.17 1"/>
  </asset>
  <worldbody>
    <light pos="0.3 0 1.2" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6"/>
    <geom name="table" type="box" pos="0.15 0 -0.03" size="0.45 0.4 0.03" material="table" friction="1 0.005 0.0001"/>
    <geom name="mat" type="box" pos="0.18 0 0.0005" size="0.12 0.16 0.0005" material="mat" contype="0" conaffinity="0"/>
    <camera name="front" pos="0.62 0 0.42" xyaxes="0 1 0 -0.62 0 0.78" fovy="55"/>
    <body name="object" pos="0.25 0 0.02">
      <freejoint name="object"/>
      {object_geom}
    </body>
  </worldbody>
</mujoco>
"""

OBJECTS = {
    # 2.5 cm cube: easy, use it first to validate the pipeline.
    "cube": '<geom name="obj" type="box" size="0.0125 0.0125 0.0125" rgba="0.85 0.15 0.15 1" '
    'mass="0.02" friction="1.5 0.01 0.001" condim="4" solref="0.006 1"/>',
    # Marker-sized pen (1.2 cm thick, 14 cm long) lying on the table along its local x axis.
    "pen": '<geom name="obj" type="capsule" size="0.006 0.065" euler="0 1.5708 0" rgba="0.1 0.3 0.85 1" '
    'mass="0.015" friction="1.5 0.01 0.001" condim="4" solref="0.006 1"/>',
}
OBJ_REST_Z = {"cube": 0.0125, "pen": 0.006}


# Half of each arm joint's travel (rad), from the official MJCF = the range swept during calibration.
ARM_HALF_RANGE = np.array([1.91986, 1.74533, 1.69, 1.65806])  # shoulder_pan, shoulder_lift, elbow_flex, wrist_flex


def lerobot_to_rad(action: np.ndarray) -> np.ndarray:
    """Real-robot LeRobot units (see module doc) -> MuJoCo joint angles / targets (rad)."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    q = np.empty(6)
    q[:4] = np.clip(a[:4], -100, 100) / 100.0 * ARM_HALF_RANGE
    q[4] = np.clip(a[4], -100, 100) / 100.0 * math.pi
    q[5] = GRIP_CLOSED + (np.clip(a[5], 0, 100) / 100.0) * (GRIP_OPEN - GRIP_CLOSED)
    return q


def rad_to_lerobot(q: np.ndarray) -> np.ndarray:
    """MuJoCo joint angles (rad) -> real-robot LeRobot units."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty(6)
    out[:4] = q[:4] / ARM_HALF_RANGE * 100.0
    out[4] = q[4] / math.pi * 100.0
    out[5] = (q[5] - GRIP_CLOSED) / (GRIP_OPEN - GRIP_CLOSED) * 100.0
    return out.astype(np.float32)


@dataclass
class Obs:
    state: np.ndarray  # (6,) float32, LeRobot units
    image: np.ndarray  # (480, 640, 3) uint8, camera "front"


class SO101PickEnv:
    """Physics runs at 500 Hz; one `step()` is one 30 Hz control tick, like the real robot loop."""

    def __init__(self, obj: str = "cube", render: bool = True):
        if obj not in OBJECTS:
            raise ValueError(f"obj must be one of {list(OBJECTS)}")
        if not (MODEL_DIR / ROBOT_XML).exists():
            raise FileNotFoundError(f"{ROBOT_XML} missing - run: python sim/fetch_model.py (--lowpoly)")
        self.obj = obj
        xml = SCENE_XML.replace("{object_geom}", OBJECTS[obj]).replace("{robot_xml}", ROBOT_XML).replace("{w}", str(CAM_W)).replace("{h}", str(CAM_H))
        # Write next to the robot MJCF so its relative mesh paths resolve.
        scene = MODEL_DIR / f"_scene_{obj}_{ROBOT_XML[:-4]}.xml"
        scene.write_text(xml)
        self.model = _add_finger_pads(mujoco.MjSpec.from_file(str(scene)))
        self.data = mujoco.MjData(self.model)
        self.n_sub = round(1.0 / FPS / self.model.opt.timestep)
        self.obj_qadr = self.model.joint("object").qposadr[0]
        self.grip_site = self.model.site("gripperframe").id
        self.renderer = mujoco.Renderer(self.model, CAM_H, CAM_W) if render else None
        self.rng = np.random.default_rng()

    # ---------------------------------------------------------------- core API

    def reset(self, seed: int | None = None, home: np.ndarray | None = None) -> Obs:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        home = HOME if home is None else home
        self.data.qpos[:6] = lerobot_to_rad(home)
        self.data.ctrl[:6] = lerobot_to_rad(home)
        # Random object pose on the mat, reachable by the arm.
        # Where real operators grasp (lerobot/svla_so101_pickplace: x 3-23 cm, y +-13 cm), minus the
        # spots right at the base the top-down expert can't reach.
        x = self.rng.uniform(0.13, 0.23)
        y = self.rng.uniform(-0.10, 0.10)
        yaw = self.rng.uniform(-math.pi / 2, math.pi / 2) if self.obj == "pen" else self.rng.uniform(-0.6, 0.6)
        a = self.obj_qadr
        self.data.qpos[a : a + 3] = [x, y, OBJ_REST_Z[self.obj] + 0.001]
        self.data.qpos[a + 3 : a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        for _ in range(20):  # let the object settle
            mujoco.mj_step(self.model, self.data)
        return self.observe()

    def step(self, action: np.ndarray, render: bool = True) -> Obs:
        self.data.ctrl[:6] = lerobot_to_rad(action)
        mujoco.mj_step(self.model, self.data, nstep=self.n_sub)
        return self.observe(render)

    def observe(self, render: bool = True) -> Obs:
        state = rad_to_lerobot(self.data.qpos[:6])
        image = self.render() if (self.renderer and render) else np.zeros((CAM_H, CAM_W, 3), np.uint8)
        return Obs(state=state, image=image)

    def render(self, camera: str = "front") -> np.ndarray:
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    # ---------------------------------------------------------------- task info

    def object_pose(self) -> tuple[np.ndarray, float]:
        a = self.obj_qadr
        pos = self.data.qpos[a : a + 3].copy()
        w, x, y, z = self.data.qpos[a + 3 : a + 7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return pos, yaw

    def object_height(self) -> float:
        return float(self.data.qpos[self.obj_qadr + 2] - OBJ_REST_Z[self.obj])

    def is_success(self) -> bool:
        """Object lifted at least 4 cm off the table."""
        return self.object_height() > 0.04

    def close(self) -> None:
        if self.renderer:
            self.renderer.close()


PAD_PARALLEL_RAD = -0.06


def _add_finger_pads(spec: mujoco.MjSpec) -> mujoco.MjModel:
    """Box finger pads for grasping.

    MuJoCo collides meshes as convex hulls. The fixed jaw is part of the big L-shaped
    `wrist_roll_follower` mesh, whose hull fills the gap between the jaws, so thin objects
    get pushed out instead of pinched. Fix: the two jaw meshes only touch the table
    (collision bit 4), and two boxes matching the measured finger shape grip the object.
    Pad poses are in the `gripperframe` site frame with the gripper joint at 0 rad
    (x = approach, z = opening direction, fixed jaw inner face at z = 0, moving at z = 0.016).
    """
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # all joints at 0
    site = model.site("gripperframe").id
    Rs, ps = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]

    def pad_in_body(body: str, center_site: np.ndarray):
        b = model.body(body).id
        Rb, pb = data.xmat[b].reshape(3, 3), data.xpos[b]
        pos = Rb.T @ (ps + Rs @ center_site - pb)
        R = Rb.T @ Rs
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, R.flatten())
        return pos, quat

    # The moving jaw rotates, so its pad is only parallel to the fixed pad at one opening.
    # Defining it with the jaw at -0.06 rad makes them parallel at a ~12 mm gap (marker / pen
    # width); a wedge-shaped pinch squeezes round objects out.
    data.qpos[model.joint("gripper").qposadr[0]] = PAD_PARALLEL_RAD
    mujoco.mj_forward(model, data)
    Rs, ps = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]
    half = [0.021, 0.0065, 0.003]  # 4.2 cm long, 1.3 cm wide, 6 mm thick; tip 6 mm past the site
    for name, body, center in [
        ("fixed_pad", "gripper", np.array([-0.015, 0.0, -0.003])),
        ("moving_pad", "moving_jaw_so101_v1", np.array([-0.015, 0.0, 0.015])),
    ]:
        pos, quat = pad_in_body(body, center)
        spec.body(body).add_geom(
            name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos, quat=quat,
            rgba=[0.2, 0.2, 0.2, 0.0], group=3, contype=1, conaffinity=1,
            friction=[1.5, 0.01, 0.001], condim=4, mass=0.0001,
        )
    for g in spec.geoms:
        mesh = getattr(g, "meshname", "")
        if mesh in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1") and g.contype:
            g.contype, g.conaffinity = 4, 4
    spec.geom("table").contype, spec.geom("table").conaffinity = 5, 5
    return spec.compile()


# Arm raised out of the camera's way, gripper open (LeRobot units). The real arm's rest pose is
# about [0, -99, 99, 74, *, 2] (gripper resting on the table); a raised start keeps the pen undisturbed.
HOME = np.array([0.0, -60.0, 60.0, 60.0, 0.0, 40.0], dtype=np.float32)
