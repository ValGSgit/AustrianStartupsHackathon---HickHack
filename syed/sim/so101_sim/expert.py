"""Scripted pick expert: top-down grasp planned with damped-least-squares IK on the `gripperframe` site.

Geometry of the SO-101 gripper (measured from the MJCF):
  site x-axis = approach direction (points out of the jaws)
  site z-axis = jaw opening direction (moving jaw is on +z, fixed jaw tip sits at the site)
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .env import GRIP_CLOSED, GRIP_OPEN, HOME, SO101PickEnv, lerobot_to_rad, rad_to_lerobot

OPEN = 55.0  # gripper command while approaching (~6 cm jaw gap)
# Median arm pose at the moment real operators close the gripper on an object, from the Hugging Face
# dataset lerobot/svla_so101_pickplace (LeRobot units). Seeding IK with it keeps the scripted grasps in
# the same arm configuration (elbow branch) that humans use when teleoperating the real SO-101.
REAL_GRASP_POSE = np.array([5.8, -7.3, 40.2, 55.4, 0.0, 7.0], dtype=np.float32)
CLOSED = 0.0  # squeeze


def solve_ik(env: SO101PickEnv, target_pos, close_axis, q0, iters: int = 200) -> np.ndarray:
    """Arm joints (rad, 5) putting the site at target_pos, approach pointing down, jaws opening along close_axis."""
    m, d = env.model, env.data
    scratch = mujoco.MjData(m)
    scratch.qpos[:] = d.qpos
    q = np.array(q0[:5], dtype=np.float64)
    lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
    jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    down = np.array([0.0, 0.0, -1.0])
    for _ in range(iters):
        scratch.qpos[:5] = q
        mujoco.mj_kinematics(m, scratch)
        mujoco.mj_comPos(m, scratch)
        R = scratch.site_xmat[env.grip_site].reshape(3, 3)
        e_pos = target_pos - scratch.site_xpos[env.grip_site]
        e_rot = 0.5 * (np.cross(R[:, 0], down) + np.cross(R[:, 2], close_axis))
        err = np.concatenate([e_pos, 0.3 * e_rot])
        if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_rot) < 1e-2:
            break
        mujoco.mj_jacSite(m, scratch, jacp, jacr, env.grip_site)
        J = np.vstack([jacp[:, :5], 0.3 * jacr[:, :5]])
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), err)
        q = np.clip(q + dq, lo, hi)
    return q


def _close_axis(yaw: float, obj: str, q_roll_hint: float) -> np.ndarray:
    """Horizontal jaw-opening direction: perpendicular to the pen; for a cube, any face normal."""
    a = yaw + (math.pi / 2 if obj == "pen" else 0.0)
    c = np.array([math.cos(a), math.sin(a), 0.0])
    return c


def plan_pick(env: SO101PickEnv, rng: np.random.Generator) -> list[np.ndarray]:
    """Full action sequence (LeRobot units, 30 Hz) for one pick-and-lift episode."""
    pos, yaw = env.object_pose()
    half_w = 0.0125 if env.obj == "cube" else 0.006
    q_home = lerobot_to_rad(HOME)[:5]

    best = None
    for flip in (1.0, -1.0):  # the two jaw orientations; keep the one with less wrist roll
        c = flip * _close_axis(yaw, env.obj, 0.0)
        # Put the fixed jaw on one side, so the object sits centered between the jaws.
        # Fixed pad's inner face sits at the site: object center = site + c * (half width + margin).
        # Site height: fingertips (5 mm past the site) end ~3 mm above the table.
        grasp = np.array([pos[0], pos[1], 0.008]) - c * (half_w + 0.003)
        q_g = solve_ik(env, grasp, c, lerobot_to_rad(REAL_GRASP_POSE)[:5])
        if best is None or abs(q_g[4]) < abs(best[1][4]):
            best = (c, q_g, grasp)
    c, q_grasp, grasp = best
    pre = grasp + np.array([0, 0, 0.07])
    lift = grasp + np.array([0, 0, 0.12])
    q_pre = solve_ik(env, pre, c, q_grasp)
    q_lift = solve_ik(env, lift, c, q_grasp)

    def deg(q5, grip):  # arm angles (rad) + gripper command (0..100) -> LeRobot action
        a = rad_to_lerobot(np.append(q5, GRIP_CLOSED))
        a[5] = grip
        return a

    def move(qa, qb, ga, gb, n):
        out = []
        for i in range(1, n + 1):
            s = i / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5  # minimum-jerk profile
            out.append(deg(qa + (qb - qa) * s, ga + (gb - ga) * s))
        return out

    jitter = lambda n: int(n * rng.uniform(0.85, 1.15))  # noqa: E731 - vary speed across demos
    acts: list[np.ndarray] = []
    acts += move(q_home, q_pre, HOME[5], OPEN, jitter(45))  # reach above the object
    acts += move(q_pre, q_grasp, OPEN, OPEN, jitter(25))  # descend
    acts += [deg(q_grasp, OPEN)] * 5
    acts += move(q_grasp, q_grasp, OPEN, CLOSED, jitter(15))  # close
    acts += [deg(q_grasp, CLOSED)] * 8
    acts += move(q_grasp, q_lift, CLOSED, CLOSED, jitter(30))  # lift
    acts += [deg(q_lift, CLOSED)] * 20  # hold
    return acts


__all__ = ["plan_pick", "solve_ik"]
