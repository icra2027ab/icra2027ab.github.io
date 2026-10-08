"""JAX-differentiable UR5e forward kinematics, for the guidance loss.

``policy_steering.guidance_fns.build_guidance_fns`` recovers an end-effector
trajectory from an action chunk before measuring distances on it. Its default
is Euler integration — right for the simulators, whose actions are Cartesian
deltas, and wrong for the UR5e, whose actions are *absolute joint targets*:
integrated, a joint target of 2 rad reads as a 2 m/s tool velocity, the loss is
measured somewhere the arm is not going, and the correction lands on the wrong
joints. This module is the ``dynamics_fn`` that makes the loss see the chunk
the way the arm does: forward kinematics of every joint target, so a chunk
that holds the current joints yields the current tool pose.

It is the differentiable twin of :mod:`policy_abstraction.dynamics.ur5_kinematics`
(same DH table, same TCP offset composition) and is checked against it in
``tests/dynamics/test_ur5_kinematics_jax.py``. The one deliberate difference is
the rotation-vector extraction: the numpy version branches on the angle, which
autodiff cannot follow, so this one goes through a quaternion instead. That
keeps the gradient finite at 180 degrees — where the UR5e's tool points at the
table, i.e. exactly the pose the arm spends its time in.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from policy_abstraction.dynamics.ur5_kinematics import UR5E_DH, tcp_offset_matrix

# Signature ``build_guidance_fns`` expects of a ``dynamics_fn``.
Ur5JaxDynamicsFn = Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], tuple]

_EPS = 1e-12


def _jax_dh_transform(theta, a: float, d: float, alpha: float) -> jnp.ndarray:
    """Standard DH transform ``Rz(theta) Tz(d) Tx(a) Rx(alpha)`` (JAX)."""
    ct, st = jnp.cos(theta), jnp.sin(theta)
    ca, sa = jnp.cos(alpha), jnp.sin(alpha)
    return jnp.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def jax_ur5e_flange_pose(q) -> jnp.ndarray:
    """Forward kinematics to the tool flange: 4x4 base->flange pose (JAX)."""
    q = jnp.asarray(q)
    T = jnp.eye(4)
    for i, (a, d, alpha) in enumerate(UR5E_DH):
        T = T @ _jax_dh_transform(q[i], a, d, alpha)
    return T


def jax_matrix_to_rotvec(R) -> jnp.ndarray:
    """3x3 rotation matrix -> rotation vector, differentiable everywhere.

    Goes through a unit quaternion (Shepperd's method: pivot on the largest of
    the trace and the three diagonal entries) rather than the ``acos(trace)``
    route, whose derivative blows up at 0 and 180 degrees. Every candidate is
    computed with its radicand floored, so the branches ``jnp.where`` does not
    select still have finite gradients. The angle is folded into ``[0, pi]``.
    """
    R = jnp.asarray(R)
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22

    def _s(radicand):
        return 2.0 * jnp.sqrt(jnp.maximum(radicand, _EPS))

    s_w = _s(1.0 + tr)
    q_w = jnp.array(
        [
            s_w / 4.0,
            (R[2, 1] - R[1, 2]) / s_w,
            (R[0, 2] - R[2, 0]) / s_w,
            (R[1, 0] - R[0, 1]) / s_w,
        ]
    )
    s_x = _s(1.0 + m00 - m11 - m22)
    q_x = jnp.array(
        [
            (R[2, 1] - R[1, 2]) / s_x,
            s_x / 4.0,
            (R[0, 1] + R[1, 0]) / s_x,
            (R[0, 2] + R[2, 0]) / s_x,
        ]
    )
    s_y = _s(1.0 + m11 - m00 - m22)
    q_y = jnp.array(
        [
            (R[0, 2] - R[2, 0]) / s_y,
            (R[0, 1] + R[1, 0]) / s_y,
            s_y / 4.0,
            (R[1, 2] + R[2, 1]) / s_y,
        ]
    )
    s_z = _s(1.0 + m22 - m00 - m11)
    q_z = jnp.array(
        [
            (R[1, 0] - R[0, 1]) / s_z,
            (R[0, 2] + R[2, 0]) / s_z,
            (R[1, 2] + R[2, 1]) / s_z,
            s_z / 4.0,
        ]
    )
    pivot = jnp.argmax(jnp.array([tr, m00, m11, m22]))
    quat = jnp.stack([q_w, q_x, q_y, q_z])[pivot]  # (4,) = [w, x, y, z]
    # w >= 0 keeps the angle in [0, pi], matching the numpy extraction.
    quat = jnp.where(quat[0] < 0.0, -quat, quat)
    quat = quat / jnp.sqrt(jnp.sum(quat * quat) + _EPS)

    v = quat[1:]
    sin_half = jnp.sqrt(jnp.sum(v * v) + _EPS)
    angle = 2.0 * jnp.arctan2(sin_half, quat[0])
    return v * (angle / sin_half)


def make_ur5e_jax_dynamics(tcp_offset=None) -> Ur5JaxDynamicsFn:
    """Build the ``dynamics_fn`` for a joint-absolute UR5e action chunk.

    Returns a function with the signature ``build_guidance_fns`` expects,
    ``(x_t, ee_pos_0, ee_ori_0, dt) -> (positions, orientations, gripper)``,
    mapping ``x_t[:, :6]`` (absolute joint targets, rad) through the forward
    kinematics — composed with the controller's ``tcp_offset`` so the poses
    match RTDE's ``getActualTCPPose()`` — and passing ``x_t[:, 6:7]`` through
    as the (absolute) gripper channel. ``ee_pos_0``, ``ee_ori_0`` and ``dt``
    are accepted for the signature and ignored: joint targets are absolute,
    so nothing is integrated from the current pose.

    The returned function is passed to ``jax.jit`` as a static argument, so it
    must be created once and reused, not rebuilt per call.
    """
    T_tcp = jnp.asarray(tcp_offset_matrix(tcp_offset))

    def _tcp_pose(q6):
        T = jax_ur5e_flange_pose(q6) @ T_tcp
        return T[:3, 3], jax_matrix_to_rotvec(T[:3, :3])

    _tcp_pose_vmap = jax.vmap(_tcp_pose)

    def dynamics_fn(x_t, ee_pos_0, ee_ori_0, dt):
        positions, orientations = _tcp_pose_vmap(x_t[:, :6])  # (T, 3), (T, 3)
        gripper = x_t[:, 6:7]  # (T, 1) — absolute, not integrated
        return positions, orientations, gripper

    return dynamics_fn


def ur5e_tcp_pose_np(q, tcp_offset=None) -> tuple[np.ndarray, np.ndarray]:
    """Numpy-in, numpy-out convenience for tests: ``(pos, rotvec)`` via JAX."""
    fn = make_ur5e_jax_dynamics(tcp_offset)
    pos, rotvec, _ = fn(
        jnp.asarray(np.asarray(q, dtype=np.float64).reshape(1, -1)),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.asarray(0.02),
    )
    return np.asarray(pos[0]), np.asarray(rotvec[0])


__all__ = [
    "Ur5JaxDynamicsFn",
    "jax_matrix_to_rotvec",
    "jax_ur5e_flange_pose",
    "make_ur5e_jax_dynamics",
    "ur5e_tcp_pose_np",
]
