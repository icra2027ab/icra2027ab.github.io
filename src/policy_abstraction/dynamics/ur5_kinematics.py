"""UR5e forward kinematics and the joint-space dynamics model built on it.

The simulated envs in this repo emit *Cartesian delta* actions, so
:class:`~policy_abstraction.dynamics.integration.IntegrationDynamicsModel` can
turn an action chunk into a predicted end-effector trajectory by plain Euler
integration. A UR5e driven through ``servoJ`` instead consumes *absolute joint
targets*, so predicting where the tool will be requires forward kinematics.

:class:`Ur5KinematicsDynamicsModel` provides that: it maps a
``[..., T, 7]`` chunk of ``[q0 … q5, gripper]`` targets to the same
``[pos(3), axis-angle(3), aperture(1)]`` state layout every other env
publishes, so ``ChunkProjector``, the Rerun visualiser and the guidance stack
consume UR5 chunks without special-casing.

Kinematic parameters
--------------------
Standard Denavit–Hartenberg parameters for the UR5e, as published by Universal
Robots (https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/):

===  ==========  ==========  =========  ========
 i    theta       a (m)       d (m)      alpha
===  ==========  ==========  =========  ========
 1    q0          0           0.1625      +pi/2
 2    q1         -0.425       0            0
 3    q2         -0.3922      0            0
 4    q3          0           0.1333      +pi/2
 5    q4          0           0.0997      -pi/2
 6    q5          0           0.0996       0
===  ==========  ==========  =========  ========

These give the pose of the **tool flange** (``tool0``) in the robot base frame.
RTDE's ``getActualTCPPose()`` reports the pose of the *configured TCP*, which is
the flange pose composed with the TCP offset set on the controller. Pass that
offset via ``tcp_offset`` (the same ``[x, y, z, rx, ry, rz]`` vector the UR
teach pendant shows) so predicted states line up with measured ones.

Everything here is numpy-only so the module can also be imported from the
``physical-ai-sdk`` venv that runs the env server.
"""

from __future__ import annotations

import math

import numpy as np

from policy_abstraction.dynamics.base import BaseDynamicsModel

# (a, d, alpha) per joint; theta comes from the joint state.
UR5E_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.1625, math.pi / 2.0),
    (-0.425, 0.0, 0.0),
    (-0.3922, 0.0, 0.0),
    (0.0, 0.1333, math.pi / 2.0),
    (0.0, 0.0997, -math.pi / 2.0),
    (0.0, 0.0996, 0.0),
)


def _dh_transform(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    """Standard DH homogeneous transform ``Rz(theta) Tz(d) Tx(a) Rx(alpha)``."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array(
        [
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotvec_to_matrix(rotvec) -> np.ndarray:
    """Rotation vector (axis * angle) → 3×3 rotation matrix (Rodrigues)."""
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = r / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → rotation vector (axis * angle, radians).

    Matches ``quat2axisangle(eef_quat)`` — the state representation the rest of
    the repo uses — and RTDE's own orientation convention.
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    # Clamp guards against a trace marginally outside [-1, 3] from round-off.
    cos_theta = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if math.pi - theta < 1e-6:
        # Near-180°: sin(theta) ≈ 0, so recover the axis from the symmetric
        # part of R instead of the (numerically dead) skew part.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        # Fix the signs from the off-diagonal entries, anchored on the largest
        # component so the division stays well conditioned.
        i = int(np.argmax(axis))
        if axis[i] > 0:
            for j in range(3):
                if j != i:
                    axis[j] = math.copysign(axis[j], A[i, j])
        return axis * theta
    skew = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    )
    return skew * (theta / (2.0 * math.sin(theta)))


def ur5e_flange_pose(q) -> np.ndarray:
    """Forward kinematics to the tool flange. Returns a 4×4 base→flange pose."""
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] < 6:
        raise ValueError(f"UR5e forward kinematics needs 6 joints; got {q.shape}")
    T = np.eye(4, dtype=np.float64)
    for i, (a, d, alpha) in enumerate(UR5E_DH):
        T = T @ _dh_transform(float(q[i]), a, d, alpha)
    return T


def tcp_offset_matrix(tcp_offset=None) -> np.ndarray:
    """``[x, y, z, rx, ry, rz]`` TCP offset → 4×4 flange→TCP transform."""
    if tcp_offset is None:
        return np.eye(4, dtype=np.float64)
    off = np.asarray(tcp_offset, dtype=np.float64).reshape(-1)
    if off.shape[0] != 6:
        raise ValueError(
            f"tcp_offset must be [x, y, z, rx, ry, rz]; got shape {off.shape}"
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotvec_to_matrix(off[3:6])
    T[:3, 3] = off[:3]
    return T


def ur5e_forward_kinematics(q, tcp_offset=None) -> tuple[np.ndarray, np.ndarray]:
    """Joint positions → ``(tcp_position, tcp_rotation_vector)`` in the base frame.

    Args:
        q: 6 joint angles in radians.
        tcp_offset: Optional ``[x, y, z, rx, ry, rz]`` TCP offset configured on
            the UR controller. ``None`` returns the bare flange pose.

    Returns:
        ``(pos, rotvec)`` — both ``(3,)`` float64 arrays, matching the layout of
        RTDE's ``getActualTCPPose()``.
    """
    T = ur5e_flange_pose(q) @ tcp_offset_matrix(tcp_offset)
    return T[:3, 3].copy(), matrix_to_rotvec(T[:3, :3])


class Ur5KinematicsDynamicsModel(BaseDynamicsModel):
    """Predict EEF states from a chunk of absolute UR5e joint targets.

    Actions are ``[q0 … q5, gripper]`` (radians; gripper openness in ``[0, 1]``)
    and predicted states are ``[x, y, z, rx, ry, rz, aperture]`` — the layout
    published by :func:`policy_abstraction.envs.ur5.config.ur5_env_spec`, so
    ``ActionLayout.eef_pos_slice``-style consumers work unchanged.

    Unlike the integration model this is *stateless*: each joint target maps to
    exactly one pose, so ``initial_state`` is only used for its dtype/shape.
    """

    def __init__(self, tcp_offset=None) -> None:
        """
        Args:
            tcp_offset: TCP offset configured on the UR controller, as
                ``[x, y, z, rx, ry, rz]``. Read it from the env spec's
                ``metadata["tcp_offset"]`` (see
                :func:`~policy_abstraction.envs.ur5.config.ur5_env_spec`) so
                predictions match ``getActualTCPPose()``.
        """
        self._tcp_offset = (
            None if tcp_offset is None else np.asarray(tcp_offset, dtype=np.float64)
        )

    def predict_states(
        self,
        actions: np.ndarray,
        initial_state: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float64)
        initial_state = np.asarray(initial_state)
        state_dim = int(initial_state.shape[0])

        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[-1] < 6:
            raise ValueError(
                f"Ur5KinematicsDynamicsModel needs >=6 action dims (joint "
                f"targets); got {actions.shape}"
            )

        flat = actions.reshape(-1, actions.shape[-1])
        out = np.zeros((flat.shape[0], state_dim), dtype=np.float64)
        for i, a in enumerate(flat):
            pos, rotvec = ur5e_forward_kinematics(a[:6], self._tcp_offset)
            n = min(3, state_dim)
            out[i, :n] = pos[:n]
            if state_dim > 3:
                m = min(3, state_dim - 3)
                out[i, 3 : 3 + m] = rotvec[:m]
        # Gripper rides through untouched in the last state slot (absolute, not
        # integrated) — same convention as IntegrationDynamicsModel.
        if state_dim > 6 and actions.shape[-1] >= 7:
            out[:, -1] = flat[:, 6]

        return out.reshape(*actions.shape[:-1], state_dim).astype(
            initial_state.dtype, copy=False
        )


__all__ = [
    "UR5E_DH",
    "Ur5KinematicsDynamicsModel",
    "matrix_to_rotvec",
    "rotvec_to_matrix",
    "tcp_offset_matrix",
    "ur5e_flange_pose",
    "ur5e_forward_kinematics",
]
