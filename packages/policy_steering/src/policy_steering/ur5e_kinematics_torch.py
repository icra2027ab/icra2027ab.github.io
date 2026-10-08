"""Torch-differentiable UR5e forward kinematics, for the guidance loss.

``policy_steering.guidance_fns_torch.build_guidance_fns`` recovers an
end-effector trajectory from an action chunk before measuring distances on
it. Its default is Euler integration: right for the simulators, whose actions
are Cartesian deltas, and wrong for the UR5e, whose actions are *absolute
joint targets*. Integrated, the base/shoulder/elbow angles read as x/y/z tool
velocities, the loss is measured somewhere the arm is not going, and -- because
an early "velocity" moves every later position -- the correction lands
hardest on the first action of the chunk, as a bias that ignores where the
arm is. That is what made guided chunks on the real arm start centimetres
away from the tool.

This module is the ``dynamics_fn`` that makes the loss see the chunk the way
the arm does: forward kinematics of every joint target, so a chunk that holds
the current joints yields the current tool pose. It is the torch twin of
:mod:`policy_abstraction.dynamics.ur5_kinematics_jax` (same DH table, same
TCP composition, same quaternion-based rotation-vector extraction so the
gradient stays finite at 180 degrees, where the tool points at the table).

Two things differ from the JAX version because of where this one runs:

* The LeRobot server steers the model's *normalized* action tensor, so the
  joint targets arrive scaled by the checkpoint's action statistics.
  :func:`make_ur5e_torch_dynamics` takes an ``unnormalize`` callable
  (see :func:`make_affine_unnormalizer`) that is applied before the
  kinematics, inside the differentiable chain.
* The kinematics run in float32 whatever the model's dtype: six chained DH
  products in bfloat16 lose millimetres, and the loss is measured in metres.

The DH table is duplicated here rather than imported because this package is
the only one bridged into the lerobot venv; ``tests/policy_steering`` checks it
against ``policy_abstraction.dynamics.ur5_kinematics.UR5E_DH``.
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch

# (a, d, alpha) per joint; theta comes from the joint target. Universal
# Robots' published UR5e parameters -- see the docstring of
# ``policy_abstraction.dynamics.ur5_kinematics`` for the table.
UR5E_DH: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.1625, math.pi / 2.0),
    (-0.425, 0.0, 0.0),
    (-0.3922, 0.0, 0.0),
    (0.0, 0.1333, math.pi / 2.0),
    (0.0, 0.0997, -math.pi / 2.0),
    (0.0, 0.0996, 0.0),
)

_EPS = 1e-12
_FK_DTYPE = torch.float32

# Signature ``build_guidance_fns`` expects of a ``dynamics_fn``:
# ``(x_t, ee_pos_0, ee_ori_0, dt) -> (positions, orientations, gripper)``.
Ur5TorchDynamicsFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]


def torch_rotvec_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Rotation vector ``(3,)`` -> rotation matrix ``(3, 3)`` (Rodrigues)."""
    rotvec = torch.as_tensor(rotvec, dtype=torch.float64).reshape(3)
    theta = torch.sqrt(torch.sum(rotvec * rotvec) + _EPS)
    k = rotvec / theta
    zero = torch.zeros((), dtype=rotvec.dtype)
    K = torch.stack(
        [
            torch.stack([zero, -k[2], k[1]]),
            torch.stack([k[2], zero, -k[0]]),
            torch.stack([-k[1], k[0], zero]),
        ]
    )
    eye = torch.eye(3, dtype=rotvec.dtype)
    return eye + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


def tcp_offset_matrix(tcp_offset: Sequence[float] | None) -> torch.Tensor:
    """``[x, y, z, rx, ry, rz]`` TCP offset -> 4x4 flange->TCP transform."""
    T = torch.eye(4, dtype=torch.float64)
    if tcp_offset is None:
        return T
    off = torch.as_tensor(list(tcp_offset), dtype=torch.float64).reshape(-1)
    if off.shape[0] != 6:
        raise ValueError(
            f"tcp_offset must be [x, y, z, rx, ry, rz]; got {tuple(off.shape)}"
        )
    T[:3, :3] = torch_rotvec_to_matrix(off[3:6])
    T[:3, 3] = off[:3]
    return T


def _dh_transform_batch(
    theta: torch.Tensor, a: float, d: float, alpha: float
) -> torch.Tensor:
    """Standard DH transform ``Rz(theta) Tz(d) Tx(a) Rx(alpha)`` for ``(N,)`` angles."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero = torch.zeros_like(theta)
    one = torch.ones_like(theta)
    rows = [
        torch.stack([ct, -st * ca, st * sa, a * ct], dim=-1),
        torch.stack([st, ct * ca, -ct * sa, a * st], dim=-1),
        torch.stack([zero, sa * one, ca * one, d * one], dim=-1),
        torch.stack([zero, zero, zero, one], dim=-1),
    ]
    return torch.stack(rows, dim=-2)  # (N, 4, 4)


def torch_ur5e_flange_pose(q: torch.Tensor) -> torch.Tensor:
    """Forward kinematics to the tool flange for ``(N, 6)`` joints -> ``(N, 4, 4)``."""
    q = torch.as_tensor(q)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    if q.shape[-1] < 6:
        raise ValueError(
            f"UR5e forward kinematics needs 6 joints; got {tuple(q.shape)}"
        )
    T = torch.eye(4, dtype=q.dtype, device=q.device).expand(q.shape[0], 4, 4)
    for i, (a, d, alpha) in enumerate(UR5E_DH):
        T = T @ _dh_transform_batch(q[:, i], a, d, alpha)
    return T


def torch_matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """``(N, 3, 3)`` rotation matrices -> ``(N, 3)`` rotation vectors, differentiable.

    Shepperd's method through a unit quaternion (pivot on the largest of the
    trace and the three diagonal entries) rather than ``acos(trace)``, whose
    derivative is infinite at 0 and 180 degrees. Every candidate is computed
    with its radicand floored so the branches the ``gather`` does not select
    still carry finite gradients. The angle is folded into ``[0, pi]``.
    """
    R = torch.as_tensor(R)
    squeeze = R.ndim == 2
    if squeeze:
        R = R.unsqueeze(0)
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    tr = m00 + m11 + m22

    def _s(radicand):
        return 2.0 * torch.sqrt(torch.clamp(radicand, min=_EPS))

    s_w = _s(1.0 + tr)
    q_w = torch.stack(
        [
            s_w / 4.0,
            (R[:, 2, 1] - R[:, 1, 2]) / s_w,
            (R[:, 0, 2] - R[:, 2, 0]) / s_w,
            (R[:, 1, 0] - R[:, 0, 1]) / s_w,
        ],
        dim=-1,
    )
    s_x = _s(1.0 + m00 - m11 - m22)
    q_x = torch.stack(
        [
            (R[:, 2, 1] - R[:, 1, 2]) / s_x,
            s_x / 4.0,
            (R[:, 0, 1] + R[:, 1, 0]) / s_x,
            (R[:, 0, 2] + R[:, 2, 0]) / s_x,
        ],
        dim=-1,
    )
    s_y = _s(1.0 + m11 - m00 - m22)
    q_y = torch.stack(
        [
            (R[:, 0, 2] - R[:, 2, 0]) / s_y,
            (R[:, 0, 1] + R[:, 1, 0]) / s_y,
            s_y / 4.0,
            (R[:, 1, 2] + R[:, 2, 1]) / s_y,
        ],
        dim=-1,
    )
    s_z = _s(1.0 + m22 - m00 - m11)
    q_z = torch.stack(
        [
            (R[:, 1, 0] - R[:, 0, 1]) / s_z,
            (R[:, 0, 2] + R[:, 2, 0]) / s_z,
            (R[:, 1, 2] + R[:, 2, 1]) / s_z,
            s_z / 4.0,
        ],
        dim=-1,
    )
    candidates = torch.stack([q_w, q_x, q_y, q_z], dim=1)  # (N, 4, 4)
    pivot = torch.argmax(torch.stack([tr, m00, m11, m22], dim=-1), dim=-1)  # (N,)
    quat = candidates[torch.arange(R.shape[0], device=R.device), pivot]  # (N, 4)
    # w >= 0 keeps the angle in [0, pi], matching the numpy extraction.
    quat = torch.where(quat[:, :1] < 0.0, -quat, quat)
    quat = quat / torch.sqrt(torch.sum(quat * quat, dim=-1, keepdim=True) + _EPS)

    v = quat[:, 1:]
    sin_half = torch.sqrt(torch.sum(v * v, dim=-1, keepdim=True) + _EPS)
    angle = 2.0 * torch.atan2(sin_half, quat[:, :1])
    rotvec = v * (angle / sin_half)
    return rotvec.squeeze(0) if squeeze else rotvec


def torch_ur5e_tcp_pose(
    q: torch.Tensor, tcp_offset: Sequence[float] | torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(N, 6)`` joints -> ``(positions (N, 3), rotation vectors (N, 3))`` of the TCP.

    ``tcp_offset`` may be the ``[x, y, z, rx, ry, rz]`` vector or an already
    built 4x4 matrix (see :func:`tcp_offset_matrix`).
    """
    q = torch.as_tensor(q)
    if isinstance(tcp_offset, torch.Tensor):
        T_tcp = (
            tcp_offset
            if tuple(tcp_offset.shape) == (4, 4)
            else tcp_offset_matrix(tcp_offset.reshape(-1).tolist())
        )
    else:
        T_tcp = tcp_offset_matrix(tcp_offset)
    T_tcp = T_tcp.to(dtype=q.dtype, device=q.device)
    T = torch_ur5e_flange_pose(q) @ T_tcp
    return T[:, :3, 3], torch_matrix_to_rotvec(T[:, :3, :3])


def make_affine_unnormalizer(
    scale: torch.Tensor, offset: torch.Tensor
) -> Callable[[torch.Tensor], torch.Tensor]:
    """``x[..., :k] * scale + offset`` for ``k = len(scale)``; later dims untouched.

    This is every normalization mode lerobot's ``UnnormalizerProcessorStep``
    implements, written as one affine map -- MEAN_STD is ``(std, mean)``,
    MIN_MAX and the quantile modes are ``((hi - lo) / 2, (hi + lo) / 2)``. The
    model's internal action tensor is padded past the real action dim (pi0 /
    pi0.5 denoise 32 channels), so only the leading ``k`` are mapped.
    """
    scale = torch.as_tensor(scale).reshape(-1)
    offset = torch.as_tensor(offset).reshape(-1)
    if scale.shape != offset.shape:
        raise ValueError(
            f"scale and offset must match; got {tuple(scale.shape)} and "
            f"{tuple(offset.shape)}"
        )
    k = int(scale.shape[0])

    def unnormalize(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < k:
            raise ValueError(
                f"unnormalize expected >= {k} action dims; got {tuple(x.shape)}"
            )
        s = scale.to(dtype=x.dtype, device=x.device)
        o = offset.to(dtype=x.dtype, device=x.device)
        head = x[..., :k] * s + o
        if x.shape[-1] == k:
            return head
        return torch.cat([head, x[..., k:]], dim=-1)

    return unnormalize


def make_ur5e_torch_dynamics(
    tcp_offset: Sequence[float] | None = None,
    unnormalize: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Ur5TorchDynamicsFn:
    """Build the ``dynamics_fn`` for a joint-absolute UR5e action chunk.

    Returns a function with the signature ``build_guidance_fns`` expects,
    ``(x_t, ee_pos_0, ee_ori_0, dt) -> (positions, orientations, gripper)``
    over a batched chunk ``x_t`` of shape ``(B, T, D)``:

    * ``unnormalize`` (optional) maps the model's normalized actions back to
      radians and gripper openness first, inside the differentiable chain;
    * dims ``0:6`` go through the forward kinematics composed with the
      controller's ``tcp_offset``, so the poses match RTDE's
      ``getActualTCPPose()``;
    * dim ``6`` passes through as the (absolute) gripper channel.

    ``ee_pos_0``, ``ee_ori_0`` and ``dt`` are accepted for the signature and
    ignored: joint targets are absolute, so nothing is integrated from the
    current pose. Outputs come back in ``x_t``'s dtype; the kinematics
    themselves run in float32.
    """
    T_tcp = tcp_offset_matrix(tcp_offset).to(_FK_DTYPE)

    def dynamics_fn(x_t, ee_pos_0, ee_ori_0, dt):
        x = unnormalize(x_t) if unnormalize is not None else x_t
        if x.ndim != 3:
            raise ValueError(
                f"dynamics_fn expects (B, T, D) actions; got {tuple(x.shape)}"
            )
        if x.shape[-1] < 7:
            raise ValueError(
                "UR5e joint-space guidance needs [q0..q5, gripper] action "
                f"channels; got {tuple(x.shape)}"
            )
        B, T, _ = x.shape
        q = x[..., :6].reshape(B * T, 6).to(_FK_DTYPE)
        positions, orientations = torch_ur5e_tcp_pose(q, T_tcp.to(q.device))
        gripper = x[..., 6:7]
        return (
            positions.reshape(B, T, 3).to(x_t.dtype),
            orientations.reshape(B, T, 3).to(x_t.dtype),
            gripper,
        )

    return dynamics_fn


__all__ = [
    "UR5E_DH",
    "Ur5TorchDynamicsFn",
    "make_affine_unnormalizer",
    "make_ur5e_torch_dynamics",
    "tcp_offset_matrix",
    "torch_matrix_to_rotvec",
    "torch_rotvec_to_matrix",
    "torch_ur5e_flange_pose",
    "torch_ur5e_tcp_pose",
]
