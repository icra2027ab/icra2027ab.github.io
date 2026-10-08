"""PyTorch port of ``kinematics.py`` for use with the torch guidance kernel.

Mirrors ``kinematics.py`` exactly so torch and JAX guidance kernels produce
numerically identical finger-FK and rotation results given the same inputs.
"""

from __future__ import annotations

import torch

from policy_steering.robot_config import (
    DEFAULT_GRIPPER,
    GripperGeometry,
    load_gripper_geometry,
)

# Default (Panda) geometry; pass ``geometry=`` to describe another gripper.
_DEFAULT_GEOMETRY = load_gripper_geometry(DEFAULT_GRIPPER)
FINGER_X_REST = _DEFAULT_GEOMETRY.finger_x_rest
FINGER_Z_FROM_GRIP = _DEFAULT_GEOMETRY.finger_z_from_grip


def torch_axis_angle_to_matrix(rotvec: torch.Tensor) -> torch.Tensor:
    """Axis-angle (3,) → rotation matrix (3, 3) via Rodrigues.

    Mirrors ``jax_axis_angle_to_matrix``: stabilises near-zero rotation
    with ``sqrt(sum(x^2) + eps)`` so the gradient stays finite at the origin.
    """
    angle_sq = torch.sum(rotvec * rotvec)
    angle = torch.sqrt(angle_sq + 1e-16)
    axis = rotvec / angle
    zero = torch.zeros((), dtype=rotvec.dtype, device=rotvec.device)
    K = torch.stack(
        [
            torch.stack([zero, -axis[2], axis[1]]),
            torch.stack([axis[2], zero, -axis[0]]),
            torch.stack([-axis[1], axis[0], zero]),
        ]
    )
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + torch.sin(angle) * K + (1.0 - torch.cos(angle)) * (K @ K)


def torch_axis_angle_to_matrix_batch(rotvecs: torch.Tensor) -> torch.Tensor:
    """Batched axis-angle (..., 3) → rotation matrix (..., 3, 3).

    Vectorised version of ``torch_axis_angle_to_matrix``; equivalent to
    ``jax.vmap`` over the leading dimensions.
    """
    angle = torch.sqrt(torch.sum(rotvecs * rotvecs, dim=-1, keepdim=True) + 1e-16)
    axis = rotvecs / angle  # (..., 3)
    ax, ay, az = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = torch.zeros_like(ax)
    K = torch.stack(
        [
            torch.stack([zero, -az, ay], dim=-1),
            torch.stack([az, zero, -ax], dim=-1),
            torch.stack([-ay, ax, zero], dim=-1),
        ],
        dim=-2,
    )  # (..., 3, 3)
    eye = torch.eye(3, dtype=rotvecs.dtype, device=rotvecs.device)
    sin_a = torch.sin(angle).unsqueeze(-1)
    cos_a = torch.cos(angle).unsqueeze(-1)
    return eye + sin_a * K + (1.0 - cos_a) * (K @ K)


def torch_finger_positions(
    ee_pos: torch.Tensor,
    ee_ori: torch.Tensor,
    gripper_val: torch.Tensor,
    geometry: GripperGeometry | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Both finger-tip positions in world frame. Mirrors ``jax_finger_positions``.

    ``gripper_val`` is the *metric* opening in metres; ``geometry`` defaults to
    the Panda.
    """
    g = _DEFAULT_GEOMETRY if geometry is None else geometry
    q1 = gripper_val / 2.0
    q2 = -gripper_val / 2.0
    dtype, device = ee_pos.dtype, ee_pos.device
    fx = torch.tensor(g.finger_x_rest, dtype=dtype, device=device)
    # Four-bar pads ride an arc, so their height varies with the opening.
    fz = torch.as_tensor(
        g.finger_z_offset(torch.abs(q1) + fx), dtype=dtype, device=device
    )
    # The jaws open along `finger_axis` of the grip-site frame; z is the tool axis.
    axis = torch.tensor(g.finger_open_direction, dtype=dtype, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    left_local = -(q1 + fx) * axis + fz.reshape(()) * ez
    right_local = (-q2 + fx) * axis + fz.reshape(()) * ez
    R = torch_axis_angle_to_matrix(ee_ori)
    return ee_pos + R @ left_local, ee_pos + R @ right_local


def torch_finger_positions_vmap(
    ee_pos: torch.Tensor,
    ee_ori: torch.Tensor,
    gripper_val: torch.Tensor,
    geometry: GripperGeometry | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised over leading time dimension. Inputs (T, 3), (T, 3), (T,).

    Equivalent to ``jax.vmap`` over the time axis in ``jax_finger_positions_vmap``.
    """
    g = _DEFAULT_GEOMETRY if geometry is None else geometry
    T = ee_pos.shape[0]
    dtype, device = ee_pos.dtype, ee_pos.device
    fx = torch.tensor(g.finger_x_rest, dtype=dtype, device=device)

    q1 = gripper_val / 2.0  # (T,)
    q2 = -gripper_val / 2.0
    # (T,) for a four-bar, scalar for a prismatic gripper — broadcast either.
    fz = torch.as_tensor(
        g.finger_z_offset(torch.abs(q1) + fx), dtype=dtype, device=device
    ).expand(T)
    axis = torch.tensor(g.finger_open_direction, dtype=dtype, device=device)
    ez = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
    left_local = (-(q1 + fx))[:, None] * axis + fz[:, None] * ez  # (T, 3)
    right_local = (-q2 + fx)[:, None] * axis + fz[:, None] * ez  # (T, 3)

    R = torch_axis_angle_to_matrix_batch(ee_ori)  # (T, 3, 3)
    left_world = ee_pos + torch.einsum("tij,tj->ti", R, left_local)
    right_world = ee_pos + torch.einsum("tij,tj->ti", R, right_local)
    return left_world, right_world
