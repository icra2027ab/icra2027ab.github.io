"""JAX-differentiable forward kinematics for parallel-jaw gripper fingers.

All functions use only ``jax.numpy`` operations and are safe to use inside
``jax.jit``, ``jax.grad``, and ``jax.vmap``.

The geometry is a parameter, not a constant: a cross-embodiment rollout
compares a Panda demo against (say) a UR5e + Robotiq 2F-140, so both sets of
finger offsets have to be available at once. The module-level ``FINGER_*``
values and the bare ``jax_finger_positions*`` functions stay bound to the Panda
for backwards compatibility; pass ``geometry=`` (or build a vmapped variant
with :func:`make_jax_finger_positions_vmap`) to describe a different gripper.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from policy_steering.robot_config import (
    DEFAULT_GRIPPER,
    GripperGeometry,
    load_gripper_geometry,
)

# Default (Panda) gripper geometry constants (metres) — configs/franka_panda.yaml.
_DEFAULT_GEOMETRY = load_gripper_geometry(DEFAULT_GRIPPER)
FINGER_X_REST = _DEFAULT_GEOMETRY.finger_x_rest
FINGER_Z_FROM_GRIP = _DEFAULT_GEOMETRY.finger_z_from_grip


def jax_axis_angle_to_matrix(rotvec):
    """Convert axis-angle vector (3,) to rotation matrix (3, 3) via Rodrigues.

    Handles the near-zero-rotation case with a stabilised angle computation:
    ``jnp.linalg.norm`` has a NaN gradient at zero (``x/||x|| = 0/0``), so we
    compute the angle as ``sqrt(sum(x^2) + eps)`` whose gradient stays finite.
    """
    angle_sq = jnp.sum(rotvec * rotvec)
    angle = jnp.sqrt(angle_sq + 1e-16)
    axis = rotvec / angle
    K = jnp.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return jnp.eye(3) + jnp.sin(angle) * K + (1.0 - jnp.cos(angle)) * (K @ K)


def jax_finger_positions(ee_pos, ee_ori, gripper_val, geometry=None):
    """Compute both finger-tip positions in world frame using JAX ops.

    Uses the symmetric approximation ``q1 = gripper_val / 2``,
    ``q2 = -gripper_val / 2`` — valid for any gripper whose jaws move
    symmetrically about the grip site, which covers both the Panda
    (``gripper_val = q1 - q2``) and the Robotiq stroke.

    Args:
        ee_pos: (3,) EE grip-site position in world frame.
        ee_ori: (3,) EE orientation as axis-angle vector.
        gripper_val: scalar *metric* gripper opening in metres. Convert from
            the policy's action channel with ``GripperGeometry.action_to_metric``
            or from an observed aperture with ``qpos_to_metric`` first.
        geometry: :class:`~policy_steering.robot_config.GripperGeometry` for the
            gripper being described. Defaults to the Panda.

    Returns:
        (left_tip, right_tip) — each (3,) in world frame.
    """
    g = _DEFAULT_GEOMETRY if geometry is None else geometry
    q1 = gripper_val / 2.0
    q2 = -gripper_val / 2.0
    # A four-bar gripper's pads swing on an arc, so their height depends on how
    # far open it is; a prismatic one returns a constant here.
    fz = g.finger_z_offset(jnp.abs(q1) + g.finger_x_rest)
    # The jaws open along `finger_axis` of the grip-site frame (x for the
    # 2F-140 as bolted to the UR5e, y for the Panda); z is the tool axis.
    axis = jnp.asarray(g.finger_open_direction)
    ez = jnp.array([0.0, 0.0, 1.0])
    left_local = -(q1 + g.finger_x_rest) * axis + fz * ez
    right_local = (-q2 + g.finger_x_rest) * axis + fz * ez
    R = jax_axis_angle_to_matrix(ee_ori)
    return ee_pos + R @ left_local, ee_pos + R @ right_local


def make_jax_finger_positions_vmap(geometry: GripperGeometry | None = None):
    """Build a time-vmapped :func:`jax_finger_positions` for one gripper.

    ``geometry`` is bound into the closure rather than passed per call so it
    stays a static (non-traced) value under ``jax.jit``.
    """
    fn = functools.partial(jax_finger_positions, geometry=geometry)
    return jax.vmap(fn, in_axes=(0, 0, 0), out_axes=(0, 0))


# Vectorized version over time dimension (Panda geometry).
jax_finger_positions_vmap = make_jax_finger_positions_vmap()
