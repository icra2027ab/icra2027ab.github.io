"""Gripper finger-tip position utilities.

Geometry is a parameter, not a constant. A rollout in this repo is
cross-embodiment: the retrieved demo comes from LIBERO's Franka Panda while the
live arm may be a UR5e with a Robotiq 2F-140, and both sides are compared at
the same instant. Every function here therefore takes a ``geometry``
(a :class:`~policy_steering.robot_config.GripperGeometry`, resolved from
``EnvSpec.robot.gripper.geometry_id``) and defaults to the Panda so existing
LIBERO-side call sites are unchanged.

Two-finger geometry
-------------------
Both supported grippers move their jaws symmetrically about the grip site, so
one aperture scalar plus two offsets describes the pair. Along the axis the
jaws open on (``GripperGeometry.finger_axis``), with the other lateral axis
zero:

  left_tip_local  = −(q1 + finger_x_rest) · axis + finger_z_from_grip · z
  right_tip_local = (−q2 + finger_x_rest) · axis + finger_z_from_grip · z

where ``q1 = +aperture/2`` and ``q2 = −aperture/2`` in metres.

For the Franka Panda (``panda_gripper.xml``) the two prismatic fingers slide in
±y of the EEF frame with a rest half-gap of 8.5 mm and a fixed 11.4 mm z-offset
from the grip site. For the Robotiq on the UR5e the TCP is configured at the
fingertip plane, so both offsets are zero, and the way it is bolted to the
flange puts its jaws on the tool frame's ±x — see the config YAMLs, which are
the single source of truth. The visualiser's ``gripper_mount_yaw_deg`` has to
agree with ``finger_axis`` or the rendered gripper and the fingertips guidance
measures from part ways by a quarter turn.

Aperture frames
---------------
``gripper_qpos`` is whatever the env reports, and differs per robot: the Panda
publishes ``[q1, q2]`` joint positions in *metres*, the UR5 adapter publishes a
single *normalised* openness in ``[0, 1]``. Both are collapsed to one scalar by
``GripperSpec.aperture_from_qpos`` and then mapped into metres by
``GripperGeometry.qpos_to_metric``, which is the identity for the Panda and a
×0.140 scale for the 2F-140. Policy *actions* live in yet another range and use
``action_to_metric``. Mixing the two silently rescales every gripper distance,
so the conversion is always explicit here.

The LIBERO dataset stores:
  obs/ee_pos          — grip-site position in world frame, shape (T, 3)
  obs/ee_ori          — grip-site orientation as axis-angle, shape (T, 3)
                        (= quat2axisangle(robot0_eef_quat))
  obs/gripper_states  — [q1, q2] raw joint positions,      shape (T, 2)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from policy_steering.robot_config import (
    DEFAULT_GRIPPER,
    GripperGeometry,
    load_gripper_geometry,
)

# Default geometry so LIBERO/Panda call sites need no argument.
_DEFAULT_GEOMETRY = load_gripper_geometry(DEFAULT_GRIPPER)

# Panda finger offsets, kept as module constants for the call sites and tests
# that reference them directly. Anything robot-agnostic should read the
# corresponding field off a resolved ``GripperGeometry`` instead.
_FINGER_X_REST = _DEFAULT_GEOMETRY.finger_x_rest
_FINGER_Z_FROM_GRIP = _DEFAULT_GEOMETRY.finger_z_from_grip


def _resolve(geometry: GripperGeometry | str | None) -> GripperGeometry:
    """Accept a geometry object, a config name, or ``None`` (→ Panda)."""
    if geometry is None:
        return _DEFAULT_GEOMETRY
    if isinstance(geometry, str):
        return load_gripper_geometry(geometry)
    return geometry


def _finger_joints(
    gripper_states: np.ndarray, geometry: GripperGeometry
) -> tuple[np.ndarray, np.ndarray]:
    """``(T, k)`` raw gripper qpos → per-finger ``(q1, q2)`` offsets in metres.

    Handles both published layouts. The single-column case is why this exists:
    indexing ``[:, 1]`` unconditionally raised ``IndexError`` on every
    real-robot observation.
    """
    gripper_states = np.atleast_2d(np.asarray(gripper_states, dtype=np.float64))
    n_cols = gripper_states.shape[1]
    if n_cols >= 2:
        # Two independently tracked fingers, already in metres (Panda). Use
        # them verbatim rather than symmetrising about the mean, so a slightly
        # asymmetric grasp keeps its real tip positions.
        return gripper_states[:, 0], gripper_states[:, 1]
    if n_cols == 1:
        # One DOF: the driver reports total openness, so split it symmetrically
        # about the grip site after scaling to metres by the stroke.
        aperture = geometry.qpos_to_metric(gripper_states[:, 0])
        return aperture / 2.0, -aperture / 2.0
    raise ValueError(
        f"gripper_states needs at least one column; got shape {gripper_states.shape}"
    )


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #


def compute_finger_tip_positions(
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper_states: np.ndarray,
    geometry: GripperGeometry | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-frame finger-tip positions for every timestep.

    Args:
        ee_pos:         ``(T, 3)`` grip-site positions in world frame.
        ee_ori:         ``(T, 3)`` grip-site orientation as axis-angle vectors.
        gripper_states: ``(T, 2)`` raw joint positions ``[q1, q2]`` in metres
                        (Panda convention), or ``(T, 1)`` normalised aperture
                        (single-DOF grippers such as the Robotiq). Converted to
                        a metric aperture via *geometry*.
        geometry:       Gripper geometry, a config name, or ``None`` for the
                        Panda default.

    Returns:
        ``(left_tips, right_tips)`` — each ``(T, 3)`` world-frame XYZ arrays.
    """
    g = _resolve(geometry)
    ee_pos = np.asarray(ee_pos, dtype=np.float64)
    # np.array (not asarray): observations arriving over the msgpack wire are
    # backed by the read-only receive buffer, and scipy's Cython rotation code
    # demands a writable one. asarray would pass a float64 input straight
    # through and raise "buffer source array is read-only" at from_rotvec.
    ee_ori = np.array(ee_ori, dtype=np.float64)

    q1, q2 = _finger_joints(gripper_states, g)  # each (T,), metres

    T = ee_pos.shape[0]
    z_off = np.full(T, g.finger_z_from_grip)

    # Lateral offsets go on whichever grip-site axis the jaws open along; the
    # other lateral axis stays zero.
    left_local = np.zeros((T, 3))
    right_local = np.zeros((T, 3))
    left_local[:, g.finger_axis_index] = -(q1 + g.finger_x_rest)
    right_local[:, g.finger_axis_index] = -q2 + g.finger_x_rest
    left_local[:, 2] = z_off
    right_local[:, 2] = z_off

    # Rotate local offsets into world frame and add grip-site origin.
    R = Rotation.from_rotvec(ee_ori).as_matrix()  # (T, 3, 3)

    # batched matmul: (T,3,3) @ (T,3,1) → (T,3)
    left_tips = ee_pos + (R @ left_local[..., np.newaxis])[..., 0]
    right_tips = ee_pos + (R @ right_local[..., np.newaxis])[..., 0]

    return left_tips, right_tips


def gripper_action_to_metric(
    action: np.ndarray | float,
    geometry: GripperGeometry | str | None = None,
) -> np.ndarray:
    """Linearly map a policy gripper-action value to a metric aperture (m).

    The policy's gripper channel lives in a normalised action space whose
    bounds are robot-specific (``[-1, 1]`` for pi0 on the Panda, ``[0, 1]`` for
    the UR5 adapter), while all distance computations — reference trajectories,
    finger FK, aperture features — operate in metres. This mapping keeps both
    sides in the same frame.

    No clipping is applied so gradients flow smoothly through the map;
    downstream FK tolerates slightly out-of-range values.
    """
    g = _resolve(geometry)
    return g.action_to_metric(np.asarray(action, dtype=np.float64))


def gripper_qpos_to_metric(
    aperture: np.ndarray | float,
    geometry: GripperGeometry | str | None = None,
) -> np.ndarray:
    """Map an *observed* aperture (post ``aperture_from_qpos``) to metres.

    Distinct from :func:`gripper_action_to_metric` because the observation and
    action ranges differ on the Panda (metric qpos, normalised actions) even
    though they coincide on the Robotiq.
    """
    g = _resolve(geometry)
    return g.qpos_to_metric(np.asarray(aperture, dtype=np.float64))


def compute_finger_tip_positions_single(
    ee_pos: np.ndarray,
    ee_ori: np.ndarray,
    gripper_qpos: np.ndarray,
    geometry: GripperGeometry | str | None = None,
) -> dict[str, np.ndarray]:
    """Compute finger-tip positions for a single timestep.

    Convenience wrapper around :func:`compute_finger_tip_positions` that
    avoids manual batch-dimension handling at every call site.

    Args:
        ee_pos:       ``(3,)`` grip-site position in world frame.
        ee_ori:       ``(3,)`` grip-site orientation as axis-angle.
        gripper_qpos: ``(2,)`` raw joint positions ``[q1, q2]`` (Panda) or
                      ``(1,)`` normalised aperture (single-DOF grippers).
        geometry:     Gripper geometry, a config name, or ``None`` for Panda.

    Returns:
        ``{"gripper_l": (3,), "gripper_r": (3,)}`` world-frame positions.
    """
    left, right = compute_finger_tip_positions(
        ee_pos[None], ee_ori[None], np.atleast_1d(gripper_qpos)[None], geometry
    )
    return {"gripper_l": left[0], "gripper_r": right[0]}
