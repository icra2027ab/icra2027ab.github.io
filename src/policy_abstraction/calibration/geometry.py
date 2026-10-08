"""Frames and projections shared by capture, solving and verification.

Conventions, once:

* A *pose* is ``[x, y, z, rx, ry, rz]`` with a rotation vector, exactly what
  RTDE's ``getActualTCPPose()`` returns. :func:`pose_to_matrix` turns it into
  the 4x4 that maps TCP coordinates into the base frame.
* ``T_a_b`` maps coordinates expressed in frame *b* into frame *a*.
* OpenCV's PnP returns ``T_cam_parent`` (``X_cam = T_cam_parent @ X_parent``).
  The env server wants the *inverse*: the camera's pose in the parent frame.
  :class:`~policy_abstraction.calibration.solve.ExtrinsicSolution` carries
  both under unambiguous names so the mix-up that bit the first calibration
  (see ``docs/ur5.md``) cannot recur silently.
"""

from __future__ import annotations

import enum

import numpy as np

from policy_abstraction.dynamics.ur5_kinematics import (
    matrix_to_rotvec,
    rotvec_to_matrix,
)


class CalibrationMode(str, enum.Enum):
    """Which camera mounting is being calibrated."""

    EXTERIOR = "exterior"
    """A camera fixed in the room, looking at the arm. The tracked point is the
    controller's TCP (the closed fingertips on this cell) and the result is the
    camera's pose in the robot ``base`` frame."""

    WRIST = "wrist"
    """A camera riding on the tool. The tracked point is a fixed mark on the
    table, recorded once in the base frame by touching it with the TCP; the
    result is the camera's pose in the TCP frame."""

    @property
    def parent_frame(self) -> str:
        return "base" if self is CalibrationMode.EXTERIOR else "tcp"


def pose_to_matrix(pose) -> np.ndarray:
    """``[x, y, z, rx, ry, rz]`` -> 4x4 ``T_base_tcp``."""
    p = np.asarray(pose, dtype=np.float64).reshape(-1)
    if p.shape[0] != 6:
        raise ValueError(f"pose must be [x, y, z, rx, ry, rz]; got shape {p.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotvec_to_matrix(p[3:6])
    T[:3, 3] = p[:3]
    return T


def matrix_to_pose(T: np.ndarray) -> np.ndarray:
    """4x4 -> ``[x, y, z, rx, ry, rz]``."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return np.concatenate([T[:3, 3], matrix_to_rotvec(T[:3, :3])])


def invert(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transform, without going through a general solve."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ T[:3, 3]
    return out


def is_rigid(T: np.ndarray, atol: float = 1e-6) -> bool:
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    R = T[:3, :3]
    return bool(
        np.allclose(R @ R.T, np.eye(3), atol=atol)
        and np.isclose(np.linalg.det(R), 1.0, atol=atol)
        and np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=atol)
    )


def rotation_angle_deg(R: np.ndarray) -> float:
    """Magnitude of a rotation matrix, in degrees."""
    cos = float(np.clip((np.trace(np.asarray(R)[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 to ``(N, 3)`` points."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return P @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]


def project(
    K: np.ndarray, T_cam_parent: np.ndarray, points_parent: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole projection of parent-frame points.

    Returns ``(uv, z)``: pixel coordinates ``(N, 2)`` and the depth of each
    point along the camera's optical axis ``(N,)``. Points behind the camera
    get ``nan`` pixels so a caller cannot mistake them for visible ones.
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    P_cam = transform_points(T_cam_parent, points_parent)
    z = P_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * P_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * P_cam[:, 1] / z + K[1, 2]
    uv = np.stack([u, v], axis=1)
    uv[z <= 1e-6] = np.nan
    return uv, z


def keypoint_in_parent(
    mode: CalibrationMode,
    tcp_pose,
    *,
    point_offset_tcp=(0.0, 0.0, 0.0),
    reference_point_base=None,
) -> np.ndarray:
    """Where the tracked physical point is, in the frame PnP solves against.

    * ``EXTERIOR``: the point rides on the tool at *point_offset_tcp* (metres,
      in the TCP frame; zero means the TCP itself). Returned in the base
      frame, which is the parent of a fixed camera.
    * ``WRIST``: the point is fixed in the room at *reference_point_base*.
      Returned in the TCP frame, which is the parent of a tool-mounted camera.
    """
    T_base_tcp = pose_to_matrix(tcp_pose)
    if mode is CalibrationMode.EXTERIOR:
        off = np.asarray(point_offset_tcp, dtype=np.float64).reshape(3)
        return T_base_tcp[:3, :3] @ off + T_base_tcp[:3, 3]
    if reference_point_base is None:
        raise ValueError(
            "wrist calibration needs the reference point's base-frame position; "
            "record it by touching the mark with the TCP first."
        )
    ref = np.asarray(reference_point_base, dtype=np.float64).reshape(3)
    return transform_points(invert(T_base_tcp), ref[None])[0]


def tracked_point_base(tcp_pose, point_offset_tcp=(0.0, 0.0, 0.0)) -> np.ndarray:
    """Base-frame position of a point carried on the tool (used to record the
    wrist mode's reference mark by touching it)."""
    return keypoint_in_parent(
        CalibrationMode.EXTERIOR, tcp_pose, point_offset_tcp=point_offset_tcp
    )


__all__ = [
    "CalibrationMode",
    "invert",
    "is_rigid",
    "keypoint_in_parent",
    "matrix_to_pose",
    "pose_to_matrix",
    "project",
    "rotation_angle_deg",
    "tracked_point_base",
    "transform_points",
]
