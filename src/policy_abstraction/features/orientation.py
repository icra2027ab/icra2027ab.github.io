"""
OrientationProjectionFeature.

Computes how a configurable body-frame vector of the end-effector is oriented
in the world frame after applying the EE rotation.

The body-frame vector defaults to ``[0, 0, 1]`` (the gripper z-axis / "approach
direction") but can be set to any unit direction via the ``ee_vector`` parameter,
e.g. ``[1, 0, 0]`` to track the gripper's x-axis.

The EE orientation arrives as an axis-angle 3-vector ``ee_ori``.  We build the
world-from-EE rotation matrix ``R = Rotation.from_rotvec(ee_ori).as_matrix()``
and compute the body vector in world coordinates as ``g = R @ ee_vector``.
``g`` is then projected onto the caller-supplied direction:

- ``axis`` mode  → returns ``dot(g, axis_hat)``: signed cosine of the angle
  between the body vector and the supplied direction (in ``[-1, 1]``).
  Use ``axis=[0, 0, -1]`` to measure "approach from above": the value is +1
  when the body vector points straight down, 0 when horizontal, -1 when
  straight up.
- ``plane`` mode → returns ``||g - (g · n_hat) n_hat||``: magnitude of the
  in-plane component of the body vector (in ``[0, 1]``).  Use
  ``plane=[0, 0, 1]`` to measure how much the body vector is tilted away from
  vertical: 0 when exactly along world z, 1 when fully horizontal.

The supplied direction is normalised at construction so callers may pass
unnormalized vectors.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from policy_abstraction.features.base import DistanceFeature


class OrientationProjectionFeature(DistanceFeature):
    """Projection of an EE body-frame vector onto a world-space direction or plane."""

    def __init__(
        self,
        direction: list[float] | np.ndarray,
        *,
        mode: str = "axis",
        ee_vector: list[float] | np.ndarray | None = None,
    ) -> None:
        if mode not in ("axis", "plane"):
            raise ValueError(
                f"OrientationProjectionFeature: mode must be 'axis' or 'plane', "
                f"got {mode!r}"
            )
        arr = np.asarray(direction, dtype=np.float64)
        norm = np.linalg.norm(arr)
        if norm < 1e-9:
            raise ValueError(
                f"OrientationProjectionFeature: direction has near-zero norm: "
                f"{direction}"
            )
        self._direction = arr / norm
        self._mode = mode

        if ee_vector is None:
            self._ee_vector = np.array([0.0, 0.0, 1.0])
        else:
            ev = np.asarray(ee_vector, dtype=np.float64)
            ev_norm = np.linalg.norm(ev)
            if ev_norm < 1e-9:
                raise ValueError(
                    f"OrientationProjectionFeature: ee_vector has near-zero norm: "
                    f"{ee_vector}"
                )
            self._ee_vector = ev / ev_norm

    def compute(
        self,
        from_entity: int | str,
        to_entity: int | str,
        pcd: np.ndarray,
        ee_pos: np.ndarray,
        finger_positions: dict[str, np.ndarray] | None = None,
        *,
        ee_ori: np.ndarray | None = None,
        gripper_state: float | None = None,
    ) -> float | None:
        if ee_ori is None:
            return None
        R = Rotation.from_rotvec(np.asarray(ee_ori, dtype=np.float64)).as_matrix()
        body_vec_world = R @ self._ee_vector
        if self._mode == "axis":
            return float(np.dot(body_vec_world, self._direction))
        in_plane = (
            body_vec_world - np.dot(body_vec_world, self._direction) * self._direction
        )
        return float(np.linalg.norm(in_plane))
