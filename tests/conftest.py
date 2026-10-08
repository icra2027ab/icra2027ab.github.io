"""
Shared pytest fixtures for the policy_abstraction test suite.

Provides:
  - synthetic_pcd          – [N, 7] float32 point cloud with two labeled clusters
  - pinhole_camera_params  – simple K + E matrices for projection round-trips
  - ee_pose                – (ee_pos, ee_ori, gripper_states) for T=5 timesteps
  - simple_pcd             – [N, 7] float32 pcd with two clusters at known centroids
  - linear_ref_traj        – (T=20, n_pairs=2) linearly decreasing reference trajectory
  - distance_pair_factory  – callable fixture for constructing DistancePair objects
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def pinhole_camera_params():
    """Return (K, E) for a simple pinhole camera at the world origin.

    K: fx=fy=500, cx=320, cy=240  (for a 640×480 image).
    E: identity – camera frame equals world frame, looking along +z.
    """
    K = np.array(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    E = np.eye(4, dtype=np.float64)
    return K, E


@pytest.fixture
def synthetic_pcd():
    """Return a [N, 7] float32 point cloud with two labeled clusters.

    Cluster 1 (seg_id=1): 200 points centred at (0.5, 0.5, 1.0), σ=0.02 m.
    Cluster 2 (seg_id=2): 200 points centred at (2.0, 2.0, 1.0), σ=0.02 m.
    One outlier at (100, 100, 100) with seg_id=1.
    """
    rng = np.random.default_rng(42)
    n = 200

    c1_xyz = rng.normal(loc=[0.5, 0.5, 1.0], scale=0.02, size=(n, 3)).astype(np.float32)
    c2_xyz = rng.normal(loc=[2.0, 2.0, 1.0], scale=0.02, size=(n, 3)).astype(np.float32)

    c1_rgb = np.tile([255, 0, 0], (n, 1)).astype(np.float32)
    c2_rgb = np.tile([0, 0, 255], (n, 1)).astype(np.float32)

    c1_ids = np.full((n, 1), 1.0, dtype=np.float32)
    c2_ids = np.full((n, 1), 2.0, dtype=np.float32)

    c1 = np.concatenate([c1_xyz, c1_rgb, c1_ids], axis=1)
    c2 = np.concatenate([c2_xyz, c2_rgb, c2_ids], axis=1)

    # One obvious outlier in cluster 1
    outlier = np.array([[100.0, 100.0, 100.0, 255.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    return np.concatenate([c1, c2, outlier], axis=0)


@pytest.fixture
def ee_pose():
    """Return (ee_pos, ee_ori, gripper_states) for T=5 timesteps.

    ee_pos:         (5, 3) – positions advancing along x from 0 to 0.4 m.
    ee_ori:         (5, 3) – all zero (identity rotation).
    gripper_states: (5, 2) – [q1=0.02, q2=-0.02] (half-open).
    """
    T = 5
    ee_pos = np.zeros((T, 3), dtype=np.float64)
    ee_pos[:, 0] = np.linspace(0.0, 0.4, T)

    ee_ori = np.zeros((T, 3), dtype=np.float64)

    q1 = np.full(T, 0.02)
    q2 = np.full(T, -0.02)
    gripper_states = np.column_stack([q1, q2])

    return ee_pos, ee_ori, gripper_states


@pytest.fixture
def simple_pcd():
    """Return a [N, 7] float32 point cloud with two clusters at known centroids.

    Cluster 1 (seg_id=1): 30 points centred at (0.0, 0.0, 0.0), σ=0.001 m.
    Cluster 2 (seg_id=2): 30 points centred at (3.0, 4.0, 0.0), σ=0.001 m.
    Centroid separation = 5.0 m in XY-plane.
    """
    rng = np.random.default_rng(7)
    n = 30

    c1_xyz = rng.normal(loc=[0.0, 0.0, 0.0], scale=0.001, size=(n, 3)).astype(
        np.float32
    )
    c2_xyz = rng.normal(loc=[3.0, 4.0, 0.0], scale=0.001, size=(n, 3)).astype(
        np.float32
    )

    c1_rgb = np.zeros((n, 3), dtype=np.float32)
    c2_rgb = np.zeros((n, 3), dtype=np.float32)

    c1_ids = np.full((n, 1), 1.0, dtype=np.float32)
    c2_ids = np.full((n, 1), 2.0, dtype=np.float32)

    c1 = np.concatenate([c1_xyz, c1_rgb, c1_ids], axis=1)
    c2 = np.concatenate([c2_xyz, c2_rgb, c2_ids], axis=1)

    return np.concatenate([c1, c2], axis=0)


@pytest.fixture
def linear_ref_traj():
    """Return a (T=20, n_pairs=2) float32 reference trajectory.

    Column 0 decreases linearly from 1.0 to 0.0.
    Column 1 decreases linearly from 2.0 to 0.0.
    """
    T = 20
    col0 = np.linspace(1.0, 0.0, T, dtype=np.float32)
    col1 = np.linspace(2.0, 0.0, T, dtype=np.float32)
    return np.stack([col0, col1], axis=1)


@pytest.fixture
def distance_pair_factory():
    """Return a factory for DistancePair objects.

    Usage::

        def test_foo(distance_pair_factory):
            pair = distance_pair_factory("my_label", from_entity="ee", to_entity=1)
    """
    from policy_abstraction.vlm.types import DistancePair, DistanceType

    def _make(
        label: str,
        from_entity: int | str = "ee",
        to_entity: int | str = 1,
        dist_type: DistanceType = DistanceType.CENTROID,
        axis: list[float] | None = None,
        plane: list[float] | None = None,
        ee_vector: list[float] | None = None,
        subtraj_idx: int | None = None,
    ) -> DistancePair:
        return DistancePair(
            from_entity=from_entity,
            to_entity=to_entity,
            label=label,
            distance_type=dist_type,
            axis=axis,
            plane=plane,
            ee_vector=ee_vector,
            subtraj_idx=subtraj_idx,
        )

    return _make
