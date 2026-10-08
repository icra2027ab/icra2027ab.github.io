"""
Tests for policy_abstraction.features.

All tests use only synthetic data (no LIBERO / GPU / network required).
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.features.base import compute_distances
from policy_abstraction.features.centroid import CentroidDistance
from policy_abstraction.features.orientation import OrientationProjectionFeature
from policy_abstraction.features.surface import MinSurfaceDistance
from policy_abstraction.vlm.types import DistancePair, DistanceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pcd(cluster_a_center, cluster_b_center, n=50):
    """Build a minimal [N, 7] point cloud with two instances (seg_id 1 and 2)."""
    rng = np.random.default_rng(0)
    a_xyz = rng.normal(loc=cluster_a_center, scale=0.01, size=(n, 3)).astype(np.float32)
    b_xyz = rng.normal(loc=cluster_b_center, scale=0.01, size=(n, 3)).astype(np.float32)

    a_rest = np.concatenate([np.zeros((n, 3)), np.full((n, 1), 1.0)], axis=1)
    b_rest = np.concatenate([np.zeros((n, 3)), np.full((n, 1), 2.0)], axis=1)

    return np.concatenate(
        [np.hstack([a_xyz, a_rest]), np.hstack([b_xyz, b_rest])], axis=0
    )


# ---------------------------------------------------------------------------
# CentroidDistance
# ---------------------------------------------------------------------------


class TestCentroidDistance:
    def test_known_distance_between_clusters(self):
        """Centroid distance equals Euclidean distance between cluster centres."""
        c_a = [0.0, 0.0, 0.0]
        c_b = [3.0, 4.0, 0.0]  # |c_b - c_a| = 5.0
        pcd = _make_pcd(c_a, c_b)
        ee_pos = np.array([0.0, 0.0, 0.0])

        dist = CentroidDistance().compute(1, 2, pcd, ee_pos)
        assert dist is not None
        assert abs(dist - 5.0) < 0.05  # σ=0.01 → centroid error < 0.05

    def test_ee_to_cluster(self):
        """Distance from 'ee' to cluster 1."""
        c_a = [1.0, 0.0, 0.0]
        pcd = _make_pcd(c_a, [5.0, 0.0, 0.0])
        ee_pos = np.array([0.0, 0.0, 0.0])

        dist = CentroidDistance().compute("ee", 1, pcd, ee_pos)
        assert dist is not None
        assert abs(dist - 1.0) < 0.05

    def test_missing_entity_returns_none(self):
        """If an entity seg_id is absent from the pcd, compute() returns None."""
        pcd = _make_pcd([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        result = CentroidDistance().compute(1, 99, pcd, ee_pos)
        assert result is None

    def test_symmetry(self):
        """Centroid distance is symmetric: d(A→B) == d(B→A)."""
        pcd = _make_pcd([0.0, 0.0, 0.0], [2.0, 1.0, 0.0])
        ee_pos = np.zeros(3)
        feat = CentroidDistance()
        d_ab = feat.compute(1, 2, pcd, ee_pos)
        d_ba = feat.compute(2, 1, pcd, ee_pos)
        assert d_ab is not None and d_ba is not None
        assert abs(d_ab - d_ba) < 1e-10


# ---------------------------------------------------------------------------
# MinSurfaceDistance
# ---------------------------------------------------------------------------


class TestMinSurfaceDistance:
    def test_min_less_than_centroid(self):
        """Min surface distance ≤ centroid distance."""
        pcd = _make_pcd([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        d_cent = CentroidDistance().compute(1, 2, pcd, ee_pos)
        d_surf = MinSurfaceDistance().compute(1, 2, pcd, ee_pos)
        assert d_surf is not None and d_cent is not None
        assert d_surf <= d_cent + 1e-9

    def test_touching_clusters_near_zero(self):
        """Clusters that overlap should have near-zero surface distance."""
        # Both clusters centred at the same point
        pcd = _make_pcd([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        dist = MinSurfaceDistance().compute(1, 2, pcd, ee_pos)
        assert dist is not None
        assert dist < 0.05  # within cluster spread

    def test_missing_entity_returns_none(self):
        """Returns None when an entity has no points in the pcd."""
        pcd = _make_pcd([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        result = MinSurfaceDistance().compute(99, 1, pcd, ee_pos)
        assert result is None

    def test_ee_to_cluster(self):
        """Min distance from EE to cluster."""
        pcd = _make_pcd([1.0, 0.0, 0.0], [5.0, 0.0, 0.0])
        ee_pos = np.array([0.0, 0.0, 0.0])
        dist = MinSurfaceDistance().compute("ee", 1, pcd, ee_pos)
        assert dist is not None
        # Cluster 1 is at ~1.0m away, so min dist < 1.1 with spread σ=0.01
        assert dist < 1.1


# ---------------------------------------------------------------------------
# compute_distances helper
# ---------------------------------------------------------------------------


class TestComputeDistances:
    def test_returns_dict_with_labels(self):
        """compute_distances returns a dict keyed by DistancePair labels."""
        pcd = _make_pcd([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        pairs = [
            DistancePair(from_entity=1, to_entity=2, label="obj1_to_obj2"),
        ]
        result = compute_distances(pairs, CentroidDistance(), pcd, ee_pos)
        assert "obj1_to_obj2" in result
        assert isinstance(result["obj1_to_obj2"], float)

    def test_missing_entity_silently_skipped(self):
        """Pairs with missing entities are not included in the result."""
        pcd = _make_pcd([0.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        pairs = [
            DistancePair(from_entity=1, to_entity=99, label="missing"),
            DistancePair(from_entity=1, to_entity=2, label="present"),
        ]
        result = compute_distances(pairs, CentroidDistance(), pcd, ee_pos)
        assert "missing" not in result
        assert "present" in result


# ---------------------------------------------------------------------------
# Gripper finger entities
# ---------------------------------------------------------------------------


class TestGripperEntities:
    """Tests for gripper_l / gripper_r entity support."""

    @pytest.fixture()
    def finger_positions(self):
        return {
            "gripper_l": np.array([0.1, 0.0, 0.0], dtype=np.float64),
            "gripper_r": np.array([-0.1, 0.0, 0.0], dtype=np.float64),
        }

    def test_gripper_aperture_centroid(self, finger_positions):
        """Centroid distance between gripper_l and gripper_r gives aperture."""
        pcd = _make_pcd([5.0, 0.0, 0.0], [6.0, 0.0, 0.0])  # irrelevant objects
        ee_pos = np.zeros(3)
        dist = CentroidDistance().compute(
            "gripper_l", "gripper_r", pcd, ee_pos, finger_positions
        )
        assert dist is not None
        assert abs(dist - 0.2) < 1e-9  # |0.1 - (-0.1)| = 0.2

    def test_gripper_to_object(self, finger_positions):
        """Centroid distance from a finger to a scene object."""
        pcd = _make_pcd([1.0, 0.0, 0.0], [5.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        dist = CentroidDistance().compute("gripper_l", 1, pcd, ee_pos, finger_positions)
        assert dist is not None
        # gripper_l is at [0.1, 0, 0], cluster 1 centroid ~[1, 0, 0]
        assert abs(dist - 0.9) < 0.05

    def test_gripper_without_finger_positions_returns_none(self):
        """Gripper entities without finger_positions return None."""
        pcd = _make_pcd([0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        dist = CentroidDistance().compute("gripper_l", 1, pcd, ee_pos)
        assert dist is None

    def test_compute_distances_with_gripper(self, finger_positions):
        """compute_distances correctly dispatches gripper pairs.

        gripper_l<->gripper_r pairs are auto-promoted to GripperApertureFeature
        so ref/live distances match the gripper-scalar semantics used by the
        JAX guidance loss. The raw scalar ``gripper_state`` is returned.
        """
        pcd = _make_pcd([1.0, 0.0, 0.0], [2.0, 0.0, 0.0])
        ee_pos = np.zeros(3)
        pairs = [
            DistancePair(
                from_entity="gripper_l",
                to_entity="gripper_r",
                label="aperture",
            ),
            DistancePair(
                from_entity="gripper_r",
                to_entity=1,
                label="finger_r_to_obj",
            ),
        ]
        result = compute_distances(
            pairs,
            CentroidDistance(),
            pcd,
            ee_pos,
            finger_positions,
            gripper_state=0.04,
        )
        assert "aperture" in result
        assert "finger_r_to_obj" in result
        assert abs(result["aperture"] - 0.04) < 1e-9


# ---------------------------------------------------------------------------
# OrientationProjectionFeature — gripper z-axis semantics
# ---------------------------------------------------------------------------


class TestOrientationProjection:
    """With ``ee_ori = 0``, the world-from-EE rotation is the identity so the
    gripper body z-axis is exactly world ``[0, 0, 1]``."""

    def test_axis_mode_identity_returns_cosine_with_axis(self):
        feat = OrientationProjectionFeature([0, 0, 1], mode="axis")
        ee_ori = np.zeros(3)
        val = feat.compute("ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=ee_ori)
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_axis_mode_approach_from_above(self):
        """``axis=[0,0,-1]`` reports +1 when the gripper points straight down."""
        from scipy.spatial.transform import Rotation

        feat = OrientationProjectionFeature([0, 0, -1], mode="axis")
        # Rotate 180° around X: gripper z-axis now points in -world-z.
        ee_ori = Rotation.from_euler("x", np.pi).as_rotvec()
        val = feat.compute("ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=ee_ori)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_plane_mode_zero_when_gripper_aligned_with_normal(self):
        feat = OrientationProjectionFeature([0, 0, 1], mode="plane")
        val = feat.compute(
            "ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=np.zeros(3)
        )
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_plane_mode_one_when_gripper_horizontal(self):
        """Rotating 90° about X drops the gripper z-axis into the XY plane,
        so its in-plane magnitude (w.r.t. plane normal = world z) is 1."""
        from scipy.spatial.transform import Rotation

        feat = OrientationProjectionFeature([0, 0, 1], mode="plane")
        ee_ori = Rotation.from_euler("x", np.pi / 2).as_rotvec()
        val = feat.compute("ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=ee_ori)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_returns_none_when_ee_ori_missing(self):
        feat = OrientationProjectionFeature([0, 0, 1], mode="axis")
        assert feat.compute("ee", "ee", np.zeros((0, 7)), np.zeros(3)) is None

    def test_dispatch_through_compute_distances_uses_axis_mode(self):
        """``compute_distances`` should dispatch ORIENTATION+axis to the axis mode."""
        pair = DistancePair(
            from_entity="ee",
            to_entity="ee",
            label="gripper_down",
            distance_type=DistanceType.ORIENTATION,
            axis=[0, 0, 1],
        )
        result = compute_distances(
            [pair],
            CentroidDistance(),
            np.zeros((0, 7)),
            np.zeros(3),
            ee_ori=np.zeros(3),
        )
        assert result["gripper_down"] == pytest.approx(1.0, abs=1e-9)

    def test_dispatch_through_compute_distances_uses_plane_mode(self):
        """``compute_distances`` should dispatch ORIENTATION+plane to the plane mode."""
        pair = DistancePair(
            from_entity="ee",
            to_entity="ee",
            label="gripper_tilt",
            distance_type=DistanceType.ORIENTATION,
            plane=[0, 0, 1],
        )
        result = compute_distances(
            [pair],
            CentroidDistance(),
            np.zeros((0, 7)),
            np.zeros(3),
            ee_ori=np.zeros(3),
        )
        assert result["gripper_tilt"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# OrientationProjectionFeature — configurable ee_vector
# ---------------------------------------------------------------------------


class TestOrientationEeVector:
    """Tests for the configurable ``ee_vector`` parameter."""

    def test_x_axis_ee_vector_identity_rotation(self):
        """With identity rotation, ee_vector=[1,0,0] → body x-axis is [1,0,0].
        Projected onto axis=[1,0,0] gives 1.0."""
        feat = OrientationProjectionFeature([1, 0, 0], mode="axis", ee_vector=[1, 0, 0])
        val = feat.compute(
            "ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=np.zeros(3)
        )
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_y_axis_ee_vector_identity_rotation(self):
        """With identity rotation, ee_vector=[0,1,0] → body y-axis is [0,1,0].
        Projected onto axis=[0,1,0] gives 1.0."""
        feat = OrientationProjectionFeature([0, 1, 0], mode="axis", ee_vector=[0, 1, 0])
        val = feat.compute(
            "ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=np.zeros(3)
        )
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_x_axis_ee_vector_orthogonal_to_z_ref(self):
        """With identity rotation, ee_vector=[1,0,0] projected onto axis=[0,0,1]
        gives 0.0 (orthogonal)."""
        feat = OrientationProjectionFeature([0, 0, 1], mode="axis", ee_vector=[1, 0, 0])
        val = feat.compute(
            "ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=np.zeros(3)
        )
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_plane_mode_with_x_axis_ee_vector(self):
        """With identity rotation, ee_vector=[1,0,0] is fully in the XY-plane
        (plane normal=[0,0,1]), so in-plane magnitude = 1.0."""
        feat = OrientationProjectionFeature(
            [0, 0, 1], mode="plane", ee_vector=[1, 0, 0]
        )
        val = feat.compute(
            "ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=np.zeros(3)
        )
        assert val == pytest.approx(1.0, abs=1e-9)

    def test_default_ee_vector_matches_z_axis(self):
        """ee_vector=None should behave identically to ee_vector=[0,0,1]."""
        feat_default = OrientationProjectionFeature([0, 0, 1], mode="axis")
        feat_explicit = OrientationProjectionFeature(
            [0, 0, 1], mode="axis", ee_vector=[0, 0, 1]
        )
        ee_ori = np.array([0.1, 0.2, 0.3])
        pcd = np.zeros((0, 7))
        ee_pos = np.zeros(3)
        val_default = feat_default.compute("ee", "ee", pcd, ee_pos, ee_ori=ee_ori)
        val_explicit = feat_explicit.compute("ee", "ee", pcd, ee_pos, ee_ori=ee_ori)
        assert val_default == pytest.approx(val_explicit, abs=1e-12)

    def test_rotated_x_axis_ee_vector(self):
        """90° rotation around Z sends body x-axis [1,0,0] → world [0,1,0].
        Projected onto axis=[0,1,0] gives 1.0."""
        from scipy.spatial.transform import Rotation

        feat = OrientationProjectionFeature([0, 1, 0], mode="axis", ee_vector=[1, 0, 0])
        ee_ori = Rotation.from_euler("z", np.pi / 2).as_rotvec()
        val = feat.compute("ee", "ee", np.zeros((0, 7)), np.zeros(3), ee_ori=ee_ori)
        assert val == pytest.approx(1.0, abs=1e-6)

    def test_dispatch_with_ee_vector(self):
        """compute_distances routes ee_vector from DistancePair correctly."""
        pair = DistancePair(
            from_entity="ee",
            to_entity="ee",
            label="x_axis_up",
            distance_type=DistanceType.ORIENTATION,
            axis=[0, 0, 1],
            ee_vector=[1, 0, 0],
        )
        result = compute_distances(
            [pair],
            CentroidDistance(),
            np.zeros((0, 7)),
            np.zeros(3),
            ee_ori=np.zeros(3),
        )
        # Identity rotation: body x=[1,0,0] dot [0,0,1] = 0
        assert result["x_axis_up"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SubtrajConstraint — 3-arg lambda signature
# ---------------------------------------------------------------------------


class TestSubtrajConstraintCompile:
    """Constraint lambdas must accept three positional arguments:
    ``ee`` (position), ``keypoints`` (centroid dict), and ``ee_R`` (3x3
    world-from-EE rotation matrix).  Older 2-arg lambdas are rejected at
    runtime."""

    def test_three_arg_lambda_compiles_and_runs(self):
        from policy_abstraction.vlm.types import SubtrajConstraint

        c = SubtrajConstraint(
            kind="sub_goal",
            subtraj_idx=0,
            label="gripper_down",
            description="",
            expression="lambda ee, keypoints, ee_R: 0.5 - ee_R[2, 2]",
        )
        fn = c.compile()
        # Identity rotation: ee_R[2, 2] = 1.0 → cost = 0.5 - 1.0 = -0.5 (satisfied).
        cost = float(fn(np.zeros(3), {}, np.eye(3)))
        assert cost == pytest.approx(-0.5, abs=1e-9)

    def test_two_arg_lambda_raises_on_invocation(self):
        from policy_abstraction.vlm.types import SubtrajConstraint

        c = SubtrajConstraint(
            kind="sub_goal",
            subtraj_idx=0,
            label="legacy",
            description="",
            expression="lambda ee, keypoints: 0.0",
        )
        fn = c.compile()
        with pytest.raises(TypeError):
            fn(np.zeros(3), {}, np.eye(3))
