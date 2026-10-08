"""Tests for AxisProjectionDistance and PlaneProjectionDistance."""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.features.axis import AxisProjectionDistance
from policy_abstraction.features.plane import PlaneProjectionDistance

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _two_cluster_pcd(c1: list[float], c2: list[float], n: int = 30) -> np.ndarray:
    """Return (2n, 7) pcd: cluster 1 at c1 (seg_id=1), cluster 2 at c2 (seg_id=2)."""
    rng = np.random.default_rng(0)
    pts1 = rng.normal(loc=c1, scale=0.001, size=(n, 3)).astype(np.float32)
    pts2 = rng.normal(loc=c2, scale=0.001, size=(n, 3)).astype(np.float32)
    ids1 = np.full((n, 1), 1.0, dtype=np.float32)
    ids2 = np.full((n, 1), 2.0, dtype=np.float32)
    rgb = np.zeros((n, 3), dtype=np.float32)
    c1_block = np.concatenate([pts1, rgb, ids1], axis=1)
    c2_block = np.concatenate([pts2, rgb, ids2], axis=1)
    return np.concatenate([c1_block, c2_block], axis=0)


_DUMMY_EE = np.array([0.5, 0.5, 0.5])


# ---------------------------------------------------------------------------
# AxisProjectionDistance
# ---------------------------------------------------------------------------


class TestAxisProjectionDistance:
    """Tests for AxisProjectionDistance."""

    def test_z_axis_vertical_separation(self):
        """Cluster at z=0 and cluster at z=5 → result ≈ 5.0 along z-axis."""
        pcd = _two_cluster_pcd([0, 0, 0], [0, 0, 5])
        feat = AxisProjectionDistance([0, 0, 1])
        result = feat.compute(1, 2, pcd, _DUMMY_EE)
        assert result is not None
        np.testing.assert_allclose(result, 5.0, atol=0.01)

    def test_signed_negative_when_reversed(self):
        """Swapping from/to entities flips the sign."""
        pcd = _two_cluster_pcd([0, 0, 0], [0, 0, 5])
        feat = AxisProjectionDistance([0, 0, 1])
        result = feat.compute(2, 1, pcd, _DUMMY_EE)
        assert result is not None
        np.testing.assert_allclose(result, -5.0, atol=0.01)

    def test_non_unit_axis_same_as_unit(self):
        """A non-unit axis (scaled) produces the same result as the unit version."""
        pcd = _two_cluster_pcd([0, 0, 0], [0, 0, 3])
        result_unit = AxisProjectionDistance([0, 0, 1]).compute(1, 2, pcd, _DUMMY_EE)
        result_scaled = AxisProjectionDistance([0, 0, 10]).compute(1, 2, pcd, _DUMMY_EE)
        assert result_unit is not None and result_scaled is not None
        np.testing.assert_allclose(result_unit, result_scaled, atol=1e-6)

    def test_perpendicular_axis_is_near_zero(self):
        """Clusters separated only along x; y-axis projection ≈ 0."""
        pcd = _two_cluster_pcd([0, 0, 0], [5, 0, 0])
        feat = AxisProjectionDistance([0, 1, 0])
        result = feat.compute(1, 2, pcd, _DUMMY_EE)
        assert result is not None
        np.testing.assert_allclose(result, 0.0, atol=0.01)

    def test_zero_axis_raises_value_error(self):
        """Constructing with a zero axis vector should raise ValueError."""
        with pytest.raises(ValueError, match="near-zero norm"):
            AxisProjectionDistance([0, 0, 0])

    def test_missing_entity_returns_none(self):
        """If an entity seg_id is not present in the pcd, return None."""
        pcd = _two_cluster_pcd([0, 0, 0], [0, 0, 5])
        feat = AxisProjectionDistance([0, 0, 1])
        result = feat.compute(1, 99, pcd, _DUMMY_EE)
        assert result is None

    def test_ee_as_from_entity(self):
        """Using 'ee' as from_entity computes projection from ee_pos to cluster centroid."""
        ee_pos = np.array([0.0, 0.0, 0.0])
        pcd = _two_cluster_pcd([0, 0, 0], [0, 0, 5])  # cluster 2 at z=5
        feat = AxisProjectionDistance([0, 0, 1])
        result = feat.compute("ee", 2, pcd, ee_pos)
        assert result is not None
        np.testing.assert_allclose(result, 5.0, atol=0.05)

    def test_diagonal_axis(self):
        """[1,1,0]/sqrt(2) axis captures diagonal projection correctly."""
        # Cluster separation: [3,4,0] — projection onto [1,1,0]/√2 = (3+4)/√2 ≈ 4.95
        pcd = _two_cluster_pcd([0, 0, 0], [3, 4, 0])
        feat = AxisProjectionDistance([1, 1, 0])
        result = feat.compute(1, 2, pcd, _DUMMY_EE)
        assert result is not None
        expected = (3 + 4) / np.sqrt(2)
        np.testing.assert_allclose(result, expected, atol=0.05)


# ---------------------------------------------------------------------------
# PlaneProjectionDistance
# ---------------------------------------------------------------------------


class TestPlaneProjectionDistance:
    """Tests for PlaneProjectionDistance."""

    def test_horizontal_distance_z_normal(self):
        """Clusters at [3,0,0] and [0,4,0] — XY-plane distance = 5.0."""
        pcd = _two_cluster_pcd([0, 0, 0], [3, 4, 0])
        feat = PlaneProjectionDistance([0, 0, 1])
        result = feat.compute(1, 2, pcd, _DUMMY_EE)
        assert result is not None
        np.testing.assert_allclose(result, 5.0, atol=0.05)

    def test_vertical_separation_removed(self):
        """Clusters separated only in z with z-normal → XY projection ≈ 0."""
        pcd = _two_cluster_pcd([0, 0, 0], [0, 0, 5])
        feat = PlaneProjectionDistance([0, 0, 1])
        result = feat.compute(1, 2, pcd, _DUMMY_EE)
        assert result is not None
        np.testing.assert_allclose(result, 0.0, atol=0.05)

    def test_non_unit_normal_same_as_unit(self):
        """Scaled normal produces the same result as the unit normal."""
        pcd = _two_cluster_pcd([0, 0, 0], [3, 4, 0])
        result_unit = PlaneProjectionDistance([0, 0, 1]).compute(1, 2, pcd, _DUMMY_EE)
        result_scaled = PlaneProjectionDistance([0, 0, 100]).compute(
            1, 2, pcd, _DUMMY_EE
        )
        assert result_unit is not None and result_scaled is not None
        np.testing.assert_allclose(result_unit, result_scaled, atol=1e-5)

    def test_zero_normal_raises_value_error(self):
        """Constructing with a zero normal vector should raise ValueError."""
        with pytest.raises(ValueError, match="near-zero norm"):
            PlaneProjectionDistance([0, 0, 0])

    def test_missing_entity_returns_none(self):
        """If an entity seg_id is not present in the pcd, return None."""
        pcd = _two_cluster_pcd([0, 0, 0], [3, 4, 0])
        feat = PlaneProjectionDistance([0, 0, 1])
        result = feat.compute(1, 99, pcd, _DUMMY_EE)
        assert result is None

    def test_always_non_negative(self):
        """Result is always ≥ 0 regardless of entity order."""
        pcd = _two_cluster_pcd([0, 0, 0], [3, 4, 0])
        feat = PlaneProjectionDistance([0, 0, 1])
        r1 = feat.compute(1, 2, pcd, _DUMMY_EE)
        r2 = feat.compute(2, 1, pcd, _DUMMY_EE)
        assert r1 is not None and r2 is not None
        assert r1 >= 0
        assert r2 >= 0

    def test_symmetric(self):
        """Swapping from/to entities gives the same magnitude."""
        pcd = _two_cluster_pcd([0, 0, 0], [3, 4, 2])
        feat = PlaneProjectionDistance([0, 0, 1])
        r1 = feat.compute(1, 2, pcd, _DUMMY_EE)
        r2 = feat.compute(2, 1, pcd, _DUMMY_EE)
        assert r1 is not None and r2 is not None
        np.testing.assert_allclose(r1, r2, atol=1e-6)

    def test_x_normal_removes_x_component(self):
        """x-normal plane removes x-separation and keeps yz in-plane distance."""
        # Separation: [5, 3, 4] → in-plane (yz) distance = sqrt(3²+4²) = 5.0
        pcd = _two_cluster_pcd([0, 0, 0], [5, 3, 4])
        feat = PlaneProjectionDistance([1, 0, 0])
        result = feat.compute(1, 2, pcd, _DUMMY_EE)
        assert result is not None
        np.testing.assert_allclose(result, 5.0, atol=0.05)
