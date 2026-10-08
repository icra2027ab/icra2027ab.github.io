"""Tests for spatial consistency scoring."""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.utils.spatial_consistency import (
    _fit_affine,
    score_spatial_consistency,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _apply_affine(pts: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
    return pts @ A.T + b


def _random_affine_2d(rng: np.random.Generator):
    A = rng.normal(size=(2, 2)) * 0.5 + np.eye(2)
    b = rng.normal(size=(2,)) * 10
    return A, b


def _random_affine_3d(rng: np.random.Generator):
    A = rng.normal(size=(3, 3)) * 0.3 + np.eye(3)
    b = rng.normal(size=(3,)) * 5
    return A, b


# ── _fit_affine ──────────────────────────────────────────────────────────────


class TestFitAffine:
    def test_identity(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64)
        A, b = _fit_affine(pts, pts)
        np.testing.assert_allclose(A, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(b, 0, atol=1e-10)

    def test_translation_only(self):
        src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        dst = src + np.array([3.0, -7.0])
        A, b = _fit_affine(src, dst)
        np.testing.assert_allclose(A, np.eye(2), atol=1e-10)
        np.testing.assert_allclose(b, [3.0, -7.0], atol=1e-10)

    def test_known_affine_2d(self):
        rng = np.random.default_rng(42)
        A_true, b_true = _random_affine_2d(rng)
        src = rng.normal(size=(10, 2))
        dst = _apply_affine(src, A_true, b_true)
        A_est, b_est = _fit_affine(src, dst)
        np.testing.assert_allclose(A_est, A_true, atol=1e-10)
        np.testing.assert_allclose(b_est, b_true, atol=1e-10)

    def test_known_affine_3d(self):
        rng = np.random.default_rng(123)
        A_true, b_true = _random_affine_3d(rng)
        src = rng.normal(size=(10, 3))
        dst = _apply_affine(src, A_true, b_true)
        A_est, b_est = _fit_affine(src, dst)
        np.testing.assert_allclose(A_est, A_true, atol=1e-10)
        np.testing.assert_allclose(b_est, b_true, atol=1e-10)


# ── score_spatial_consistency ────────────────────────────────────────────────


class TestSpatialConsistency:
    def test_perfect_affine_all_high(self):
        """All points follow the same affine -> all scores should be 1.0."""
        rng = np.random.default_rng(0)
        A, b = _random_affine_2d(rng)
        ref = rng.normal(size=(15, 2)) * 50
        obs = _apply_affine(ref, A, b)
        scores = score_spatial_consistency(ref, obs)
        assert scores.shape == (15,)
        np.testing.assert_allclose(scores, 1.0, atol=1e-6)

    def test_outliers_get_low_scores(self):
        """Inject outliers; they should score much lower than inliers."""
        rng = np.random.default_rng(7)
        A, b = _random_affine_2d(rng)
        N = 20
        ref = rng.normal(size=(N, 2)) * 50
        obs = _apply_affine(ref, A, b)

        # Corrupt the last 3 points with large random offsets
        obs[-3:] += rng.normal(size=(3, 2)) * 200

        scores = score_spatial_consistency(ref, obs)
        # Inlier scores (indices 3:-3, excluding anchors) should be high
        assert np.all(scores[3:-3] > 0.9)
        # Outlier scores should be low
        assert np.all(scores[-3:] < 0.3)

    def test_3d_points(self):
        """Works with 3-D correspondences."""
        rng = np.random.default_rng(99)
        A, b = _random_affine_3d(rng)
        ref = rng.normal(size=(12, 3)) * 10
        obs = _apply_affine(ref, A, b)

        # Corrupt last point
        obs[-1] += 100.0

        scores = score_spatial_consistency(ref, obs, n_anchors=4)
        assert scores.shape == (12,)
        # Anchors are 1.0
        np.testing.assert_allclose(scores[:4], 1.0)
        # Good points score high
        assert np.all(scores[4:-1] > 0.9)
        # Outlier scores low
        assert scores[-1] < 0.1

    def test_custom_n_anchors(self):
        rng = np.random.default_rng(5)
        A, b = _random_affine_2d(rng)
        ref = rng.normal(size=(10, 2)) * 30
        obs = _apply_affine(ref, A, b)
        scores = score_spatial_consistency(ref, obs, n_anchors=5)
        np.testing.assert_allclose(scores[:5], 1.0)
        assert np.all(scores[5:] > 0.9)

    def test_minimum_points(self):
        """With exactly n_anchors points, all should score 1.0."""
        ref = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float64)
        obs = ref * 2 + 3  # simple affine
        scores = score_spatial_consistency(ref, obs)
        np.testing.assert_allclose(scores, 1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="Shape mismatch"):
            score_spatial_consistency(np.zeros((5, 2)), np.zeros((5, 3)))

    def test_too_few_points_raises(self):
        with pytest.raises(ValueError, match="Not enough points"):
            score_spatial_consistency(np.zeros((2, 2)), np.zeros((2, 2)))

    def test_too_few_anchors_raises(self):
        with pytest.raises(ValueError, match="Need at least"):
            score_spatial_consistency(np.zeros((5, 3)), np.zeros((5, 3)), n_anchors=2)
