"""
Tests for policy_abstraction.utils.segmentation.

All tests use only synthetic data (no LIBERO / GPU / network required).
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.utils.segmentation import (
    build_segmented_pointcloud,
    colorize_segmentation,
    find_interacting_instance_ids,
    merge_pointclouds,
    remove_statistical_outliers,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rgbd_seg(H=16, W=16, depth_val=1.0):
    """Return (rgb, depth, seg) for a simple synthetic scene."""
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    rgb[: H // 2, :, 0] = 200  # top half red → seg_id 1
    rgb[H // 2 :, :, 1] = 200  # bottom half green → seg_id 2

    depth = np.full((H, W), depth_val, dtype=np.float64)

    seg = np.zeros((H, W), dtype=np.int32)
    seg[: H // 2, :] = 1
    seg[H // 2 :, :] = 2

    return rgb, depth, seg


def _make_identity_K(H=16, W=16):
    """Simple pinhole K centred at (W/2, H/2)."""
    return np.array(
        [[200.0, 0.0, W / 2], [0.0, 200.0, H / 2], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# build_segmented_pointcloud
# ---------------------------------------------------------------------------


class TestBuildSegmentedPointcloud:
    def test_output_shape_and_dtype(self):
        """Output is [M, 7] float32 with M > 0."""
        H, W = 16, 16
        rgb, depth, seg = _make_rgbd_seg(H, W)
        K = _make_identity_K(H, W)
        E = np.eye(4)

        pcd = build_segmented_pointcloud(rgb, depth, seg, K, E)
        assert pcd.ndim == 2
        assert pcd.shape[1] == 7
        assert pcd.shape[0] > 0
        assert pcd.dtype == np.float32

    def test_seg_ids_present(self):
        """Output contains both seg_id 1 and 2 from the test scene."""
        H, W = 16, 16
        rgb, depth, seg = _make_rgbd_seg(H, W)
        K = _make_identity_K(H, W)
        E = np.eye(4)

        pcd = build_segmented_pointcloud(rgb, depth, seg, K, E)
        unique_ids = set(pcd[:, 6].astype(int))
        assert 1 in unique_ids
        assert 2 in unique_ids

    def test_channel_dim_depth_squeezed(self):
        """Depth with shape [H, W, 1] is accepted (squeezed internally)."""
        H, W = 8, 8
        rgb, depth, seg = _make_rgbd_seg(H, W)
        K = _make_identity_K(H, W)
        E = np.eye(4)

        pcd_3d = build_segmented_pointcloud(rgb, depth[..., np.newaxis], seg, K, E)
        pcd_2d = build_segmented_pointcloud(rgb, depth, seg, K, E)
        assert pcd_3d.shape == pcd_2d.shape


# ---------------------------------------------------------------------------
# merge_pointclouds
# ---------------------------------------------------------------------------


class TestMergePointclouds:
    def test_concatenation(self, synthetic_pcd):
        """Merging a cloud with itself doubles the row count."""
        merged = merge_pointclouds(synthetic_pcd, synthetic_pcd)
        assert merged.shape[0] == 2 * synthetic_pcd.shape[0]
        assert merged.shape[1] == 7

    def test_empty_input(self):
        """Merging no non-empty cloud returns empty (0, 7) array."""
        empty = np.empty((0, 7), dtype=np.float32)
        result = merge_pointclouds(empty, empty)
        assert result.shape == (0, 7)

    def test_single_cloud_passthrough(self, synthetic_pcd):
        """Merging a single cloud returns it unchanged."""
        result = merge_pointclouds(synthetic_pcd)
        np.testing.assert_array_equal(result, synthetic_pcd)


# ---------------------------------------------------------------------------
# remove_statistical_outliers
# ---------------------------------------------------------------------------


class TestRemoveStatisticalOutliers:
    def test_removes_obvious_outlier(self, synthetic_pcd):
        """The point at (100, 100, 100) should be removed."""
        before = synthetic_pcd.shape[0]
        filtered = remove_statistical_outliers(
            synthetic_pcd, nb_neighbors=20, std_ratio=2.0
        )
        # The outlier should be gone
        assert filtered.shape[0] < before
        assert not np.any(filtered[:, 0] > 10.0)

    def test_preserves_inliers(self, synthetic_pcd):
        """Most inlier points should survive outlier removal."""
        filtered = remove_statistical_outliers(
            synthetic_pcd, nb_neighbors=20, std_ratio=2.0
        )
        # 400 inlieri points, 1 outlier → at least 350 should survive
        assert filtered.shape[0] >= 350

    def test_small_cloud_passthrough(self):
        """When pcd has ≤ nb_neighbors rows, it is returned unchanged."""
        tiny = np.ones((5, 7), dtype=np.float32)
        result = remove_statistical_outliers(tiny, nb_neighbors=20)
        np.testing.assert_array_equal(result, tiny)


# ---------------------------------------------------------------------------
# find_interacting_instance_ids
# ---------------------------------------------------------------------------


class TestFindInteractingInstanceIds:
    def _make_pcd_with_two_instances(self):
        """Two clusters: instance 1 near origin, instance 2 far away."""
        rng = np.random.default_rng(1)
        n = 60

        c1_xyz = rng.normal([0.0, 0.0, 0.0], 0.01, (n, 3)).astype(np.float32)
        c2_xyz = rng.normal([5.0, 5.0, 5.0], 0.01, (n, 3)).astype(np.float32)

        c1 = np.concatenate([c1_xyz, np.zeros((n, 3)), np.full((n, 1), 1.0)], axis=1)
        c2 = np.concatenate([c2_xyz, np.zeros((n, 3)), np.full((n, 1), 2.0)], axis=1)
        return np.concatenate([c1, c2], axis=0)

    def test_nearby_instance_detected(self):
        """Instance 1 (near origin) is returned when finger tip is at origin."""
        pcd = self._make_pcd_with_two_instances()
        finger_tips = np.array([[0.0, 0.0, 0.0]])  # right on top of cluster 1
        result = find_interacting_instance_ids(
            pcd, finger_tips, threshold_m=0.1, min_points=30
        )
        assert 1 in result

    def test_far_instance_not_detected(self):
        """Instance 2 (at [5,5,5]) is not returned when finger is at origin."""
        pcd = self._make_pcd_with_two_instances()
        finger_tips = np.array([[0.0, 0.0, 0.0]])
        result = find_interacting_instance_ids(
            pcd, finger_tips, threshold_m=0.1, min_points=30
        )
        assert 2 not in result

    def test_min_points_filter(self):
        """Requiring more points than are nearby excludes the instance."""
        pcd = self._make_pcd_with_two_instances()
        finger_tips = np.array([[0.0, 0.0, 0.0]])
        # With min_points=1000 (more than any cluster), nothing should be returned
        result = find_interacting_instance_ids(
            pcd, finger_tips, threshold_m=0.1, min_points=1000
        )
        assert len(result) == 0

    def test_background_zero_excluded(self):
        """seg_id=0 (background) is never returned."""
        n = 60
        rng = np.random.default_rng(2)
        xyz = rng.normal([0.0, 0.0, 0.0], 0.01, (n, 3)).astype(np.float32)
        pcd = np.concatenate([xyz, np.zeros((n, 3)), np.zeros((n, 1))], axis=1)
        finger_tips = np.array([[0.0, 0.0, 0.0]])
        result = find_interacting_instance_ids(
            pcd, finger_tips, threshold_m=0.1, min_points=1
        )
        assert 0 not in result

    def test_empty_pcd_returns_empty_set(self):
        """Empty point cloud → empty result."""
        pcd = np.empty((0, 7), dtype=np.float32)
        result = find_interacting_instance_ids(pcd, np.array([[0.0, 0.0, 0.0]]))
        assert result == set()


# ---------------------------------------------------------------------------
# colorize_segmentation (visualization)
# ---------------------------------------------------------------------------


class TestColorizeSegmentation:
    def test_output_shape_and_dtype(self):
        """Output is [H, W, 3] uint8."""
        seg = np.array([[1, 2], [0, 1]], dtype=np.int32)
        out = colorize_segmentation(seg, {1, 2})
        assert out.shape == (2, 2, 3)
        assert out.dtype == np.uint8

    def test_background_has_bg_gray(self):
        """Background pixels (seg_id not in interacting_ids) get bg_gray colour."""
        seg = np.array([[0, 1]], dtype=np.int32)
        out = colorize_segmentation(seg, {1}, bg_gray=50)
        np.testing.assert_array_equal(out[0, 0], [50, 50, 50])

    def test_empty_interacting_ids(self):
        """With no interacting IDs everything becomes background."""
        seg = np.array([[1, 2], [3, 4]], dtype=np.int32)
        out = colorize_segmentation(seg, set(), bg_gray=30)
        assert (out == 30).all()
