from __future__ import annotations

import numpy as np
from open_perception.reconstruction.rgbd import (
    mask_to_pointcloud,
    remove_statistical_outliers,
)


def test_pointcloud_count_matches_mask(synthetic_depth, identity_intrinsics):
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:4, 0:4] = True
    pts = mask_to_pointcloud(synthetic_depth, mask, identity_intrinsics)
    assert pts.shape == (16, 3)
    # All points have z == constant depth.
    np.testing.assert_allclose(pts[:, 2], 0.5, atol=1e-5)


def test_pointcloud_drops_nonpositive_depth(identity_intrinsics):
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[3:5, 3:5] = 0.5
    mask = np.ones((8, 8), dtype=bool)
    pts = mask_to_pointcloud(depth, mask, identity_intrinsics)
    assert pts.shape == (4, 3)


def test_pointcloud_extrinsics_transforms_to_world(
    synthetic_depth, identity_intrinsics
):
    mask = np.zeros((32, 32), dtype=bool)
    mask[15:17, 15:17] = True
    extrinsics = np.eye(4, dtype=np.float32)
    extrinsics[:3, 3] = (10.0, 20.0, 30.0)
    pts = mask_to_pointcloud(
        synthetic_depth, mask, identity_intrinsics, extrinsics=extrinsics
    )
    # All points should be shifted by the translation.
    assert np.all(pts[:, 0] > 9) and np.all(pts[:, 0] < 11)
    assert np.all(pts[:, 1] > 19) and np.all(pts[:, 1] < 21)
    assert np.all(pts[:, 2] > 29.4) and np.all(pts[:, 2] < 30.6)


def test_remove_statistical_outliers_drops_flyers():
    rng = np.random.default_rng(0)
    cluster = rng.normal(scale=0.01, size=(200, 3)).astype(np.float32)
    flyers = np.array([[5.0, 5.0, 5.0], [-4.0, -4.0, -4.0]], dtype=np.float32)
    pts = np.vstack([cluster, flyers])
    filtered, keep = remove_statistical_outliers(pts, k=16, std_ratio=2.0)
    assert filtered.shape[0] < pts.shape[0]
    # The flyers (last two rows) must be removed.
    assert not keep[-1] and not keep[-2]
    # The bulk of cluster points should survive.
    assert keep[:200].sum() >= 190


def test_remove_statistical_outliers_passthrough_small_cloud():
    pts = np.zeros((4, 3), dtype=np.float32)
    filtered, keep = remove_statistical_outliers(pts, k=16, std_ratio=2.0)
    assert filtered.shape == pts.shape
    assert keep.all()


def test_remove_statistical_outliers_keeps_uniform_cloud():
    rng = np.random.default_rng(1)
    pts = rng.normal(scale=0.05, size=(300, 3)).astype(np.float32)
    _, keep = remove_statistical_outliers(pts, k=16, std_ratio=3.0)
    # Uniform Gaussian cluster — almost everything should survive at 3-sigma.
    assert keep.mean() > 0.95
