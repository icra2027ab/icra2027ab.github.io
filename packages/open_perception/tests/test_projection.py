"""Unit tests for ``utils.projection``."""

from __future__ import annotations

import numpy as np
from open_perception.types import CameraInfo
from open_perception.utils.projection import (
    projected_points_to_bbox,
    projected_points_to_mask,
    world_to_image,
)


def _identity_camera(image_hw: tuple[int, int] = (32, 32)) -> CameraInfo:
    H, W = image_hw
    K = np.array(
        [[20.0, 0.0, W / 2], [0.0, 20.0, H / 2], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    return CameraInfo(intrinsics=K, extrinsics=np.eye(4, dtype=np.float32))


def test_world_to_image_principal_point() -> None:
    """A point on the optical axis at +Z should project to the principal point."""
    cam = _identity_camera()
    pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    proj = world_to_image(pts, cam, image_hw=(32, 32))
    assert proj.uv.shape == (1, 2)
    # cx, cy = 16, 16
    assert proj.uv[0, 0] == 16
    assert proj.uv[0, 1] == 16
    assert proj.keep_mask.tolist() == [True]


def test_world_to_image_drops_behind_camera() -> None:
    cam = _identity_camera()
    pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    proj = world_to_image(pts, cam, image_hw=(32, 32))
    assert proj.uv.shape[0] == 1
    assert proj.keep_mask.tolist() == [False, True]


def test_world_to_image_drops_out_of_bounds() -> None:
    cam = _identity_camera(image_hw=(32, 32))
    # Far off-axis at unit depth → projects far outside the 32x32 image.
    pts = np.array([[10.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    proj = world_to_image(pts, cam, image_hw=(32, 32))
    assert proj.keep_mask.tolist() == [False, True]


def test_world_to_image_occlusion_culls_far_points() -> None:
    cam = _identity_camera()
    pts = np.array([[0.0, 0.0, 2.0]], dtype=np.float32)  # 2m deep
    depth_map = np.full((32, 32), 1.0, dtype=np.float32)  # observed: 1m
    proj = world_to_image(
        pts, cam, image_hw=(32, 32), depth_map=depth_map, occlusion_tolerance=0.05
    )
    # Projected point at 2m sits behind the observed surface at 1m → drop.
    assert proj.uv.shape[0] == 0
    assert proj.keep_mask.tolist() == [False]


def test_world_to_image_occlusion_keeps_near_points() -> None:
    cam = _identity_camera()
    pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    depth_map = np.full((32, 32), 1.0, dtype=np.float32)
    proj = world_to_image(
        pts, cam, image_hw=(32, 32), depth_map=depth_map, occlusion_tolerance=0.05
    )
    assert proj.uv.shape[0] == 1


def test_world_to_image_missing_depth_does_not_drop() -> None:
    """observed_z<=0 means the secondary has no depth there — be permissive."""
    cam = _identity_camera()
    pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    depth_map = np.zeros((32, 32), dtype=np.float32)  # all zeros = "no observation"
    proj = world_to_image(
        pts, cam, image_hw=(32, 32), depth_map=depth_map, occlusion_tolerance=0.05
    )
    assert proj.uv.shape[0] == 1


def test_world_to_image_with_translated_extrinsics() -> None:
    """Camera at (1, 0, 0) looking down +Z should see a point at (1, 0, 1) on-axis."""
    K = np.array(
        [[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    # world_T_camera: camera origin at world (1, 0, 0), no rotation.
    extr = np.eye(4, dtype=np.float32)
    extr[0, 3] = 1.0
    cam = CameraInfo(intrinsics=K, extrinsics=extr)
    pts = np.array([[1.0, 0.0, 1.0]], dtype=np.float32)
    proj = world_to_image(pts, cam, image_hw=(32, 32))
    # In camera frame: world_pt - cam_origin = (0, 0, 1) → principal point.
    assert proj.uv[0, 0] == 16
    assert proj.uv[0, 1] == 16


def test_world_to_image_empty_input() -> None:
    cam = _identity_camera()
    proj = world_to_image(np.zeros((0, 3), dtype=np.float32), cam, image_hw=(32, 32))
    assert proj.uv.shape == (0, 2)
    assert proj.z_cam.shape == (0,)
    assert proj.keep_mask.shape == (0,)


def test_projected_points_to_mask_basic() -> None:
    uv = np.array([[5, 5], [10, 10]], dtype=np.int32)
    mask = projected_points_to_mask(uv, image_hw=(32, 32), dilate=0)
    assert mask.dtype == bool
    assert mask[5, 5]  # (y, x)
    assert mask[10, 10]
    assert mask.sum() == 2


def test_projected_points_to_mask_dilates() -> None:
    uv = np.array([[15, 15]], dtype=np.int32)
    mask_no_dilate = projected_points_to_mask(uv, image_hw=(32, 32), dilate=0)
    mask_dilated = projected_points_to_mask(uv, image_hw=(32, 32), dilate=2)
    assert mask_no_dilate.sum() == 1
    # 2-px dilation around a single pixel = a 5x5 block.
    assert mask_dilated.sum() == 25


def test_projected_points_to_mask_empty() -> None:
    mask = projected_points_to_mask(np.zeros((0, 2), dtype=np.int32), image_hw=(32, 32))
    assert mask.shape == (32, 32)
    assert not mask.any()


def test_projected_points_to_bbox_basic() -> None:
    uv = np.array([[5, 10], [15, 20]], dtype=np.int32)
    bbox = projected_points_to_bbox(uv)
    assert bbox is not None
    y0, x0, y1, x1 = bbox.tolist()
    assert (y0, x0, y1, x1) == (10, 5, 21, 16)


def test_projected_points_to_bbox_empty() -> None:
    bbox = projected_points_to_bbox(np.zeros((0, 2), dtype=np.int32))
    assert bbox is None
