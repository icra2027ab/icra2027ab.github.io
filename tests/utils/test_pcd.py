"""
Tests for policy_abstraction.utils.pcd.

All tests use only synthetic data (no LIBERO / GPU / network required).
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.utils.pcd import (
    pixels_to_3d,
    project_points_to_pixels,
    rgbd_to_pointcloud,
)

W, H = 640, 480


class TestProjectPixelsRoundTrip:
    """project_points_to_pixels → pixels_to_3d should recover the original 3D point."""

    def _exact_pixel_points(self):
        """World points whose projections fall on exact integer pixels (with K below).

        With K: fx=fy=500, cx=320, cy=240 and E=identity:
          u_raw = fx*x/z + cx  must be integer
          v_raw = fy*y/z + cy  must be integer
        """
        return np.array(
            [
                [0.0, 0.0, 2.0],  # → u_raw=320, v_raw=240
                [0.2, 0.2, 2.0],  # → u_raw=370, v_raw=290
                [-0.1, 0.1, 1.0],  # → u_raw=270, v_raw=290
                [0.0, 0.1, 1.0],  # → u_raw=320, v_raw=290
                [0.1, 0.0, 1.0],  # → u_raw=370, v_raw=240
            ],
            dtype=np.float64,
        )

    def test_round_trip_preprocessed(self, pinhole_camera_params):
        """Project then unproject recovers original world-frame points."""
        K, E = pinhole_camera_params
        pts3d = self._exact_pixel_points()

        pts_2d, valid = project_points_to_pixels(pts3d, K, E, H, W, preprocessed=True)
        assert valid.all(), "All test points should be in front of the camera"

        # z_cam = pts3d[:, 2] when E=identity (camera frame == world frame)
        z_cam = pts3d[:, 2]
        recovered = pixels_to_3d(pts_2d, z_cam, K, E, W, preprocessed=True)

        np.testing.assert_allclose(recovered, pts3d, atol=1e-9)

    def test_round_trip_not_preprocessed(self, pinhole_camera_params):
        """Round-trip works with preprocessed=False as well."""
        K, E = pinhole_camera_params
        pts3d = self._exact_pixel_points()

        pts_2d, valid = project_points_to_pixels(pts3d, K, E, H, W, preprocessed=False)
        z_cam = pts3d[:, 2]
        recovered = pixels_to_3d(pts_2d, z_cam, K, E, W, preprocessed=False)

        np.testing.assert_allclose(recovered, pts3d, atol=1e-9)

    def test_behind_camera_is_invalid(self, pinhole_camera_params):
        """Points with z ≤ 0 in camera frame are flagged as invalid."""
        K, E = pinhole_camera_params
        # With E=identity, camera frame = world frame → z=-1 is behind camera
        pts3d = np.array([[0.0, 0.0, -1.0]])
        _, valid = project_points_to_pixels(pts3d, K, E, H, W)
        assert not valid[0]

    def test_preprocessed_flips_u_axis(self, pinhole_camera_params):
        """preprocessed=True mirrors u relative to preprocessed=False."""
        K, E = pinhole_camera_params
        pts3d = np.array([[0.2, 0.0, 2.0]])

        pts_prep, _ = project_points_to_pixels(pts3d, K, E, H, W, preprocessed=True)
        pts_raw, _ = project_points_to_pixels(pts3d, K, E, H, W, preprocessed=False)

        # u_prep = (W-1) - u_raw
        assert pts_prep[0, 0] == (W - 1) - pts_raw[0, 0]
        # v should be unchanged
        assert pts_prep[0, 1] == pts_raw[0, 1]


class TestRgbdToPointcloud:
    def _make_rgbd(self, H=16, W=16):
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        rgb[:, :, 0] = 128
        depth = np.ones((H, W), dtype=np.float64) * 2.0
        return rgb, depth

    def test_output_shape(self, pinhole_camera_params):
        """Output is [M, 6] with M ≤ H*W."""
        K, E = pinhole_camera_params
        rgb, depth = self._make_rgbd()
        pcd = rgbd_to_pointcloud(rgb, depth, K, E)
        assert pcd.ndim == 2
        assert pcd.shape[1] == 6
        assert pcd.shape[0] > 0
        assert pcd.shape[0] <= rgb.shape[0] * rgb.shape[1]

    def test_output_dtype(self, pinhole_camera_params):
        """Output dtype is float32."""
        K, E = pinhole_camera_params
        rgb, depth = self._make_rgbd()
        pcd = rgbd_to_pointcloud(rgb, depth, K, E)
        assert pcd.dtype == np.float32

    def test_depth_min_filter(self, pinhole_camera_params):
        """Pixels with depth ≤ depth_min are excluded."""
        K, E = pinhole_camera_params
        H_, W_ = 8, 8
        rgb = np.ones((H_, W_, 3), dtype=np.uint8)
        # Half pixels at depth 0.5, half at depth 2.0
        depth = np.ones((H_, W_)) * 2.0
        depth[:, : W_ // 2] = 0.5

        pcd_all = rgbd_to_pointcloud(rgb, depth, K, E, depth_min=0.0)
        pcd_some = rgbd_to_pointcloud(rgb, depth, K, E, depth_min=1.0)
        # With depth_min=1.0 the 0.5-depth pixels are filtered out
        assert pcd_some.shape[0] < pcd_all.shape[0]

    def test_colors_preserved(self, pinhole_camera_params):
        """RGB channels in output match the source image's pixel colors."""
        K, E = pinhole_camera_params
        H_, W_ = 4, 4
        rgb = np.zeros((H_, W_, 3), dtype=np.uint8)
        rgb[:, :, 0] = 200  # red channel
        depth = np.ones((H_, W_)) * 1.0

        pcd = rgbd_to_pointcloud(rgb, depth, K, E)
        # All output R values should be 200, G=0, B=0
        np.testing.assert_allclose(pcd[:, 3], 200.0)
        np.testing.assert_allclose(pcd[:, 4], 0.0)
        np.testing.assert_allclose(pcd[:, 5], 0.0)
