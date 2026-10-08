"""Utilities for 3D ↔ pixel coordinate conversions and point-cloud generation.

Coordinate conventions (robosuite / this codebase)
---------------------------------------------------
* **Intrinsic** ``K`` — standard 3×3 pinhole matrix, pixels, top-left origin,
  OpenCV convention.  ``K[0,0]`` = fx, ``K[1,1]`` = fy, ``K[0,2]`` = cx,
  ``K[1,2]`` = cy.
* **Extrinsic** ``E`` — 4×4 *camera-to-world* pose returned by robosuite
  (with its axis-correction already applied).  To go from world → camera,
  invert ``E``.
* **Preprocessed images** — observations are stored/displayed after a 180°
  rotation (``obs[key][::-1, ::-1]``), which converts OpenGL's bottom-left
  origin to a top-left-origin image **and** mirrors it horizontally.
  The intrinsic ``K`` already encodes the OpenCV (top-left) v-axis, so only
  the u-axis needs mirroring: ``u_prep = (W - 1) - u_raw``.

All three public functions accept an optional ``preprocessed`` flag
(default ``True``) that applies or reverses this U-axis correction.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 3-D → pixel
# ---------------------------------------------------------------------------


def project_points_to_pixels(
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_height: int,
    image_width: int,
    *,
    preprocessed: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world-frame 3-D points to pixel coordinates.

    Args:
        points_3d:    ``[N, 3]`` XYZ positions in world frame.
        intrinsic:    ``[3, 3]`` pinhole intrinsic K.
        extrinsic:    ``[4, 4]`` camera-to-world pose (robosuite convention).
        image_height: Image height in pixels (used for bounds context only).
        image_width:  Image width in pixels.
        preprocessed: When ``True`` (default), applies the u-axis mirror
                      correction so that returned pixels match the
                      180°-rotated preprocessed image layout.

    Returns:
        ``(pts_2d, valid)``

        * ``pts_2d`` — ``[N, 2]`` int32 array of (col, row) = (u, v) pixel
          coordinates.  Invalid points (behind the camera) are set to 0.
        * ``valid`` — ``[N]`` boolean mask; ``True`` where the point is in
          front of the camera (``z_cam > 0``).
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    N = len(points_3d)
    pts_h = np.hstack([points_3d, np.ones((N, 1))])  # [N, 4]

    # robosuite returns camera-to-world; invert to get world-to-camera.
    E_world2cam = np.linalg.inv(np.asarray(extrinsic, dtype=np.float64))
    pts_cam = (E_world2cam @ pts_h.T).T  # [N, 4]
    valid = pts_cam[:, 2] > 1e-3

    pts_2d = np.zeros((N, 2), dtype=np.float64)
    if valid.any():
        p = (np.asarray(intrinsic, dtype=np.float64) @ pts_cam[valid, :3].T).T  # [K, 3]
        pts_2d[valid, 0] = p[:, 0] / p[:, 2]  # u (col, left→right)
        pts_2d[valid, 1] = p[:, 1] / p[:, 2]  # v (row, top→bottom)

    if preprocessed:
        # [::-1, ::-1] preprocessing mirrors the u-axis only (axis-corrected K
        # already handles the v-axis flip from OpenGL to OpenCV convention).
        pts_2d[:, 0] = (image_width - 1) - pts_2d[:, 0]

    return pts_2d.astype(np.int32), valid


# ---------------------------------------------------------------------------
# pixel + depth → 3-D
# ---------------------------------------------------------------------------


def pixels_to_3d(
    pixels: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    image_width: int,
    *,
    preprocessed: bool = True,
) -> np.ndarray:
    """Back-project pixel coordinates and depth values to world-frame 3-D points.

    This is the inverse of :func:`project_points_to_pixels`.

    Args:
        pixels:       ``[N, 2]`` array of (col, row) = (u, v) integer pixel
                      coordinates in the (optionally preprocessed) image.
        depth:        ``[N]`` array of depth values in world-frame metric units
                      (i.e. the z-coordinate of the corresponding camera-space
                      point, *not* the Euclidean distance to the camera).
        intrinsic:    ``[3, 3]`` pinhole intrinsic K.
        extrinsic:    ``[4, 4]`` camera-to-world pose (robosuite convention).
        image_width:  Width of the image in pixels (needed to undo the u-axis
                      mirror when ``preprocessed=True``).
        preprocessed: When ``True`` (default), reverses the u-axis mirror
                      applied by the 180°-rotation preprocessing step before
                      unprojecting.

    Returns:
        ``[N, 3]`` float64 array of XYZ positions in world frame.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(intrinsic, dtype=np.float64)
    E = np.asarray(extrinsic, dtype=np.float64)

    u = pixels[:, 0].copy()
    v = pixels[:, 1]

    if preprocessed:
        # Undo the u-axis mirror introduced by [::-1, ::-1] preprocessing.
        u = (image_width - 1) - u

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Unproject from pixel + depth → camera-space XYZ.
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth

    pts_cam_h = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)  # [N, 4]

    # Transform from camera frame → world frame using the camera-to-world pose.
    pts_world = (E @ pts_cam_h.T).T  # [N, 4]
    return pts_world[:, :3]


# ---------------------------------------------------------------------------
# RGBD → point cloud
# ---------------------------------------------------------------------------


def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    *,
    depth_scale: float = 1.0,
    depth_min: float = 1e-3,
    depth_max: float = np.inf,
    preprocessed: bool = True,
) -> np.ndarray:
    """Convert an RGBD image pair into a coloured 3-D point cloud.

    Args:
        rgb:          ``[H, W, 3]`` uint8 RGB image.
        depth:        ``[H, W]`` float depth image.  Pixel values are
                      multiplied by *depth_scale* to obtain metric units.
                      Pixels with ``depth <= depth_min`` (after scaling)
                      are discarded.
        intrinsic:    ``[3, 3]`` pinhole intrinsic K.
        extrinsic:    ``[4, 4]`` camera-to-world pose (robosuite convention).
        depth_scale:  Multiply raw depth values by this factor to convert to
                      metres (or whatever world-frame units K uses).  Set to
                      ``1.0`` (default) when depth is already in world units.
        depth_min:    Depth threshold below which pixels are ignored
                      (default ``1e-3``).
        depth_max:    Depth threshold above which pixels are ignored
                      (default no limit).
        preprocessed: When ``True`` (default), reverses the u-axis mirror
                      applied by the 180°-rotation preprocessing step before
                      unprojecting.

    Returns:
        ``[M, 6]`` float32 array with columns ``[X, Y, Z, R, G, B]``.
        ``R, G, B`` are in the range ``[0, 255]``.  ``M`` is the number of
        valid (non-masked) pixels.
    """
    rgb = np.asarray(rgb)
    depth = np.asarray(depth, dtype=np.float64) * depth_scale

    H, W = depth.shape

    # Build pixel-grid and mask out invalid depth values.
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))  # [H, W] each
    mask = (depth > depth_min) & (depth < depth_max)

    u_flat = cols[mask].astype(np.float64)  # [M]
    v_flat = rows[mask].astype(np.float64)  # [M]
    d_flat = depth[mask]  # [M]

    # Vectorised back-projection.
    pixels_flat = np.stack([u_flat, v_flat], axis=-1)  # [M, 2]
    xyz = pixels_to_3d(
        pixels_flat,
        d_flat,
        intrinsic,
        extrinsic,
        image_width=W,
        preprocessed=preprocessed,
    )  # [M, 3]

    # Gather colours for valid pixels.
    colors = rgb[mask].astype(np.float32)  # [M, 3]

    return np.concatenate([xyz.astype(np.float32), colors], axis=-1)  # [M, 6]
