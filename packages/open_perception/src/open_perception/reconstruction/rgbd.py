"""RGBD -> (K, 3) point cloud per mask.

Keeps it simple: depth is interpreted as metric distance along the camera Z
axis (MuJoCo/OpenGL depth callers should convert upstream). The transform
into world frame only happens when ``extrinsics`` is given.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def mask_to_pointcloud(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray | None = None,
    stride: int = 1,
    *,
    flip_u: bool = False,
) -> np.ndarray:
    """Return ``(K, 3)`` float32 point cloud from pixels where ``mask`` is True.

    ``intrinsics`` is a standard pinhole (3, 3). ``extrinsics`` is the
    ``world_T_camera`` 4×4; when supplied, the points are returned in world
    frame. Invalid (non-positive) depths are dropped.

    Set ``flip_u=True`` when ``depth`` and ``mask`` come from a
    ``[::-1, ::-1]`` (180°-rotated) image while ``intrinsics`` still
    matches the unrotated layout — the robosuite/LIBERO convention. The
    u-axis mirror (``u = (W-1) - u``) is then undone before unprojection
    so the resulting world points line up with ``extrinsics``.
    """
    if depth.shape[:2] != mask.shape[:2]:
        raise ValueError(f"depth {depth.shape} and mask {mask.shape} shapes differ")
    ys, xs = np.where(mask)
    if stride > 1:
        ys = ys[::stride]
        xs = xs[::stride]
    z = depth[ys, xs].astype(np.float32)
    valid = z > 0
    ys, xs, z = ys[valid], xs[valid], z[valid]
    if z.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    xs_K = (depth.shape[1] - 1) - xs if flip_u else xs
    X = (xs_K.astype(np.float32) - cx) * z / fx
    Y = (ys.astype(np.float32) - cy) * z / fy
    pts = np.stack([X, Y, z], axis=1)

    if extrinsics is not None:
        R = extrinsics[:3, :3].astype(np.float32)
        t = extrinsics[:3, 3].astype(np.float32)
        pts = pts @ R.T + t
    return pts


def remove_statistical_outliers(
    points: np.ndarray,
    k: int = 16,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop points whose mean distance to ``k`` neighbors is a global outlier.

    For each point, compute the mean distance to its ``k`` nearest neighbors;
    points whose mean exceeds ``mu + std_ratio * sigma`` of the global
    distribution are removed. Mirrors Open3D / PCL ``StatisticalOutlierRemoval``.

    Returns the filtered ``(M, 3)`` points and a boolean ``(N,)`` keep-mask.
    Pointclouds with ``<= k`` points are returned unchanged.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got shape {points.shape}")
    n = points.shape[0]
    if n <= k or std_ratio <= 0:
        return points, np.ones((n,), dtype=bool)

    tree = cKDTree(points)
    # Query k+1 neighbors — the first is the point itself (distance 0).
    dists, _ = tree.query(points, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)
    mu = float(mean_dists.mean())
    sigma = float(mean_dists.std())
    threshold = mu + std_ratio * sigma
    keep = mean_dists <= threshold
    return points[keep], keep
