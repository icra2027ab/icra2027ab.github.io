"""World-frame point projection into camera images, with optional occlusion test.

Inverse of ``reconstruction.rgbd.mask_to_pointcloud`` — given world-frame
points and a target camera, returns the surviving image-space pixels.
Used by the multi-camera fusion stage to find where a main-camera
instance lands in a secondary camera, accounting for occlusions via the
secondary's depth map.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from ..types import CameraInfo


@dataclass(frozen=True)
class ProjectedPoints:
    """Result of projecting (K, 3) world points into a target camera.

    ``uv`` and ``z_cam`` are the surviving (after culling) pixels and
    their depths in the target camera frame. ``keep_mask`` is over the
    *input* points and lets the caller cross-reference to the original
    3D set (e.g. to colour matched points in a visualization).
    """

    uv: np.ndarray  # (M, 2) int32 — (x, y) pixel coords
    z_cam: np.ndarray  # (M,) float32 — depth in target camera frame
    keep_mask: np.ndarray  # (K,) bool


def world_to_image(
    points_world: np.ndarray,
    camera: CameraInfo,
    image_hw: tuple[int, int],
    depth_map: np.ndarray | None = None,
    occlusion_tolerance: float = 0.05,
    flip_u: bool = False,
) -> ProjectedPoints:
    """Project ``(K, 3)`` world points into ``camera`` and optionally cull occlusions.

    ``camera.extrinsics`` is interpreted as ``world_T_camera`` (matching
    ``mask_to_pointcloud``); we invert it for the projection. Points
    behind the image plane (``Z <= 0``) and out-of-bounds pixels are
    dropped. When ``depth_map`` is given, pixels whose projected depth
    exceeds the observed depth by more than ``occlusion_tolerance``
    metres are dropped (the projected point sits behind whatever the
    camera actually sees).

    Set ``flip_u=True`` to mirror returned ``u`` coordinates across the
    image width — pair this with the reconstruct stage's matching
    ``flip_u`` so projections land in the *stored* RGB/depth frame
    (datasets like robosuite/LIBERO store images 180°-rotated relative
    to their intrinsics).
    """
    H, W = image_hw
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    K_total = pts.shape[0]
    if K_total == 0:
        return ProjectedPoints(
            uv=np.zeros((0, 2), dtype=np.int32),
            z_cam=np.zeros((0,), dtype=np.float32),
            keep_mask=np.zeros((0,), dtype=bool),
        )

    if camera.extrinsics is not None:
        cam_T_world = np.linalg.inv(camera.extrinsics).astype(np.float32)
        R = cam_T_world[:3, :3]
        t = cam_T_world[:3, 3]
        pts_cam = pts @ R.T + t
    else:
        pts_cam = pts.copy()

    Z = pts_cam[:, 2]
    in_front = Z > 1e-6

    fx = float(camera.intrinsics[0, 0])
    fy = float(camera.intrinsics[1, 1])
    cx = float(camera.intrinsics[0, 2])
    cy = float(camera.intrinsics[1, 2])

    safe_Z = np.where(in_front, Z, 1.0)
    u = pts_cam[:, 0] * fx / safe_Z + cx
    v = pts_cam[:, 1] * fy / safe_Z + cy
    u_i = np.round(u).astype(np.int32)
    v_i = np.round(v).astype(np.int32)
    if flip_u:
        # Convert sensor-frame u into stored-image u (mirror across width)
        # so subsequent depth-map lookups and the returned uv match the
        # caller's image layout.
        u_i = (W - 1) - u_i
    in_bounds = (u_i >= 0) & (u_i < W) & (v_i >= 0) & (v_i < H)

    keep = in_front & in_bounds

    if depth_map is not None and keep.any():
        u_keep = u_i[keep]
        v_keep = v_i[keep]
        observed = depth_map[v_keep, u_keep].astype(np.float32)
        projected = Z[keep]
        # observed <= 0 means the secondary has no depth there — be permissive
        # (treat as not occluded) so missing-depth regions don't drop everything.
        not_occluded = (observed <= 0) | (projected <= observed + occlusion_tolerance)
        keep_idxs = np.where(keep)[0]
        keep[keep_idxs[~not_occluded]] = False

    uv = np.stack([u_i[keep], v_i[keep]], axis=1).astype(np.int32)
    z_cam = Z[keep].astype(np.float32)
    return ProjectedPoints(uv=uv, z_cam=z_cam, keep_mask=keep)


def projected_points_to_mask(
    uv: np.ndarray, image_hw: tuple[int, int], dilate: int = 2
) -> np.ndarray:
    """Render ``(M, 2)`` ``(x, y)`` pixel coords as a sparse boolean mask.

    A small dilation (default 2 px) gives the sparse footprint enough
    area to IoU meaningfully against a dense tracked mask. ``dilate <= 0``
    skips dilation.
    """
    H, W = image_hw
    mask = np.zeros((H, W), dtype=bool)
    if uv.shape[0] == 0:
        return mask
    xs = uv[:, 0]
    ys = uv[:, 1]
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    mask[ys[valid], xs[valid]] = True
    if dilate > 0:
        structure = np.ones((2 * dilate + 1, 2 * dilate + 1), dtype=bool)
        mask = binary_dilation(mask, structure=structure)
    return mask


def projected_points_to_bbox(uv: np.ndarray, pad: int = 0) -> np.ndarray | None:
    """Return ``[y0, x0, y1, x1]`` over projected pixels, or ``None`` if empty.

    Mirrors the contract of ``utils.geometry.mask_to_bbox``.
    """
    if uv.shape[0] == 0:
        return None
    xs = uv[:, 0]
    ys = uv[:, 1]
    y0 = int(ys.min()) - pad
    x0 = int(xs.min()) - pad
    y1 = int(ys.max()) + pad + 1
    x1 = int(xs.max()) + pad + 1
    return np.array([y0, x0, y1, x1], dtype=np.int32)
