"""Client for the SAM3D WebSocket server plus depth-alignment and 3D tracking.

The SAM3D server (``scripts/serve_sam3d.py``) reconstructs a 3D Gaussian
splat from a single RGB image and a binary mask.  This module provides:

* :class:`SAM3DClient` — synchronous WebSocket client that wraps the RPC.
* :func:`align_gaussians_to_depth` — scales and positions the canonical
  SAM3D Gaussians to match the metric depth image from a LIBERO demo.
* :func:`track_object_centroids` — tracks the 3D centroid of an object
  (given per-frame depth + masks) across a demo sequence.
* :func:`transform_gaussians_to_camera_space` — applies the SAM3D-estimated
  pose to obtain Gaussian positions in MoGe camera space.

Coordinate conventions
-----------------------
* **HDF5 images** are stored with a 180° rotation (both axes).  After
  ``img[::-1, ::-1]`` (or the depth equivalent) they are in display
  convention (row 0 = top-left, standard OpenCV image layout).
* **Extrinsic** matrices in the HDF5 are camera-to-world 4×4 transforms
  (robosuite convention, already axis-corrected).
* **SAM3D rotation / translation** are estimated in MoGe-PyTorch3D camera
  space.  :func:`transform_gaussians_to_camera_space` returns positions in
  this space; the conversion to LIBERO camera space applies sign flips on Y
  and Z (OpenCV vs PyTorch3D axis convention).
"""

from __future__ import annotations

import asyncio
import logging
import pickle
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WebSocket client
# ---------------------------------------------------------------------------


class SAM3DClient:
    """Synchronous WebSocket client for the SAM3D inference server.

    Parameters
    ----------
    host:   Hostname or IP of the server (default ``"localhost"``).
    port:   WebSocket port (default ``8766``).
    """

    def __init__(self, host: str = "localhost", port: int = 8766) -> None:
        self._uri = f"ws://{host}:{port}"
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the WebSocket connection (blocks until ready)."""
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async_connect())

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None and self._loop is not None:
            self._loop.run_until_complete(self._ws.close())
        self._ws = None

    def __enter__(self) -> "SAM3DClient":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.disconnect()

    async def _async_connect(self) -> None:
        import websockets

        self._ws = await websockets.connect(
            self._uri,
            max_size=None,
            compression=None,
            open_timeout=30,
        )
        # Read handshake
        raw = await self._ws.recv()
        assert isinstance(raw, bytes)
        handshake = pickle.loads(raw)
        logger.info(
            "[SAM3DClient] Connected to %s (%s)", self._uri, handshake.get("server")
        )

    # ------------------------------------------------------------------
    # RPC
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        seed: int = 42,
        return_mesh: bool = False,
        pointmap: np.ndarray | None = None,
        with_mesh_postprocess: bool = False,
        with_texture_baking: bool = False,
        use_vertex_color: bool = True,
    ) -> dict[str, Any]:
        """Reconstruct 3D Gaussians (and optionally a mesh) from an RGB image and mask.

        Parameters
        ----------
        image:        ``[H, W, 3]`` uint8 RGB image (display convention, row 0 = top).
        mask:         ``[H, W]`` bool mask for the target object.
        seed:         Random seed for the diffusion model.
        return_mesh:  If True, the result includes ``mesh_glb_bytes`` — raw GLB
                      file bytes for the FlexiCubes mesh decoded from the same
                      latent features as the Gaussians.  Save with
                      :meth:`save_mesh` or directly::

                          with open("object.glb", "wb") as f:
                              f.write(result["mesh_glb_bytes"])

        pointmap:             ``[H, W, 3]`` float32 3-D point map in **PyTorch3D
                              camera space**.  When provided, MoGe depth
                              estimation is skipped.  Pass the ``pointmap``
                              field from a previous call or a depth-derived map
                              (see :func:`depth_to_sam3d_pointmap`).
        with_mesh_postprocess: Run FlexiCubes hole-filling + simplification
                               (slow; default False).
        with_texture_baking:  Bake UV texture from 100 Hammersley views
                               (slow; default False).
        use_vertex_color:     Store vertex colours instead of UV texture when
                               not baking (default True; ignored when
                               ``with_texture_baking=True``).

        Returns
        -------
        dict with keys:

        * ``gs_xyz``          [N, 3] float32 — canonical Gaussian positions
        * ``gs_colors``       [N, 3] float32 — RGB in [0, 1]
        * ``gs_opacity``      [N, 1] float32 — opacity in [0, 1]
        * ``gs_scaling``      [N, 3] float32 — Gaussian scale
        * ``gs_rotation``     [N, 4] float32 — per-splat quaternion (w, x, y, z)
        * ``rotation``        [4]    float32 — object rotation quaternion (w, x, y, z)
        * ``translation``     [3]    float32 — object translation in MoGe cam-space
        * ``scale``           [3]    float32 — object scale
        * ``pointmap``        [H, W, 3] float32 — 3D point map (PyTorch3D space)
        * ``mesh_glb_bytes``  bytes | None — raw GLB bytes (only when return_mesh=True)
        """
        if self._ws is None or self._loop is None:
            raise RuntimeError("Not connected. Call connect() first.")
        kwargs: dict[str, Any] = {
            "image": image,
            "mask": mask,
            "seed": seed,
            "return_mesh": return_mesh,
            "with_mesh_postprocess": with_mesh_postprocess,
            "with_texture_baking": with_texture_baking,
            "use_vertex_color": use_vertex_color,
        }
        if pointmap is not None:
            kwargs["pointmap"] = np.asarray(pointmap, dtype=np.float32)
        return self._loop.run_until_complete(self._async_call("reconstruct", kwargs))

    @staticmethod
    def save_mesh(mesh_glb_bytes: bytes, path: str) -> None:
        """Write raw GLB bytes returned by :meth:`reconstruct` to *path*.

        Parameters
        ----------
        mesh_glb_bytes: Bytes from ``result["mesh_glb_bytes"]``.
        path:           Destination file path (should end in ``.glb``).
        """
        with open(path, "wb") as fh:
            fh.write(mesh_glb_bytes)

    async def _async_call(self, cmd: str, kwargs: dict) -> Any:
        assert self._ws is not None
        payload = pickle.dumps({"cmd": cmd, "kwargs": kwargs})
        await self._ws.send(payload)
        raw = await self._ws.recv()
        assert isinstance(raw, bytes)
        msg = pickle.loads(raw)
        if msg.get("error"):
            raise RuntimeError(f"SAM3DServer error:\n{msg['error']}")
        return msg["result"]


# ---------------------------------------------------------------------------
# Camera / pointmap utilities
# ---------------------------------------------------------------------------


def depth_to_sam3d_pointmap(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    preprocessed: bool = True,
) -> np.ndarray:
    """Convert a metric depth image to a SAM3D pointmap in PyTorch3D camera space.

    SAM3D's ``compute_pointmap`` bypasses MoGe when a pre-computed pointmap is
    supplied.  The expected format is ``[H, W, 3]`` float32 in **PyTorch3D
    camera space** (X right, Y up, Z toward the viewer).

    This function unpacks a pinhole-camera depth image (LIBERO / OpenCV
    convention: Y down, Z forward) and rotates it into PyTorch3D space.

    Parameters
    ----------
    depth:        ``[H, W]`` or ``[H, W, 1]`` float32  metric depth in metres
                  (z-component along optical axis, not Euclidean distance).
    intrinsic:    ``[3, 3]`` float64  pinhole K matrix matching the image
                  coordinate convention described by *preprocessed*.
    preprocessed: If True (default), the depth image has been preprocessed
                  with the LIBERO 180°-rotation (``[::-1, ::-1]``).  The
                  correction mirrors the u-axis and v-axis back to the
                  original camera convention before unprojecting.

    Returns
    -------
    ``[H, W, 3]`` float32 pointmap in PyTorch3D camera space.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3:
        depth = depth[..., 0]
    H, W = depth.shape

    K = np.asarray(intrinsic, dtype=np.float64)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(W, dtype=np.float64)  # [W]
    v = np.arange(H, dtype=np.float64)  # [H]
    u_grid, v_grid = np.meshgrid(u, v)  # [H, W]

    # The 180° preprocessing only mirrors the U-axis (pcd.py convention):
    # u_pp = W-1-u_orig, but v_pp == v_cv (K already encodes the OpenCV V-axis).
    # Therefore only cx changes: cx_eff = W-1-cx.  cy is unchanged.
    if preprocessed:
        cx_eff = (W - 1) - cx
    else:
        cx_eff = cx
    cy_eff = cy  # V-axis is the same in both preprocessed and original K convention

    d = depth.astype(np.float64)
    # Unproject in the preprocessed image's R3 (OpenCV-like) camera space:
    #   X_ppR3 = (u - cx_eff) * d / fx   (right in preprocessed image)
    #   Y_ppR3 = (v - cy)     * d / fy   (down in preprocessed image)
    #   Z_ppR3 = d                         (forward)
    x_r3 = (u_grid - cx_eff) * d / fx
    y_r3 = (v_grid - cy_eff) * d / fy
    z_r3 = d

    # R3 → PyTorch3D: negate Y (down→up) and Z (forward→toward viewer)
    pointmap = np.stack([x_r3, -y_r3, -z_r3], axis=-1)  # [H, W, 3]
    return pointmap.astype(np.float32)


# ---------------------------------------------------------------------------
# Gaussian transform utilities
# ---------------------------------------------------------------------------


def transform_gaussians_to_camera_space(
    gs_xyz: np.ndarray,
    rotation_wxyz: np.ndarray,
    translation: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """Apply the SAM3D-estimated pose to canonical Gaussians.

    The transform is ``p_cam = (p_local * scale) @ R + translation``
    matching the PyTorch3D ``compose_transform(scale, R, T).transform_points``
    convention used internally by the SAM3D pipeline.

    Parameters
    ----------
    gs_xyz:         [N, 3] float32  — canonical Gaussian positions.
    rotation_wxyz:  [4]    float32  — object rotation quaternion in (w, x, y, z).
    translation:    [3]    float32  — object translation.
    scale:          [3]    float32  — isotropic (or per-axis) scale.

    Returns
    -------
    [N, 3] float32 positions in MoGe-PyTorch3D camera space.
    """
    from scipy.spatial.transform import Rotation as ScipyR

    # PyTorch3D uses (w, x, y, z); scipy uses (x, y, z, w)
    q = rotation_wxyz  # [4]
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    R_mat = ScipyR.from_quat(q_xyzw).as_matrix()  # [3, 3]
    R_mat = moge_to_libero_camera(R_mat)

    # gs_xyz = gs_xyz @ np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64)  # mirror Y and Z to match the preprocessed image convention
    # compose_transform(.scale().rotate().translate()): p_out = p * s @ R + T
    pts_scaled = gs_xyz.astype(np.float64) * scale.astype(np.float64)  # [N, 3]
    pts_rotated = pts_scaled @ R_mat  # [N, 3]
    # Apply the same M=diag([-1,1,-1]) flip to translation so the output is a
    # consistent coordinate frame (X right, Y up, Z forward).  Previously only
    # R_mat was right-multiplied by M (via moge_to_libero_camera) but T was left
    # in PyTorch3D space, making the coordinate system inconsistent.
    T = translation.astype(np.float64)
    T_converted = T * np.array([-1.0, 1.0, -1.0])  # apply M to translation
    pts_cam = pts_rotated + T_converted  # [N, 3]
    return pts_cam.astype(np.float32)


def moge_to_libero_camera(pts: np.ndarray) -> np.ndarray:
    """Convert 3D points from preprocessed-image PyTorch3D space to LIBERO/OpenCV.

    The SAM3D pipeline operates in PyTorch3D camera space of the **preprocessed**
    (180°-rotated) image.  PyTorch3D uses (X right, Y up, Z toward viewer);
    LIBERO/OpenCV uses (X right, Y down, Z forward).

    Empirically, negating X (U-axis mirror from preprocessing) and Z
    (forward ↔ toward-viewer flip) gives correct object orientation.
    Y-axis is preserved (PyTorch3D Y-up vs OpenCV Y-down cancel out in
    the preprocessed-image frame due to the 180°-rotation).

    Parameters
    ----------
    pts: [N, 3] in preprocessed-image PyTorch3D camera space.

    Returns
    -------
    [N, 3] in original LIBERO/OpenCV camera space (matching the extrinsic).
    """
    out = pts.copy()
    out[:, 0] *= -1  # negate X (U-axis mirror)
    out[:, 2] *= -1  # negate Z (forward ↔ toward-viewer)
    return out


# ---------------------------------------------------------------------------
# Depth-based 3D utilities
# ---------------------------------------------------------------------------


def get_masked_depth_world_points(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    preprocessed: bool = True,
) -> np.ndarray:
    """Back-project masked depth pixels to world-frame 3D points.

    Parameters
    ----------
    depth:        [H, W] or [H, W, 1] float32  metric depth (metres).
    mask:         [H, W] bool  pixel mask for the object.
    intrinsic:    [3, 3] float64  pinhole intrinsic K (display convention).
    extrinsic:    [4, 4] float64  camera-to-world (robosuite convention).
    preprocessed: If True, reverses the u-axis mirror from 180°-rotation
                  preprocessing before unprojecting (see ``pcd.py``).

    Returns
    -------
    [M, 3] float64 world-frame XYZ of the masked pixels.
    """
    from policy_abstraction.utils.pcd import pixels_to_3d

    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[..., 0]

    mask = np.asarray(mask, dtype=bool)
    H, W = depth.shape

    rows, cols = np.where(mask)
    d_vals = depth[rows, cols]
    valid = d_vals > 1e-3
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float64)

    pixels = np.stack([cols[valid], rows[valid]], axis=1).astype(np.float64)
    return pixels_to_3d(
        pixels,
        d_vals[valid],
        intrinsic,
        extrinsic,
        image_width=W,
        preprocessed=preprocessed,
    )


def get_object_centroid_world(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    preprocessed: bool = True,
) -> np.ndarray:
    """Compute the median 3D world position of a masked object from depth.

    Returns
    -------
    [3] float64 world-frame centroid, or zeros if no valid pixels.
    """
    pts = get_masked_depth_world_points(
        depth, mask, intrinsic, extrinsic, preprocessed=preprocessed
    )
    if len(pts) == 0:
        return np.zeros(3, dtype=np.float64)
    return np.median(pts, axis=0)


# ---------------------------------------------------------------------------
# Scale alignment
# ---------------------------------------------------------------------------


def _icp(
    source: np.ndarray,
    target: np.ndarray,
    max_dist: float,
    max_iter: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Point-to-point ICP core. Returns (R [3,3], t [3]) in column-vector convention.

    Finds the rigid transform such that ``R @ source[i] + t`` is close to the
    nearest point in *target*.  Both arrays are treated as sets of 3-D points.

    Parameters
    ----------
    source:   [N, 3] float64 — source point cloud.
    target:   [M, 3] float64 — target point cloud.
    max_dist: Maximum correspondence distance (metres).
    max_iter: Maximum iterations.

    Returns
    -------
    R: [3, 3] float64 — rotation matrix.
    t: [3]    float64 — translation vector.
    """
    from scipy.spatial import KDTree

    src = source.copy().astype(np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    tree = KDTree(tgt)

    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)

    for _ in range(max_iter):
        dists, idx = tree.query(src, k=1, workers=-1)
        valid = dists < max_dist
        if valid.sum() < 3:
            break

        src_m = src[valid]
        tgt_m = tgt[idx[valid]]

        c_s = src_m.mean(0)
        c_t = tgt_m.mean(0)

        # SVD-based optimal rotation (Procrustes)
        H = (src_m - c_s).T @ (tgt_m - c_t)
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        t = c_t - R @ c_s

        prev = src.copy()
        src = (R @ src.T).T + t

        R_total = R @ R_total
        t_total = R @ t_total + t

        if np.mean(np.linalg.norm(src - prev, axis=1)) < 1e-7:
            break

    return R_total, t_total


def _icp_refine(
    gs_xyz_all: np.ndarray,
    gs_xyz_active: np.ndarray,
    target: np.ndarray,
    max_dist: float,
    max_iter: int = 50,
) -> np.ndarray:
    """Thin wrapper around :func:`_icp` that applies the transform to all Gaussians."""
    R, t = _icp(gs_xyz_active, target, max_dist=max_dist, max_iter=max_iter)
    return ((R @ gs_xyz_all.astype(np.float64).T).T + t).astype(np.float64)


# ---------------------------------------------------------------------------
# Mesh utilities
# ---------------------------------------------------------------------------


def load_mesh_vertices_from_glb(mesh_glb_bytes: bytes) -> np.ndarray:
    """Load vertex positions from raw GLB bytes.

    Uses Open3D to parse the GLB file.  The returned vertices are in the same
    coordinate frame as what was stored in the GLB (i.e. after SAM3D's
    ``to_glb()`` z-up → y-up rotation has been applied).

    Parameters
    ----------
    mesh_glb_bytes: Raw bytes of a GLB file (e.g. from ``SAM3DClient.reconstruct``).

    Returns
    -------
    [V, 3] float64 vertex positions in GLB/mesh space.
    """
    import os
    import tempfile

    import open3d as o3d

    with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as fh:
        fh.write(mesh_glb_bytes)
        tmp = fh.name
    try:
        mesh = o3d.io.read_triangle_mesh(tmp)
        return np.asarray(mesh.vertices, dtype=np.float64)
    finally:
        os.unlink(tmp)


def align_mesh_to_depth(
    mesh_verts_world: np.ndarray,
    depth_pts: np.ndarray,
    adjust_scale: bool = True,
    extent_percentile: float = 90.0,
    max_dist: float | None = None,
    max_iter: int = 50,
) -> dict:
    """Align mesh world-vertices to a depth point cloud (optional scale + rigid ICP).

    Pipeline:

    1. If *adjust_scale*, uniformly rescale the mesh about its centroid so its
       spatial extent matches the depth cloud, then translate to the depth centroid.
    2. Run point-to-point ICP to find the residual rigid correction.

    Parameters
    ----------
    mesh_verts_world:  [V, 3] float64 — mesh vertices already in world space
                       (via the SAM3D pose transform).
    depth_pts:         [P, 3] float64 — masked depth back-projection in world space.
    adjust_scale:      Pre-scale mesh extent to match depth extent (default True).
    extent_percentile: Percentile of point radii used as the characteristic size.
    max_dist:          ICP max correspondence distance. Defaults to
                       ``percentile(depth_radii, extent_percentile) * 0.5``.
    max_iter:          ICP iterations (default 50).

    Returns
    -------
    dict with keys:

    * ``R``                  [3, 3] — rigid rotation from ICP step.
    * ``t``                  [3]    — rigid translation from ICP step.
    * ``scale``              float  — uniform scale applied before ICP (1.0 if
                             *adjust_scale* is False).
    * ``c_mesh``             [3]    — mesh centroid used as the scale pivot.
    * ``c_depth``            [3]    — depth centroid (scale target).
    * ``mesh_verts_aligned`` [V, 3] — final aligned mesh vertices.

    To apply the same composite transform to other world-space points *pts*::

        pts_pre     = (pts - c_mesh) * scale + c_depth
        pts_aligned = (R @ pts_pre.T).T + t
    """
    mesh_verts_world = np.asarray(mesh_verts_world, dtype=np.float64)
    depth_pts = np.asarray(depth_pts, dtype=np.float64)

    c_mesh = np.median(mesh_verts_world, axis=0)
    c_depth = np.median(depth_pts, axis=0)

    scale = 1.0
    if adjust_scale:
        r_mesh = np.linalg.norm(mesh_verts_world - c_mesh, axis=1)
        r_depth = np.linalg.norm(depth_pts - c_depth, axis=1)
        ext_mesh = float(np.percentile(r_mesh, extent_percentile))
        ext_depth = float(np.percentile(r_depth, extent_percentile))
        scale = ext_depth / (ext_mesh + 1e-8)

    mesh_pre = (mesh_verts_world - c_mesh) * scale + c_depth

    if max_dist is None:
        r_depth = np.linalg.norm(depth_pts - c_depth, axis=1)
        max_dist = float(np.percentile(r_depth, extent_percentile)) * 0.5

    R, t = _icp(mesh_pre, depth_pts, max_dist=max_dist, max_iter=max_iter)
    mesh_aligned = (R @ mesh_pre.T).T + t

    return {
        "R": R,
        "t": t,
        "scale": scale,
        "c_mesh": c_mesh,
        "c_depth": c_depth,
        "mesh_verts_aligned": mesh_aligned,
    }


def track_object_with_mesh_icp(
    mesh_verts_world: np.ndarray,
    gs_xyz_ref: np.ndarray,
    depth_frames: np.ndarray,
    mask_frames: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_frames: np.ndarray,
    preprocessed: bool = True,
    max_dist: float | None = None,
    max_iter: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Track Gaussian positions per-frame by rigid ICP of the mesh to each depth cloud.

    For each timestep, finds the rigid transform (R_t, t_t) that maps the
    reference-frame aligned mesh vertices to the frame's masked depth point
    cloud, then applies the same transform to all Gaussians.

    Falls back to the reference-frame positions for frames with too few valid
    depth pixels.

    Parameters
    ----------
    mesh_verts_world: [V, 3] float64 — reference-frame aligned mesh vertices
                      (output of :func:`align_mesh_to_depth`).
    gs_xyz_ref:       [N, 3] float32 or float64 — reference-frame aligned
                      Gaussian positions.
    depth_frames:     [T, H, W] or [T, H, W, 1] float32 metric depth.
    mask_frames:      [T, H, W] bool per-frame object mask.
    intrinsic:        [3, 3] or [T, 3, 3] float64 pinhole K.
    extrinsic_frames: [T, 4, 4] float64 camera-to-world.
    preprocessed:     U-axis mirror correction flag (default True).
    max_dist:         ICP max correspondence distance. Defaults to
                      ``1.5 × 90th-percentile radius`` of *mesh_verts_world*.
    max_iter:         ICP iterations per frame (default 30).

    Returns
    -------
    gs_xyz_per_frame: [T, N, 3] float32 — per-frame world-space Gaussian positions.
    centroids:        [T, 3] float64 — per-frame median Gaussian centroid
                      (NaN for frames with no valid depth).
    """
    mesh_verts_world = np.asarray(mesh_verts_world, dtype=np.float64)
    gs_ref = np.asarray(gs_xyz_ref, dtype=np.float64)

    T = len(depth_frames)
    N = len(gs_ref)
    gs_out = np.zeros((T, N, 3), dtype=np.float32)
    centroids_out = np.full((T, 3), np.nan, dtype=np.float64)

    K_is_static = np.asarray(intrinsic).ndim == 2

    if max_dist is None:
        c = np.median(mesh_verts_world, axis=0)
        radii = np.linalg.norm(mesh_verts_world - c, axis=1)
        max_dist = float(np.percentile(radii, 90)) * 1.5

    for t in range(T):
        K = intrinsic if K_is_static else intrinsic[t]
        E = np.asarray(extrinsic_frames[t], dtype=np.float64)

        depth_pts = get_masked_depth_world_points(
            depth_frames[t], mask_frames[t], K, E, preprocessed=preprocessed
        )

        if len(depth_pts) < 3:
            gs_out[t] = gs_ref.astype(np.float32)
            continue

        R_t, t_t = _icp(
            mesh_verts_world, depth_pts, max_dist=max_dist, max_iter=max_iter
        )
        gs_t = ((R_t @ gs_ref.T).T + t_t).astype(np.float32)
        gs_out[t] = gs_t
        centroids_out[t] = np.median(gs_t, axis=0)

    return gs_out, centroids_out


def align_gaussians_to_depth(
    reconstruction: dict[str, np.ndarray],
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic: np.ndarray,
    preprocessed: bool = True,
    opacity_threshold: float = 0.5,
    extent_percentile: float = 90.0,
    refine_icp: bool = True,
    icp_max_distance: float | None = None,
    icp_max_iterations: int = 50,
) -> dict[str, Any]:
    """Align SAM3D Gaussians to the LIBERO metric depth image.

    Uses SAM3D's estimated SO3 pose (rotation quaternion) to orient the
    canonical Gaussians into world space via the camera extrinsic, then
    corrects the metric scale with depth back-projection, and optionally
    refines with point-to-point ICP so the splat surface fits the depth cloud.

    Parameters
    ----------
    reconstruction:     Output of :meth:`SAM3DClient.reconstruct`.
    depth:              [H, W] or [H, W, 1] float32  metric depth (metres).
    mask:               [H, W] bool  object mask in the same image convention.
    intrinsic:          [3, 3] float64  pinhole intrinsic K.
    extrinsic:          [4, 4] float64  camera-to-world (robosuite convention).
    preprocessed:       Passed through to :func:`get_masked_depth_world_points`.
    opacity_threshold:  Gaussians with opacity below this value are excluded
                        from scale computation and ICP source cloud.
    extent_percentile:  Percentile of radii used as the characteristic object
                        size for scale matching (default 90th).
    refine_icp:         Run ICP refinement after pose + scale initialisation
                        (default True).
    icp_max_distance:   Maximum correspondence distance for ICP (metres).
                        Defaults to ``0.4 * extent_depth`` when None.
    icp_max_iterations: Maximum ICP iterations (default 50).

    Returns
    -------
    dict with:

    * ``gs_xyz_world``   [N, 3] float32 — depth-aligned world-frame positions
    * ``scale_factor``   float — depth_extent / world-space Gaussian extent
    * ``centroid_world`` [3] float64 — world-frame object centroid from depth
    * ``depth_world_pts`` [M, 3] float64 — all masked depth points in world
    """
    gs_xyz = reconstruction["gs_xyz"]  # [N, 3] canonical
    gs_opacity = reconstruction["gs_opacity"]  # [N, 1]
    rotation_wxyz = reconstruction["rotation"]  # [4]  SAM3D estimated
    translation = reconstruction["translation"]  # [3]
    scale = reconstruction["scale"]  # [3]

    # --- 1. Back-project masked depth → world-space target cloud -------------
    depth_pts = get_masked_depth_world_points(
        depth, mask, intrinsic, extrinsic, preprocessed=preprocessed
    )
    if len(depth_pts) < 3:
        logger.warning(
            "[align_gaussians_to_depth] Too few depth points; identity fallback."
        )
        return {
            "gs_xyz_world": gs_xyz.copy(),
            "scale_factor": 1.0,
            "centroid_world": np.zeros(3, dtype=np.float64),
            "depth_world_pts": depth_pts,
        }

    centroid_depth = np.median(depth_pts, axis=0)
    radii_depth = np.linalg.norm(depth_pts - centroid_depth, axis=1)
    extent_depth = float(np.percentile(radii_depth, extent_percentile))

    # --- 2. SO3 alignment: canonical → MoGe cam → LIBERO cam → world --------
    # SAM3D's estimated rotation brings the canonical splat into MoGe camera
    # space.  We then flip Y/Z to go to LIBERO/OpenCV camera space, and finally
    # apply the camera-to-world extrinsic.
    gs_xyz_cam = transform_gaussians_to_camera_space(
        gs_xyz, rotation_wxyz, translation, scale
    )
    gs_xyz_libero = moge_to_libero_camera(gs_xyz_cam)

    R_c2w = extrinsic[:3, :3].astype(np.float64)
    t_c2w = extrinsic[:3, 3].astype(np.float64)
    gs_xyz_world = (R_c2w @ gs_xyz_libero.astype(np.float64).T).T + t_c2w  # [N, 3]

    # --- 3. Metric scale correction ------------------------------------------
    # SAM3D's pose is in MoGe normalized units, not metric.  Rescale so that
    # the world-space Gaussian extent matches the depth-cloud extent.
    active = gs_opacity.squeeze() > opacity_threshold
    pts_active = gs_xyz_world[active] if active.any() else gs_xyz_world

    centroid_gs = np.median(pts_active, axis=0)
    radii_gs = np.linalg.norm(pts_active - centroid_gs, axis=1)
    extent_gs = float(np.percentile(radii_gs, extent_percentile))
    scale_factor = extent_depth / (extent_gs + 1e-8)

    # Scale around Gaussian centroid, translate to depth centroid
    gs_xyz_world = (gs_xyz_world - centroid_gs) * scale_factor + centroid_depth

    # --- 4. ICP refinement ---------------------------------------------------
    if refine_icp:
        print("\n\nRunning ICP refinement...\n\n")
        max_dist = (
            icp_max_distance if icp_max_distance is not None else 0.4 * extent_depth
        )
        active_after_scale = gs_xyz_world[active] if active.any() else gs_xyz_world
        gs_xyz_world = _icp_refine(
            gs_xyz_all=gs_xyz_world,
            gs_xyz_active=active_after_scale,
            target=depth_pts,
            max_dist=max_dist,
            max_iter=icp_max_iterations,
        )

    return {
        "gs_xyz_world": gs_xyz_world.astype(np.float32),
        "scale_factor": float(scale_factor),
        "centroid_world": centroid_depth,
        "depth_world_pts": depth_pts,
    }


# ---------------------------------------------------------------------------
# 3D tracking
# ---------------------------------------------------------------------------


def track_object_centroids(
    depth_frames: np.ndarray,
    mask_frames: np.ndarray,
    intrinsic: np.ndarray,
    extrinsic_frames: np.ndarray,
    preprocessed: bool = True,
) -> np.ndarray:
    """Track object 3D centroid across frames using depth + segmentation masks.

    For each timestep, back-projects the masked depth pixels to world space
    and computes the median (robust centroid).

    Parameters
    ----------
    depth_frames:     [T, H, W] or [T, H, W, 1] float32  metric depth.
    mask_frames:      [T, H, W] bool  per-frame object mask.
    intrinsic:        [3, 3] or [T, 3, 3] float64  pinhole intrinsic K.
    extrinsic_frames: [T, 4, 4] float64  per-frame camera-to-world.
    preprocessed:     U-axis mirror correction flag (default True).

    Returns
    -------
    [T, 3] float64  world-frame centroid per timestep.  Frames with no
    valid depth pixels yield NaN.
    """
    T = len(depth_frames)
    centroids = np.full((T, 3), np.nan, dtype=np.float64)

    K_is_static = np.asarray(intrinsic).ndim == 2

    for t in range(T):
        dep = np.asarray(depth_frames[t], dtype=np.float64)
        msk = np.asarray(mask_frames[t], dtype=bool)
        K = intrinsic if K_is_static else intrinsic[t]
        E = np.asarray(extrinsic_frames[t], dtype=np.float64)

        if not msk.any():
            continue

        centroids[t] = get_object_centroid_world(
            dep, msk, K, E, preprocessed=preprocessed
        )

    return centroids


def propagate_gaussians(
    gs_xyz_aligned: np.ndarray,
    reference_centroid: np.ndarray,
    tracked_centroids: np.ndarray,
) -> np.ndarray:
    """Apply tracked centroid offsets to pre-aligned Gaussians.

    Given a reference frame's aligned Gaussian cloud and the tracked centroid
    trajectory, returns per-frame world-space Gaussian positions by translating
    the cloud to follow the object.

    Parameters
    ----------
    gs_xyz_aligned:      [N, 3] float32 — Gaussians aligned at the reference
                         frame (from :func:`align_gaussians_to_depth`).
    reference_centroid:  [3] float64 — centroid at the reference frame.
    tracked_centroids:   [T, 3] float64 — centroid at each timestep (NaN = skip).

    Returns
    -------
    [T, N, 3] float32  per-frame world-space Gaussian positions.
    """
    T = len(tracked_centroids)
    N = len(gs_xyz_aligned)
    all_xyz = np.zeros((T, N, 3), dtype=np.float32)

    gs_centered = gs_xyz_aligned - reference_centroid.astype(np.float32)  # [N, 3]
    for t in range(T):
        if np.any(np.isnan(tracked_centroids[t])):
            # Fall back to reference position
            all_xyz[t] = gs_xyz_aligned
        else:
            all_xyz[t] = gs_centered + tracked_centroids[t].astype(np.float32)

    return all_xyz
