"""SAM3D 3D object reconstruction and tracking demo on LIBERO PRO data.

This script demonstrates the full SAM3D pipeline:

1. **Load** a LIBERO PRO demo HDF5 file with RGB, metric depth, and
   instance segmentation.
2. **Reconstruct** a 3D Gaussian splat (and optionally a mesh) for a chosen
   object using the SAM3D server (must be running; see ``scripts/serve_sam3d.py``).
   The LIBERO metric depth at the reference frame is converted to a PyTorch3D
   pointmap and passed to the server so MoGe depth estimation is skipped and
   the reconstruction is conditioned on the actual sensor depth.
3. **Align** the reconstructed mesh to the LIBERO depth point cloud via
   scale correction + point-to-point ICP, correcting position, orientation,
   and scale offsets from SAM3D's pose estimate.
4. **Track** the object across all demo timesteps by rigid ICP of the mesh
   against each frame's masked depth point cloud.
5. **Visualise** everything in the Rerun viewer: camera images with proper
   pinhole calibration, the aligned Gaussian point cloud, the tracked
   centroid trajectory, and the full scene depth point cloud.

Usage
-----
Start the SAM3D server in one terminal::

    third_party/sam-3d-objects/.venv/bin/python scripts/serve_sam3d.py \\
        --config third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml

Then run this demo::

    python examples/sam3d_libero_demo.py \\
        --demo datasets/libero_object_new/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5 \\
        --object tomato_sauce_1 \\
        --save-mesh /tmp/object.glb

Optional flags::

    --demo-key   demo_0          # which demo trajectory to use
    --host       localhost        # SAM3D server host
    --port       8766             # SAM3D server port
    --ref-frame  0               # frame used for 3D reconstruction
    --connect    0.0.0.0:9876    # rerun gRPC server (omit to spawn viewer)
    --stride     1               # only process every N-th frame (faster)
    --save-mesh  PATH            # write GLB mesh to PATH (e.g. /tmp/obj.glb)
    --normalize-pointmap         # normalize depth pointmap scale before SAM3D
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import h5py
import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from policy_abstraction.perception.sam3d_client import (
    SAM3DClient,
    align_mesh_to_depth,
    depth_to_sam3d_pointmap,
    get_masked_depth_world_points,
    load_mesh_vertices_from_glb,
    track_object_with_mesh_icp,
    transform_gaussians_to_camera_space,
)
from policy_abstraction.utils.pcd import rgbd_to_pointcloud

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3D 3D reconstruction + tracking on a LIBERO PRO demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        required=True,
        help="Path to a LIBERO PRO demo HDF5 file.",
    )
    parser.add_argument(
        "--object",
        dest="object_name",
        default="basket",
        help=(
            "Object name to reconstruct (from segmentation_name_mapping attr). "
            "Prefix matching is used (e.g. 'basket' matches 'basket_1'). "
            "Defaults to 'basket'."
        ),
    )
    parser.add_argument(
        "--demo-key",
        default="demo_0",
        help="Key of the demo trajectory inside f['data'].",
    )
    parser.add_argument(
        "--ref-frame",
        type=int,
        default=0,
        help="Timestep used for the 3D reconstruction reference.",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="SAM3D server hostname.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="SAM3D server port.",
    )
    parser.add_argument(
        "--connect",
        metavar="HOST:PORT",
        default=None,
        help="Connect to a running rerun gRPC server (e.g. 0.0.0.0:9876). Spawns a new viewer if omitted.",
    )
    parser.add_argument(
        "--app-id",
        default="sam3d_libero",
        help="Rerun application ID.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Process every N-th frame (1 = all frames).",
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=3.0,
        help="Maximum depth (metres) for scene point-cloud visualisation.",
    )
    parser.add_argument(
        "--save-mesh",
        metavar="PATH",
        default=None,
        help="Write the reconstructed GLB mesh to this path (e.g. /tmp/object.glb).",
    )
    parser.add_argument(
        "--no-rerun",
        action="store_true",
        default=False,
        help="Skip Rerun visualisation entirely (for fast iteration / debugging).",
    )
    parser.add_argument(
        "--no-align",
        action="store_true",
        default=False,
        help="Skip mesh-to-depth ICP alignment and use the raw SAM3D pose.",
    )
    parser.add_argument(
        "--normalize-pointmap",
        action="store_true",
        default=False,
        help=(
            "Normalize the depth-derived pointmap so median |Z| ≈ 1 before "
            "passing to SAM3D (mimics MoGe's internal scale). Denormalizes "
            "SAM3D's translation/scale outputs so world positions stay metric."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _choose_object(
    seg_name_map: dict[str, str], requested: str | None
) -> tuple[int, str]:
    """Return (seg_id, name) for the target object."""
    _skip = {"OnTheGroundPanda", "NullMount", "Panda"}

    if requested is not None:
        for sid, name in seg_name_map.items():
            if name == requested or name.startswith(requested):
                return int(sid), name
        raise ValueError(
            f"Object {requested!r} not found in segmentation map: {seg_name_map}"
        )

    # Pick the first non-robot/non-ground object
    for sid, name in sorted(seg_name_map.items(), key=lambda kv: int(kv[0])):
        if not any(skip in name for skip in _skip):
            return int(sid), name

    raise ValueError(f"No suitable object found in segmentation map: {seg_name_map}")


def _prepare_image(rgb_raw: np.ndarray) -> np.ndarray:
    """Apply 180° preprocessing to match display convention (row 0 = top-left)."""
    return np.ascontiguousarray(rgb_raw[::-1, ::-1])


def _prepare_depth(depth_raw: np.ndarray) -> np.ndarray:
    """Apply 180° preprocessing and squeeze channel dim."""
    dep = np.ascontiguousarray(depth_raw[::-1, ::-1])
    if dep.ndim == 3:
        dep = dep[..., 0]
    return dep.astype(np.float32)


def _prepare_seg(seg_raw: np.ndarray) -> np.ndarray:
    """Apply 180° preprocessing and squeeze channel dim."""
    seg = np.ascontiguousarray(seg_raw[::-1, ::-1])
    if seg.ndim == 3:
        seg = seg[..., 0]
    return seg


# ---------------------------------------------------------------------------
# Rerun logging helpers
# ---------------------------------------------------------------------------


def _log_camera(entity: str, K: np.ndarray, E: np.ndarray, H: int, W: int) -> None:
    """Log camera calibration in Rerun (pinhole + camera-to-world transform)."""
    # E is camera-to-world; log the camera's world pose
    rr.log(
        entity,
        rr.Transform3D(translation=E[:3, 3], mat3x3=E[:3, :3]),
        static=True,
    )
    rr.log(entity, rr.Pinhole(image_from_camera=K, width=W, height=H), static=True)


def _log_scene_pointcloud(
    entity: str,
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    depth_max: float = 3.0,
) -> None:
    """Log a coloured RGBD point cloud to Rerun."""
    pcd = rgbd_to_pointcloud(rgb, depth, K, E, depth_max=depth_max, preprocessed=True)
    if len(pcd) == 0:
        return
    rr.log(
        entity,
        rr.Points3D(
            positions=pcd[:, :3],
            colors=(pcd[:, 3:6]).astype(np.uint8),
            radii=0.002,
        ),
    )


def _log_object_gaussians(
    entity: str,
    gs_xyz_world: np.ndarray,
    gs_colors: np.ndarray,
    gs_opacity: np.ndarray,
    min_opacity: float = 0.3,
    radius_scale: float = 0.003,
) -> None:
    """Log Gaussian splat positions as a coloured point cloud."""
    active = gs_opacity.squeeze() > min_opacity
    pts = gs_xyz_world[active]
    cols = gs_colors[active]
    if len(pts) == 0:
        return
    rr.log(
        entity,
        rr.Points3D(
            positions=pts,
            colors=(cols * 255).clip(0, 255).astype(np.uint8),
            radii=radius_scale,
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    demo_path = Path(args.demo)
    if not demo_path.exists():
        logger.error("Demo file not found: %s", demo_path)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load demo data
    # ------------------------------------------------------------------
    logger.info("Loading demo: %s / %s", demo_path.name, args.demo_key)
    with h5py.File(demo_path, "r") as f:
        data_grp = f["data"]
        demo = data_grp[args.demo_key]
        obs = demo["obs"]

        # Dataset-level metadata
        depth_near = float(data_grp.attrs.get("depth_near", 0.0106))
        depth_far = float(data_grp.attrs.get("depth_far", 530.49))
        seg_map_json = data_grp.attrs.get("segmentation_name_mapping", "{}")
        seg_name_map: dict[str, str] = json.loads(seg_map_json)

        T = obs["agentview_rgb"].shape[0]
        logger.info("Demo length: %d frames, seg map: %s", T, seg_name_map)

        # Load all arrays into memory (h5py doesn't support negative strides)
        rgb_raw = obs["agentview_rgb"][:]  # [T, H, W, 3]
        depth_raw = obs["agentview_depth"][:]  # [T, H, W, 1]
        seg_raw = obs["agentview_segmentation"][:]  # [T, H, W, 1]
        K_frames = obs["agentview_intrinsic"][:]  # [T, 3, 3]
        E_frames = obs["agentview_extrinsic"][:]  # [T, 4, 4] cam-to-world

    H, W = rgb_raw.shape[1:3]
    K0 = K_frames[0]  # intrinsic is constant across frames

    # Depth is already in metric units for LIBERO PRO datasets
    # (the values stored are actual metres, not normalised [0, 1])
    depth_is_normalised = depth_raw.max() <= 1.0
    if depth_is_normalised:
        from policy_abstraction.utils.depth import mujoco_depth_to_metric

        logger.info("Converting normalised depth to metric …")
        depth_raw = mujoco_depth_to_metric(depth_raw, depth_near, depth_far)

    # Apply 180° preprocessing to all frames
    rgb_pp = np.stack([_prepare_image(rgb_raw[t]) for t in range(T)])  # [T, H, W, 3]
    depth_pp = np.stack([_prepare_depth(depth_raw[t]) for t in range(T)])  # [T, H, W]
    seg_pp = np.stack([_prepare_seg(seg_raw[t]) for t in range(T)])  # [T, H, W]

    # ------------------------------------------------------------------
    # Choose object
    # ------------------------------------------------------------------
    seg_id, obj_name = _choose_object(seg_name_map, args.object_name)
    logger.info("Target object: %r (seg_id=%d)", obj_name, seg_id)

    mask_frames: np.ndarray = seg_pp == seg_id  # [T, H, W] bool

    # ------------------------------------------------------------------
    # 3D reconstruction via SAM3D server
    # ------------------------------------------------------------------
    ref_t = args.ref_frame
    ref_rgb = rgb_pp[ref_t]  # [H, W, 3] uint8
    ref_mask = mask_frames[ref_t]  # [H, W] bool
    ref_depth = depth_pp[ref_t]  # [H, W] float32
    ref_E = E_frames[ref_t]  # [4, 4] cam-to-world

    if not ref_mask.any():
        logger.error(
            "Object %r not visible at frame %d. Try --ref-frame N.", obj_name, ref_t
        )
        sys.exit(1)

    # Build a PyTorch3D pointmap from the LIBERO metric depth so the server
    # uses the sensor depth instead of running MoGe estimation.
    logger.info("Building SAM3D pointmap from LIBERO depth (frame %d) …", ref_t)
    ref_pointmap = depth_to_sam3d_pointmap(ref_depth, K0, preprocessed=True)

    # --- Pointmap experiment flags -------------------------------------------
    # Compute pm_scale for denormalization (used later for both flags)
    pm_scale = 1.0

    if args.normalize_pointmap:
        z_vals = ref_pointmap[..., 2].ravel()
        z_nonzero = z_vals[np.abs(z_vals) > 1e-4]
        if len(z_nonzero) > 0:
            pm_scale = float(np.median(np.abs(z_nonzero)))
            ref_pointmap = (ref_pointmap / pm_scale).astype(np.float32)
            logger.info(
                "--normalize-pointmap: scale=%.4f (median |Z|=%.4f → 1.0)",
                pm_scale,
                pm_scale,
            )
        else:
            logger.warning(
                "--normalize-pointmap: no valid Z values, skipping normalization."
            )

    logger.info("Connecting to SAM3D server at %s:%d …", args.host, args.port)
    with SAM3DClient(host=args.host, port=args.port) as sam3d_client:
        logger.info("Running 3D reconstruction for %r (frame %d) …", obj_name, ref_t)
        # log image and mask to Rerun for debugging

        recon = sam3d_client.reconstruct(
            ref_rgb,
            ref_mask,
            seed=42,
            pointmap=ref_pointmap,
            return_mesh=True,
        )

    logger.info(
        "Reconstruction done: %d Gaussians, pointmap %s",
        len(recon["gs_xyz"]),
        recon["pointmap"].shape,
    )

    # Optionally save the mesh
    if args.save_mesh is not None:
        mesh_bytes = recon.get("mesh_glb_bytes")
        if mesh_bytes:
            SAM3DClient.save_mesh(mesh_bytes, args.save_mesh)
            logger.info("Saved mesh to %s", args.save_mesh)
        else:
            logger.warning("No mesh bytes in reconstruction result; skipping save.")

    # ------------------------------------------------------------------
    # Initial world-space Gaussian + mesh transform from SAM3D pose
    # ------------------------------------------------------------------
    from scipy.spatial.transform import Rotation as ScipyR

    R_c2w = ref_E[:3, :3].astype(np.float64)
    t_c2w = ref_E[:3, 3].astype(np.float64)

    q = recon["rotation"]
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    R_sam3d = ScipyR.from_quat(q_xyzw).as_matrix()  # [3, 3]
    scale_vec = recon["scale"].astype(np.float64) * pm_scale  # [3] metric
    T_sam3d = recon["translation"].astype(np.float64) * pm_scale  # [3] metric

    # M_flip: preprocessed-PyTorch3D → original-OpenCV (negate X and Z, keep Y)
    M_flip = np.diag([-1.0, 1.0, -1.0])
    # M_gc: GLB z-up → y-up rotation baked into vertex storage by to_glb()
    # (column-vector form: p_canonical = M_gc @ p_mesh)
    M_gc = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

    mat3x3_pose = R_c2w @ M_flip @ R_sam3d.T @ np.diag(scale_vec) @ M_gc
    mesh_t_pose = R_c2w @ M_flip @ T_sam3d + t_c2w

    # Gaussians in world space (SAM3D pose only, no alignment yet)
    # If the pointmap was normalised, SAM3D's translation/scale are in
    # normalised units; pm_scale has already been applied above.
    gs_xyz_out = transform_gaussians_to_camera_space(
        recon["gs_xyz"],
        recon["rotation"],
        recon["translation"] * pm_scale,
        recon["scale"] * pm_scale,
    )
    gs_xyz_cam = gs_xyz_out
    # gs_xyz_cam is in (X right, Y up, Z forward) — a consistent intermediate space
    # produced by transform_gaussians_to_camera_space (moge_to_libero_camera negates
    # PyTorch3D X and Z, and the translation is now also converted consistently).
    # The extrinsic R_c2w expects OpenCV camera convention (X right, Y down, Z forward),
    # so flip Y before transforming to world space.
    gs_xyz_cam_cv = gs_xyz_cam.astype(np.float64).copy()
    gs_xyz_cam_cv[:, 1] *= -1  # Y up → Y down (PyTorch3D Y-up → OpenCV Y-down)
    gs_xyz_world_pose = (R_c2w @ gs_xyz_cam_cv.T).T + t_c2w

    # ------------------------------------------------------------------
    # Mesh-to-depth alignment (scale + rigid ICP) at reference frame
    # ------------------------------------------------------------------
    mesh_bytes = recon.get("mesh_glb_bytes")
    ref_depth_pts = get_masked_depth_world_points(ref_depth, ref_mask, K0, ref_E)

    if mesh_bytes and len(ref_depth_pts) >= 3 and not args.no_align:
        logger.info("Aligning mesh to reference depth cloud …")
        verts_mesh = load_mesh_vertices_from_glb(mesh_bytes)  # [V, 3] GLB space
        verts_world_pose = (
            mat3x3_pose @ verts_mesh.T
        ).T + mesh_t_pose  # [V, 3] world (pose only)

        align_res = align_mesh_to_depth(
            verts_world_pose, ref_depth_pts, adjust_scale=True
        )

        R_al = align_res["R"]  # [3, 3] rigid from ICP
        t_al = align_res["t"]  # [3]
        s_al = align_res["scale"]  # uniform scale
        c_mesh = align_res["c_mesh"]
        c_depth = align_res["c_depth"]
        verts_world_aligned = align_res["mesh_verts_aligned"]  # [V, 3]

        # Apply same composite transform to Gaussians:
        #   pts_pre = (pts - c_mesh) * scale + c_depth
        #   pts_aligned = (R @ pts_pre.T).T + t
        gs_pre = (gs_xyz_world_pose - c_mesh) * s_al + c_depth
        gs_xyz_aligned = ((R_al @ gs_pre.T).T + t_al).astype(np.float32)

        # Update mesh Rerun transform to reflect alignment:
        #   p_world = R_al @ ((mat3x3_pose @ p_mesh + mesh_t_pose - c_mesh)*s + c_depth) + t_al
        #           = (s_al * R_al @ mat3x3_pose) @ p_mesh
        #             + R_al @ ((mesh_t_pose - c_mesh)*s_al + c_depth) + t_al
        mat3x3_final = s_al * R_al @ mat3x3_pose
        mesh_t_final = R_al @ ((mesh_t_pose - c_mesh) * s_al + c_depth) + t_al

        logger.info(
            "Mesh alignment: scale=%.4f  depth centroid: %s",
            s_al,
            np.round(c_depth, 3),
        )
    else:
        if args.no_align:
            logger.info("--no-align set; using raw SAM3D pose.")
        elif not mesh_bytes:
            logger.warning("No mesh in reconstruction; skipping mesh-based alignment.")
        else:
            logger.warning("Too few depth points for alignment; using raw SAM3D pose.")
        verts_world_aligned = None
        gs_xyz_aligned = gs_xyz_world_pose.astype(np.float32)
        mat3x3_final = mat3x3_pose
        mesh_t_final = mesh_t_pose
        s_al = 1.0  # no alignment applied

    # ------------------------------------------------------------------
    # Per-frame tracking via mesh ICP
    # ------------------------------------------------------------------
    if verts_world_aligned is not None:
        logger.info("Tracking object across %d frames via mesh ICP …", T)
        gs_xyz_per_frame, centroids = track_object_with_mesh_icp(
            verts_world_aligned,
            gs_xyz_aligned,
            depth_frames=depth_pp[:2],
            mask_frames=mask_frames,
            intrinsic=K0,
            extrinsic_frames=E_frames,
            preprocessed=True,
        )  # [T, N, 3], [T, 3]
    else:
        # Fallback: static (no tracking)
        logger.warning("Mesh not available; Gaussians will remain at reference pose.")
        gs_xyz_per_frame = (
            np.broadcast_to(gs_xyz_aligned[None], (T, *gs_xyz_aligned.shape))
            .copy()
            .astype(np.float32)
        )

        centroids = np.full((T, 3), np.nan, dtype=np.float64)

    valid_centroids = centroids[~np.any(np.isnan(centroids), axis=1)]
    logger.info("Valid tracking frames: %d / %d", len(valid_centroids), T)

    # ------------------------------------------------------------------
    # Debug: alignment quality at reference frame
    # ------------------------------------------------------------------
    active = recon["gs_opacity"].squeeze() > 0.3
    gs_active = gs_xyz_aligned[active] if active.any() else gs_xyz_aligned
    gs_centroid = np.median(gs_active, axis=0)
    ref_centroid = (
        np.median(ref_depth_pts, axis=0) if len(ref_depth_pts) else np.zeros(3)
    )
    logger.info("Gaussian centroid (world):  %s", np.round(gs_centroid, 4))
    logger.info("Depth ref centroid (world): %s", np.round(ref_centroid, 4))
    logger.info(
        "Centroid L2 distance:       %.4f m",
        float(np.linalg.norm(gs_centroid - ref_centroid)),
    )

    if args.no_rerun:
        logger.info("--no-rerun set, skipping visualisation.")
        return

    # ------------------------------------------------------------------
    # Rerun visualisation
    # ------------------------------------------------------------------
    logger.info("Setting up Rerun viewer …")
    rr.init(args.app_id, spawn=False)
    if args.connect:
        rr.connect_grpc(f"rerun+http://{args.connect}/proxy")
        logger.info("Connected to Rerun gRPC at %s", args.connect)
    else:
        rr.spawn()
        logger.info("Spawned Rerun viewer")

    rr.log("debug/reference_rgb", rr.Image(ref_rgb), static=True)
    rr.log(
        "debug/reference_mask", rr.Image((ref_mask * 255).astype(np.uint8)), static=True
    )
    rr.log(
        "debug/reference_pointmap",
        rr.Points3D(
            positions=ref_pointmap.reshape(-1, 3),
            colors=np.tile(ref_rgb.reshape(-1, 3), (1, 1)),
        ),
        static=True,
    )
    # log the splat in original SAM3D pose for debugging
    _log_object_gaussians(
        "debug/gaussians_sam3d_pose",
        gs_xyz_cam.astype(np.float32),
        recon["gs_colors"],
        recon["gs_opacity"],
    )

    rr.log("task", rr.TextDocument(demo_path.stem.replace("_", " ")), static=True)
    mesh_info = f"Mesh: {args.save_mesh}" if args.save_mesh else "Mesh: not saved"
    align_info = (
        f"Mesh-depth alignment scale: {s_al:.4f}"
        if verts_world_aligned is not None
        else "Alignment: mesh unavailable"
    )
    rr.log(
        "scene/object/info",
        rr.TextDocument(
            f"Object: {obj_name}\n"
            f"Seg ID: {seg_id}\n"
            f"Gaussians: {len(recon['gs_xyz'])}\n"
            f"{align_info}\n"
            f"Reference frame: {ref_t}\n"
            f"{mesh_info}",
            media_type="text/plain",
        ),
        static=True,
    )

    # Log static camera calibration for agentview (fixed camera)
    _log_camera("camera/agentview", K0, E_frames[0], H, W)

    # Log SAM3D reference reconstruction (at reference frame, static)
    rr.log(
        "scene/object/reference_reconstruction",
        rr.Points3D(
            positions=gs_xyz_aligned[recon["gs_opacity"].squeeze() > 0.3],
            colors=(recon["gs_colors"][recon["gs_opacity"].squeeze() > 0.3] * 255)
            .clip(0, 255)
            .astype(np.uint8),
            radii=0.003,
        ),
        static=True,
    )

    # Log reconstructed mesh in world space (static)
    # mat3x3_final / mesh_t_final already incorporate the mesh-depth alignment.
    if mesh_bytes:
        rr.log(
            "scene/object/mesh",
            rr.Transform3D(mat3x3=mat3x3_final, translation=mesh_t_final),
            static=True,
        )
        rr.log(
            "scene/object/mesh",
            rr.Asset3D(contents=mesh_bytes, media_type=rr.MediaType.GLB),
            static=True,
        )
        logger.info("Logged mesh to Rerun at scene/object/mesh")

    # Log the full centroid trajectory (static)
    valid_mask = ~np.any(np.isnan(centroids), axis=1)
    if valid_mask.sum() >= 2:
        traj = centroids[valid_mask]
        segments = [
            [traj[i].tolist(), traj[i + 1].tolist()] for i in range(len(traj) - 1)
        ]
        rr.log(
            "scene/object/trajectory",
            rr.LineStrips3D(segments, colors=[[255, 200, 0, 200]]),
            static=True,
        )

    # Log reference-frame depth world points for the object (static)
    # ref_depth_pts was already computed during mesh-depth alignment.
    if len(ref_depth_pts) > 0:
        ref_obj_colour = ref_rgb[ref_mask].mean(axis=0).astype(np.uint8)
        rr.log(
            "scene/object/reference_depth_points",
            rr.Points3D(
                positions=ref_depth_pts.astype(np.float32),
                colors=np.tile(ref_obj_colour, (len(ref_depth_pts), 1)),
                radii=0.003,
            ),
            static=True,
        )

    # Per-frame logging
    logger.info("Streaming frames to Rerun …")
    frame_indices = range(0, T, args.stride)
    for t in frame_indices[:1]:
        rr.set_time("frame", sequence=t)

        # Camera image
        rr.log("camera/agentview", rr.Image(rgb_pp[t]))

        # Depth overlay on the object mask
        dep_t = depth_pp[t]
        msk_t = mask_frames[t]
        dep_vis = np.zeros((*dep_t.shape, 3), dtype=np.uint8)
        if msk_t.any():
            import cv2

            d_norm = np.clip(dep_t / args.depth_max, 0, 1)
            dep_colour = cv2.applyColorMap(
                (d_norm * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            dep_vis[msk_t] = dep_colour[msk_t]
        rr.log("camera/agentview/depth_mask_overlay", rr.Image(dep_vis))

        # Object mask
        mask_img = msk_t[..., None] * np.array([0, 200, 255], dtype=np.uint8)
        rr.log("camera/agentview/object_mask", rr.Image(mask_img.astype(np.uint8)))

        # Scene point cloud (agentview)
        _log_scene_pointcloud(
            "scene/pointcloud/agentview",
            rgb_pp[t],
            dep_t,
            K0,
            E_frames[t],
            depth_max=args.depth_max,
        )

        # Tracked object Gaussians
        _log_object_gaussians(
            "scene/object/gaussians",
            gs_xyz_per_frame[t],
            recon["gs_colors"],
            recon["gs_opacity"],
        )

        # Tracked centroid marker
        c = centroids[t]
        if not np.any(np.isnan(c)):
            rr.log(
                "scene/object/centroid",
                rr.Points3D(
                    positions=[c.tolist()],
                    colors=[[255, 100, 0, 255]],
                    radii=0.01,
                ),
            )
            rr.log("scene/object/centroid_x", rr.Scalars(float(c[0])))
            rr.log("scene/object/centroid_y", rr.Scalars(float(c[1])))
            rr.log("scene/object/centroid_z", rr.Scalars(float(c[2])))

    logger.info("Done. Close the Rerun window to exit.")

    # Keep Rerun alive if spawned
    if not args.connect:
        import time

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
