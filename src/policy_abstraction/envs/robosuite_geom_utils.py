"""Robosuite/MuJoCo helpers shared across env-server backends.

Every function here only touches plain robosuite/MuJoCo state (``env.sim``,
``env.segmentation_id_mapping``, ``env.instances_to_ids``) so the same code
serves both LIBERO and RoboCasa servers without modification.
"""

from __future__ import annotations

import numpy as np

# MuJoCo geom type constants (mjtGeom enum)
_GEOM_PLANE = 0
_GEOM_SPHERE = 2
_GEOM_CAPSULE = 3
_GEOM_CYLINDER = 5
_GEOM_BOX = 6
_GEOM_MESH = 7


# ---------------------------------------------------------------------------
# Camera intrinsics / extrinsics
# ---------------------------------------------------------------------------


def get_camera_intrinsic_matrix(sim, camera_name, camera_height, camera_width):
    """3×3 pinhole intrinsic matrix from a MuJoCo camera's fovy."""
    cam_id = sim.model.camera_name2id(camera_name)
    fovy = sim.model.cam_fovy[cam_id]
    f = 0.5 * camera_height / np.tan(fovy * np.pi / 360)
    return np.array([[f, 0, camera_width / 2], [0, f, camera_height / 2], [0, 0, 1]])


def get_camera_extrinsic_matrix(sim, camera_name):
    """4×4 world→camera extrinsic with the OpenCV-style axis correction."""
    import robosuite.utils.transform_utils as T  # local import keeps module light

    cam_id = sim.model.camera_name2id(camera_name)
    camera_pos = sim.data.cam_xpos[cam_id]
    camera_rot = sim.data.cam_xmat[cam_id].reshape(3, 3)
    R = T.make_pose(camera_pos, camera_rot)

    # MuJoCo camera body axis correction so x/y align with the image and z
    # points along the viewpoint.
    camera_axis_correction = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return R @ camera_axis_correction


# ---------------------------------------------------------------------------
# GT object point clouds
# ---------------------------------------------------------------------------


def _sample_sphere(radius: float, n: int, rng: np.random.Generator) -> np.ndarray:
    phi = rng.uniform(0, 2 * np.pi, n)
    cos_theta = rng.uniform(-1, 1, n)
    sin_theta = np.sqrt(1 - cos_theta**2)
    return radius * np.column_stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta]
    )


def _sample_box(
    half_extents: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    hx, hy, hz = half_extents
    face_areas = np.array([hy * hz, hx * hz, hx * hy]) * 4
    probs = face_areas / face_areas.sum()
    counts = rng.multinomial(n, probs)
    pts = []
    for axis, cnt in enumerate(counts):
        if cnt == 0:
            continue
        u = rng.uniform(-1, 1, cnt)
        v = rng.uniform(-1, 1, cnt)
        sign = rng.choice([-1.0, 1.0], cnt)
        p = np.zeros((cnt, 3), dtype=np.float64)
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3
        p[:, axis] = sign * half_extents[axis]
        p[:, a1] = u * half_extents[a1]
        p[:, a2] = v * half_extents[a2]
        pts.append(p)
    return np.concatenate(pts, axis=0)


def _sample_cylinder(
    radius: float, half_len: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    cap_area = np.pi * radius**2
    barrel_area = 2 * np.pi * radius * (2 * half_len)
    total = 2 * cap_area + barrel_area
    n_barrel = int(n * barrel_area / total)
    n_caps = n - n_barrel
    pts = []
    if n_barrel > 0:
        theta = rng.uniform(0, 2 * np.pi, n_barrel)
        z = rng.uniform(-half_len, half_len, n_barrel)
        pts.append(np.column_stack([radius * np.cos(theta), radius * np.sin(theta), z]))
    if n_caps > 0:
        r = radius * np.sqrt(rng.uniform(0, 1, n_caps))
        theta = rng.uniform(0, 2 * np.pi, n_caps)
        sign = rng.choice([-1.0, 1.0], n_caps)
        pts.append(
            np.column_stack([r * np.cos(theta), r * np.sin(theta), sign * half_len])
        )
    return np.concatenate(pts, axis=0)


def _sample_geom_surface(
    sim, geom_id: int, n_points: int, rng: np.random.Generator
) -> "np.ndarray | None":
    gtype = int(sim.model.geom_type[geom_id])
    size = sim.model.geom_size[geom_id]

    if gtype == _GEOM_SPHERE:
        return _sample_sphere(size[0], n_points, rng)
    if gtype == _GEOM_BOX:
        return _sample_box(size[:3], n_points, rng)
    if gtype in (_GEOM_CYLINDER, _GEOM_CAPSULE):
        return _sample_cylinder(size[0], size[1], n_points, rng)
    if gtype == _GEOM_MESH:
        mesh_id = sim.model.geom_dataid[geom_id]
        if mesh_id < 0:
            return None
        vert_adr = sim.model.mesh_vertadr[mesh_id]
        vert_num = sim.model.mesh_vertnum[mesh_id]
        verts = sim.model.mesh_vert[vert_adr : vert_adr + vert_num].copy()
        if verts.shape[0] == 0:
            return None
        idx = rng.choice(verts.shape[0], size=n_points, replace=True)
        return verts[idx]
    return None


def get_gt_object_pointclouds(env, n_points_per_geom: int = 500) -> dict:
    """World-frame GT point clouds for each segmented object.

    Returns ``{seg_id: {"name": str, "points": [N, 3] float32,
    "colors": [3] float32}}`` keyed by mask IDs (raw_seg_id + 1).
    Compatible with both LIBERO's ``SegmentationRenderEnv`` and RoboCasa's
    robosuite envs that expose ``segmentation_id_mapping`` and
    ``instances_to_ids``.
    """
    sim = env.sim
    rng = np.random.default_rng(0)

    instances_to_ids: dict = {}
    for src in [env, getattr(env, "env", None)]:
        if src is not None:
            cand = getattr(src, "instances_to_ids", None)
            if cand:
                instances_to_ids = cand
                break
    if not instances_to_ids:
        model = getattr(getattr(env, "env", None), "model", None)
        if model is not None:
            instances_to_ids = getattr(model, "instances_to_ids", {})

    seg_mapping: dict = getattr(env, "segmentation_id_mapping", {})

    result: dict = {}
    for seg_id, instance_name in seg_mapping.items():
        mask_id = seg_id + 1

        id_groups = instances_to_ids.get(instance_name, {})
        geom_ids = (
            id_groups.get("geom", []) if isinstance(id_groups, dict) else id_groups
        )
        if not geom_ids:
            continue

        all_pts = []
        for gid in geom_ids:
            local_pts = _sample_geom_surface(sim, gid, n_points_per_geom, rng)
            if local_pts is None:
                continue
            pos = sim.data.geom_xpos[gid]
            rot = sim.data.geom_xmat[gid].reshape(3, 3)
            world_pts = (rot @ local_pts.T).T + pos
            all_pts.append(world_pts)

        if not all_pts:
            continue

        points = np.concatenate(all_pts, axis=0).astype(np.float32)
        rgba = sim.model.geom_rgba[geom_ids[0]]
        color = (rgba[:3] * 255).astype(np.float32)

        result[str(mask_id)] = {
            "name": instance_name,
            "points": points,
            "colors": color,
        }

    return result


__all__ = [
    "get_camera_intrinsic_matrix",
    "get_camera_extrinsic_matrix",
    "get_gt_object_pointclouds",
]
