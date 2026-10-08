"""Source-agnostic view of one demonstration.

Demo analysis and reference-trajectory computation used to read LIBERO HDF5
groups directly, which is why they only ever worked for one source. Everything
they actually need is small and common to every format:

* an upright RGB frame,
* metric depth registered to that frame,
* a segmentation labelling for that frame, plus id → name,
* the camera intrinsic and extrinsic that place the depth in the robot frame,
* the end-effector position over time.

:class:`DemoEpisode` is that contract. Each adapter owns the quirks of its own
format — LIBERO's 180° frame convention and normalised MuJoCo depth buffer, a
``.vla``'s millimetre depth on a stride — so callers never branch on the
source. Adding a format means adding an adapter here and a line in
:func:`build_demo_episode`; nothing downstream changes.

Two things vary by source rather than being universal, and the contract makes
both explicit instead of assuming the simulator's answer:

``frames_with_depth()``
    Which timesteps carry 3-D. A simulator renders depth every frame; a real
    recording stores it on a stride because it is uncompressed. Callers iterate
    this rather than ``range(len(episode))``.

``segmentation()``
    Ground truth in a simulator, and nothing at all on a real robot. A real
    episode is handed a *segmenter* that produces masks from RGB-D; without one
    it reports no segmentation rather than inventing labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


@runtime_checkable
class DemoEpisode(Protocol):
    """One demonstration, in the terms analysis needs.

    Every image-shaped return is **upright** and mutually registered: the
    adapter has already applied whatever flip its format stores frames in, to
    RGB, depth and segmentation alike.
    """

    source_kind: str
    task: str

    frames_preprocessed: bool
    """Whether this format stores frames 180°-rotated.

    The adapter un-rotates them either way, so callers see upright images
    regardless — but back-projection still has to mirror the u axis to undo
    that rotation, and doing it for a camera that publishes upright frames
    mirrors every reconstructed object about the image centre-line while
    leaving the scene superficially plausible. This is the same distinction
    ``--perception.flip-u auto`` makes for the live cameras, carried on the
    demo so a retrieved episode answers it for itself.
    """

    def __len__(self) -> int:
        """Number of timesteps."""
        ...

    def rgb(self, t: int) -> np.ndarray:
        """``(H, W, 3)`` uint8."""
        ...

    def depth_m(self, t: int) -> np.ndarray | None:
        """``(H, W)`` float32 depth in **metres**, or ``None``."""
        ...

    def segmentation(self, t: int) -> np.ndarray | None:
        """``(H, W)`` integer instance labels (0 = background), or ``None``."""
        ...

    def intrinsic(self, t: int) -> np.ndarray | None:
        """``(3, 3)`` pinhole intrinsic for the primary camera."""
        ...

    def extrinsic(self, t: int) -> np.ndarray | None:
        """``(4, 4)`` camera pose in the robot base frame."""
        ...

    def ee_positions(self) -> np.ndarray:
        """``(T, 3)`` float32 end-effector positions."""
        ...

    def ee_orientations(self) -> np.ndarray | None:
        """``(T, 3)`` float32 end-effector orientations as axis-angle vectors.

        ``None`` when the format stores no orientation. Together with
        :meth:`gripper_states` this is what turns an EE track into finger-tip
        positions, so a pair naming ``gripper_l``/``gripper_r`` can be measured
        against the demo at all.
        """
        ...

    def gripper_states(self) -> np.ndarray | None:
        """``(T, k)`` raw gripper joint positions, or ``None``.

        ``k == 2`` for a two-finger arm publishing both joints (Panda),
        ``k == 1`` for a single-DOF driver publishing total openness
        (Robotiq). :func:`~policy_abstraction.utils.gripper._finger_joints`
        accepts both, given the matching :meth:`gripper_geometry`.
        """
        ...

    def gripper_geometry(self) -> str | None:
        """Name of the gripper geometry this demo was recorded with.

        ``None`` means the Panda default. A real recording names its own
        (``"robotiq_2f140"``), and using the live robot's geometry instead
        would re-measure the demo with the wrong fingers.
        """
        ...

    def actions(self) -> np.ndarray | None:
        """``(T-1, action_dim)`` recorded actions, or ``None``."""
        ...

    def seg_name_map(self) -> dict[int, str]:
        """``{instance id: object name}`` for :meth:`segmentation`."""
        ...

    def frames_with_depth(self) -> list[int]:
        """Timesteps for which :meth:`depth_m` returns an array."""
        ...


# ---------------------------------------------------------------------------
# LIBERO / RoboCasa HDF5
# ---------------------------------------------------------------------------


class LiberoHDF5Episode:
    """A demo inside a LIBERO-schema HDF5 file.

    Frames are stored 180°-rotated and depth as a normalised MuJoCo buffer;
    both are corrected here so callers see the same upright metric world every
    other source presents.
    """

    source_kind = "libero"
    # MuJoCo renders these upside down; the adapter un-rotates them, and
    # back-projection must still undo the u mirror that rotation implies.
    frames_preprocessed = True

    def __init__(
        self,
        source_path: str,
        demo_key: str,
        task: str = "",
        camera: str = "agentview",
    ) -> None:
        self.source_path = str(source_path)
        self.demo_key = str(demo_key)
        self.task = task
        self._camera = camera
        self._cache: dict[str, np.ndarray] = {}
        self._names: dict[int, str] | None = None

    # -- internals ---------------------------------------------------------

    def _obs(self, key: str) -> np.ndarray | None:
        """Read one observation array, applying the format's corrections."""
        if key in self._cache:
            return self._cache[key]
        import h5py

        with h5py.File(self.source_path, "r") as f:
            group = f["data"][self.demo_key]["obs"]  # type: ignore[index]
            if key not in group:
                return None
            arr = np.asarray(group[key][:])
        if key.endswith(("_rgb", "_depth", "_segmentation")):
            # The stored frames are rotated 180°; every consumer wants upright.
            arr = arr[:, ::-1, ::-1, ...]
        if key.endswith("_depth"):
            arr = self._to_metric(arr)
        self._cache[key] = arr
        return arr

    @staticmethod
    def _to_metric(depth: np.ndarray) -> np.ndarray:
        """MuJoCo writes a normalised buffer; convert only if it looks like one."""
        from policy_abstraction.envs.libero.config import (
            LIBERO_DEPTH_FAR,
            LIBERO_DEPTH_NEAR,
        )
        from policy_abstraction.utils.depth import mujoco_depth_to_metric

        arr = np.asarray(depth, dtype=np.float32)
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = mujoco_depth_to_metric(
                arr, near=LIBERO_DEPTH_NEAR, far=LIBERO_DEPTH_FAR
            )
        return arr.astype(np.float32)

    # -- DemoEpisode -------------------------------------------------------

    def __len__(self) -> int:
        arr = self._obs(f"{self._camera}_rgb")
        return 0 if arr is None else int(arr.shape[0])

    def rgb(self, t: int) -> np.ndarray:
        arr = self._obs(f"{self._camera}_rgb")
        if arr is None:
            raise KeyError(f"{self._camera}_rgb missing from {self.demo_key}")
        return np.ascontiguousarray(arr[t])

    def depth_m(self, t: int) -> np.ndarray | None:
        arr = self._obs(f"{self._camera}_depth")
        if arr is None:
            return None
        frame = arr[t]
        return np.ascontiguousarray(frame[..., 0] if frame.ndim == 3 else frame)

    def segmentation(self, t: int) -> np.ndarray | None:
        arr = self._obs(f"{self._camera}_segmentation")
        if arr is None:
            return None
        frame = arr[t]
        return np.ascontiguousarray(frame[..., 0] if frame.ndim == 3 else frame)

    def intrinsic(self, t: int) -> np.ndarray | None:
        arr = self._obs(f"{self._camera}_intrinsic")
        if arr is None:
            return None
        K = np.asarray(arr[t], dtype=np.float64)
        return K[:3, :3] if K.shape == (4, 4) else K

    def extrinsic(self, t: int) -> np.ndarray | None:
        arr = self._obs(f"{self._camera}_extrinsic")
        return None if arr is None else np.asarray(arr[t], dtype=np.float64)

    def ee_positions(self) -> np.ndarray:
        arr = self._obs("ee_pos")
        if arr is None:
            raise KeyError(f"ee_pos missing from {self.demo_key}")
        return np.asarray(arr, dtype=np.float32)

    def ee_orientations(self) -> np.ndarray | None:
        arr = self._obs("ee_ori")
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    def gripper_states(self) -> np.ndarray | None:
        arr = self._obs("gripper_states")
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    def gripper_geometry(self) -> str | None:
        # LIBERO and the RoboCasa HDF5 collection both record a Panda, which
        # is the default geometry.
        return None

    def actions(self) -> np.ndarray | None:
        import h5py

        try:
            with h5py.File(self.source_path, "r") as f:
                grp = f["data"][self.demo_key]  # type: ignore[index]
                if "actions" not in grp:
                    return None
                return np.asarray(grp["actions"][:], dtype=np.float32)
        except Exception:
            return None

    def seg_name_map(self) -> dict[int, str]:
        if self._names is None:
            import h5py

            names: dict[int, str] = {}
            try:
                with h5py.File(self.source_path, "r") as f:
                    raw = f["data"].attrs.get("segmentation_name_mapping")  # type: ignore[union-attr]
                if raw:
                    names = {int(k): v for k, v in json.loads(raw).items()}
            except Exception:
                names = {}
            self._names = names
        return dict(self._names)

    def frames_with_depth(self) -> list[int]:
        # A simulator renders depth for every frame it renders colour.
        return (
            list(range(len(self)))
            if self._obs(f"{self._camera}_depth") is not None
            else []
        )


# ---------------------------------------------------------------------------
# .vla rollout
# ---------------------------------------------------------------------------


class VlaEpisode:
    """A ``.vla`` rollout recorded by ``scripts/evaluate.py``.

    Depth is stored as uint16 millimetres on a stride, indexed by the
    ``observation/depth_timestep`` stream, so :meth:`frames_with_depth` is a
    sparse subset of the timeline rather than all of it. Real cameras publish
    no ground-truth masks, so :meth:`segmentation` needs a *segmenter*.
    """

    source_kind = "vla"
    # A real camera publishes upright frames (CameraSpec.image_transform
    # "none"), so there is no rotation for back-projection to undo.
    frames_preprocessed = False

    def __init__(
        self,
        source_path: str,
        demo_key: str,
        segmenter=None,
    ) -> None:
        """
        Parameters
        ----------
        source_path:
            Directory holding the ``.vla`` files.
        demo_key:
            Episode filename stem.
        segmenter:
            Optional ``(rgb, depth_m, K, E, task) -> (labels, {id: name})``.
            Called in increasing ``t`` so a tracking backend sees the episode
            in order. Without one the episode reports no segmentation, which
            is honest: nothing in a real recording labels its own pixels.
        """
        self._root = Path(source_path).resolve()
        self.source_path = str(self._root)
        self.demo_key = str(demo_key)
        self._segmenter = segmenter
        self._data: dict | None = None
        self._meta: dict | None = None
        self._seg_cache: dict[int, np.ndarray | None] = {}
        self._names: dict[int, str] = {}

    # -- internals ---------------------------------------------------------

    @property
    def _sidecar(self) -> dict:
        if self._meta is None:
            path = self._root / f"{self.demo_key}.json"
            if not path.exists():
                raise FileNotFoundError(f"episode {self.demo_key!r} has no sidecar")
            with path.open() as fh:
                self._meta = json.load(fh)
        return self._meta

    @property
    def task(self) -> str:  # type: ignore[override]
        return str(self._sidecar.get("task_description", ""))

    @property
    def _loaded(self) -> dict:
        if self._data is None:
            import robodm

            traj = robodm.Trajectory(str(self._root / f"{self.demo_key}.vla"), mode="r")
            try:
                self._data = traj.load()
            finally:
                traj.close()
            if not self._data:
                raise RuntimeError(f"no data decoded from {self.demo_key}.vla")
        return self._data

    @property
    def _primary_camera(self) -> str:
        for cam in self._sidecar.get("env_spec", {}).get("cameras", []):
            if cam.get("is_primary"):
                return str(cam["name"])
        raise KeyError(f"{self.demo_key!r}: EnvSpec names no primary camera")

    def _depth_index(self) -> dict[int, int]:
        """``{timestep: row in the depth stream}``."""
        idx = self._loaded.get("observation/depth_timestep")
        key = f"observation/{self._primary_camera}_depth"
        if idx is None or key not in self._loaded:
            return {}
        steps = np.asarray(idx).ravel().astype(int)
        n = len(np.asarray(self._loaded[key]))
        return {int(t): i for i, t in enumerate(steps[:n])}

    def _camera_params(self, t: int) -> dict:
        params = self._loaded.get("camera_params")
        if params is None or t >= len(params):
            return {}
        entry = params[t]
        # Stored one dict per step, wrapped in a length-1 object array.
        while isinstance(entry, np.ndarray) and entry.dtype == object and entry.size:
            entry = entry.ravel()[0]
        if not isinstance(entry, dict):
            return {}
        return entry.get(self._primary_camera, {}) or {}

    # -- DemoEpisode -------------------------------------------------------

    def __len__(self) -> int:
        arr = self._loaded.get(f"observation/{self._primary_camera}_image")
        return 0 if arr is None else int(len(arr))

    def rgb(self, t: int) -> np.ndarray:
        key = f"observation/{self._primary_camera}_image"
        arr = self._loaded.get(key)
        if arr is None:
            raise KeyError(f"{key} missing from {self.demo_key}.vla")
        return np.ascontiguousarray(np.asarray(arr[t]))

    def depth_m(self, t: int) -> np.ndarray | None:
        row = self._depth_index().get(int(t))
        if row is None:
            return None
        key = f"observation/{self._primary_camera}_depth"
        mm = np.asarray(self._loaded[key][row], dtype=np.float32)
        # 0 is "no return", not "at the camera"; NaN keeps it out of the cloud.
        metres = mm / 1000.0
        metres[mm == 0] = np.nan
        return np.ascontiguousarray(metres)

    def segmentation(self, t: int) -> np.ndarray | None:
        if self._segmenter is None:
            return None
        if t in self._seg_cache:
            return self._seg_cache[t]
        depth = self.depth_m(t)
        try:
            labels, names = self._segmenter(
                self.rgb(t), depth, self.intrinsic(t), self.extrinsic(t), self.task
            )
        except Exception:
            self._seg_cache[t] = None
            return None
        if names:
            self._names.update({int(k): str(v) for k, v in names.items()})
        labels = None if labels is None else np.ascontiguousarray(labels)
        self._seg_cache[t] = labels
        return labels

    def intrinsic(self, t: int) -> np.ndarray | None:
        K = self._camera_params(t).get("intrinsic")
        if K is None:
            return None
        K = np.asarray(K, dtype=np.float64)
        return K[:3, :3] if K.shape == (4, 4) else K

    def extrinsic(self, t: int) -> np.ndarray | None:
        E = self._camera_params(t).get("extrinsic")
        if E is None:
            return None
        return self._to_base_frame(np.asarray(E, dtype=np.float64))

    def _to_base_frame(self, E: np.ndarray) -> np.ndarray:
        """Put a recorded extrinsic into the frame proprioception is reported in.

        The env server is supposed to publish extrinsics already in the
        controller's ``base`` frame, so the normal case is a no-op. An episode
        recorded before that correction existed — or with the wrong
        ``--env.calibration-base-frame`` — carries a ``base_link`` extrinsic
        instead, which mirrors the whole scene in x and y while leaving the
        table horizontal. That survives a plane-fit sanity check and only
        shows up against the robot's own TCP.

        Rather than rewrite the (large) ``.vla``, such an episode is repaired
        by adding ``"extrinsic_frame": "base_link"`` to its JSON sidecar.
        Absent the key the extrinsic is taken at face value.
        """
        frame = str(self._sidecar.get("extrinsic_frame", "base"))
        if frame == "base":
            return E
        if frame != "base_link":
            raise ValueError(
                f"{self.demo_key!r}: sidecar 'extrinsic_frame' must be 'base' "
                f"or 'base_link', got {frame!r}"
            )
        return _BASE_LINK_TO_BASE @ E

    def ee_positions(self) -> np.ndarray:
        """End-effector positions, read through the recorded state layout.

        The layout is stored with the episode, so this does not assume the
        UR5e's ``[pos, axis-angle, aperture]`` packing — it finds whichever
        field holds the position.
        """
        col = self._state_slice("pos")
        if col is not None:
            return col
        # No layout, or one with no position field: the leading 3 columns are
        # the position under every layout PA currently records.
        return np.ascontiguousarray(self._state()[:, :3])

    def ee_orientations(self) -> np.ndarray | None:
        return self._state_slice("quat_to_axisangle")

    def gripper_states(self) -> np.ndarray | None:
        """``(T, 1)`` aperture, the single DOF a real driver publishes."""
        return self._state_slice("gripper_aperture")

    def gripper_geometry(self) -> str | None:
        gripper = (
            self._sidecar.get("env_spec", {}).get("robot", {}).get("gripper") or {}
        )
        geometry_id = gripper.get("geometry_id")
        return str(geometry_id) if geometry_id else None

    def actions(self) -> np.ndarray | None:
        arr = self._loaded.get("action")
        return None if arr is None else np.asarray(arr, dtype=np.float32)

    def _state(self) -> np.ndarray:
        state = self._loaded.get("observation/state")
        if state is None:
            raise KeyError(f"observation/state missing from {self.demo_key}.vla")
        return np.asarray(state, dtype=np.float32)

    def _state_slice(self, op: str) -> np.ndarray | None:
        """Columns of ``observation/state`` holding the field named *op*.

        Returns ``None`` when the recorded layout does not carry that field,
        which is how a format that publishes no orientation or no gripper
        reports the fact rather than handing back the wrong columns.
        """
        state = self._state()
        offset = 0
        for field in self._state_fields():
            width = _STATE_FIELD_WIDTHS.get(field)
            if width is None:
                # An op this reader does not know the width of: every field
                # after it would be at an unknown offset, so stop rather than
                # return a mis-aligned slice.
                return None
            if field == op:
                if offset + width > state.shape[1]:
                    return None
                return np.ascontiguousarray(state[:, offset : offset + width])
            offset += width
        return None

    def _state_fields(self) -> list[str]:
        spec = self._sidecar.get("env_spec", {}).get("robot", {})
        layout = spec.get("state_layout") or {}
        return [str(f.get("op", "")) for f in layout.get("fields", [])]

    def seg_name_map(self) -> dict[int, str]:
        return dict(self._names)

    def frames_with_depth(self) -> list[int]:
        return sorted(self._depth_index())


# Column counts for the state ops a recorded state layout can carry.
_STATE_FIELD_WIDTHS = {
    "pos": 3,
    "quat_to_axisangle": 3,
    "gripper_aperture": 1,
}

# Universal Robots' two base frames sit 180 degrees apart about z: `base` is
# what RTDE reports poses in, `base_link` is the ROS/URDF convention that
# hand-eye tools built on a URDF (Kalib among them) produce. The rotation is
# its own inverse, so the same matrix converts either way.
_BASE_LINK_TO_BASE = np.diag([-1.0, -1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_demo_episode(result: dict, segmenter=None) -> DemoEpisode:
    """Build the right adapter for a retrieval result dict.

    *result* carries ``source_kind``, ``source_path``, ``demo_key`` and
    ``task`` — the same fields the embedding cache round-trips.
    """
    kind = result.get("source_kind", "libero")
    source_path = result["source_path"]
    demo_key = result["demo_key"]
    task = result.get("task", "")

    if kind in ("libero", "robocasa_hdf5"):
        episode = LiberoHDF5Episode(source_path, demo_key, task=task)
        episode.source_kind = kind
        return episode
    if kind == "vla":
        return VlaEpisode(source_path, demo_key, segmenter=segmenter)
    raise ValueError(
        f"no DemoEpisode adapter for source_kind {kind!r}. Add one in "
        "policy_abstraction.analysis.demo_episode and register it here."
    )


# ---------------------------------------------------------------------------
# Calibration sanity
# ---------------------------------------------------------------------------


def extrinsic_disagreement(episode, max_frames: int = 12) -> float | None:
    """How far the calibration puts the end-effector from the surface seen there.

    A wrong hand-eye calibration does not fail loudly: depth still looks
    plausible, reconstruction still returns points, and guidance still emits
    numbers — they are just measured in a frame that does not match the robot.
    This is the cheap check that catches it.

    The end-effector's position is known exactly from proprioception. Put it in
    the camera frame with the calibration, project it, and read the depth image
    where it lands. The gripper is the nearest surface along that ray, so a
    correct calibration agrees with the depth to within a few centimetres.

    The statistic is the **best** agreement over the sampled frames, not the
    median. A single-pixel depth probe was tried first and false-alarms badly:
    an arm that leaves the camera's view, or a gripper occluded behind the
    object it is reaching for, reads the background metres away, so on a
    calibration verified correct to 2 cm the median came out at 1.2 m.
    Occlusion and out-of-view frames can only ever make one frame's number
    worse, never better, whereas a genuinely wrong extrinsic — a rotation, a
    translation, a mirrored base frame — is wrong in *every* frame by roughly
    the same amount. The minimum is therefore the statistic that is immune to
    the nuisance and still sensitive to the fault.

    Returns
    -------
    float or None
        Closest approach, in metres, between the end-effector and the surface
        reconstructed around it. ``inf`` when the calibration never places the
        end-effector in view across the whole episode — a workspace camera that
        never sees the gripper is itself evidence the extrinsic is wrong.
        ``None`` when the check cannot run at all (no depth, or no
        calibration).
    """
    frames = episode.frames_with_depth()
    if not frames:
        return None

    step = max(1, len(frames) // max_frames)
    sampled = frames[::step][:max_frames]

    try:
        ee_all = np.asarray(episode.ee_positions(), dtype=np.float64)
    except Exception:
        return None

    errors: list[float] = []
    checkable = False
    for t in sampled:
        K, E = episode.intrinsic(t), episode.extrinsic(t)
        depth = episode.depth_m(t)
        if K is None or E is None or depth is None or t >= len(ee_all):
            continue
        checkable = True
        K = np.asarray(K, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        # The extrinsic is the camera's pose in the robot base frame, so its
        # inverse takes a base-frame point into the camera.
        p_cam = E[:3, :3].T @ (ee_all[t] - E[:3, 3])
        if p_cam[2] <= 0:
            continue
        pixel = K @ (p_cam / p_cam[2])
        u, v = int(round(pixel[0])), int(round(pixel[1]))
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            continue
        nearest = _nearest_surface_distance(depth, K, E, (u, v), ee_all[t])
        if nearest is not None:
            errors.append(nearest)

    if not checkable:
        return None
    if not errors:
        return float("inf")
    return float(np.min(errors))


def _nearest_surface_distance(
    depth: np.ndarray,
    K: np.ndarray,
    E: np.ndarray,
    pixel: tuple[int, int],
    point: np.ndarray,
    window: int = 20,
) -> float | None:
    """Distance from *point* to the nearest surface reconstructed near *pixel*.

    Only a ``window``-radius patch is back-projected, which keeps this to a
    couple of thousand points rather than the whole frame, and confines the
    answer to the surface the end-effector actually projects onto.
    """
    u, v = pixel
    h, w = depth.shape[:2]
    u0, u1 = max(0, u - window), min(w, u + window + 1)
    v0, v1 = max(0, v - window), min(h, v + window + 1)
    patch = depth[v0:v1, u0:u1].astype(np.float64)
    vv, uu = np.mgrid[v0:v1, u0:u1]
    ok = np.isfinite(patch) & (patch > 0)
    if not ok.any():
        return None
    z = patch[ok]
    x = (uu[ok] - K[0, 2]) * z / K[0, 0]
    y = (vv[ok] - K[1, 2]) * z / K[1, 1]
    cam_pts = np.stack([x, y, z], axis=1)
    world = cam_pts @ E[:3, :3].T + E[:3, 3]
    return float(np.min(np.linalg.norm(world - np.asarray(point), axis=1)))
