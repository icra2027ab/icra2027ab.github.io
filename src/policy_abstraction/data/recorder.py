"""Recorder components for saving episode trajectories.

The Recorder is responsible for logger observations, actions, states,
and other rollout data to disk.  The default implementation uses robodm.
"""

from __future__ import annotations

import json
import pathlib
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import robodm
from loguru import logger


class BaseRecorder(ABC):
    """Abstract base class for episode recorders."""

    @abstractmethod
    def initialize(self, episode_data_path: str, episode_metadata: dict) -> None:
        """
        Set up the recorder for a new episode.

        Args:
            episode_data_path: Filesystem path (or prefix) where data should be saved.
            episode_metadata:  Arbitrary key/value metadata about the episode
                               (task description, seed, hyper-params, …).
        """

    @abstractmethod
    def record(self, **kwargs: Any) -> None:
        """
        Add one timestep of data.

        All keyword arguments are implementation-defined.  Common keys:
            observation/agentview_image, observation/wrist_image,
            observation/state, action, reward, done, timestep,
            policy/action_chunks.
        """

    @abstractmethod
    def save(self) -> None:
        """Flush and close the recording."""

    def update_metadata(self, key: str, value: Any) -> None:  # noqa: B027
        """Update a metadata entry (no-op by default; override to implement)."""
        pass


# ---------------------------------------------------------------------------
# Default implementation: robodm
# ---------------------------------------------------------------------------

class RobodmRecorder(BaseRecorder):
    """Records episode trajectories using robodm.

    This is the drop-in replacement for the original inline robodm logic
    that previously lived inside ``EpisodeRunner``.
    """

    def __init__(self, video_codec: str = "libx264") -> None:
        self._video_codec = video_codec
        self._trajectory = None
        self._metadata: dict | None = None
        self._data_path: pathlib.Path | None = None

    # ------------------------------------------------------------------
    # BaseRecorder interface
    # ------------------------------------------------------------------

    def initialize(self, episode_data_path: str, episode_metadata: dict) -> None:
        """
        Create a robodm Trajectory and store metadata.

        Args:
            episode_data_path: Path passed to ``robodm.Trajectory``.
            episode_metadata:  Dict that will be written as a JSON sidecar.
        """

        self._data_path = pathlib.Path(episode_data_path)
        self._metadata = dict(episode_metadata)

        try:
            self._trajectory = robodm.Trajectory(
                path=episode_data_path,
                mode="w",
                video_codec=self._video_codec,
            )
            logger.info(f"[RobodmRecorder] Initialized trajectory at {episode_data_path}")
        except Exception as exc:
            logger.warning(f"[RobodmRecorder] Failed to initialize trajectory: {exc}")
            self._trajectory = None

    def update_metadata(self, key: str, value: Any) -> None:
        """Add or update a key in the episode metadata JSON sidecar.

        Must be called after :meth:`initialize`.  The updated metadata is
        written to disk when :meth:`save` is called.

        Args:
            key:   Top-level key in the metadata dict.
            value: JSON-serialisable value.
        """
        if self._metadata is None:
            logger.warning("[RobodmRecorder] update_metadata called before initialize; ignoring.")
            return
        self._metadata[key] = value

    def record(self, **kwargs: Any) -> None:
        """
        Append a timestep to the trajectory.

        Recognises the following keyword arguments (all optional):

        - ``camera_image_streams``         (dict[str, np.ndarray]) – per-camera
          uint8 RGB images, keyed by camera name (any number of cameras).
          Each entry is written to ``observation/<cam_name>_image``.
        - ``camera_depth_streams``         (dict[str, np.ndarray]) – per-camera
          metric depth in **metres**, keyed by camera name. Written to
          ``observation/<cam_name>_depth`` as uint16 millimetres.
        - ``depth_timestep``               (int) – the timestep the depth frames
          in this call belong to. Written to ``observation/depth_timestep``.
          Depth is recorded on a stride (it is stored uncompressed and dwarfs
          the h264 colour streams), so this index is what lets a reader map a
          depth frame back to its timestep rather than assume a cadence.
        - ``ref_camera_image_streams``     (dict[str, np.ndarray]) – per-camera
          retrieved reference uint8 RGB images, keyed by camera name.
          Written to ``abstraction/ref_<cam_name>_image``.
        - ``camera_chunk_overlay_streams`` (dict[str, np.ndarray]) – per-camera
          action-chunk overlay uint8 RGB images. Written to
          ``policy/<cam_name>_chunk_overlay``.
        - ``state``              (np.ndarray) – proprioceptive robot state vector.
        - ``action``             (np.ndarray) – executed action.
        - ``reward``             (float | np.ndarray) – scalar reward.
        - ``done``               (bool | np.ndarray) – episode termination flag.
        - ``timestep``           (int) – current time index.
        - ``action_chunks``      (np.ndarray) – full action-chunk prediction from the policy.
        - ``camera_params``      (dict) – camera intrinsics/extrinsics for projection.
        - ``retrieval_figure``   (np.ndarray) – uint8 RGB retrieval grid image (stored once).
        - ``analysis_seg_image`` (np.ndarray) – uint8 RGB seg-ID annotated image (stored once).
        """
        if self._trajectory is None:
            kwargs.pop("abstraction_object_pointclouds", None)
            kwargs.pop("abstraction_scene_pointcloud", None)
            kwargs.pop("abstraction_entity_centroids", None)
            kwargs.pop("camera_image_streams", None)
            kwargs.pop("camera_depth_streams", None)
            kwargs.pop("ref_camera_image_streams", None)
            kwargs.pop("camera_chunk_overlay_streams", None)
            return

        # Variable-shape pointcloud dict (M points/object varies per replan)
        # doesn't fit robodm's fixed-shape numeric streams. Wrap the dict in a
        # 0-d ``dtype=object`` ndarray so robodm's rawvideo path stores it
        # opaquely; on load it comes back as ``(N,) object`` of dicts.
        pcds = kwargs.pop("abstraction_object_pointclouds", None)
        if pcds is not None:
            try:
                payload = np.empty((), dtype=object)
                payload[()] = {k: np.asarray(v) for k, v in pcds.items()}
                self._trajectory.add("abstraction/object_pointclouds", payload)
            except Exception as exc:
                logger.warning(
                    f"[RobodmRecorder] Failed to record object_pointclouds: {exc}"
                )

        # Full-scene fused cloud — a single ``[M, 7]`` array per replan, wrapped
        # in a 0-d object ndarray like the per-object clouds above.
        scene_pcd = kwargs.pop("abstraction_scene_pointcloud", None)
        if scene_pcd is not None:
            try:
                payload = np.empty((), dtype=object)
                payload[()] = np.asarray(scene_pcd, dtype=np.float32)
                self._trajectory.add("abstraction/scene_pointcloud", payload)
            except Exception as exc:
                logger.warning(
                    f"[RobodmRecorder] Failed to record scene_pointcloud: {exc}"
                )

        # Per-entity 3-D centroids ({seg-id | "ee" | "gripper_l/r": [3]}) used
        # to draw the distance-pair lines on replay. Keys are stringified so the
        # mixed int/str dict survives the object-ndarray round trip; the loader
        # revives numeric keys back to int.
        ecs = kwargs.pop("abstraction_entity_centroids", None)
        if ecs is not None:
            try:
                payload = np.empty((), dtype=object)
                payload[()] = {
                    str(k): np.asarray(v, dtype=np.float32).ravel()[:3]
                    for k, v in ecs.items()
                }
                self._trajectory.add("abstraction/entity_centroids", payload)
            except Exception as exc:
                logger.warning(
                    f"[RobodmRecorder] Failed to record entity_centroids: {exc}"
                )

        # Per-camera image streams (live, ref, chunk-overlay). Each dict
        # is keyed by camera name, written to a stream named after the
        # camera so the loader can iterate the env's EnvSpec at replay
        # time without assuming any fixed primary/wrist layout.
        for stream_kwarg, key_fmt in (
            ("camera_image_streams",         "observation/{name}_image"),
            ("ref_camera_image_streams",     "abstraction/ref_{name}_image"),
            ("camera_chunk_overlay_streams", "policy/{name}_chunk_overlay"),
        ):
            stream_dict = kwargs.pop(stream_kwarg, None)
            if not stream_dict:
                continue
            for cam_name, img in stream_dict.items():
                if img is None:
                    continue
                self._record_image_stream(key_fmt.format(name=cam_name), img)

        # Depth: metric metres in, uint16 millimetres out. robodm stores these
        # raw, so millimetres halve the footprint against float32 and are still
        # lossless at the sensor's resolution (a D435 quantises well above 1mm).
        depth_streams = kwargs.pop("camera_depth_streams", None)
        depth_t = kwargs.pop("depth_timestep", None)
        if depth_streams:
            wrote_depth = False
            for cam_name, depth in depth_streams.items():
                if depth is None:
                    continue
                try:
                    arr = np.asarray(depth, dtype=np.float32)
                    # NaN/inf mark "no return"; 0 is the convention for that in
                    # a uint16 depth map, and clipping keeps a stray far value
                    # from wrapping around into a plausible near one.
                    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
                    mm = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
                    self._trajectory.add(
                        f"observation/{cam_name}_depth", np.ascontiguousarray(mm)
                    )
                    wrote_depth = True
                except Exception as exc:
                    logger.warning(
                        f"[RobodmRecorder] Failed to record depth for {cam_name}: {exc}"
                    )
            if wrote_depth and depth_t is not None:
                try:
                    self._trajectory.add(
                        "observation/depth_timestep",
                        np.array([depth_t], dtype=np.int32),
                    )
                except Exception as exc:
                    logger.warning(
                        f"[RobodmRecorder] Failed to record depth_timestep: {exc}"
                    )

        _MAP = {
            "state":                        "observation/state",
            "robot_joints":                 "observation/robot_joints",
            "action":                       "action",
            "reward":                       "reward",
            "done":                         "done",
            "timestep":                     "timestep",
            "action_chunks":                "policy/action_chunks",
            "camera_params":                "camera_params",
            "retrieval_figure":             "retrieval/query_grid",
            "analysis_seg_image":           "analysis/seg_annotated_image",
            "vlm_scene_image":              "vlm/scene_image",
            "vlm_subtraj_image":            "vlm/subtraj_image",
            # Abstraction / MTA
            "abstraction_live_distances":   "abstraction/live_distances",
            "abstraction_aligned_t":        "abstraction/aligned_t",
            "abstraction_ref_distances":    "abstraction/ref_distances",
            "abstraction_errors":           "abstraction/distance_errors",
            "abstraction_progress":         "abstraction/mta_progress",
            "abstraction_alignment_mse_high": "abstraction/alignment_mse_high",
            "abstraction_ref_image":        "abstraction/ref_image_annotated",
            "abstraction_live_image":       "abstraction/live_image_annotated",
            "abstraction_alignment_plot":   "abstraction/alignment_plot",
            "abstraction_constraint_costs": "abstraction/constraint_costs",
            # Perception (predicted-seg) overlays + per-stage history grids
            "perception_overlay_agentview": "perception/overlay_agentview",
            "perception_overlay_wrist":     "perception/overlay_wrist",
            "perception_history_agentview": "perception/history_agentview",
            "perception_history_wrist":     "perception/history_wrist",
            "perception_debug_agentview":   "perception/debug_agentview",
            "perception_debug_wrist":       "perception/debug_wrist",
            # Interaction / low-level abstraction
            "interaction_live_distances":   "interaction/live_distances",
            "interaction_aligned_t":        "interaction/aligned_t",
            "interaction_ref_distances":    "interaction/ref_distances",
            "abstraction_alignment_mse_low": "interaction/alignment_mse_low",
            "interaction_mta_progress":     "interaction/mta_progress",
            # Guidance parameters
            "guidance_weights":             "guidance/weights",
            "guidance_factor":              "guidance/factor",
            "guidance_relevances":          "guidance/relevances",
            "guidance_ori_weight":          "guidance/ori_weight",
            "guidance_gripper_weight":      "guidance/gripper_weight",
            "guidance_ll_factor":           "guidance/ll_factor",
        }

        _DTYPE_CAST = {
            "observation/state":              np.float32,
            "observation/robot_joints":       np.float32,
            "action":                         np.float32,
            "reward":                         np.float32,
            "done":                           bool,
            "timestep":                       np.int32,
            "policy/action_chunks":           np.float32,
            "camera_params":                  None,          # keep as-is (dict)
            "retrieval/query_grid":           None,          # keep as-is (uint8)
            "analysis/seg_annotated_image":   None,          # keep as-is (uint8)
            "vlm/scene_image":                None,          # keep as-is (uint8)
            "vlm/subtraj_image":              None,          # keep as-is (uint8)
            # Abstraction / MTA
            "abstraction/live_distances":     np.float32,
            "abstraction/aligned_t":          np.int32,
            "abstraction/ref_distances":      np.float32,
            "abstraction/distance_errors":    np.float32,
            "abstraction/mta_progress":       np.float32,
            "abstraction/alignment_mse_high": np.float32,
            "abstraction/ref_image_annotated": None,    # keep as-is (uint8)
            "abstraction/live_image_annotated": None,  # keep as-is (uint8)
            "abstraction/alignment_plot":     None,    # keep as-is (uint8)
            "abstraction/constraint_costs":   np.float32,
            # Perception overlays / history grids (uint8 RGB, kept as-is)
            "perception/overlay_agentview":   None,
            "perception/overlay_wrist":       None,
            "perception/history_agentview":   None,
            "perception/history_wrist":       None,
            "perception/debug_agentview":     None,
            "perception/debug_wrist":         None,
            # Interaction / low-level abstraction
            "interaction/live_distances":     np.float32,
            "interaction/aligned_t":          np.int32,
            "interaction/ref_distances":      np.float32,
            "interaction/alignment_mse_low":  np.float32,
            "interaction/mta_progress":       np.float32,
            # Guidance parameters
            "guidance/weights":               np.float32,
            "guidance/factor":                np.float32,
            "guidance/relevances":            np.float32,
            "guidance/ori_weight":            np.float32,
            "guidance/gripper_weight":        np.float32,
            "guidance/ll_factor":             np.float32,
        }

        for kwarg_key, traj_key in _MAP.items():
            if kwarg_key not in kwargs:
                continue
            value = kwargs[kwarg_key]
            dtype = _DTYPE_CAST[traj_key]
            try:
                if isinstance(value, np.ndarray):
                    arr = value.astype(dtype) if dtype is not None else value
                    # Crop images to even dimensions so libx264 codec works.
                    if arr.ndim == 3 and arr.shape[2] in (3, 4):
                        h, w = arr.shape[:2]
                        if h % 2 or w % 2:
                            arr = arr[: h - h % 2, : w - w % 2]
                else:
                    # Scalars → 1-element arrays
                    arr = np.array([value], dtype=dtype if dtype is not None else type(value))
                self._trajectory.add(traj_key, arr)
            except Exception as exc:
                logger.warning(f"[RobodmRecorder] Failed to record '{traj_key}': {exc}")

    def _record_image_stream(self, traj_key: str, img: np.ndarray) -> None:
        """Append one uint8 RGB image to ``traj_key`` with even-crop for libx264.

        Shared by the per-camera image / ref-image / chunk-overlay paths so the
        encoder hits the same shape constraints regardless of camera count.
        """
        try:
            arr = np.asarray(img)
            if arr.ndim == 3 and arr.shape[2] in (3, 4):
                h, w = arr.shape[:2]
                if h % 2 or w % 2:
                    arr = arr[: h - h % 2, : w - w % 2]
            self._trajectory.add(traj_key, np.ascontiguousarray(arr))
        except Exception as exc:
            logger.warning(f"[RobodmRecorder] Failed to record '{traj_key}': {exc}")

    def save(self) -> None:
        """Close the robodm trajectory and write the JSON metadata sidecar."""
        if self._trajectory is None:
            logger.debug("[RobodmRecorder] Nothing to save (trajectory was never initialised).")
            return
        try:
            self._trajectory.close()
            logger.info(f"[RobodmRecorder] Saved trajectory to {self._data_path}")
        except Exception as exc:
            logger.error(f"[RobodmRecorder] Failed to close trajectory: {exc}")

        if self._metadata and self._data_path is not None:
            metadata_path = self._data_path.with_suffix(".json")
            try:
                with open(metadata_path, "w") as fh:
                    json.dump(self._metadata, fh, indent=2)
                logger.info(f"[RobodmRecorder] Saved metadata to {metadata_path}")
            except Exception as exc:
                logger.error(f"[RobodmRecorder] Failed to save metadata: {exc}")
