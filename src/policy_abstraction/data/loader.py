"""VLA episode loader backed by robodm."""

from __future__ import annotations

import json
from loguru import logger
import pathlib
from typing import Any, Dict, Iterator, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Recorder-side feature-key conventions
# ---------------------------------------------------------------------------
# The recorder squeezes the env's camera streams into two fixed feature keys
# regardless of the actual env: the primary camera goes to ``RECORDED_PRIMARY``
# and the wrist camera (when present) goes to ``RECORDED_WRIST``. At load time
# we route them back to the canonical obs/image keys derived from the saved
# EnvSpec — see :meth:`VLALoader._resolve_env_spec`.
RECORDED_PRIMARY = "observation/agentview_image"
RECORDED_WRIST = "observation/wrist_image"


class VLALoader:
    """Loads a robodm ``.vla`` episode file and exposes its data for replay.

    Usage
    -----
    >>> loader = VLALoader("path/to/episode.vla").load()
    >>> for t, frame in loader.iter_frames():
    ...     visualizer.update(timestep=t, **frame)
    >>> loader.close()

    Or as a context manager::

        with VLALoader("path/to/episode.vla") as loader:
            for t, frame in loader.iter_frames():
                ...

    Frame dict
    ----------
    Each frame returned by :meth:`get_frame` / :meth:`iter_frames` is a dict
    with keys matching the keyword arguments of
    :class:`~policy_abstraction.visualization.base.BaseVisualizer.update`:

    ``obs``
        Raw observation dict with keys ``agentview_image`` and
        ``robot0_eye_in_hand_image`` (when present in the file).
    ``robot_state``
        1-D ``np.float32`` proprioceptive state vector, or ``None``.
    ``action``
        1-D ``np.float32`` action vector, or ``None``.
    ``reward``
        Scalar float, or ``None``.
    ``done``
        Bool termination flag.
    ``action_chunks``
        Action-chunk array ``[T, D]`` (from the policy prediction logged at
        that step), or ``None``.
    """

    def __init__(self, path: str) -> None:
        self.path = pathlib.Path(path)
        self._data: Optional[Dict[str, np.ndarray]] = None
        self._metadata: Optional[Dict[str, Any]] = None
        self._length: int = 0
        # Resolved lazily after load() from metadata["env_spec"] or, for
        # pre-env_spec recordings, a LIBERO heuristic based on feature keys.
        self._env_spec: Optional[Any] = None
        # Distance pairs reconstructed once from the ``translated_pairs``
        # sidecar metadata (same for every frame). ``...`` is the "not yet
        # computed" sentinel; ``None`` means no pairs are available.
        self._pairs_cache: Any = ...

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def task_description(self) -> str:
        """Human-readable task string from the sidecar JSON, or empty."""
        if self._metadata:
            return str(self._metadata.get("task_description", ""))
        return ""

    @property
    def metadata(self) -> Dict[str, Any]:
        """Full metadata dict (may be empty if no sidecar JSON was found)."""
        return dict(self._metadata) if self._metadata else {}

    @property
    def length(self) -> int:
        """Number of timesteps available in this episode."""
        return self._length

    @property
    def feature_keys(self):
        """List of raw feature keys present in the trajectory."""
        if self._data is None:
            return []
        return list(self._data.keys())

    @property
    def env_spec(self) -> Optional[Any]:
        """Reconstructed :class:`EnvSpec` for the recorded episode.

        Preferred source: ``metadata["env_spec"]`` written by recent recorders.
        Fallback: a LIBERO :class:`EnvSpec` when only the legacy
        ``observation/agentview_image`` + ``observation/wrist_image`` feature
        streams are present (pre-env_spec recordings). Returns ``None`` when
        neither source is available — callers fall back to legacy obs keys.
        """
        return self._env_spec

    @property
    def robot_base(self) -> Dict[str, Optional[np.ndarray]]:
        """Robot base pose saved in the sidecar JSON.

        Returns a dict with keys ``base_pos`` (``[3]`` float64) and
        ``base_ori`` (``[3]`` float64 axis-angle or rotation-matrix),
        or ``None`` values when the sidecar was not found / not recorded.
        """
        pos = ori = None
        if self._metadata:
            raw_pos = self._metadata.get("robot_base_pos")
            raw_ori = self._metadata.get("robot_base_ori")
            if raw_pos is not None:
                pos = np.asarray(raw_pos, dtype=np.float64)
            if raw_ori is not None:
                ori = np.asarray(raw_ori, dtype=np.float64)
        return {"base_pos": pos, "base_ori": ori}

    @property
    def retrieval_data(self) -> Dict[str, Any]:
        """Episode-level retrieval info saved during evaluation.

        Returns a dict with a subset of:

        ``query_grid``
            uint8 RGB retrieval-figure image (``H×W×3``), or absent.
        ``seg_annotated_image``
            uint8 RGB segmentation overlay image, or absent.
        ``retrieval_results``
            List of ranked retrieved-demo dicts from the JSON sidecar.
        ``vlm_analysis``
            VLM analysis dict from the JSON sidecar.
        ``multi_demo_stats``
            Dict of DTW-aligned multi-demo distance statistics (numpy arrays
            revived from JSON lists), suitable for
            :meth:`RerunVisualizer.log_retrieval_info`.
        """
        result: Dict[str, Any] = {}
        if self._data is not None:
            qg = self._data.get("retrieval/query_grid")
            if qg is not None and len(qg) > 0:
                result["query_grid"] = qg[0]
            seg = self._data.get("analysis/seg_annotated_image")
            if seg is not None and len(seg) > 0:
                result["seg_annotated_image"] = seg[0]
            scene = self._data.get("vlm/scene_image")
            if scene is not None and len(scene) > 0:
                result["vlm_scene_image"] = scene[0]
            subtraj = self._data.get("vlm/subtraj_image")
            if subtraj is not None and len(subtraj) > 0:
                result["vlm_subtraj_image"] = subtraj[0]
        if self._metadata:
            if "retrieval_results" in self._metadata:
                result["retrieval_results"] = self._metadata["retrieval_results"]
            if "vlm_analysis" in self._metadata:
                result["vlm_analysis"] = self._metadata["vlm_analysis"]
            if "vlm_conversation" in self._metadata:
                result["vlm_conversation"] = self._metadata["vlm_conversation"]
            if "iterative_analysis" in self._metadata:
                # Written by every iterative-strategy run; without this a
                # replay silently dropped the whole iteration history that the
                # live viewer had shown.
                result["iterative_analysis"] = self._metadata["iterative_analysis"]
            if "multi_demo_stats" in self._metadata:
                _mds = self._metadata["multi_demo_stats"]
                result["multi_demo_stats"] = {
                    "ref_traj": np.asarray(_mds["ref_traj"], dtype=np.float32),
                    "mean": np.asarray(_mds["mean"], dtype=np.float32),
                    "std": np.asarray(_mds["std"], dtype=np.float32),
                    "min": np.asarray(_mds["min"], dtype=np.float32),
                    "max": np.asarray(_mds["max"], dtype=np.float32),
                    "pair_labels": list(_mds["pair_labels"]),
                    "n_demos": int(_mds["n_demos"]),
                }
        return result

    @property
    def abstraction_setup_data(self) -> Dict[str, Any]:
        """Episode-level abstraction-setup info for ``log_abstraction_setup``.

        Returns kwargs suitable for
        :meth:`RerunVisualizer.log_abstraction_setup`:

        ``mapping``
            Object exposing the same attributes as
            :class:`~policy_abstraction.vlm.equivalence_mapper.ObjectMapping`,
            reconstructed from the JSON sidecar.
        ``constraints``
            List of constraint-like objects with ``kind``, ``subtraj_idx``,
            ``label``, ``description``, ``pairs_affected``, ``reasoning``.
        """
        result: Dict[str, Any] = {}
        if not self._metadata:
            return result

        eq = self._metadata.get("equivalence_mapping")
        if eq is not None:
            from types import SimpleNamespace

            def _maybe_int(s):
                try:
                    return int(s)
                except (TypeError, ValueError):
                    return s

            result["mapping"] = SimpleNamespace(
                retrieved_objects={
                    _maybe_int(k): v for k, v in eq.get("retrieved_objects", {}).items()
                },
                live_objects={
                    _maybe_int(k): v for k, v in eq.get("live_objects", {}).items()
                },
                id_map={
                    _maybe_int(k): _maybe_int(v)
                    for k, v in eq.get("id_map", {}).items()
                },
                unmapped_retrieved_ids=list(eq.get("unmapped_retrieved_ids", [])),
                vlm_prompt_text=eq.get("vlm_prompt_text", "") or "",
                vlm_response_json=eq.get("vlm_response_json", "") or "",
                conversation=None,
            )

        equiv_conv = self._metadata.get("equivalence_conversation")
        if equiv_conv is not None:
            result["equivalence_conversation"] = {
                "model": equiv_conv.get("model", "") or "",
                "system_prompt": equiv_conv.get("system_prompt", "") or "",
                "user_text": equiv_conv.get("user_text", "") or "",
                "user_images_b64": list(equiv_conv.get("user_images_b64", []) or []),
                "assistant_response": equiv_conv.get("assistant_response", "") or "",
            }

        constraints_meta = self._metadata.get("abstraction_constraints")
        if constraints_meta:
            from types import SimpleNamespace

            result["constraints"] = [
                SimpleNamespace(
                    kind=c.get("kind", "path"),
                    subtraj_idx=c.get("subtraj_idx"),
                    label=c.get("label", ""),
                    description=c.get("description", ""),
                    expression=c.get("expression", ""),
                    pairs_affected=list(c.get("pairs_affected", [])),
                    reasoning=c.get("reasoning"),
                )
                for c in constraints_meta
            ]

        return result

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> "VLALoader":
        """Open the ``.vla`` file, decode all features and return *self*.

        Raises
        ------
        FileNotFoundError
            If the ``.vla`` path does not exist.
        RuntimeError
            If robodm fails to open or decode the file.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Episode file not found: {self.path}")

        try:
            import robodm
        except ImportError as exc:
            raise ImportError(
                "robodm is required for VLALoader. "
                "Install it from the third_party/robodm directory."
            ) from exc

        logger.info(f"[VLALoader] Opening {self.path}")
        traj = robodm.Trajectory(str(self.path), mode="r")
        try:
            self._data = traj.load()  # type: ignore[assignment]
        finally:
            traj.close()

        if not self._data:
            raise RuntimeError(f"[VLALoader] No data loaded from {self.path}")

        # Determine episode length from a canonical per-timestep key, falling
        # back to the longest array.  This avoids picking up short arrays that
        # are only recorded at replan steps (e.g. action_chunks, abstraction/*).
        for key in (
            "timestep",
            "observation/agentview_image",
            "observation/state",
            "action",
        ):
            arr = self._data.get(key)
            if arr is not None and hasattr(arr, "__len__") and len(arr) > 0:
                self._length = len(arr)
                break
        else:
            self._length = max(
                (len(v) for v in self._data.values() if hasattr(v, "__len__")),
                default=0,
            )

        logger.info(
            f"[VLALoader] Loaded {self._length} timesteps, "
            f"features: {list(self._data.keys())}"
        )

        # Load optional JSON sidecar (written by RobodmRecorder)
        meta_path = self.path.with_suffix(".json")
        if meta_path.exists():
            try:
                with open(meta_path) as fh:
                    self._metadata = json.load(fh)
                logger.info(f"[VLALoader] Loaded metadata from {meta_path}")
            except Exception as exc:
                logger.warning(f"[VLALoader] Could not read metadata: {exc}")

        self._env_spec = self._resolve_env_spec()

        return self

    def _camera_image_array(
        self, cam: Any, primary_name: Optional[str]
    ) -> Optional[np.ndarray]:
        """Per-camera live image stream, with legacy slot fallback for old files."""
        arr = self._data.get(f"observation/{cam.name}_image")
        if arr is not None:
            return arr
        # Legacy squeezed schema: the primary camera was stored under
        # ``observation/agentview_image`` and the wrist under
        # ``observation/wrist_image`` regardless of the camera's real name.
        if cam.name == primary_name:
            arr = self._data.get(RECORDED_PRIMARY)
            if arr is not None:
                return arr
        if cam.role == "wrist":
            arr = self._data.get(RECORDED_WRIST)
        return arr

    def _camera_ref_image_array(self, cam: Any) -> Optional[np.ndarray]:
        """Per-camera reference image stream (replan cadence), with legacy fallback."""
        arr = self._data.get(f"abstraction/ref_{cam.name}_image")
        if arr is not None:
            return arr
        if cam.role == "wrist":
            return self._data.get("abstraction/ref_wrist_image")
        return None

    def _camera_chunk_overlay_array(self, cam: Any) -> Optional[np.ndarray]:
        """Per-camera action-chunk overlay stream (replan cadence), with legacy fallback."""
        arr = self._data.get(f"policy/{cam.name}_chunk_overlay")
        if arr is not None:
            return arr
        if cam.role == "wrist":
            return self._data.get("policy/wrist_chunk_overlay")
        return None

    def _resolve_env_spec(self) -> Optional[Any]:
        """Best-effort EnvSpec reconstruction (see :attr:`env_spec`)."""
        if self._metadata and self._metadata.get("env_spec"):
            try:
                from policy_abstraction.envs.spec import EnvSpec

                return EnvSpec.from_dict(self._metadata["env_spec"])
            except Exception as exc:
                logger.warning(
                    f"[VLALoader] Failed to decode env_spec from metadata: {exc}"
                )
        # Heuristic fallback for pre-env_spec recordings.
        if self._data and RECORDED_PRIMARY in self._data:
            try:
                from policy_abstraction.envs.libero.config import libero_env_spec

                logger.info(
                    "[VLALoader] No env_spec in metadata; assuming LIBERO env_spec "
                    "from feature-key heuristic."
                )
                return libero_env_spec()
            except Exception as exc:
                logger.warning(f"[VLALoader] LIBERO env_spec fallback failed: {exc}")
        return None

    def close(self) -> None:
        """Release in-memory data arrays."""
        self._data = None
        logger.debug("[VLALoader] Closed.")

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "VLALoader":
        return self.load()

    def __exit__(self, *_) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def _reconstruct_pairs(self) -> Optional[list]:
        """Rebuild the episode's :class:`DistancePair` list from sidecar metadata.

        The recorder writes ``translated_pairs`` (a ``dataclasses.asdict`` dump
        of the post-equivalence-mapping pairs) to the JSON sidecar. We revive
        them so the replay visualiser can draw the per-pair 3-D lines exactly
        as the live rollout did. Result is cached — the pairs are static across
        the episode. Returns ``None`` when the metadata is absent (e.g. an
        episode recorded before the field existed, or a no-abstraction run).
        """
        if self._pairs_cache is not ...:
            return self._pairs_cache

        self._pairs_cache = None
        tp_meta = (self._metadata or {}).get("translated_pairs")
        if not tp_meta:
            return None

        import dataclasses as _dc

        from policy_abstraction.vlm.types import DistancePair, DistanceType

        valid_fields = {f.name for f in _dc.fields(DistancePair)}
        pairs: list = []
        for raw in tp_meta:
            if not isinstance(raw, dict):
                continue
            kw = {k: v for k, v in raw.items() if k in valid_fields}
            dt = kw.get("distance_type")
            if isinstance(dt, str):
                try:
                    kw["distance_type"] = DistanceType(dt)
                except ValueError:
                    kw.pop("distance_type", None)
            try:
                pairs.append(DistancePair(**kw))
            except Exception as exc:
                logger.warning(f"[VLALoader] Could not rebuild DistancePair: {exc}")
        self._pairs_cache = pairs or None
        return self._pairs_cache

    def _scene_pcd_at(self, replan_idx: int) -> Optional[np.ndarray]:
        """Full-scene ``[M, 7]`` point cloud for the given replan, or ``None``.

        Stored by the recorder as a 0-d object ndarray per replan under
        ``abstraction/scene_pointcloud``. Absent for episodes recorded without
        scene-pcd capture; empty arrays (a replan where the build failed) read
        back as ``None``.
        """
        scene_arr = (
            self._data.get("abstraction/scene_pointcloud") if self._data else None
        )
        if scene_arr is None or len(scene_arr) == 0:
            return None
        item = scene_arr[min(replan_idx, len(scene_arr) - 1)]
        snap = item.item() if hasattr(item, "item") else item
        snap = np.asarray(snap)
        if snap.ndim == 2 and snap.shape[0] > 0:
            return snap
        return None

    def _entity_centroids_at(
        self,
        replan_idx: int,
        obj_pcds: Optional[Dict[str, np.ndarray]],
        robot_state: Optional[np.ndarray],
    ) -> Optional[Dict[Any, np.ndarray]]:
        """World-frame 3-D centroid per distance-pair entity for this replan.

        Prefers the recorded ``abstraction/entity_centroids`` stream (which
        also carries the robot gripper-finger tips). Falls back, for episodes
        recorded before that stream existed, to rebuilding object centroids
        from the per-object point clouds (keyed by their integer seg-id, read
        from column 6) plus the EE position from the proprioceptive state.
        Returns ``None`` when nothing can be recovered.
        """
        ec_arr = self._data.get("abstraction/entity_centroids") if self._data else None
        if ec_arr is not None and len(ec_arr) > 0:
            item = ec_arr[min(replan_idx, len(ec_arr) - 1)]
            snap = item.item() if hasattr(item, "item") else item
            if isinstance(snap, dict) and snap:
                out: Dict[Any, np.ndarray] = {}
                for k, v in snap.items():
                    # seg-ids were stringified for storage — revive to int so
                    # they match ``DistancePair.from_entity`` lookups.
                    key = (
                        int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k
                    )
                    out[key] = np.asarray(v, dtype=np.float64).ravel()[:3]
                if out:
                    return self._track_robot_entities(out, robot_state)

        # Fallback reconstruction from object point clouds + EE state.
        out = {}
        if obj_pcds:
            for pcd in obj_pcds.values():
                pcd = np.asarray(pcd)
                if pcd.ndim == 2 and pcd.shape[1] > 6 and pcd.shape[0] > 0:
                    seg_id = int(round(float(pcd[0, 6])))
                    out[seg_id] = pcd[:, :3].mean(axis=0).astype(np.float64)
        # Older recordings never wrote the finger tips either, so the robot
        # entities (EE and, when the env spec declares them, both finger
        # tips) are rebuilt from proprioception here as well; otherwise every
        # finger-to-object pair is silently dropped from the 3-D replay.
        out = self._track_robot_entities(out, robot_state)
        return out or None

    def _track_robot_entities(
        self, centroids: Dict[Any, np.ndarray], robot_state: Optional[np.ndarray]
    ) -> Dict[Any, np.ndarray]:
        """Move the EE and finger-tip entities to this step's proprioception.

        The recorded ``entity_centroids`` stream is written once per replan,
        so between replans its robot entries lag the arm by up to a chunk and
        every EE / finger pair in the 3-D view freezes until the next chunk.
        The objects genuinely are only re-perceived per replan, but the robot
        state is recorded every step, so the robot-side entities are rebuilt
        from it here, the same way the live abstractor computed them.
        """
        if robot_state is None or len(robot_state) < 3:
            return centroids
        rs = np.asarray(robot_state, dtype=np.float64)
        centroids["ee"] = rs[:3]
        has_fingers = "gripper_l" in centroids or "gripper_r" in centroids
        if not has_fingers and self._env_spec is not None:
            # A recording without the centroid stream (or one written before
            # finger tips were tracked) still names its finger entities in
            # the env spec, so rebuild them whenever the spec declares them.
            spec_entities = getattr(self._env_spec.robot, "entity_names", ()) or ()
            has_fingers = "gripper_l" in spec_entities or "gripper_r" in spec_entities
        if has_fingers and len(rs) >= 7 and self._env_spec is not None:
            from policy_abstraction.utils.gripper import (
                compute_finger_tip_positions_single,
            )

            geometry_id = getattr(
                getattr(self._env_spec.robot, "gripper", None), "geometry_id", None
            )
            try:
                tips = compute_finger_tip_positions_single(
                    rs[:3], rs[3:6], rs[6:7], geometry_id
                )
            except Exception as exc:  # pragma: no cover - geometry lookup
                logger.debug(f"[VLALoader] finger tips not rebuilt: {exc}")
            else:
                centroids.update(tips)
        return centroids

    def get_frame(self, t: int) -> Dict[str, Any]:
        """Return one timestep of data as a dict ready for :meth:`BaseVisualizer.update`.

        Parameters
        ----------
        t:
            Zero-based timestep index.

        Returns
        -------
        dict
            Keys: ``obs``, ``robot_state``, ``action``, ``reward``, ``done``,
            ``action_chunks``.
        """
        if self._data is None:
            raise RuntimeError("Call load() before accessing frame data.")
        if not (0 <= t < self._length):
            raise IndexError(f"Timestep {t} out of range [0, {self._length})")

        # --- Obs images --------------------------------------------------
        # New recordings save one image stream per camera (``observation/<name>_image``).
        # Older recordings used two fixed slots (``observation/agentview_image`` for the
        # primary, ``observation/wrist_image`` for the wrist) — fall back to those when
        # the per-camera stream is absent.
        spec = self._env_spec
        cameras = tuple(spec.cameras) if spec is not None else ()
        primary_name = spec.primary_camera.name if spec is not None else None
        obs: Dict[str, np.ndarray] = {}
        images_per_cam: Dict[str, np.ndarray] = {}
        for cam in cameras:
            arr = self._camera_image_array(cam, primary_name)
            if arr is None or t >= len(arr):
                continue
            obs[cam.image_key] = arr[t]
            images_per_cam[cam.name] = arr[t]

        # --- Proprioceptive state ----------------------------------------
        state_arr = self._data.get("observation/state")
        robot_state: Optional[np.ndarray] = (
            state_arr[t].astype(np.float32) if state_arr is not None else None
        )

        # --- Action ------------------------------------------------------
        action_arr = self._data.get("action")
        action: Optional[np.ndarray] = (
            action_arr[t].astype(np.float32) if action_arr is not None else None
        )

        # --- Reward ------------------------------------------------------
        reward_arr = self._data.get("reward")
        reward: Optional[float] = (
            float(np.asarray(reward_arr[t]).flat[0]) if reward_arr is not None else None
        )

        # --- Done --------------------------------------------------------
        done_arr = self._data.get("done")
        done: bool = bool(done_arr[t]) if done_arr is not None else False

        # --- Robot joints ------------------------------------------------
        joints_arr = self._data.get("observation/robot_joints")
        robot_joints: Optional[np.ndarray] = (
            joints_arr[t].astype(np.float32) if joints_arr is not None else None
        )

        # --- Action chunks -----------------------------------------------
        # action_chunks are logged only at replan timesteps; map via replan_steps.
        replan_steps = int((self._metadata or {}).get("replan_steps", 1)) or 1
        chunks_arr = self._data.get("policy/action_chunks")
        action_chunks: Optional[np.ndarray] = None
        if chunks_arr is not None and len(chunks_arr) > 0:
            chunk_idx = min(t // replan_steps, len(chunks_arr) - 1)
            # Only expose action_chunks at the replan boundary timestep
            if t % replan_steps == 0:
                action_chunks = chunks_arr[chunk_idx]

        # --- Abstraction / MTA data (only present when recorded) --------
        # Two recording cadences in the .vla:
        #   * per-step: live_distances, ref_distances, distance_errors,
        #     aligned_t, mta_progress, constraint_costs.
        #   * per-replan: ref_image_annotated, ref_wrist_image,
        #     live_image_annotated, alignment_plot, wrist_chunk_overlay,
        #     and the perception/* overlays.
        pair_labels: list[str] = (
            list(self._metadata.get("abstraction_pair_labels", []))
            if self._metadata
            else []
        )
        constraint_keys: list[str] = (
            list(self._metadata.get("abstraction_constraint_keys", []))
            if self._metadata
            else []
        )
        abstraction_result = None
        frame_bundle = None
        if pair_labels:
            from policy_abstraction.evaluation.abstraction import AbstractionResult
            from policy_abstraction.visualization.frame_bundle import FrameBundle

            live_d_arr = self._data.get("abstraction/live_distances")
            ref_d_arr = self._data.get("abstraction/ref_distances")
            err_arr = self._data.get("abstraction/distance_errors")
            aligned_arr = self._data.get("abstraction/aligned_t")
            prog_arr = self._data.get("abstraction/mta_progress")
            cost_arr = self._data.get("abstraction/constraint_costs")
            ref_img_arr = self._data.get("abstraction/ref_image_annotated")
            live_img_arr = self._data.get("abstraction/live_image_annotated")
            align_plot_arr = self._data.get("abstraction/alignment_plot")
            persp_ag_arr = self._data.get("perception/overlay_agentview")
            persp_wr_arr = self._data.get("perception/overlay_wrist")

            if live_d_arr is not None and len(live_d_arr) > 0:
                # Per-step fields use ``t`` directly (one entry per timestep).
                # Per-replan fields use ``t // replan_steps`` to index the
                # replan-cadence arrays.
                replan_steps = int((self._metadata or {}).get("replan_steps", 1)) or 1
                step_idx = min(t, len(live_d_arr) - 1)
                replan_idx = min(t // replan_steps, max(0, len(live_d_arr) - 1))

                def _to_dict(arr, labels, idx):
                    if arr is None or idx >= len(arr):
                        return {}
                    return {
                        lbl: float(v)
                        for lbl, v in zip(labels, np.asarray(arr[idx]).flat)
                    }

                constraint_costs: dict[str, float] = {}
                if constraint_keys and cost_arr is not None and len(cost_arr) > 0:
                    cost_idx = min(t, len(cost_arr) - 1)
                    row = np.asarray(cost_arr[cost_idx]).flat
                    constraint_costs = {
                        k: float(v)
                        for k, v in zip(constraint_keys, row, strict=False)
                        if not np.isnan(v)
                    }

                # Per-replan perception overlays (index 0=agentview, 1=wrist).
                perception_overlays: list[np.ndarray] | None = None
                ag_img = (
                    persp_ag_arr[replan_idx]
                    if persp_ag_arr is not None
                    and len(persp_ag_arr) > 0
                    and replan_idx < len(persp_ag_arr)
                    else None
                )
                wr_img = (
                    persp_wr_arr[replan_idx]
                    if persp_wr_arr is not None
                    and len(persp_wr_arr) > 0
                    and replan_idx < len(persp_wr_arr)
                    else None
                )
                if ag_img is not None or wr_img is not None:
                    perception_overlays = [
                        img for img in (ag_img, wr_img) if img is not None
                    ]

                # Per-replan pointcloud feature: one ``{name: [M, 7]}`` dict
                # per replan boundary, stored as a 0-d object ndarray by the
                # recorder. Centroids are recoverable from XYZ.
                obj_pcds: dict[str, np.ndarray] | None = None
                obj_centroids: dict[str, np.ndarray] | None = None
                pcd_arr = self._data.get("abstraction/object_pointclouds")
                if pcd_arr is not None and len(pcd_arr) > 0:
                    pcd_idx = min(replan_idx, len(pcd_arr) - 1)
                    item = pcd_arr[pcd_idx]
                    snapshot = item.item() if hasattr(item, "item") else item
                    if isinstance(snapshot, dict) and snapshot:
                        obj_pcds = snapshot
                        obj_centroids = {
                            name: pcd[:, :3].mean(axis=0)
                            for name, pcd in snapshot.items()
                            if pcd.ndim == 2 and pcd.shape[0] > 0
                        }

                # Full-scene fused point cloud (every valid-depth pixel, not
                # just the relevant objects). Recorded as a 0-d object ndarray
                # of an ``[M, 7]`` array, one per replan. Only present when the
                # episode was recorded with scene-pcd capture enabled.
                scene_pcd = self._scene_pcd_at(replan_idx)

                # Distance-pair geometry for the 3-D overlay. ``pairs`` come
                # from the static sidecar metadata; ``entity_centroids`` from
                # the recorded per-replan stream when available, else rebuilt
                # from the object point clouds (seg-id centroids) + the EE
                # position (proprioceptive state) so existing episodes still
                # draw object/EE pairs.
                pairs = self._reconstruct_pairs()
                entity_centroids = self._entity_centroids_at(
                    replan_idx, obj_pcds, robot_state
                )

                abstraction_result = AbstractionResult(
                    live_distances=_to_dict(live_d_arr, pair_labels, step_idx),
                    aligned_t=int(np.asarray(aligned_arr[step_idx]).flat[0])
                    if aligned_arr is not None and step_idx < len(aligned_arr)
                    else 0,
                    ref_distances=_to_dict(ref_d_arr, pair_labels, step_idx),
                    distance_errors=_to_dict(err_arr, pair_labels, step_idx),
                    mta_progress=float(np.asarray(prog_arr[step_idx]).flat[0])
                    if prog_arr is not None and step_idx < len(prog_arr)
                    else 0.0,
                    pair_labels=pair_labels,
                    constraint_costs=constraint_costs,
                    perception_overlays=perception_overlays,
                    object_pointclouds=obj_pcds,
                    object_centroids=obj_centroids,
                    scene_pointcloud=scene_pcd,
                    pairs=pairs,
                    entity_centroids=entity_centroids,
                )

                # Per-camera ref-image + chunk-overlay streams (replan
                # cadence). The primary camera's reference frame is the
                # standalone ``abstraction/ref_image_annotated`` stream;
                # ``ref_images`` carries the auxiliary cameras (wrist, side
                # views, …).
                chunk_overlays: dict[str, np.ndarray] = {}
                ref_images: dict[str, np.ndarray] = {}
                for cam in cameras:
                    overlay_arr = self._camera_chunk_overlay_array(cam)
                    if overlay_arr is not None and replan_idx < len(overlay_arr):
                        chunk_overlays[cam.name] = overlay_arr[replan_idx]
                    if cam.name == primary_name:
                        continue
                    ref_arr = self._camera_ref_image_array(cam)
                    if ref_arr is not None and replan_idx < len(ref_arr):
                        ref_images[cam.name] = ref_arr[replan_idx]

                # FrameBundle.primary_image expects ``images[primary_name]``
                # to exist, so insert a zero-pixel placeholder when the
                # primary camera produced no frame at this timestep.
                images_for_bundle = dict(images_per_cam)
                if primary_name and primary_name not in images_for_bundle:
                    images_for_bundle[primary_name] = np.zeros(
                        (1, 1, 3), dtype=np.uint8
                    )

                frame_bundle = FrameBundle(
                    images=images_for_bundle,
                    primary_name=primary_name or "",
                    chunk_overlays=chunk_overlays,
                    ref_images=ref_images,
                    ref_image_annotated=ref_img_arr[replan_idx]
                    if ref_img_arr is not None and replan_idx < len(ref_img_arr)
                    else None,
                    live_image_annotated=live_img_arr[replan_idx]
                    if live_img_arr is not None and replan_idx < len(live_img_arr)
                    else None,
                    alignment_plot=align_plot_arr[replan_idx]
                    if align_plot_arr is not None and replan_idx < len(align_plot_arr)
                    else None,
                )

        # --- Guidance info (rebuilt from per-step scalars) --------------
        guidance_info: dict | None = None
        gf_arr = self._data.get("guidance/factor")
        ll_arr = self._data.get("guidance/ll_factor")
        if gf_arr is not None and len(gf_arr) > 0:
            gf_idx = min(t, len(gf_arr) - 1)
            guidance_info = {
                "guidance_factor": float(np.asarray(gf_arr[gf_idx]).flat[0]),
            }
            if ll_arr is not None and len(ll_arr) > 0:
                ll_idx = min(t, len(ll_arr) - 1)
                guidance_info["ll_factor"] = float(np.asarray(ll_arr[ll_idx]).flat[0])

        # Static (one-shot) perception artefacts: debug panels + history grids.
        # These are only relevant on the first frame; the visualizer pins them
        # statically across the timeline anyway.
        if t == 0 and abstraction_result is not None:
            hist_ag = self._data.get("perception/history_agentview")
            hist_wr = self._data.get("perception/history_wrist")
            history_grids: list[np.ndarray] = []
            if hist_ag is not None and len(hist_ag) > 0:
                history_grids.append(hist_ag[0])
            if hist_wr is not None and len(hist_wr) > 0:
                history_grids.append(hist_wr[0])
            if history_grids:
                abstraction_result.perception_history_grids = history_grids

            dbg_ag = self._data.get("perception/debug_agentview")
            dbg_wr = self._data.get("perception/debug_wrist")
            debug_imgs: list[np.ndarray] = []
            if dbg_ag is not None and len(dbg_ag) > 0:
                debug_imgs.append(dbg_ag[0])
            if dbg_wr is not None and len(dbg_wr) > 0:
                debug_imgs.append(dbg_wr[0])
            if debug_imgs:
                abstraction_result.perception_debug_images = debug_imgs

            if self._metadata and "perception_timings_md" in self._metadata:
                abstraction_result.perception_timings_md = self._metadata[
                    "perception_timings_md"
                ]

        return {
            "obs": obs,
            "robot_state": robot_state,
            "robot_joints": robot_joints,
            "action": action,
            "reward": reward,
            "done": done,
            "action_chunks": action_chunks,
            "abstraction_result": abstraction_result,
            "frame_bundle": frame_bundle,
            "guidance_info": guidance_info,
        }

    def camera_params_at(self, t: int) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """The camera matrices recorded at timestep ``t``.

        Maps camera name to ``{"intrinsic": 3x3, "extrinsic": 4x4}`` in the
        convention :class:`ChunkProjector` expects (extrinsic = camera-to-world),
        or ``None`` for recordings that carry no ``camera_params`` stream.
        """
        arr = self._data.get("camera_params")
        if arr is None or len(arr) == 0:
            return None
        entry = arr[min(int(t), len(arr) - 1)]
        while isinstance(entry, np.ndarray):
            if entry.size == 0:
                return None
            entry = entry.flat[0]
        return entry if isinstance(entry, dict) and entry else None

    def iter_frames(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """Iterate over all timesteps, yielding ``(t, frame_dict)`` pairs."""
        for t in range(self._length):
            yield t, self.get_frame(t)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = f"{self._length} steps" if self._data is not None else "not loaded"
        return f"VLALoader(path={self.path!r}, status={status!r})"
