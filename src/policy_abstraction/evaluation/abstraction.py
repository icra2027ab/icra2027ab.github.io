"""Rollout abstraction pipeline: live-scene analysis, object mapping, and MTA.

Components orchestrating the rollout abstraction step that bridges the
retrieved demonstration and the current live episode:

* :class:`RolloutAbstractor` — one-time setup (VLM-based live-name
  inference, object mapping, reference trajectory precomputation) followed
  by per-step distance computation and Monotonic Trajectory Alignment (MTA).

* :class:`AbstractionResult` — structured output returned by
  :meth:`RolloutAbstractor.step` at each replanning boundary.

* :class:`LiveObsAnalyzer` — legacy GT-name analyser kept for tests;
  :class:`RolloutAbstractor` no longer uses it.
"""

from __future__ import annotations

import dataclasses
import enum
import re
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Callable, Protocol

import cv2
import h5py
import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation

from policy_abstraction.analysis.demo_episode import build_demo_episode
from policy_abstraction.dynamics import BaseDynamicsModel, IntegrationDynamicsModel
from policy_abstraction.envs.spec import CameraSpec, EnvSpec
from policy_abstraction.evaluation.alignment import (
    AlignmentMode,
    MonotonicTrajectoryAligner,
)
from policy_abstraction.features.base import DistanceFeature, compute_distances
from policy_abstraction.features.centroid import CentroidDistance
from policy_abstraction.utils.depth import mujoco_depth_to_metric
from policy_abstraction.utils.gripper import (
    compute_finger_tip_positions,
    compute_finger_tip_positions_single,
    gripper_action_to_metric,
    gripper_qpos_to_metric,
)
from policy_abstraction.utils.pcd import project_points_to_pixels
from policy_abstraction.utils.segmentation import (
    build_segmented_pointcloud,
    draw_all_segment_ids,
    erode_segmentation,
    get_robot_segment_ids,
)
from policy_abstraction.utils.trajectory import compute_subtrajectories
from policy_abstraction.utils.transforms import quat2axisangle
from policy_abstraction.visualization.drawing import (
    compute_centroids_2d,
    compute_entity_centroids_3d,
    draw_pairs_on_image,
    draw_subtraj_overlay,
)
from policy_abstraction.vlm.equivalence_mapper import (
    ObjectEquivalenceMapper,
    ObjectMapping,
)
from policy_abstraction.vlm.live_object_inferer import LiveObjectInferer
from policy_abstraction.vlm.types import ROBOT_ENTITIES, DistancePair, DistanceType
from policy_steering.robot_config import load_gripper_geometry


class PerceptionMode(str, enum.Enum):
    """How live-observation point clouds are produced.

    GT_POINTCLOUD
        Ground-truth 3-D object point clouds supplied directly by the
        simulator (no camera reconstruction).  The observation must contain
        an ``"gt_object_pointclouds"`` key — a dict mapping
        ``seg_id (int) → [N, 3] float32`` world-frame points per object.

    GT_SEG_RECONSTRUCT
        Ground-truth segmentation masks + depth → camera-based point cloud
        reconstruction.  This is the original (default) behaviour.

    REAL_PERCEPTION
        A learned segmentation model (e.g. SAM-3) predicts per-object masks
        from RGB, which are then combined with depth for reconstruction.
        Requires a ``segmentation_fn`` callable.
    """

    GT_POINTCLOUD = "gt_pointcloud"
    GT_SEG_RECONSTRUCT = "gt_seg_reconstruct"
    REAL_PERCEPTION = "REAL_PERCEPTION"


class SegmentationFn(Protocol):
    """Callable that predicts per-object masks from an RGB image.

    Parameters
    ----------
    rgb : np.ndarray
        ``[H, W, 3]`` uint8 image (already preprocessed / flipped).
    object_names : list[str]
        Names of objects to segment.

    Returns
    -------
    dict[str, np.ndarray]
        ``{object_name: [H, W] bool}`` binary masks.
    """

    def __call__(
        self,
        rgb: np.ndarray,
        object_names: list[str],
    ) -> dict[str, np.ndarray]: ...


class MultiViewSegmentationFn(Protocol):
    """Multi-camera variant of :class:`SegmentationFn`.

    Receives all views in a single call so the underlying detector can fuse
    detections across cameras (consistent ``instance_id`` per physical
    object). ``cameras`` carry intrinsics + extrinsics in the implementation's
    expected type (e.g. ``open_perception.CameraInfo``); kept as ``Any`` here
    to avoid a hard import dependency on the perception package.

    The fn is stateful: tracks objects across frames between :meth:`reset`
    calls. Callers must invoke :meth:`reset` at episode start.

    Returns one dict per view, keyed by ``instance_id`` (unique per physical
    object after multi-view fusion). Each entry pairs the matching query name
    with its ``[H, W]`` boolean mask. Multiple entries can share a query name
    when the same query matched several physical instances.
    """

    def __call__(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        cameras: list[Any],
        object_names: list[str],
    ) -> list[dict[int, tuple[str, np.ndarray]]]: ...

    def reset(self) -> None: ...


if TYPE_CHECKING:
    from policy_abstraction.analysis.demo_analyzer import DemoAnalysisResult


def _normalize_object_name(name: str) -> str:
    """Strip digits, replace underscores with spaces, collapse whitespace.

    Mirrors ``open_perception.utils.images.normalize_name``: makes
    VLM-inferred names (which omit version suffixes per the system prompt)
    comparable against GT seg-name-map entries (which retain ``_<digit>``
    suffixes from the simulator/HDF5).
    """
    return " ".join(re.sub(r"\d+", "", name).replace("_", " ").split())


def _ids_referenced_in_tasks(
    objects: dict[int, str], task_strs: list[str | None]
) -> set[int]:
    """Return ids of objects whose name stem appears in any of *task_strs*.

    Used to extend the equivalence-mapping prompt with spatial landmarks
    referenced by the task instruction (e.g. "next to the cookie box"
    → keep ``cookies_1``) so the retrieved and live object lists share
    referents and the VLM can disambiguate multi-instance matches.
    """
    task_text = " ".join(_normalize_object_name(t).lower() for t in task_strs if t)
    if task_text.strip() == "":
        return set()
    task_tokens = [t for t in task_text.split() if len(t) >= 4]
    ids: set[int] = set()
    for sid, name in objects.items():
        norm = _normalize_object_name(name).lower()
        if not norm:
            continue
        name_tokens = [t for t in norm.split() if len(t) >= 4]
        # Match when an object-name token shares a 4+ character prefix with a
        # task token. Handles trivial singular/plural variation like
        # "cookies" (object name) ↔ "cookie box" (task phrasing).
        for ntok in name_tokens:
            matched = False
            for ttok in task_tokens:
                prefix = min(len(ntok), len(ttok))
                if prefix >= 4 and ntok[:prefix] == ttok[:prefix]:
                    matched = True
                    break
            if matched:
                ids.add(int(sid))
                break
    return ids


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class LiveAnalysisResult:
    """Output from :class:`LiveObsAnalyzer.analyze`.

    Describes the current live scene's objects as seen in the first frame of
    the episode (after the wait phase).
    """

    objects: dict[int, str]
    """``{seg_id: name}`` for all visible non-background objects in the frame.

    Names come from the environment's segmentation name map when available,
    falling back to ``str(seg_id)`` for unknown IDs.
    """

    scene_image: np.ndarray | None
    """``[H, W, 3]`` uint8 RGB of the first live frame (180°-flipped, upright)."""

    seg_index_image: np.ndarray | None
    """``[H, W, 3]`` uint8 RGB segmentation overlay with all IDs labelled."""

    all_seg_ids: set[int]
    """All unique non-zero segmentation IDs visible in the frame."""


@dataclasses.dataclass
class AbstractionResult:
    """Per-replan output from :class:`RolloutAbstractor.step`.

    Contains live distances, the MTA-aligned reference timestep, and derived
    quantities suitable for recording and visualisation.
    """

    live_distances: dict[str, float]
    """``{pair_label: distance}`` computed from the current live point cloud."""

    aligned_t: int
    """Reference trajectory index determined by MTA (monotonically non-decreasing)."""

    ref_distances: dict[str, float]
    """Reference distances at ``aligned_t`` (from the precomputed ref trajectory)."""

    distance_errors: dict[str, float]
    """``live_distances[k] - ref_distances[k]`` for each shared key."""

    mta_progress: float
    """Fractional completion ``aligned_t / (T_ref - 1)`` in ``[0, 1]``."""

    pair_labels: list[str]
    """Ordered list of pair labels (column order for the numpy arrays)."""

    subtrajectories: list[tuple[int, int]] | None = None
    """List of ``(start, end)`` index pairs defining sub-trajectory segments in the
    reference trajectory (end is exclusive).  Boundaries are determined by drops in
    gripper (EE) absolute speed below a threshold."""

    current_subtraj_idx: int | None = None
    """Index into ``subtrajectories`` of the segment that contains ``aligned_t``."""

    object_pointclouds: dict[str, np.ndarray] | None = None
    """``{object_name: [M, 7]}`` point clouds (XYZ+RGB+seg_id) for each relevant object
    in the current live frame.  Keyed by human-readable object name."""

    object_centroids: dict[str, np.ndarray] | None = None
    """``{object_name: [3]}`` world-frame centroid positions for each relevant object."""

    scene_pointcloud: np.ndarray | None = None
    """``[M, 7]`` full-scene fused point cloud (XYZ+RGB+seg_id) — every
    valid-depth pixel across all cameras, not just the relevant objects.
    ``None`` unless scene-pcd capture is enabled. Rendered as ``scene/full_pcd``
    (separate from the per-object :attr:`object_pointclouds`) so the whole
    scene is visible while replaying. Sub-sampled to a point cap at record time
    to bound ``.vla`` size."""

    pairs: list[DistancePair] | None = None
    """Translated distance pairs used at this step (from :attr:`RolloutAbstractor._translated_pairs`).
    Needed by :func:`~policy_abstraction.evaluation.guidance_builder.build_distance_guidance`
    to recover pair geometry (axis/plane) and EE involvement."""

    entity_centroids: dict[int | str, np.ndarray] | None = None
    """``{entity: [3]}`` world-frame centroids keyed by ``int`` seg-id or ``"ee"``.
    Covers every entity referenced by :attr:`pairs`, plus the EE position.
    Used by :func:`~policy_abstraction.evaluation.guidance_builder.build_distance_guidance`."""

    alignment_mse: float = float("inf")
    """MSE of the MTA alignment at the current step.  Lower = better alignment with the
    reference trajectory.  Used to dynamically scale guidance intensity via exponential
    decay: ``guidance_factor = base * exp(-k * alignment_mse)``."""

    constraint_costs: dict[str, float] = dataclasses.field(default_factory=dict)
    """``{constraint_label: cost}`` for every evaluated constraint at this step.
    A cost ≤ 0 means the constraint is satisfied; cost > 0 means it is violated.
    Sub-goal constraints are keyed with a ``"sg/"`` prefix, path constraints with
    ``"pc/"``."""

    aligned_ts: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(0, dtype=np.int32)
    )
    """Per-pair aligned reference timestep indices ``(n_pairs,)`` int32.
    In UNIFIED mode all entries equal ``aligned_t``; in PER_FEATURE/MIXED
    modes each entry is independently aligned."""

    guidance_weights: np.ndarray = dataclasses.field(
        default_factory=lambda: np.ones(0, dtype=np.float32)
    )
    """Per-pair temporal guidance weight array ``(n_pairs,)`` computed from
    the per-feature alignment spread via softmax-over-lag weighting.
    Uniform (all ``1.0``) in UNIFIED mode."""

    relevances: dict[str, float] = dataclasses.field(default_factory=dict)
    """Per-object high-level relevance ``{object_name: score}`` in ``[0, 1]``.
    Computed for objects that appear in a distance pair with any robot entity
    (``"ee"``, ``"gripper_l"``, or ``"gripper_r"``).  Score is 1.0 when the EE
    is ≥ 10 cm away from the object and smoothly drops to 0.0 when the EE is
    ≤ 5 cm away."""

    perception_overlays: list[np.ndarray] | None = None
    """Per-view detection-overlay images from the latest predicted-segmentation
    call. Index 0 = agentview, 1 = wrist (when both views are fused). ``None``
    when ``perception_mode != REAL_PERCEPTION`` or the segmentation
    fn does not expose render helpers."""

    perception_history_grids: list[np.ndarray] | None = None
    """Per-view per-stage history grid (horizontal concat of the pipeline's
    stage snapshots). Useful for inspecting the SAM3 → filter → assign →
    reconstruct → fuse_views evolution at each replan. ``None`` when not
    available."""

    perception_debug_images: list[np.ndarray] | None = None
    """Per-view ``render_debug_image`` panels (matplotlib renderings of the
    detect frame's stage_history with raw scores / metrics). ``None`` unless
    ``debug_perception`` is enabled on the perception client."""

    perception_timings_md: str | None = None
    """Markdown-formatted per-stage timings of the latest perception call,
    suitable for a rerun ``TextDocument``."""

    perception_latency: float = 0.0
    """Wall-clock seconds spent in :meth:`RolloutAbstractor._build_live_pcd`
    (segmentation + reconstruction) at this step.  ``0.0`` when not measured."""


def _gripper_aperture_scalar(gripper_states) -> np.ndarray | None:
    """``(T, k)`` raw gripper qpos → the ``(T, 1)`` scalar in the robot's own units.

    Two published layouts, one meaning. A Panda reports both finger joints, of
    which the second is negative, so their difference is the total aperture. A
    single-DOF driver (the cell's Robotiq) already reports that total, and
    subtracting a column that does not exist is what used to raise here.

    The result stays in the units the recording used: metres for the Panda,
    normalised openness for the Robotiq. That is the frame the dynamics
    roll-out wants for its initial state (its gripper channel is mapped with
    ``gripper_action_to_metric`` afterwards). Anything compared against a live
    distance goes through :func:`_gripper_aperture_metric` instead.
    """
    if gripper_states is None:
        return None
    arr = np.atleast_2d(np.asarray(gripper_states, dtype=np.float32))
    if arr.shape[1] >= 2:
        return (arr[:, 0:1] - arr[:, 1:2]).astype(np.float32)
    if arr.shape[1] == 1:
        return arr.astype(np.float32)
    return None


def _gripper_aperture_metric(
    gripper_states, geometry: str | None = None
) -> np.ndarray | None:
    """``(T, k)`` raw gripper qpos → the ``(T, 1)`` aperture in metres.

    The live side measures its aperture with ``gripper_qpos_to_metric``, so
    the reference has to be in the same frame or the two never agree. For the
    Panda the joint difference already is metres; a single-DOF Robotiq value
    is a normalised openness that has to be scaled by the stroke of the
    gripper the *demo* was recorded with. Comparing the raw openness (0..1)
    against a live gap in metres (0..0.14) made the aperture pair read the
    live gripper as "at the start of the demo" whatever it did, which pinned
    the alignment cursor at the start of the approach and had the guidance
    drag the arm back to the demo's standoff at every replan.
    """
    scalar = _gripper_aperture_scalar(gripper_states)
    if scalar is None:
        return None
    arr = np.atleast_2d(np.asarray(gripper_states))
    if arr.shape[1] >= 2:
        return scalar
    return np.asarray(
        gripper_qpos_to_metric(scalar[:, 0], geometry), dtype=np.float32
    ).reshape(-1, 1)


# ---------------------------------------------------------------------------
# High-level relevance
# ---------------------------------------------------------------------------


def _states_after_actions(
    dynamics_model: BaseDynamicsModel,
    actions: np.ndarray,
    initial_state: np.ndarray,
    dt: float = 1.0 / 50,
) -> np.ndarray:
    """``[initial_state, state after actions[0], state after actions[1], ...]``.

    ``predict_states`` is given a leading pad action so the roll-out has one
    more state than actions, and that pad row is then pinned to
    ``initial_state``. The pinning cannot be left to the model: the
    integration model reads the pad as a zero velocity and holds the initial
    state, but the UR5e kinematics model reads it as joint targets, and
    all-zero joints put the tool ~0.8 m away at the fully extended arm. A
    reference chunk whose first row sat there told the guidance that the very
    first action of every chunk was metres off target, which came out as a
    one-step joint spike the env server had to clamp. Both models also pass
    the pad's gripper channel (0, closed) through, so the gripper is pinned too.
    """
    actions = np.asarray(actions)
    padded = np.concatenate(
        [np.zeros((1, actions.shape[-1]), dtype=actions.dtype), actions], axis=0
    )
    states = np.array(
        dynamics_model.predict_states(padded, initial_state, dt=dt), copy=True
    )
    states[0] = initial_state
    return states


def compute_hl_relevances(
    pairs: list,
    ee_pos: np.ndarray,
    entity_centroids: dict[int | str, np.ndarray],
    int_name_map: dict[int, str],
    *,
    near_threshold: float = 0.05,
    far_threshold: float = 0.15,
) -> dict[str, float]:
    """Compute per-object high-level relevance for robot-involving distance pairs.

    Relevance captures how necessary the guidance is for a given object: it is
    high when the end-effector is far from the object (the robot still needs to
    be guided towards it) and drops to zero when the EE is already close
    (guidance is no longer needed).

    The score is linearly interpolated between *near_threshold* and
    *far_threshold* and clamped to ``[0, 1]``.

    Pairs whose robot side is ``"ee"``, ``"gripper_l"``, or ``"gripper_r"`` are
    all considered — for grasping tasks the VLM frequently picks the fingers
    rather than the EE, and those pairs still need a non-zero HL factor.
    Robot-to-robot pairs (e.g. ``gripper_l``<->``gripper_r`` aperture) have no
    spatial target and are skipped.

    Parameters
    ----------
    pairs:
        Distance pairs (from ``RolloutAbstractor._translated_pairs``).
    ee_pos:
        ``(3,)`` end-effector world-frame position.  Used as the proxy robot
        position for the relevance distance (finger tips are within a few cm,
        so the coarse near/far heuristic is unchanged).
    entity_centroids:
        ``{entity_id: (3,)}`` world-frame centroids keyed by seg-id or
        ``"ee"``.
    int_name_map:
        ``{seg_id: object_name}`` mapping from segmentation IDs to
        human-readable names.
    near_threshold:
        Distance (m) at or below which relevance is 0.  Default 5 cm.
    far_threshold:
        Distance (m) at or above which relevance is 1.  Default 10 cm.

    Returns
    -------
    dict[str, float]
        ``{object_name: relevance}`` for each unique object that shares a
        pair with a robot entity and whose centroid is available.
    """
    relevances: dict[str, float] = {}
    span = far_threshold - near_threshold

    for pair in pairs:
        from_is_robot = pair.from_entity in ROBOT_ENTITIES
        to_is_robot = pair.to_entity in ROBOT_ENTITIES
        if not (from_is_robot or to_is_robot):
            continue
        # Robot-to-robot pairs (e.g. gripper aperture) have no spatial target.
        if from_is_robot and to_is_robot:
            continue
        other = pair.to_entity if from_is_robot else pair.from_entity

        obj_name = (
            int_name_map.get(other, str(other))
            if isinstance(other, int)
            else str(other)
        )
        if obj_name in relevances:
            continue

        centroid = entity_centroids.get(other)
        if centroid is None:
            continue

        dist = float(np.linalg.norm(ee_pos - centroid))
        if span <= 0.0:
            # Degenerate ramp: treat relevance as a hard step at near_threshold.
            relevances[obj_name] = 1.0 if dist > near_threshold else 0.0
        else:
            relevances[obj_name] = float(
                np.clip((dist - near_threshold) / span, 0.0, 1.0)
            )

    return relevances


# ---------------------------------------------------------------------------
# Live-scene helpers
# ---------------------------------------------------------------------------


def _extract_live_scene_image(obs: dict, camera: CameraSpec) -> np.ndarray | None:
    """Upright RGB from *camera* in a live observation dict.

    Both the observation key and the orientation fix come from the camera
    spec: a simulator publishes its frames upside-down and the spec marks
    that with ``image_transform="flip_vh"``, while a real sensor publishes
    them upright. Reading through the spec is what lets one scene-image
    path serve every backend.
    """
    raw_img = obs.get(camera.image_key)
    if raw_img is None:
        return None
    return np.ascontiguousarray(camera.apply_image_transform(np.asarray(raw_img)))


def _render_relevant_seg_image(
    obs: dict,
    relevant_seg_ids: set[int],
    camera: CameraSpec,
) -> np.ndarray | None:
    """Render an overlay highlighting only ``relevant_seg_ids`` on *camera*.

    Used by GT modes to give :class:`ObjectEquivalenceMapper` a labelled
    image scoped to the VLM-inferred subset (instead of the full scene).
    Returns ``None`` if either RGB or seg mask is missing — including on a
    camera that publishes no segmentation at all, which is every real one.
    """
    rgb = _extract_live_scene_image(obs, camera)
    seg_raw = obs.get(camera.segmentation_key) if camera.segmentation_key else None
    if rgb is None or seg_raw is None:
        return None
    seg_mask = np.ascontiguousarray(camera.apply_image_transform(np.asarray(seg_raw)))
    seg_2d = seg_mask[..., 0] if seg_mask.ndim == 3 else seg_mask
    visible_ids = {int(v) for v in np.unique(seg_2d) if int(v) != 0}
    exclude_ids = visible_ids - relevant_seg_ids
    try:
        seg_bgr = draw_all_segment_ids(seg_mask, rgb, exclude_ids=exclude_ids)
        return cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning(f"[abstraction] _render_relevant_seg_image failed: {exc}")
        return None


def _render_demo_seg_image(
    seg_mask: np.ndarray | None,
    scene_image: np.ndarray | None,
    relevant_seg_ids: set[int],
) -> np.ndarray | None:
    """Re-render the demo seg overlay scoped to ``relevant_seg_ids``.

    Mirrors :func:`_render_relevant_seg_image` for the retrieved-demo side,
    where the raw mask and RGB come from :class:`DemoAnalysisResult` rather
    than a live obs dict.  Returns ``None`` if either input is missing.
    """
    if seg_mask is None or scene_image is None:
        return None
    seg_2d = seg_mask[..., 0] if seg_mask.ndim == 3 else seg_mask
    visible_ids = {int(v) for v in np.unique(seg_2d) if int(v) != 0}
    exclude_ids = visible_ids - relevant_seg_ids
    try:
        seg_bgr = draw_all_segment_ids(seg_mask, scene_image, exclude_ids=exclude_ids)
        return cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        logger.warning(f"[abstraction] _render_demo_seg_image failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# LiveObsAnalyzer (legacy — kept for tests; new pipeline uses the helpers above)
# ---------------------------------------------------------------------------


class LiveObsAnalyzer:
    """Identifies task-relevant objects in the first live observation frame.

    Unlike :class:`~policy_abstraction.analysis.demo_analyzer.DemoAnalyzer`,
    this class operates on a single live obs dict (one frame) rather than a
    full HDF5 rollout, so gripper-interaction detection is not possible.

    Object discovery relies on:
    1. The environment's segmentation name map (passed at construction),
       filtered to only IDs that are actually visible in the current frame.
    2. Fallback: ``str(seg_id)`` for any visible ID absent from the map.

    The result is used by :class:`RolloutAbstractor` to feed the VLM
    equivalence-mapping step.

    Parameters
    ----------
    seg_name_map:
        Optional ``{str(seg_id): name}`` mapping read from the environment
        or from the retrieved demo's HDF5 ``data.attrs["segmentation_name_mapping"]``.
    """

    def __init__(self, seg_name_map: dict[str, str] | None = None) -> None:
        self._name_map: dict[int, str] = (
            {int(k): v for k, v in seg_name_map.items()} if seg_name_map else {}
        )

    def analyze(self, obs: dict, task_description: str) -> LiveAnalysisResult:
        """Extract visible objects from the live observation.

        Parameters
        ----------
        obs:
            Live observation dict from the LIBERO environment.
            Expected keys: ``"agentview_image"``, ``"agentview_segmentation"``
            (optional — returns an empty result if absent).
        task_description:
            Natural-language task string (currently unused, reserved for
            future VLM-based filtering).

        Returns
        -------
        LiveAnalysisResult
        """
        # ---- Scene image (upright) -----------------------------------------
        raw_img = obs.get("agentview_image")
        scene_image: np.ndarray | None = None
        if raw_img is not None:
            scene_image = np.ascontiguousarray(np.asarray(raw_img)[::-1, ::-1])

        # ---- Segmentation mask ---------------------------------------------
        seg_raw = obs.get("agentview_segmentation_instance")
        if seg_raw is None:
            logger.warning(
                "[LiveObsAnalyzer] 'agentview_segmentation_instance' not found in live obs. "
                "Enable segmentation observations in the environment config."
            )
            return LiveAnalysisResult(
                objects={},
                scene_image=scene_image,
                seg_index_image=None,
                all_seg_ids=set(),
            )

        seg_mask = np.ascontiguousarray(
            np.asarray(seg_raw)[::-1, ::-1]
        )  # [H, W] or [H, W, 1]

        # ---- Collect visible seg IDs & exclude robot/gripper ----------------
        all_vis_ids = set(int(x) for x in np.unique(seg_mask) if int(x) != 0)
        robot_ids = get_robot_segment_ids(self._name_map) if self._name_map else set()
        unique_ids = all_vis_ids - robot_ids

        # ---- Build objects dict from name map ------------------------------
        objects: dict[int, str] = {
            sid: self._name_map.get(sid, str(sid)) for sid in unique_ids
        }

        # ---- Segmentation index image for VLM context ----------------------
        seg_index_image: np.ndarray | None = None
        if scene_image is not None and seg_mask is not None:
            try:
                seg_bgr = draw_all_segment_ids(
                    seg_mask, scene_image, exclude_ids=robot_ids
                )
                seg_index_image = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
            except Exception as exc:
                logger.warning(f"[LiveObsAnalyzer] draw_all_segment_ids failed: {exc}")

        return LiveAnalysisResult(
            objects=objects,
            scene_image=scene_image,
            seg_index_image=seg_index_image,
            all_seg_ids=unique_ids,
        )


# ---------------------------------------------------------------------------
# RolloutAbstractor
# ---------------------------------------------------------------------------


class RolloutAbstractor:
    """Orchestrates rollout abstraction and per-step MTA for a single episode.

    Lifecycle
    ---------
    1. Construct with *top_result* (retrieval output) and *analysis*
       (:class:`~policy_abstraction.analysis.demo_analyzer.DemoAnalysisResult`).
    2. Call :meth:`setup` once on the first real observation frame.  This
       runs :class:`~policy_abstraction.vlm.live_object_inferer.LiveObjectInferer`
       to discover relevant live-scene object names, resolves those names to
       seg IDs (per perception mode), runs
       :class:`~policy_abstraction.vlm.equivalence_mapper.ObjectEquivalenceMapper`,
       and precomputes the reference distance trajectory from the retrieved HDF5.
    3. Call :meth:`step` at each replanning boundary (before the policy query)
       to compute live distances and obtain the MTA-aligned reference timestep.

    Parameters
    ----------
    top_result:
        Dict from the retriever containing at least ``"source_path"``,
        ``"demo_key"``, and ``"task"`` (libero source only — demo replay
        on calvin/robocasa is Phase 2).
    analysis:
        :class:`~policy_abstraction.analysis.demo_analyzer.DemoAnalysisResult`
        from the demo analyzer run on *top_result*.
    vlm_model:
        OpenAI model name for the equivalence mapping VLM call.
    feature:
        :class:`~policy_abstraction.features.base.DistanceFeature` instance
        for computing pairwise distances.  Defaults to
        :class:`~policy_abstraction.features.centroid.CentroidDistance`.
    live_object_inferer:
        :class:`~policy_abstraction.vlm.live_object_inferer.LiveObjectInferer`
        used in :meth:`setup` to discover the relevant live-scene object names
        from (task, retrieved demo, live RGB). Required — abstraction is
        disabled for the episode if missing.
    gt_name_lookup:
        Callable returning the live environment's full ``{str(seg_id): name}``
        mapping. Used only by :data:`PerceptionMode.GT_SEG_RECONSTRUCT` and
        :data:`PerceptionMode.GT_POINTCLOUD` modes to resolve the inferred names
        to seg IDs. Must be ``None`` for :data:`PerceptionMode.REAL_PERCEPTION`
        — that mode never consults the simulator's name map.
    mta_window:
        Lookahead window for greedy MTA (default 10).
    """

    def __init__(
        self,
        top_result: dict,
        analysis: "DemoAnalysisResult",
        *,
        env_spec: EnvSpec,
        vlm_model: str = "gpt-5.4",
        feature: DistanceFeature | None = None,
        live_object_inferer: LiveObjectInferer | None = None,
        gt_name_lookup: Callable[[], dict[str, str]] | None = None,
        mta_window: int | None = 10,
        gripper_speed_threshold: float = 0.002,
        subtraj_min_length: int = 20,
        erode_px: int = 3,
        perception_mode: PerceptionMode = PerceptionMode.GT_SEG_RECONSTRUCT,
        segmentation_fn: SegmentationFn | None = None,
        multi_view_segmentation_fn: MultiViewSegmentationFn | None = None,
        mta_max_advance: float | None = None,
        subtraj_time_weight: float = 1.0,
        alignment_mode: AlignmentMode = AlignmentMode.UNIFIED,
        mixed_variation: int = 5,
        temporal_factor_temperature: float = 1.0,
        temporal_factor_floor: float = 0.05,
        sg_switch_persistence: int = 5,
        relevance_near_threshold: float = 0.05,
        relevance_far_threshold: float = 0.15,
        dynamics_model: BaseDynamicsModel | None = None,
        subtraj_boundaries: list[tuple[int, int]] | None = None,
        record_scene_pointcloud: bool = False,
        scene_pcd_max_points: int = 30000,
        demo_episode=None,
    ) -> None:
        self._top_result = top_result
        self._analysis = analysis
        self._env_spec = env_spec
        # The already-opened retrieved demo, when the caller has one. Analysis
        # built it a moment ago and, for a real recording, filled its
        # segmentation cache along the way -- re-opening here would re-run the
        # tracker over the whole episode and hand back different instance ids
        # than the ones the VLM's pairs were written against.
        self._demo_episode = demo_episode
        self._subtraj_boundaries_override: list[tuple[int, int]] | None = (
            [(int(s), int(e)) for s, e in subtraj_boundaries]
            if subtraj_boundaries
            else None
        )
        self._primary_camera: CameraSpec = env_spec.primary_camera
        self._wrist_camera: CameraSpec | None = env_spec.wrist_camera
        # Finger geometry of the *live* robot, selected by the env spec. The
        # reference demo keeps its own (Panda) geometry — see the reference
        # trajectory builder — because a cross-embodiment rollout compares two
        # different grippers at the same instant.
        self._gripper_geometry = load_gripper_geometry(
            env_spec.robot.gripper.geometry_id
        )
        self._vlm_model = vlm_model
        self._feature: DistanceFeature = feature or CentroidDistance()
        self._live_object_inferer = live_object_inferer
        self._gt_name_lookup = gt_name_lookup
        # Populated in setup() once VLM-inferred names are resolved against the
        # appropriate ID source (synthesized for PREDICTED, GT lookup for GT).
        self._int_name_map: dict[int, str] = {}
        self._mta_window = mta_window
        self._speed_threshold = gripper_speed_threshold
        self._subtraj_min_length = subtraj_min_length
        self._erode_px = erode_px
        self._perception_mode = perception_mode
        self._segmentation_fn = segmentation_fn
        self._multi_view_segmentation_fn = multi_view_segmentation_fn
        self._mta_max_advance = mta_max_advance
        self._subtraj_time_weight = subtraj_time_weight
        self._alignment_mode = alignment_mode
        self._mixed_variation = mixed_variation
        self._temporal_factor_temperature = temporal_factor_temperature
        self._temporal_factor_floor = temporal_factor_floor
        self._sg_switch_persistence = sg_switch_persistence
        self._relevance_near_threshold = relevance_near_threshold
        self._relevance_far_threshold = relevance_far_threshold
        self._record_scene_pointcloud = record_scene_pointcloud
        self._scene_pcd_max_points = scene_pcd_max_points
        self._dynamics_model: BaseDynamicsModel = (
            dynamics_model or IntegrationDynamicsModel()
        )

        if (
            perception_mode == PerceptionMode.REAL_PERCEPTION
            and segmentation_fn is None
            and multi_view_segmentation_fn is None
        ):
            raise ValueError(
                "REAL_PERCEPTION mode requires either segmentation_fn "
                "or multi_view_segmentation_fn."
            )

        # Set after setup()
        self._live_analysis: LiveAnalysisResult | None = None
        self._mapping: ObjectMapping | None = None
        self._translated_pairs: list[DistancePair] = []
        self._pair_labels: list[str] = []
        self._relevant_seg_ids: set[int] = set()
        self._live_to_demo_ids: dict[int | str, list[int | str]] = {}
        self._ref_traj: np.ndarray | None = None  # [T_ref, n_pairs]
        self._ref_std: np.ndarray | None = (
            None  # [T_ref, n_pairs] per-pair uncertainty from multi-demo alignment
        )
        self._aligner: MonotonicTrajectoryAligner | None = None
        self._ready: bool = False
        self._ref_actions: np.ndarray | None = None  # [T-1, action_dim] demo actions
        self._ref_entity_centroids: dict[int | str, np.ndarray] | None = (
            None  # {seg_id_or_robot_entity: [T, 3]}
        )

        # Sub-trajectory segmentation (set after precomputation)
        self._subtrajectories: list[tuple[int, int]] = []
        self._ref_ee_pos: np.ndarray | None = None  # [T, 3]
        self._ref_ee_ori: np.ndarray | None = None  # [T, 3] axis-angle
        self._ref_gripper: np.ndarray | None = None  # [T, 1] scalar gripper state
        self._ref_ee_2d: np.ndarray | None = None  # [T, 2] pixel coords
        self._ref_ee_valid: np.ndarray | None = None  # [T] bool mask

        # Logging helpers
        self._depth_warn_logged: bool = False
        self._no_relevant_names_logged: bool = False

        # Latest perception artefacts (populated by _pcd_from_predicted_seg_multiview
        # when the multi-view segmentation fn exposes render_overlays /
        # render_history_grids / last_timings_md helpers). Used to surface
        # detection/tracking visualisations to rerun and the recorder.
        self._perception_overlays: list[np.ndarray] | None = None
        self._perception_history_grids: list[np.ndarray] | None = None
        self._perception_debug_images: list[np.ndarray] | None = None
        self._perception_timings_md: str | None = None
        # Latest agentview predicted-seg array (synthetic IDs), refreshed every
        # step in _pcd_from_predicted_seg_multiview. Read by VisualizationHandler
        # to draw the live agentview overlay without leaking GT seg.
        self._latest_agentview_predicted_seg: np.ndarray | None = None

        # Cache for reference-frame data to avoid repeated HDF5 reads
        # (aligned_t, rgb, dep, seg, K, E, ee_pos, wrist_rgb, wrist_K, wrist_E)
        self._ref_frame_cache: tuple | None = None

        # Compiled constraint callables: list of (kind, label, subtraj_idx, callable)
        self._compiled_constraints: list[tuple[str, str, int | None, Callable]] = []
        # Sub-goal fns indexed by subtraj_idx for fast satisfaction checks.
        self._subtraj_sg_fns: dict[int, list] = {}
        for c in analysis.constraints:
            try:
                fn = c.compile()
                self._compiled_constraints.append((c.kind, c.label, c.subtraj_idx, fn))
                if c.kind == "sub_goal" and c.subtraj_idx is not None:
                    self._subtraj_sg_fns.setdefault(c.subtraj_idx, []).append(fn)
            except Exception as exc:
                logger.error(
                    f"[RolloutAbstractor] Failed to compile constraint '{c.label}': {exc}"
                )
                raise

    # ------------------------------------------------------------------
    # Properties for callers (e.g., episode_runner, visualizer)
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def pair_labels(self) -> list[str]:
        return self._pair_labels

    @property
    def ref_traj(self) -> np.ndarray | None:
        return self._ref_traj

    @property
    def ref_std(self) -> np.ndarray | None:
        """Per-pair uncertainty ``[T_ref, n_pairs]`` from multi-demo DTW alignment."""
        return self._ref_std

    @property
    def ref_ee_ori(self) -> np.ndarray | None:
        """Reference EE orientation ``[T_ref, 3]`` in axis-angle."""
        return self._ref_ee_ori

    @property
    def ref_gripper(self) -> np.ndarray | None:
        """Reference gripper state ``[T_ref, 1]`` (scalar q1 - q2)."""
        return self._ref_gripper

    @property
    def live_analysis(self) -> LiveAnalysisResult | None:
        return self._live_analysis

    @property
    def mapping(self) -> ObjectMapping | None:
        return self._mapping

    @property
    def aligner(self) -> MonotonicTrajectoryAligner | None:
        return self._aligner

    # ------------------------------------------------------------------
    # One-time setup
    # ------------------------------------------------------------------

    def setup(
        self,
        obs: dict,
        task_description: str,
        camera_params: dict[str, dict] | None = None,
    ) -> bool:
        """Initialise the abstractor for the current episode.

        Parameters
        ----------
        obs:
            First live observation (after the wait phase).
        task_description:
            Natural-language task string for the current episode.

        Returns
        -------
        bool
            ``True`` if setup succeeded and the abstractor is ready to use;
            ``False`` on any failure (episode continues without abstraction).
        """
        # 1. Capture live scene image and reset stateful perception backends.
        # The reset must run *before* any setup-time detect so the per-name
        # instance_ids returned below come from a fresh tracker.
        live_scene_image = _extract_live_scene_image(obs, self._primary_camera)
        if live_scene_image is None:
            logger.warning(
                f"[RolloutAbstractor] Primary camera "
                f"{self._primary_camera.name!r} publishes no image at obs key "
                f"{self._primary_camera.image_key!r} (obs has "
                f"{sorted(obs)[:12]}…); abstraction disabled for this episode."
            )
            return False
        if self._multi_view_segmentation_fn is not None:
            self._multi_view_segmentation_fn.reset()

        # 2. VLM live-name inference --------------------------------------
        if self._live_object_inferer is None:
            logger.warning(
                "[RolloutAbstractor] No LiveObjectInferer provided; "
                "abstraction disabled for this episode."
            )
            return False
        cache_key = f"{task_description}::{self._top_result.get('demo_key', 'unknown')}"
        live_object_vocabulary: list[str] | None = None
        if self._gt_name_lookup is not None:
            try:
                live_object_vocabulary = sorted(
                    {n for n in (self._gt_name_lookup() or {}).values() if n}
                )
            except Exception as exc:
                logger.warning(
                    f"[RolloutAbstractor] gt_name_lookup failed while building "
                    f"live-object vocabulary: {exc}"
                )
                live_object_vocabulary = None
        relevant_names = self._live_object_inferer.infer(
            task_description,
            self._analysis,
            live_scene_image,
            cache_key=cache_key,
            live_object_vocabulary=live_object_vocabulary,
            retrieved_task=self._top_result.get("task"),
        )
        if not relevant_names:
            logger.warning(
                "[RolloutAbstractor] LiveObjectInferer returned no names; "
                "abstraction disabled for this episode."
            )
            return False

        # 3. Resolve names → (live_objects, live_seg_image) per perception mode.
        live_objects, live_seg_image = self._resolve_live_objects(
            relevant_names, obs, camera_params
        )
        if not live_objects:
            logger.warning(
                "[RolloutAbstractor] No live objects resolved from VLM names "
                f"({relevant_names}); abstraction disabled for this episode."
            )
            return False
        self._int_name_map = dict(live_objects)
        self._live_analysis = LiveAnalysisResult(
            objects=dict(live_objects),
            scene_image=live_scene_image,
            seg_index_image=live_seg_image,
            all_seg_ids=set(live_objects.keys()),
        )

        # 4. Map demo objects → live objects via VLM equivalence ----------
        # Restrict the equivalence-mapping prompt to retrieved objects that
        # actually appear in the distance pairs PLUS any retrieved demo
        # object whose name-stem is referenced in the retrieved or live task
        # instruction (e.g. "next to the cookie box" pulls in ``cookies_1``).
        # Symmetric landmarks let the VLM disambiguate multi-instance
        # matches via spatial referring expressions.
        retrieved_task_str = self._top_result.get("task")
        interacting_ids = {
            sid
            for sid in self._analysis.interacting_ids
            if sid in self._analysis.objects
        }
        landmark_ids = _ids_referenced_in_tasks(
            self._analysis.objects, [retrieved_task_str]
        )
        relevant_retrieved_ids = (interacting_ids | landmark_ids) & set(
            self._analysis.objects
        )
        if relevant_retrieved_ids:
            retrieved_objects = {
                sid: name
                for sid, name in self._analysis.objects.items()
                if sid in relevant_retrieved_ids
            }
            retrieved_seg_image = _render_demo_seg_image(
                self._analysis.seg_mask,
                self._analysis.scene_image,
                relevant_retrieved_ids,
            )
            if retrieved_seg_image is None:
                retrieved_seg_image = self._analysis.seg_index_image
        else:
            retrieved_objects = self._analysis.objects
            retrieved_seg_image = self._analysis.seg_index_image

        retrieved_centroids_2d: dict[int, tuple[int, int]] | None = None
        if self._analysis.seg_mask is not None:
            retrieved_centroids_2d = compute_centroids_2d(
                self._analysis.seg_mask, set(retrieved_objects.keys())
            )

        live_centroids_2d, live_centroids_3d = self._compute_live_centroids(
            list(live_objects.keys()), obs, camera_params
        )

        equiv_mapper = ObjectEquivalenceMapper(model=self._vlm_model)
        self._mapping = equiv_mapper.map_objects(
            retrieved_objects=retrieved_objects,
            live_objects=live_objects,
            retrieved_scene_image=self._analysis.scene_image,
            live_scene_image=live_scene_image,
            retrieved_seg_image=retrieved_seg_image,
            live_seg_image=live_seg_image,
            retrieved_task=retrieved_task_str,
            live_task=task_description,
            retrieved_centroids_2d=retrieved_centroids_2d,
            live_centroids_2d=live_centroids_2d,
            live_centroids_3d=live_centroids_3d,
        )

        # 3. Translate distance pairs -------------------------------------
        self._translated_pairs = equiv_mapper.translate_pairs(
            self._analysis.distance_pairs, self._mapping
        )

        if not self._translated_pairs:
            logger.warning(
                "[RolloutAbstractor] No translated distance pairs after equivalence mapping; "
                "abstraction disabled for this episode. "
                f"Retrieved pairs: {self._analysis.distance_pairs}, "
                f"id_map: {self._mapping.id_map}"
            )
            return False

        self._pair_labels = [p.label for p in self._translated_pairs]

        dropped = len(self._analysis.distance_pairs) - len(self._translated_pairs)
        if dropped > 0:
            kept = {p.label for p in self._translated_pairs}
            logger.warning(
                f"[RolloutAbstractor] Equivalence mapping dropped {dropped} of "
                f"{len(self._analysis.distance_pairs)} distance pairs: "
                f"{[p.label for p in self._analysis.distance_pairs if p.label not in kept]}. "
                f"id_map={self._mapping.id_map}, "
                f"unmapped={self._mapping.unmapped_retrieved_ids}."
            )
        if not any(
            isinstance(e, int)
            for p in self._translated_pairs
            for e in (p.from_entity, p.to_entity)
        ):
            # Every surviving pair relates robot entities to each other (an
            # aperture, a gripper-orientation term). Those carry no information
            # about the scene, so guidance would run all episode and steer
            # toward nothing task-relevant. Better to disable it and say why
            # than to produce a plausible-looking but vacuous run.
            logger.error(
                "[RolloutAbstractor] No distance pair references an object after "
                "equivalence mapping — only robot-internal pairs survived "
                f"({self._pair_labels}). Guidance would carry no scene "
                "information, so abstraction is disabled for this episode. "
                "Usual causes: the demo analysis found no objects (check its "
                "'objects' field), or its cached pairs name segment ids this "
                "scene does not have."
            )
            return False

        # Check once whether any pair references a gripper finger entity
        self._has_finger_pair = any(
            e in ("gripper_l", "gripper_r")
            for p in self._translated_pairs
            for e in (p.from_entity, p.to_entity)
        )

        # Check if any pair needs EE orientation or gripper state for distance
        # computation (orientation, gripper aperture, or finger-entity pairs).
        from policy_abstraction.vlm.types import DistanceType

        # Constraints always receive ``ee_R`` (the world-from-EE rotation
        # matrix) as a third lambda argument, so any constraint at all forces
        # ``ee_ori`` to be available each step.
        self._needs_ee_ori = (
            self._has_finger_pair
            or any(
                p.distance_type
                in (DistanceType.ORIENTATION, DistanceType.GRIPPER_APERTURE)
                for p in self._translated_pairs
            )
            or len(self._compiled_constraints) > 0
        )

        # Precompute the set of relevant seg IDs for object point cloud extraction
        self._relevant_seg_ids = {
            int(entity)
            for pair in self._translated_pairs
            for entity in (pair.from_entity, pair.to_entity)
            if isinstance(entity, int)
        }

        # Build reverse mapping (live → demo) so constraint lambdas
        # (which reference demo entity IDs) can resolve against the
        # live-scene entity_centroids_by_id dict.
        self._live_to_demo_ids: dict[int | str, list[int | str]] = {}
        for demo_id, live_id in self._mapping.id_map.items():
            self._live_to_demo_ids.setdefault(live_id, []).append(demo_id)

        # 4. Precompute reference distance trajectory ---------------------
        t0 = time.perf_counter()
        self._ref_traj = self._precompute_ref_traj(
            self._top_result, self._analysis.distance_pairs
        )
        logger.info(
            f"[RolloutAbstractor] Reference trajectory precomputed: "
            f"{self._ref_traj.shape} in {time.perf_counter() - t0:.2f}s"
        )

        # 5. Construct aligner --------------------------------------------
        # Build explicit subtraj→pair-column map from DistancePair.subtraj_idx.
        # Pairs with subtraj_idx=None are global (omitted from the map so
        # the aligner includes them in every sub-trajectory's mask).
        subtraj_pair_map: dict[int, list[int]] = {}
        for col_idx, pair in enumerate(self._analysis.distance_pairs):
            if pair.subtraj_idx is not None:
                subtraj_pair_map.setdefault(pair.subtraj_idx, []).append(col_idx)

        self._aligner = MonotonicTrajectoryAligner(
            self._ref_traj,
            self._pair_labels,
            window=self._mta_window,
            subtrajectories=self._subtrajectories if self._subtrajectories else None,
            subtraj_pair_map=subtraj_pair_map if subtraj_pair_map else None,
            max_advance=self._mta_max_advance,
            subtraj_time_weight=self._subtraj_time_weight,
            alignment_mode=self._alignment_mode,
            mixed_variation=self._mixed_variation,
            temporal_factor_temperature=self._temporal_factor_temperature,
            temporal_factor_floor=self._temporal_factor_floor,
            sg_switch_persistence=self._sg_switch_persistence,
        )

        # Note: stateful multi-view perception clients are reset at the *start*
        # of setup() — before the live-name resolution step that issues the
        # setup-time detect() call. The first per-step call therefore runs
        # track() against the anchors seeded during setup.

        self._ready = True
        return True

    def set_ref_std(
        self,
        std: np.ndarray,
        source_pair_labels: list[str],
    ) -> None:
        """Set per-pair uncertainty from multi-demo DTW-aligned statistics.

        The *std* array may have columns corresponding to a different (superset)
        set of pair labels than ``self._pair_labels`` (e.g. if some pairs were
        dropped during equivalence-mapping translation).  This method selects
        and reorders columns to match ``self._pair_labels``.

        Parameters
        ----------
        std:
            ``[T_ref_stats, n_source_pairs]`` float32 std-dev from
            :func:`~policy_abstraction.evaluation.retrieval.compute_multi_demo_stats`.
        source_pair_labels:
            Column labels of *std* (same order as ``analysis.distance_pairs``).
        """
        if self._ref_traj is None:
            logger.warning(
                "[RolloutAbstractor] set_ref_std called before ref_traj is ready."
            )
            return

        src_label_to_col = {lbl: i for i, lbl in enumerate(source_pair_labels)}
        T_ref = self._ref_traj.shape[0]
        T_std = std.shape[0]
        n_pairs = len(self._pair_labels)

        mapped = np.zeros((T_ref, n_pairs), dtype=np.float32)
        for i, lbl in enumerate(self._pair_labels):
            src_col = src_label_to_col.get(lbl)
            if src_col is not None:
                # Linearly resample if T_std != T_ref (shouldn't normally differ)
                col = std[:, src_col].astype(np.float32)
                if T_std == T_ref:
                    mapped[:, i] = col
                else:
                    mapped[:, i] = np.interp(
                        np.linspace(0, T_std - 1, T_ref),
                        np.arange(T_std),
                        col,
                    )

        self._ref_std = mapped
        logger.info(
            f"[RolloutAbstractor] ref_std set: shape {mapped.shape}, "
            f"mean uncertainty {np.nanmean(mapped):.4f}"
        )

    # ------------------------------------------------------------------
    # Dynamics-integrated guidance chunks
    # ------------------------------------------------------------------

    def compute_dynamics_ref_chunk(
        self,
        aligned_ts: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray:
        """Compute dynamics-model reference distances for a guidance chunk.

        For each pair, starts from the **achieved** robot state at
        ``aligned_ts[col_idx]``, integrates the reference actions forward for
        *chunk_size* steps using the dynamics model, then computes distances at
        each step using the integrated EE positions and per-frame entity
        centroids stored during precomputation.

        The returned array is shaped ``(chunk_size, n_pairs)`` and is meant to
        be passed as ``ref_traj`` to
        :func:`~policy_abstraction.evaluation.guidance_builder.build_distance_guidance_params`
        with ``aligned_ts`` set to all-zeros (the crop is already applied).

        Parameters
        ----------
        aligned_ts:
            Per-pair aligned reference timestep indices ``(n_pairs,)`` int32.
        chunk_size:
            Number of forward steps to integrate.

        Returns
        -------
        np.ndarray
            ``(chunk_size, n_pairs)`` float32.
        """
        n_pairs = len(self._pair_labels)
        result = np.full((chunk_size, n_pairs), np.nan, dtype=np.float32)

        # Fallback when precomputation data is not available
        if (
            self._ref_actions is None
            or self._ref_ee_pos is None
            or self._ref_ee_ori is None
            or self._ref_gripper is None
            or self._ref_entity_centroids is None
            or self._ref_traj is None
        ):
            if self._ref_traj is not None:
                for ci in range(n_pairs):
                    pt = max(0, min(int(aligned_ts[ci]), self._ref_traj.shape[0] - 1))
                    end = min(pt + chunk_size, self._ref_traj.shape[0])
                    sl = self._ref_traj[pt:end, ci]
                    result[: len(sl), ci] = sl
                    if len(sl) < chunk_size and len(sl) > 0:
                        result[len(sl) :, ci] = sl[-1]
            return result

        T_ref = self._ref_ee_pos.shape[0]
        pairs = self._analysis.distance_pairs

        # Identify surface-type pair columns.  The synthetic PCD built below
        # only contains entity centroids (one point each), so
        # MinSurfaceDistance degenerates to centroid distance — producing
        # expected values that are systematically larger than the true surface
        # distance the guidance function computes.  For these columns we fall
        # back to the precomputed ref_traj which was built from full point
        # clouds.
        surface_cols: set[int] = set()
        for ci in range(n_pairs):
            p = pairs[ci]
            if (
                p.distance_type == DistanceType.SURFACE
                and p.axis is None
                and p.plane is None
            ):
                surface_cols.add(ci)

        # Pre-fill surface columns from the precomputed ref_traj
        for ci in surface_cols:
            pt = max(0, min(int(aligned_ts[ci]), T_ref - 1))
            end = min(pt + chunk_size, T_ref)
            sl = self._ref_traj[pt:end, ci]
            result[: len(sl), ci] = sl
            if len(sl) < chunk_size and len(sl) > 0:
                result[len(sl) :, ci] = sl[-1]

        # Group columns by unique aligned_t to share dynamics integrations
        unique_ts: dict[int, list[int]] = {}
        for ci in range(n_pairs):
            if ci in surface_cols:
                continue  # already filled above
            t = max(0, min(int(aligned_ts[ci]), T_ref - 1))
            unique_ts.setdefault(t, []).append(ci)

        for start_t, col_indices in unique_ts.items():
            # Initial state from achieved reference
            initial_state = np.concatenate(
                [
                    self._ref_ee_pos[start_t],
                    self._ref_ee_ori[start_t],
                    self._ref_gripper[start_t],
                ]
            ).astype(np.float32)  # [7]

            # Slice reference actions for the chunk
            act_end = min(start_t + chunk_size, len(self._ref_actions))
            chunk_actions = self._ref_actions[start_t:act_end]

            if len(chunk_actions) == 0:
                # At trajectory end — fill with last achieved distance
                for ci in col_indices:
                    result[:, ci] = self._ref_traj[min(start_t, T_ref - 1), ci]
                continue

            states = _states_after_actions(
                self._dynamics_model, chunk_actions, initial_state, dt=1.0 / 50
            )
            ee_dyn = states[:, :3].astype(np.float32)  # [n_integrated, 3]
            ori_dyn = states[:, 3:6].astype(np.float32)  # [n_integrated, 3]

            # Gripper: both dynamics models leave channel 6 as the raw action
            # (absolute, never integrated).  Map it to the metric q1-q2 frame
            # so finger FK and aperture distances here match the JAX-side
            # guidance, which applies the same normalisation.
            grip_metric_dyn = gripper_action_to_metric(
                states[:, 6], self._gripper_geometry
            ).astype(np.float32)  # [n_integrated]
            q1_dyn = grip_metric_dyn / 2.0
            q2_dyn = -grip_metric_dyn / 2.0
            qpos_dyn = np.stack([q1_dyn, q2_dyn], axis=-1)  # [n_integrated, 2]
            finger_l_dyn, finger_r_dyn = compute_finger_tip_positions(
                ee_dyn, ori_dyn, qpos_dyn, self._gripper_geometry
            )  # each [n_integrated, 3]

            for k in range(chunk_size):
                ref_frame = start_t + k
                if ref_frame >= T_ref:
                    # Pad with last computed value
                    if k > 0:
                        for ci in col_indices:
                            result[k, ci] = result[k - 1, ci]
                    continue

                _k_idx = min(k, len(ee_dyn) - 1)
                ee_k = ee_dyn[_k_idx]

                # Build synthetic pcd from stored per-frame entity centroids.
                # Robot entities (ee, gripper_l, gripper_r) are resolved via
                # the integrated EE state + FK-derived finger tips below —
                # not from the pre-stored reference centroids — so that the
                # reference trajectory is geometrically symmetric with the
                # JAX live path.
                pcd_rows: list[np.ndarray] = []
                for eid, centroids in self._ref_entity_centroids.items():
                    if isinstance(eid, str) and eid in ROBOT_ENTITIES:
                        continue
                    c = centroids[ref_frame]
                    if np.any(np.isnan(c)):
                        continue
                    pcd_rows.append(
                        np.array(
                            [c[0], c[1], c[2], 0, 0, 0, float(eid)],
                            dtype=np.float32,
                        )
                    )

                if not pcd_rows:
                    continue

                finger_pos = {
                    "gripper_l": np.asarray(finger_l_dyn[_k_idx], dtype=np.float32),
                    "gripper_r": np.asarray(finger_r_dyn[_k_idx], dtype=np.float32),
                }

                synthetic_pcd = np.stack(pcd_rows)
                relevant_pairs = [pairs[ci] for ci in col_indices]
                dists = compute_distances(
                    relevant_pairs,
                    self._feature,
                    synthetic_pcd,
                    ee_k,
                    finger_positions=finger_pos,
                    ee_ori=ori_dyn[_k_idx],
                    gripper_state=float(grip_metric_dyn[_k_idx]),
                )
                for ci in col_indices:
                    lbl = self._pair_labels[ci]
                    if lbl in dists:
                        result[k, ci] = dists[lbl]

        # NaN fill: leading NaNs from achieved ref_traj, then carry-forward
        for ci in range(n_pairs):
            col = result[:, ci]
            if np.isnan(col[0]):
                pt = max(0, min(int(aligned_ts[ci]), T_ref - 1))
                col[0] = self._ref_traj[pt, ci]
            for k in range(1, chunk_size):
                if np.isnan(col[k]):
                    col[k] = col[k - 1]

        return result

    def crop_ref_std(
        self,
        aligned_ts: np.ndarray,
        chunk_size: int,
    ) -> np.ndarray | None:
        """Crop ``ref_std`` to a ``(chunk_size, n_pairs)`` window matching
        :meth:`compute_dynamics_ref_chunk` so both can be passed to the guidance
        builder with ``aligned_ts=0``.

        Returns ``None`` when ``ref_std`` is not set.
        """
        if self._ref_std is None:
            return None
        n_pairs = self._ref_std.shape[1]
        T_std = self._ref_std.shape[0]
        out = np.zeros((chunk_size, n_pairs), dtype=np.float32)
        for ci in range(n_pairs):
            pt = max(0, min(int(aligned_ts[ci]), T_std - 1))
            end = min(pt + chunk_size, T_std)
            sl = self._ref_std[pt:end, ci]
            out[: len(sl), ci] = sl
            if len(sl) < chunk_size:
                out[len(sl) :, ci] = sl[-1] if len(sl) > 0 else 0.0
        return out

    # ------------------------------------------------------------------
    # Per-step operation
    # ------------------------------------------------------------------

    def step(
        self,
        obs: dict,
        camera_params: dict[str, dict],
    ) -> AbstractionResult | None:
        """Compute live distances and run MTA at one replanning boundary.

        Parameters
        ----------
        obs:
            Current live observation dict.
        camera_params:
            ``{cam_name: {"intrinsic": K, "extrinsic": E}}`` from
            :class:`~policy_abstraction.visualization.chunk_projector.ChunkProjector`.

        Returns
        -------
        AbstractionResult or ``None``
            ``None`` when the abstractor is not ready, required observation
            keys are absent, or the computed point cloud is empty.
        """
        if not self._ready or self._aligner is None:
            return None

        # ---- Build point cloud (mode-dependent) -------------------------
        _t_perception = time.perf_counter()
        pcd = self._build_live_pcd(obs, camera_params)
        perception_latency = time.perf_counter() - _t_perception
        if pcd is None or pcd.shape[0] == 0:
            if not self._depth_warn_logged:
                logger.warning(
                    f"[RolloutAbstractor] Could not build live point cloud "
                    f"(mode={self._perception_mode.value}); skipping step."
                )
                self._depth_warn_logged = True
            return None

        # ---- Compute live distances -------------------------------------
        robot = self._env_spec.robot
        ee_pos = np.asarray(obs[robot.eef_pos_key], dtype=np.float32)

        # Compute EE orientation and gripper state when needed by any pair type
        finger_positions: dict[str, np.ndarray] | None = None
        ee_ori: np.ndarray | None = None
        gripper_qpos: np.ndarray | None = None
        _gripper_scalar: float | None = None
        if self._needs_ee_ori:
            ee_ori = quat2axisangle(
                np.asarray(obs[robot.eef_quat_key], dtype=np.float64)
            )
            gripper_qpos = np.asarray(obs[robot.gripper_qpos_key], dtype=np.float64)
            # The aperture scalar is compared against the demo's, which is in
            # metres, so the observed value has to leave the robot's own units
            # first: identity for the Panda (qpos already metric), a stroke
            # scale for the Robotiq (normalised openness).
            _gripper_scalar = float(
                gripper_qpos_to_metric(
                    robot.gripper.aperture_from_qpos(gripper_qpos),
                    self._gripper_geometry,
                )
            )
            if self._has_finger_pair:
                finger_positions = compute_finger_tip_positions_single(
                    ee_pos.astype(np.float64),
                    ee_ori,
                    gripper_qpos,
                    self._gripper_geometry,
                )

        # World-from-EE rotation matrix passed to constraint lambdas as the
        # third positional argument (so they can reference the gripper z-axis
        # via ``ee_R[:, 2]``).  Identity when no orientation is available.
        ee_R = (
            Rotation.from_rotvec(ee_ori).as_matrix()
            if ee_ori is not None
            else np.eye(3)
        )

        live_distances = compute_distances(
            self._translated_pairs,
            self._feature,
            pcd,
            ee_pos,
            finger_positions,
            ee_ori=ee_ori,
            gripper_state=_gripper_scalar,
        )

        if not live_distances:
            logger.warning(
                "[RolloutAbstractor] All distance pairs returned None; skipping step."
            )
            return None

        # ---- Extract per-object point clouds for 3-D visualization ----------
        object_pointclouds: dict[str, np.ndarray] = {}
        object_centroids: dict[str, np.ndarray] = {}
        entity_centroids_by_id: dict[int | str, np.ndarray] = {"ee": ee_pos}
        if finger_positions is not None:
            entity_centroids_by_id.update(finger_positions)
        for seg_id in self._relevant_seg_ids:
            obj_pts = pcd[pcd[:, 6] == seg_id]
            if obj_pts.shape[0] == 0:
                continue
            obj_name = self._int_name_map.get(seg_id, str(seg_id))
            object_pointclouds[obj_name] = obj_pts
            centroid = obj_pts[:, :3].mean(axis=0)
            object_centroids[obj_name] = centroid
            entity_centroids_by_id[seg_id] = centroid

        # Full-scene fused cloud (all pixels) for the ``scene/full_pcd`` overlay.
        # Built only when enabled — it back-projects the entire depth image, so
        # it is heavier than the relevant-object clouds above.
        scene_pointcloud = (
            self._build_scene_pcd(obs, camera_params)
            if self._record_scene_pointcloud
            else None
        )

        # Add demo-ID aliases so constraint lambdas (which use demo entity
        # IDs like keypoints[1]) can resolve against this dict.
        for live_id, centroid_val in list(entity_centroids_by_id.items()):
            for demo_id in self._live_to_demo_ids.get(live_id, []):
                if demo_id not in entity_centroids_by_id:
                    entity_centroids_by_id[demo_id] = centroid_val

        # ---- Sub-goal satisfaction for current & previous sub-trajectories ---
        # Evaluated before alignment so the aligner can use it to gate subtraj
        # switches (backward allowed only when previous sub-goal not satisfied).
        subtraj_sg_map: dict[int, bool] = {}
        cur_subtraj_for_sg = self._aligner.current_subtraj_idx
        if cur_subtraj_for_sg is not None:
            for sg_idx in (cur_subtraj_for_sg, cur_subtraj_for_sg - 1):
                if sg_idx < 0 or sg_idx not in self._subtraj_sg_fns:
                    continue
                fns = self._subtraj_sg_fns[sg_idx]
                try:
                    subtraj_sg_map[sg_idx] = all(
                        float(fn(ee_pos, entity_centroids_by_id, ee_R)) <= 0.0
                        for fn in fns
                    )
                except Exception as exc:
                    logger.debug(
                        f"[RolloutAbstractor] Sub-goal satisfaction check failed "
                        f"for subtraj {sg_idx}: {exc}"
                    )

        # ---- MTA alignment (with sub-goal hint) -----------------------------
        live_vec = np.array(
            [live_distances.get(lbl, np.nan) for lbl in self._pair_labels],
            dtype=np.float32,
        )
        aligned_t, aligned_ts, guidance_weights = self._aligner.align_all(
            live_vec, subtraj_sg_satisfied=subtraj_sg_map
        )
        ref_distances = self._aligner.ref_distances_at(aligned_t)
        mta_progress = self._aligner.progress()

        # ---- Distance errors --------------------------------------------
        distance_errors: dict[str, float] = {
            k: float(live_distances[k]) - float(ref_distances[k])
            for k in live_distances
            if k in ref_distances and not np.isnan(ref_distances[k])
        }

        current_subtraj_idx = self._aligner.current_subtraj_idx

        # ---- Evaluate all constraints --------------------------------------
        constraint_costs: dict[str, float] = {}
        for kind, label, _subtraj_idx, fn in self._compiled_constraints:
            prefix = "sg" if kind == "sub_goal" else "pc"
            try:
                cost = float(fn(ee_pos, entity_centroids_by_id, ee_R))
                constraint_costs[f"{prefix}/{label}"] = cost
            except Exception as exc:
                logger.debug(
                    f"[RolloutAbstractor] Constraint '{label}' eval failed: {exc}"
                )

        # ---- High-level relevance per object (robot-entity-involving pairs) --
        hl_relevances = compute_hl_relevances(
            self._translated_pairs,
            ee_pos,
            entity_centroids_by_id,
            self._int_name_map,
            near_threshold=self._relevance_near_threshold,
            far_threshold=self._relevance_far_threshold,
        )

        return AbstractionResult(
            live_distances=live_distances,
            aligned_t=aligned_t,
            ref_distances=ref_distances,
            distance_errors=distance_errors,
            mta_progress=mta_progress,
            pair_labels=self._pair_labels,
            subtrajectories=self._subtrajectories,
            current_subtraj_idx=current_subtraj_idx,
            object_pointclouds=object_pointclouds if object_pointclouds else None,
            object_centroids=object_centroids if object_centroids else None,
            scene_pointcloud=scene_pointcloud,
            pairs=self._translated_pairs,
            entity_centroids=entity_centroids_by_id,
            alignment_mse=self._aligner.current_mse,
            constraint_costs=constraint_costs,
            aligned_ts=aligned_ts,
            guidance_weights=guidance_weights,
            relevances=hl_relevances,
            perception_overlays=self._perception_overlays,
            perception_history_grids=self._perception_history_grids,
            perception_debug_images=self._perception_debug_images,
            perception_timings_md=self._perception_timings_md,
            perception_latency=perception_latency,
        )

    # ------------------------------------------------------------------
    # Image annotation
    # ------------------------------------------------------------------

    def draw_annotated_live_image(
        self,
        rgb: np.ndarray,
        seg: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
        ee_pos: np.ndarray,
        result,  # AbstractionResult
    ) -> np.ndarray | None:
        if result.pairs is None:
            return None
        img = rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb.copy()
        img = np.ascontiguousarray(img)
        H, W = img.shape[:2]
        K3 = K[:3, :3] if K.shape == (4, 4) else K
        seg_arr = np.squeeze(np.asarray(seg, dtype=np.int32))
        draw_pairs_on_image(
            img,
            result.pairs,
            seg_arr,
            K3,
            E,
            ee_pos,
            H,
            W,
            result.distance_errors,
            result.live_distances,
            result.entity_centroids,
            current_subtraj_idx=result.current_subtraj_idx,
            # `rgb` is the live primary frame, already through that camera's
            # own transform.
            preprocessed=self._primary_camera.pixels_mirrored,
        )
        return img

    @staticmethod
    def _nearest_depth_frame(episode, t: int) -> int:
        """Timestep at or nearest *t* that the episode can reconstruct.

        A simulator renders depth every frame and this is the identity. A real
        rollout stores it on a stride, so the aligned reference step usually
        falls between two depth frames.
        """
        frames = episode.frames_with_depth()
        if not frames:
            return t
        return min(frames, key=lambda f: abs(f - t))

    def _load_hdf5_wrist_frame(self, aligned_t: int):
        """Auxiliary wrist view for a LIBERO-schema demo.

        Kept on the HDF5 path only: it probes a handful of naming conventions
        that exist because the simulator schemas disagree with each other, and
        no real recording carries wrist depth, masks or intrinsics for it to
        find.
        """
        wrist_rgb = wrist_dep = wrist_seg = None
        wrist_K = wrist_E = None
        with h5py.File(self._top_result["source_path"], "r") as f:
            obs: Any = f["data"][self._top_result["demo_key"]]["obs"]  # type: ignore[index]
            for wrist_prefix in ("eye_in_hand", "robot0_eye_in_hand", "wrist"):
                rgb_key = f"{wrist_prefix}_rgb"
                if rgb_key not in obs:
                    # Try _image suffix (live env naming)
                    rgb_key = f"{wrist_prefix}_image"
                if rgb_key in obs:
                    wrist_rgb = obs[rgb_key][aligned_t][::-1, ::-1].copy()
                    dep_key = f"{wrist_prefix}_depth"
                    if dep_key in obs:
                        wrist_dep = obs[dep_key][aligned_t][::-1, ::-1].copy()
                    seg_key = f"{wrist_prefix}_segmentation"
                    if seg_key in obs:
                        wrist_seg = obs[seg_key][aligned_t][::-1, ::-1].copy()
                    break
            for wrist_K_key, wrist_E_key in (
                ("eye_in_hand_intrinsic", "eye_in_hand_extrinsic"),
                ("robot0_eye_in_hand_intrinsic", "robot0_eye_in_hand_extrinsic"),
            ):
                if wrist_K_key in obs and wrist_E_key in obs:
                    wrist_K = obs[wrist_K_key][aligned_t].copy()
                    wrist_E = obs[wrist_E_key][aligned_t].copy()
                    break
        return wrist_rgb, wrist_dep, wrist_seg, wrist_K, wrist_E

    @property
    def _ref_frames_preprocessed(self) -> bool:
        """Whether the retrieved demo's frames are stored 180°-rotated.

        The reference frames come from a recorded dataset, so the answer is a
        property of that dataset rather than of the live camera — an HDF5 demo
        recorded in simulation is pre-flipped; one recorded on the arm is not.
        Every projection onto a reference frame must consult this, or the
        overlay lands mirrored on half the sources.
        """
        return bool(
            getattr(self._demo_episode, "frames_preprocessed", True)
            if self._demo_episode is not None
            else True
        )

    def draw_annotated_ref_image(
        self,
        aligned_t: int,
        distance_errors: dict[str, float],
        current_subtraj_idx: int | None = None,
        chunk_length: int | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (agentview_annotated, wrist_annotated) ref frames at *aligned_t*.

        Loads RGB, segmentation, and wrist image from the retrieved HDF5 demo.
        Results are cached per ``aligned_t`` to avoid repeated HDF5 reads.

        When *chunk_length* is given, the next *chunk_length* steps of the
        reference EE trajectory are projected onto the wrist image in the same
        style as the live action-chunk overlay.
        """
        if self._ref_frame_cache is not None and self._ref_frame_cache[0] == aligned_t:
            (
                _,
                rgb,
                dep,
                seg,
                K,
                E,
                ee_pos,
                wrist_rgb,
                wrist_dep,
                wrist_seg,
                wrist_K,
                wrist_E,
            ) = self._ref_frame_cache
        else:
            try:
                episode = self._demo_episode
                if episode is None:
                    episode = build_demo_episode(self._top_result)
                    self._demo_episode = episode

                # Depth may be strided (a real recording), so snap to the
                # nearest reconstructable frame and take RGB from there too:
                # a colour frame paired with another step's depth would draw
                # the pair lines against the wrong geometry.
                frame_t = self._nearest_depth_frame(episode, aligned_t)
                rgb = np.ascontiguousarray(episode.rgb(frame_t))
                dep = episode.depth_m(frame_t)
                seg = episode.segmentation(frame_t)
                K = episode.intrinsic(frame_t)
                E = episode.extrinsic(frame_t)
                ee_all = episode.ee_positions()
                ee_pos = np.asarray(ee_all[min(aligned_t, len(ee_all) - 1)])

                # The auxiliary (wrist) view is only carried by the simulator
                # schemas; a real recording publishes no wrist depth, mask or
                # intrinsic, so there is nothing to reconstruct from it.
                wrist_rgb = wrist_dep = wrist_seg = None
                wrist_K = wrist_E = None
                if episode.source_kind in ("libero", "robocasa_hdf5"):
                    (
                        wrist_rgb,
                        wrist_dep,
                        wrist_seg,
                        wrist_K,
                        wrist_E,
                    ) = self._load_hdf5_wrist_frame(aligned_t)
            except Exception as exc:
                logger.debug(
                    f"[RolloutAbstractor] Could not load ref frame at t={aligned_t}: {exc}"
                )
                return None, None
            self._ref_frame_cache = (
                aligned_t,
                rgb,
                dep,
                seg,
                K,
                E,
                ee_pos,
                wrist_rgb,
                wrist_dep,
                wrist_seg,
                wrist_K,
                wrist_E,
            )

        img = rgb.astype(np.uint8) if rgb.dtype != np.uint8 else rgb.copy()
        img = np.ascontiguousarray(img)
        H, W = img.shape[:2]
        # Depth, masks and the camera matrices can all be absent for a real
        # recording (no segmenter, no calibration). The colour frame is still
        # worth showing, so this degrades to an un-annotated reference frame
        # rather than returning nothing.
        seg_arr = (
            np.squeeze(np.asarray(seg, dtype=np.int32))
            if seg is not None
            else np.zeros((H, W), dtype=np.int32)
        )
        K3: np.ndarray | None = None
        if K is not None:
            K_arr = np.asarray(K)
            K3 = K_arr[:3, :3] if K_arr.shape == (4, 4) else K_arr
        E_arr = None if E is None else np.asarray(E)
        preprocessed = self._ref_frames_preprocessed

        # Build pcd from the reference frame for 3-D geometry overlays
        ref_entity_3d: dict[int | str, np.ndarray] | None = None
        try:
            if dep is None or seg is None or K3 is None or E_arr is None:
                raise ValueError("reference frame carries no depth/mask/calibration")
            # The episode adapter already published depth in metres.
            ref_pcd = self._build_pcd(
                rgb, dep, seg, K3, E_arr, preprocessed=preprocessed
            )
            # Merge wrist camera PCD if available
            if (
                wrist_rgb is not None
                and wrist_dep is not None
                and wrist_seg is not None
                and wrist_K is not None
                and wrist_E is not None
            ):
                w_dep_metric = wrist_dep.copy()
                if w_dep_metric.max() <= 1.0:
                    _wcam = self._wrist_camera or self._primary_camera
                    w_dep_metric = mujoco_depth_to_metric(
                        w_dep_metric,
                        near=_wcam.depth_near,
                        far=_wcam.depth_far,
                    )
                wK3 = wrist_K[:3, :3] if wrist_K.shape == (4, 4) else wrist_K
                wrist_pcd = self._build_pcd(
                    wrist_rgb,
                    w_dep_metric,
                    wrist_seg,
                    wK3,
                    wrist_E,
                    preprocessed=preprocessed,
                )
                if wrist_pcd.shape[0] > 0 and ref_pcd.shape[0] > 0:
                    ref_pcd = np.concatenate([ref_pcd, wrist_pcd], axis=0)
                elif wrist_pcd.shape[0] > 0:
                    ref_pcd = wrist_pcd
            if ref_pcd.shape[0] > 0:
                # Get finger positions at aligned_t from precomputed ref data
                ref_finger_pos: dict[str, np.ndarray] | None = None
                if self._has_finger_pair and self._ref_entity_centroids is not None:
                    fp: dict[str, np.ndarray] = {}
                    for k in ("gripper_l", "gripper_r"):
                        if k in self._ref_entity_centroids:
                            fp[k] = self._ref_entity_centroids[k][aligned_t]
                    if fp:
                        ref_finger_pos = fp
                ref_entity_3d = compute_entity_centroids_3d(
                    self._analysis.distance_pairs, ref_pcd, ee_pos, ref_finger_pos
                )
        except Exception as exc:
            logger.debug(
                f"[RolloutAbstractor] ref entity_3d computation failed at t={aligned_t}: {exc}"
            )

        draw_subtraj_overlay(
            img,
            self._subtrajectories,
            self._ref_ee_2d,
            self._ref_ee_valid,
            current_subtraj_idx,
        )
        if K3 is not None and E_arr is not None:
            # Without a camera there is nothing to project the pairs onto; the
            # colour frame and the sub-trajectory overlay above still stand.
            draw_pairs_on_image(
                img,
                self._analysis.distance_pairs,
                seg_arr,
                K3,
                E_arr,
                ee_pos,
                H,
                W,
                distance_errors,
                entity_3d=ref_entity_3d,
                current_subtraj_idx=current_subtraj_idx,
                # `img` is a reference frame out of the retrieved demo, so it
                # follows that dataset's convention, not the live camera's.
                preprocessed=preprocessed,
            )

        wrist_out: np.ndarray | None = None
        if wrist_rgb is not None:
            wrist_out = np.ascontiguousarray(wrist_rgb.astype(np.uint8))
            # Project the next chunk_length reference EE positions onto the wrist image.
            if (
                chunk_length is not None
                and self._ref_ee_pos is not None
                and wrist_K is not None
                and wrist_E is not None
            ):
                try:
                    wrist_out = self._draw_ref_traj_on_wrist(
                        wrist_out, aligned_t, chunk_length, wrist_K, wrist_E
                    )
                except Exception as exc:
                    logger.debug(
                        f"[RolloutAbstractor] ref wrist traj projection failed: {exc}"
                    )

        return img, wrist_out

    def _draw_ref_traj_on_wrist(
        self,
        img: np.ndarray,
        aligned_t: int,
        chunk_length: int,
        wrist_K: np.ndarray,
        wrist_E: np.ndarray,
    ) -> np.ndarray:
        """Project the next *chunk_length* ref EE positions onto the wrist image.

        Uses the same line-segment style as :class:`ChunkProjector` with a
        fixed white colour (no gripper-state info available for the reference).
        """
        import cv2

        assert self._ref_ee_pos is not None
        T_ref = self._ref_ee_pos.shape[0]
        t_end = min(aligned_t + chunk_length, T_ref)
        future_ee = self._ref_ee_pos[aligned_t:t_end].astype(np.float64)  # [n, 3]
        if future_ee.shape[0] < 2:
            return img

        H, W = img.shape[:2]
        wrist_K3 = wrist_K[:3, :3] if wrist_K.shape == (4, 4) else wrist_K
        pts_2d, valid = project_points_to_pixels(
            future_ee,
            wrist_K3,
            wrist_E,
            H,
            W,
            preprocessed=self._ref_frames_preprocessed,
        )
        in_bounds = (
            valid
            & (pts_2d[:, 0] >= 0)
            & (pts_2d[:, 0] < W)
            & (pts_2d[:, 1] >= 0)
            & (pts_2d[:, 1] < H)
        )

        out = img.copy()
        overlay = out.copy()
        color = (255, 255, 255)  # white trajectory for reference
        n = len(pts_2d)
        for i in range(n - 1):
            if not (in_bounds[i] and in_bounds[i + 1]):
                continue
            cv2.line(
                overlay,
                tuple(pts_2d[i]),
                tuple(pts_2d[i + 1]),
                color,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        # End-of-chunk marker
        visible = np.where(in_bounds)[0]
        if len(visible) > 0:
            lx, ly = int(pts_2d[visible[-1], 0]), int(pts_2d[visible[-1], 1])
            sq = 4
            cv2.rectangle(
                overlay, (lx - sq, ly - sq), (lx + sq, ly + sq), color, thickness=-1
            )

        cv2.addWeighted(overlay, 0.65, out, 0.35, 0, out)
        return out

    # ------------------------------------------------------------------
    # Shared point-cloud construction
    # ------------------------------------------------------------------

    def _build_pcd(
        self,
        rgb: np.ndarray,
        dep: np.ndarray,
        seg: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
        *,
        preprocessed: bool = True,
    ) -> np.ndarray:
        """Build a segmented point cloud with consistent erosion applied.

        *preprocessed* undoes the u-axis mirror introduced by the 180°-rotation
        transform, so it must track the camera's own ``image_transform``: the
        simulators pre-flip their frames, the real UR5e does not. Getting it
        wrong mirrors every object cloud about the image centre-line while
        leaving the scene superficially plausible.

        The default is ``True`` because the *reference demo* is always a
        pre-flipped LIBERO frame no matter what the live robot is. Live call
        sites hold a :class:`CameraSpec` and must pass its own value.
        """
        if self._erode_px > 0:
            seg = erode_segmentation(seg, self._erode_px)
        return build_segmented_pointcloud(
            rgb, dep, seg, K, E, preprocessed=preprocessed
        )

    def _build_scene_pcd(
        self,
        obs: dict,
        camera_params: dict[str, dict],
    ) -> np.ndarray | None:
        """Build the full-scene ``[M, 7]`` fused point cloud for this frame.

        Back-projects *every* valid-depth pixel across all spec cameras —
        segmentation-agnostic (``seg_id`` column is 0 everywhere; the cloud is
        coloured by RGB), so it captures the whole scene (table, background,
        every object) rather than only the relevant objects in
        :attr:`object_pointclouds`. Stride-subsampled to
        :attr:`_scene_pcd_max_points` to bound recording size. Returns an empty
        ``[0, 7]`` array (never ``None``) when the build yields nothing so the
        recorder keeps one entry per replan and the per-replan index stays
        aligned on replay.
        """
        empty = np.zeros((0, 7), dtype=np.float32)
        try:
            views = self._iter_camera_views(obs, camera_params)
        except Exception as exc:
            logger.warning(
                f"[RolloutAbstractor] scene-pcd view iteration failed: {exc}"
            )
            return empty
        parts: list[np.ndarray] = []
        for cam, rgb, dep, _seg, K3, E in views:
            seg_zero = np.zeros(dep.shape[:2], dtype=np.int32)
            try:
                pcd = build_segmented_pointcloud(
                    rgb,
                    dep,
                    seg_zero,
                    K3,
                    E,
                    preprocessed=cam.pixels_mirrored,
                )
            except Exception as exc:
                logger.warning(
                    f"[RolloutAbstractor] scene-pcd build failed for "
                    f"camera {cam.name!r}: {exc}"
                )
                continue
            if pcd.shape[0] > 0:
                parts.append(pcd)
        if not parts:
            return empty
        scene = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        cap = self._scene_pcd_max_points
        if cap and scene.shape[0] > cap:
            stride = int(np.ceil(scene.shape[0] / cap))
            scene = scene[::stride]
        return scene

    # ------------------------------------------------------------------
    # Episode setup: VLM-name → live entity ID resolution
    # ------------------------------------------------------------------

    def _resolve_live_objects(
        self,
        relevant_names: list[str],
        obs: dict,
        camera_params: dict[str, dict] | None,
    ) -> tuple[dict[int, str], np.ndarray | None]:
        """Resolve VLM-inferred names to ``(live_objects, live_seg_image)``.

        Branches on :attr:`_perception_mode`:

        * **REAL_PERCEPTION** — runs the multi-view perception
          fn once to seed the tracker, assigns synthetic integer IDs per
          name, and renders an annotated overlay from the segmenter.
        * **GT modes** — calls ``gt_name_lookup()`` (constructor closure)
          to obtain the simulator's full ``{seg_id: name}`` map, then
          intersects with *relevant_names* (case-insensitive). Renders
          a relevant-only overlay from the GT seg image.
        """
        if self._perception_mode == PerceptionMode.REAL_PERCEPTION:
            return self._resolve_live_objects_predicted(
                relevant_names, obs, camera_params or {}
            )
        return self._resolve_live_objects_gt(relevant_names, obs)

    def _iter_camera_views(
        self,
        obs: dict,
        camera_params: dict[str, dict],
        *,
        require_segmentation: bool = False,
    ) -> list[
        tuple[
            CameraSpec,
            np.ndarray,
            np.ndarray,
            np.ndarray | None,
            np.ndarray,
            np.ndarray,
        ]
    ]:
        """Yield ``(camera, rgb, depth, seg, K, E)`` for every spec camera with full data.

        Iteration order is **primary first, then the rest** — matches the
        original LIBERO ordering where view 0 was always agentview. RGB and
        depth are passed through the camera's ``image_transform`` (e.g.
        ``flip_vh``), and depth is converted to metres using the camera's
        ``depth_near`` / ``depth_far``. Cameras missing any required key in
        *obs* or *camera_params* are silently skipped.
        """
        ordered = [self._primary_camera] + [
            c for c in self._env_spec.cameras if not c.is_primary
        ]
        out: list[
            tuple[
                CameraSpec,
                np.ndarray,
                np.ndarray,
                np.ndarray | None,
                np.ndarray,
                np.ndarray,
            ]
        ] = []
        for cam in ordered:
            if cam.depth_key is None or cam.depth_key not in obs:
                continue
            if cam.image_key not in obs:
                continue
            if cam.name not in camera_params:
                continue
            seg_arr: np.ndarray | None = None
            if require_segmentation:
                if cam.segmentation_key is None or cam.segmentation_key not in obs:
                    continue
                seg_arr = np.ascontiguousarray(
                    cam.apply_image_transform(np.asarray(obs[cam.segmentation_key]))
                )
            cam_info = camera_params[cam.name]
            K = cam_info.get("intrinsic")
            E = cam_info.get("extrinsic")
            if K is None or E is None:
                continue
            K = np.asarray(K)
            E = np.asarray(E)
            rgb = np.ascontiguousarray(
                cam.apply_image_transform(np.asarray(obs[cam.image_key]))
            )
            dep = np.ascontiguousarray(
                cam.apply_image_transform(np.asarray(obs[cam.depth_key]))
            )
            # Simulators emit a normalised z-buffer; real sensors emit metres.
            # The range sniff below is kept for specs that predate
            # `depth_is_metric`, but a camera that declares metric depth is
            # never converted — a close-up real scene can legitimately be
            # entirely under 1 m.
            if not cam.depth_is_metric and dep.max() <= 1.0:
                dep = mujoco_depth_to_metric(
                    dep, near=cam.depth_near, far=cam.depth_far
                )
            K3 = K[:3, :3] if K.shape == (4, 4) else K
            out.append((cam, rgb, dep, seg_arr, K3, E))
        return out

    def _resolve_live_objects_predicted(
        self,
        relevant_names: list[str],
        obs: dict,
        camera_params: dict[str, dict],
    ) -> tuple[dict[int, str], np.ndarray | None]:
        """Setup-time multi-view detect → ``{instance_id: name}`` + overlay.

        Uses the perception tracker's ``instance_id`` directly as the live
        seg ID — multi-instance friendly (e.g. four bowls share a query but
        each gets its own ID) and consistent with the IDs shown in the
        ``fuse_views`` / ``render_overlays`` debug images.
        """
        if self._multi_view_segmentation_fn is None:
            logger.warning(
                "[RolloutAbstractor] PREDICTED mode without multi_view_segmentation_fn; "
                "name → ID resolution unavailable."
            )
            return {}, None

        from open_perception import CameraInfo  # lazy: sys.path-mounted

        views = self._iter_camera_views(obs, camera_params)
        rgbs: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        cameras: list[Any] = []
        for _cam, rgb, dep, _seg, K3, E in views:
            rgbs.append(rgb)
            depths.append(dep)
            cameras.append(CameraInfo(intrinsics=K3, extrinsics=E))

        if not rgbs:
            logger.warning(
                "[RolloutAbstractor] No camera views available for PREDICTED resolution."
            )
            return {}, None

        masks_per_view = self._multi_view_segmentation_fn(
            rgbs, depths, cameras, relevant_names
        )

        # Build live_objects from the primary view's per-instance detections.
        # Each detected physical object gets one entry, keyed by the tracker's
        # ``instance_id``. Multiple entries can share a name when several
        # instances of the same query were detected.
        primary_view_masks: dict[int, tuple[str, np.ndarray]] = (
            masks_per_view[0] if masks_per_view else {}
        )
        live_objects: dict[int, str] = {
            inst_id: name for inst_id, (name, _mask) in primary_view_masks.items()
        }

        detected_names = {name for name, _ in primary_view_masks.values()}
        missing = [n for n in relevant_names if n not in detected_names]
        if missing:
            logger.warning(
                f"[RolloutAbstractor] PREDICTED setup: perception did not detect "
                f"{missing} on the seed frame; they will be unavailable for "
                f"distance computation this episode."
            )

        # Render the agentview overlay in the same style as the retrieved
        # scene's `seg_index_image` (masks blended onto the RGB scene with the
        # integer ID drawn at each region's centroid, no name labels). The IDs
        # are the same ``instance_id``s we expose in `live_objects`, so the
        # VLM sees consistent IDs in the prompt text and the segmentation
        # overlay image.
        live_seg_image: np.ndarray | None = None
        primary_rgb = rgbs[0]
        synth_seg = np.zeros(primary_rgb.shape[:2], dtype=np.int32)
        for inst_id, (_name, mask) in primary_view_masks.items():
            if mask is None or not bool(mask.any()):
                continue
            synth_seg[np.asarray(mask).astype(bool)] = inst_id
        try:
            seg_bgr = draw_all_segment_ids(synth_seg, primary_rgb)
            live_seg_image = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            logger.warning(f"[RolloutAbstractor] live overlay render failed: {exc}")

        return live_objects, live_seg_image

    def _resolve_live_objects_gt(
        self,
        relevant_names: list[str],
        obs: dict,
    ) -> tuple[dict[int, str], np.ndarray | None]:
        """GT lookup: intersect VLM names with the simulator's name map."""
        if self._gt_name_lookup is None:
            logger.warning(
                "[RolloutAbstractor] GT mode without gt_name_lookup; "
                "cannot resolve VLM-inferred names to seg IDs."
            )
            return {}, None

        try:
            full_map = self._gt_name_lookup() or {}
        except Exception as exc:
            logger.warning(f"[RolloutAbstractor] gt_name_lookup failed: {exc}")
            return {}, None

        wanted = {_normalize_object_name(n) for n in relevant_names if n.strip()}
        wanted.discard("")
        live_objects: dict[int, str] = {}
        for sid_str, name in full_map.items():
            try:
                sid = int(sid_str)
            except (TypeError, ValueError):
                continue
            if _normalize_object_name(name) in wanted:
                live_objects[sid] = name

        live_seg_image = _render_relevant_seg_image(
            obs, set(live_objects.keys()), self._primary_camera
        )
        return live_objects, live_seg_image

    def _compute_live_centroids(
        self,
        live_ids: list[int],
        obs: dict,
        camera_params: dict[str, dict] | None,
    ) -> tuple[
        dict[int, tuple[int, int]] | None,
        dict[int, np.ndarray] | None,
    ]:
        """Compute live-scene 2D and 3D centroids for the equivalence prompt.

        3D centroids come from the freshly built segmented point cloud
        (mode-agnostic). 2D centroids are obtained by projecting the 3D
        centroids onto the primary camera's image plane, so both modes
        produce centroids in the same coordinate space without needing to
        plumb raw seg masks out of ``_resolve_live_objects``.

        Returns ``(None, None)`` if the pcd or primary-camera parameters
        are unavailable; callers should pass the result straight to
        :meth:`ObjectEquivalenceMapper.map_objects`, which tolerates
        ``None`` for either dict.
        """
        if not live_ids:
            return None, None
        params = camera_params or {}
        # Every live object is a candidate for the equivalence prompt, and the
        # pair-derived object set does not exist yet at this point of setup.
        pcd = self._build_live_pcd(
            obs, params, relevant_seg_ids=[int(sid) for sid in live_ids]
        )
        if pcd is None or pcd.size == 0:
            return None, None

        centroids_3d: dict[int, np.ndarray] = {}
        for sid in live_ids:
            pts = pcd[pcd[:, 6] == int(sid), :3]
            if pts.shape[0] == 0:
                continue
            centroids_3d[int(sid)] = pts.mean(axis=0).astype(np.float64)
        if not centroids_3d:
            return None, None

        centroids_2d: dict[int, tuple[int, int]] | None = None
        primary = self._primary_camera
        cam_info = params.get(primary.name)
        rgb = obs.get(primary.image_key) if primary.image_key else None
        if cam_info is not None and rgb is not None:
            K = cam_info.get("intrinsic")
            E = cam_info.get("extrinsic")
            if K is not None and E is not None:
                K_arr = np.asarray(K)
                K3 = K_arr[:3, :3] if K_arr.shape == (4, 4) else K_arr
                E_arr = np.asarray(E)
                rgb_t = primary.apply_image_transform(np.asarray(rgb))
                H, W = rgb_t.shape[:2]
                ids_ordered = list(centroids_3d.keys())
                pts_world = np.stack([centroids_3d[i] for i in ids_ordered], axis=0)
                pts_px, valid = project_points_to_pixels(
                    pts_world,
                    K3,
                    E_arr,
                    H,
                    W,
                    # This lands on `rgb_t`, so it follows that camera's own
                    # transform -- upright on the real cell, flipped in sim.
                    preprocessed=primary.pixels_mirrored,
                )
                centroids_2d = {}
                for sid, (u, v), ok in zip(ids_ordered, pts_px, valid, strict=False):
                    if not bool(ok):
                        continue
                    centroids_2d[int(sid)] = (int(u), int(v))
                if not centroids_2d:
                    centroids_2d = None

        return centroids_2d, centroids_3d

    # ------------------------------------------------------------------
    # Mode-dependent live point-cloud construction
    # ------------------------------------------------------------------

    def _build_live_pcd(
        self,
        obs: dict,
        camera_params: dict[str, dict],
        relevant_seg_ids: Iterable[int] | None = None,
    ) -> np.ndarray | None:
        """Build a ``[M, 7]`` segmented point cloud from a live observation.

        ``relevant_seg_ids`` overrides the pair-derived set of objects to
        segment (REAL_PERCEPTION only). :meth:`setup` needs it: it builds the
        equivalence prompt's live centroids before any pair has been
        translated, so the pair-derived set is still empty there.

        Dispatches to the appropriate strategy based on
        :attr:`_perception_mode`:

        * **GT_POINTCLOUD** — assembles per-object point clouds supplied by
          the simulator in ``obs["gt_object_pointclouds"]``.
        * **GT_SEG_RECONSTRUCT** — reconstructs from GT depth + GT
          segmentation (original behaviour).
        * **REAL_PERCEPTION** — runs ``segmentation_fn`` on RGB,
          then reconstructs from predicted masks + depth.
        """
        mode = self._perception_mode

        if mode == PerceptionMode.GT_POINTCLOUD:
            return self._pcd_from_gt(obs)

        if mode == PerceptionMode.REAL_PERCEPTION:
            return self._pcd_from_predicted_seg(
                obs, camera_params, relevant_seg_ids=relevant_seg_ids
            )

        # Default: GT_SEG_RECONSTRUCT
        return self._pcd_from_gt_seg(obs, camera_params)

    # -- Mode 1: GT point clouds from simulator ----------------------------

    def _pcd_from_gt(self, obs: dict) -> np.ndarray | None:
        """Assemble ``[M, 7]`` from pre-computed GT object point clouds.

        Expects ``obs["gt_object_pointclouds"]`` as a dict mapping
        ``int(seg_id) → [N, 3] float32`` (world-frame XYZ).  Colour
        channels are filled from ``obs["gt_object_colors"]`` if available,
        otherwise set to a neutral grey.
        """
        gt_pcds: dict | None = obs.get("gt_object_pointclouds")
        if gt_pcds is None:
            logger.warning(
                "[RolloutAbstractor] GT_POINTCLOUD mode but "
                "'gt_object_pointclouds' not in obs."
            )
            return None

        gt_colors: dict | None = obs.get("gt_object_colors")
        parts: list[np.ndarray] = []
        for seg_id, pts in gt_pcds.items():
            seg_id = int(seg_id)
            pts = np.asarray(pts, dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
                continue
            N = pts.shape[0]
            if gt_colors is not None and seg_id in gt_colors:
                rgb = np.broadcast_to(
                    np.asarray(gt_colors[seg_id], dtype=np.float32).reshape(1, 3),
                    (N, 3),
                ).copy()
            else:
                rgb = np.full((N, 3), 180.0, dtype=np.float32)
            seg_col = np.full((N, 1), seg_id, dtype=np.float32)
            parts.append(np.hstack([pts, rgb, seg_col]))

        if not parts:
            return None
        return np.concatenate(parts, axis=0)

    # -- Mode 2: GT segmentation + camera depth ----------------------------

    def _pcd_from_gt_seg(
        self,
        obs: dict,
        camera_params: dict[str, dict],
    ) -> np.ndarray | None:
        """Reconstruct point cloud from GT segmentation + depth.

        Merges point clouds from every spec camera that exposes RGB, depth,
        segmentation and camera params on this step. Cameras with no
        ``segmentation_key`` (e.g. LIBERO's wrist) are silently skipped.
        """
        views = self._iter_camera_views(obs, camera_params, require_segmentation=True)
        if not views:
            logger.warning(
                "[RolloutAbstractor] No camera views with full GT seg data "
                "available for PCD construction."
            )
            return None

        pcds: list[np.ndarray] = []
        for _cam, rgb, dep, seg, K3, E in views:
            assert seg is not None  # require_segmentation=True
            pcd = self._build_pcd(
                rgb, dep, seg, K3, E, preprocessed=_cam.pixels_mirrored
            )
            if pcd.shape[0] > 0:
                pcds.append(pcd)

        if not pcds:
            return None
        if len(pcds) == 1:
            return pcds[0]
        return np.concatenate(pcds, axis=0)

    # -- Mode 3: predicted segmentation + camera depth ---------------------

    def _pcd_from_predicted_seg(
        self,
        obs: dict,
        camera_params: dict[str, dict],
        relevant_seg_ids: Iterable[int] | None = None,
    ) -> np.ndarray | None:
        """Segment with the configured segmentation fn, then reconstruct from depth.

        Two paths:

        * ``multi_view_segmentation_fn`` — every spec camera with full RGB/
          depth/intrinsics/extrinsics is fed through one fused detector call;
          per-view masks build per-view PCDs that are concatenated.
        * ``segmentation_fn`` — primary camera only, single-view legacy path.
        """
        seg_ids = (
            self._relevant_seg_ids
            if relevant_seg_ids is None
            else {int(sid) for sid in relevant_seg_ids}
        )
        relevant_names = [
            self._int_name_map[sid] for sid in seg_ids if sid in self._int_name_map
        ]
        if not relevant_names:
            # Nothing to look for, so perception is never even called. This
            # used to return silently and surface one step later as "could not
            # build live point cloud", which points at the camera and is the
            # wrong place to look: the cause is upstream, in pairs that name no
            # object or an equivalence mapping that resolved none of them.
            if not self._no_relevant_names_logged:
                logger.warning(
                    "[RolloutAbstractor] No object names to segment, so the live "
                    "scene is never queried and guidance cannot apply. Pairs in "
                    f"use reference segment ids {sorted(seg_ids)}; "
                    f"the live name map has {sorted(self._int_name_map)}. Either "
                    "the demo's pairs relate only robot entities to each other, "
                    "or equivalence mapping matched none of its objects to the "
                    "live scene."
                )
                self._no_relevant_names_logged = True
            return None
        name_to_seg_id = {v: k for k, v in self._int_name_map.items()}

        if self._multi_view_segmentation_fn is not None:
            return self._pcd_from_predicted_seg_multiview(
                obs, camera_params, relevant_names
            )

        assert self._segmentation_fn is not None

        # Single-view legacy path: use the primary camera only.
        views = self._iter_camera_views(obs, camera_params)
        primary_view = next((v for v in views if v[0].is_primary), None)
        if primary_view is None:
            logger.warning(
                "[RolloutAbstractor] Primary camera missing for predicted seg; skipping."
            )
            return None
        _cam, rgb, dep, _seg, K3, E = primary_view

        masks: dict[str, np.ndarray] = self._segmentation_fn(rgb, relevant_names)
        if not masks:
            return None

        # Build a synthetic segmentation array from the predicted masks
        # using the original seg_id for each object so downstream code
        # (distance computation, object extraction) works unchanged.
        H, W = rgb.shape[:2]
        seg = np.zeros((H, W), dtype=np.int32)
        for name, mask in masks.items():
            sid = name_to_seg_id.get(name)
            if sid is not None:
                seg[mask.astype(bool)] = sid

        return self._build_pcd(rgb, dep, seg, K3, E, preprocessed=_cam.pixels_mirrored)

    def _pcd_from_predicted_seg_multiview(
        self,
        obs: dict,
        camera_params: dict[str, dict],
        relevant_names: list[str],
    ) -> np.ndarray | None:
        """Multi-view fused segmentation → per-view reconstruction → concat.

        Every camera in :class:`EnvSpec` that exposes RGB, depth, and camera
        params on this step contributes one view. The primary camera (e.g.
        LIBERO's agentview) is iterated first so the perception fn can use
        it as the fusion main camera; if it isn't available the call is
        aborted.
        """
        from open_perception import (
            CameraInfo,  # lazy: in-repo package, sys.path-mounted
        )

        views = self._iter_camera_views(obs, camera_params)
        if not views or not views[0][0].is_primary:
            logger.warning(
                "[RolloutAbstractor] primary camera missing for multi-view "
                "predicted seg; skipping."
            )
            return None

        cams: list[CameraSpec] = [v[0] for v in views]
        rgbs: list[np.ndarray] = [v[1] for v in views]
        depths: list[np.ndarray] = [v[2] for v in views]
        Ks: list[np.ndarray] = [v[4] for v in views]
        Es: list[np.ndarray] = [v[5] for v in views]
        cameras: list[CameraInfo] = [
            CameraInfo(intrinsics=K, extrinsics=E) for K, E in zip(Ks, Es, strict=True)
        ]
        primary_view_idx = next((i for i, c in enumerate(cams) if c.is_primary), 0)

        # Reset the primary-camera predicted-seg cache before this step's
        # perception call. Refilled below when the primary view returns
        # masks; stays None if perception fails or returns no primary masks,
        # which signals VisualizationHandler to skip the live overlay rather
        # than reuse a stale frame.
        self._latest_agentview_predicted_seg = None

        assert self._multi_view_segmentation_fn is not None
        masks_per_view = self._multi_view_segmentation_fn(
            rgbs, depths, cameras, relevant_names
        )

        # Capture detection/tracking visualisations from the perception client
        # (duck-typed so any MultiViewSegmentationFn impl that doesn't render
        # falls through silently). These flow up through AbstractionResult and
        # are consumed by the rerun visualiser + the episode recorder.
        fn = self._multi_view_segmentation_fn
        render_overlays = getattr(fn, "render_overlays", None)
        if callable(render_overlays):
            try:
                self._perception_overlays = render_overlays() or None
            except Exception as exc:
                logger.warning(
                    f"[RolloutAbstractor] perception render_overlays failed: {exc}"
                )
                self._perception_overlays = None
        render_history_grids = getattr(fn, "render_history_grids", None)
        if callable(render_history_grids):
            try:
                self._perception_history_grids = render_history_grids() or None
            except Exception as exc:
                logger.warning(
                    f"[RolloutAbstractor] perception render_history_grids failed: {exc}"
                )
                self._perception_history_grids = None
        render_debug_images = getattr(fn, "render_debug_images", None)
        if callable(render_debug_images):
            try:
                self._perception_debug_images = render_debug_images() or None
            except Exception as exc:
                logger.warning(
                    f"[RolloutAbstractor] perception render_debug_images failed: {exc}"
                )
                self._perception_debug_images = None
        last_timings_md = getattr(fn, "last_timings_md", None)
        if callable(last_timings_md):
            try:
                self._perception_timings_md = last_timings_md()
            except Exception:
                self._perception_timings_md = None

        # Stamp the per-view seg array with each detection's perception
        # ``instance_id`` directly — same IDs used as keys in `_int_name_map`,
        # so the centroid/PCD lookups downstream see consistent IDs across the
        # 2D overlay, the VLM prompt, and the 3D pipeline.
        pcds: list[np.ndarray] = []
        for v, masks in enumerate(masks_per_view):
            if not masks:
                continue
            H, W = rgbs[v].shape[:2]
            seg = np.zeros((H, W), dtype=np.int32)
            for inst_id, (_name, mask) in masks.items():
                if mask is not None:
                    seg[mask.astype(bool)] = inst_id
            if v == primary_view_idx:
                self._latest_agentview_predicted_seg = seg.copy()
            pcd = self._build_pcd(
                rgbs[v],
                depths[v],
                seg,
                Ks[v],
                Es[v],
                preprocessed=cams[v].pixels_mirrored,
            )
            if pcd is not None and pcd.shape[0] > 0:
                pcds.append(pcd)

        if not pcds:
            return None
        if len(pcds) == 1:
            return pcds[0]
        return np.concatenate(pcds, axis=0)

    # ------------------------------------------------------------------
    # Reference trajectory precomputation
    # ------------------------------------------------------------------

    def _precompute_ref_traj(
        self,
        top_result: dict,
        pairs: list[DistancePair],
    ) -> np.ndarray:
        """Replay the retrieved demo frame-by-frame and compute distances.

        Returns
        -------
        np.ndarray
            ``[T_ref, n_pairs]`` float32 array.  NaN entries are
            carry-forwarded from the previous valid timestep; first-frame
            NaNs are filled with ``0.0``.
        """
        n_pairs = len(self._pair_labels)
        try:
            # Read through the DemoEpisode contract rather than the HDF5
            # groups this used to open directly: the retrieved demo may be a
            # simulator recording or a real `.vla` rollout, and each format's
            # quirks (LIBERO's 180 degree frames and normalised depth buffer, a
            # `.vla`'s millimetre depth on a stride) belong to its adapter.
            episode = self._demo_episode
            if episode is None:
                episode = build_demo_episode(top_result)

            T = len(episode)
            if T == 0:
                raise RuntimeError(
                    f"retrieved demo {top_result.get('demo_key')!r} has no frames"
                )
            H, W = episode.rgb(0).shape[:2]

            # Only these timesteps can be reconstructed: every frame for a
            # simulator, the recorded stride for a real rollout. The gaps are
            # closed by the NaN carry-forward below, which already covered
            # frames whose reconstruction failed.
            depth_frames = [t for t in episode.frames_with_depth() if t < T]
            if not depth_frames:
                raise RuntimeError(
                    f"retrieved demo {top_result.get('demo_key')!r} carries no "
                    "depth, so no reference distances can be measured from it. "
                    "A rollout recorded with --output.depth-stride 0 cannot be "
                    "used as a guidance demo."
                )

            # Use achieved EE positions/orientations/gripper for the reference
            # trajectory (MTA alignment).  The dynamics model is used only when
            # *cropping* guidance chunks — see compute_dynamics_ref_chunk().
            ee_pos = np.asarray(episode.ee_positions(), dtype=np.float32)  # [T, 3]
            _ee_ori = episode.ee_orientations()
            ee_ori = None if _ee_ori is None else np.asarray(_ee_ori, dtype=np.float32)
            gripper_states = episode.gripper_states()
            # Raw units feed the dynamics roll-out's initial state; metres
            # feed the distances that are compared against the live ones.
            gripper_scalar = _gripper_aperture_scalar(gripper_states)  # [T, 1] or None
            gripper_metric = _gripper_aperture_metric(
                gripper_states, episode.gripper_geometry()
            )  # [T, 1] or None
            actions = episode.actions()
            # Back-projection must undo the u mirror only for a format that
            # stores its frames rotated; a real camera publishes them upright.
            preprocessed = bool(getattr(episode, "frames_preprocessed", True))

            # Collect entity IDs referenced by any distance pair (excluding robot entities)
            _entity_ids: set[int] = set()
            for pair in pairs:
                for ent in (pair.from_entity, pair.to_entity):
                    if isinstance(ent, int):
                        _entity_ids.add(int(ent))
            ref_entity_centroids: dict[int | str, np.ndarray] = {
                eid: np.full((T, 3), np.nan, dtype=np.float32) for eid in _entity_ids
            }

            # Precompute finger-tip trajectories for the reference demo
            _has_finger = any(
                e in ("gripper_l", "gripper_r")
                for p in pairs
                for e in (p.from_entity, p.to_entity)
            )
            ref_finger_l: np.ndarray | None = None
            ref_finger_r: np.ndarray | None = None
            if _has_finger and ee_ori is not None and gripper_states is not None:
                # The demo's own geometry, not ``self._gripper_geometry``:
                # the reference is whatever robot recorded it (LIBERO's Panda,
                # the cell's Robotiq), and measuring it with the *live*
                # robot's fingers would put the reference tips in the wrong
                # place relative to the objects they are compared against.
                ref_finger_l, ref_finger_r = compute_finger_tip_positions(
                    ee_pos.astype(np.float64),
                    ee_ori.astype(np.float64),
                    np.asarray(gripper_states, dtype=np.float64),
                    geometry=episode.gripper_geometry(),
                )
                ref_entity_centroids["gripper_l"] = ref_finger_l.astype(np.float32)
                ref_entity_centroids["gripper_r"] = ref_finger_r.astype(np.float32)
            elif _has_finger:
                logger.warning(
                    "[RolloutAbstractor] Pairs reference gripper_l/gripper_r but "
                    f"demo {top_result.get('demo_key')!r} publishes no "
                    f"{'orientation' if ee_ori is None else 'gripper state'}; "
                    "those pairs cannot be measured against it and will read 0."
                )

            traj = np.full((T, n_pairs), np.nan, dtype=np.float32)
            for t in depth_frames:
                try:
                    seg_t = episode.segmentation(t)
                    dep_t = episode.depth_m(t)
                    K_t = episode.intrinsic(t)
                    E_t = episode.extrinsic(t)
                    if seg_t is None or dep_t is None or K_t is None or E_t is None:
                        continue
                    K_t = np.asarray(K_t)
                    K_t = K_t[:3, :3] if K_t.shape == (4, 4) else K_t
                    pcd = self._build_pcd(
                        episode.rgb(t),
                        dep_t,
                        seg_t,
                        K_t,
                        np.asarray(E_t),
                        preprocessed=preprocessed,
                    )
                    if pcd.shape[0] == 0:
                        continue
                    # Store per-entity centroids for dynamics chunk computation
                    seg_col = pcd[:, 6].astype(np.int32)
                    for eid in _entity_ids:
                        mask = seg_col == eid
                        if mask.any():
                            ref_entity_centroids[eid][t] = pcd[mask, :3].mean(axis=0)
                    # Build per-frame finger positions for distance computation
                    finger_pos_t: dict[str, np.ndarray] | None = None
                    if (
                        _has_finger
                        and ref_finger_l is not None
                        and ref_finger_r is not None
                    ):
                        finger_pos_t = {
                            "gripper_l": ref_finger_l[t].astype(np.float32),
                            "gripper_r": ref_finger_r[t].astype(np.float32),
                        }
                    _ref_gripper_scalar = (
                        float(gripper_metric[t, 0])
                        if gripper_metric is not None
                        else None
                    )
                    dists = compute_distances(
                        pairs,
                        self._feature,
                        pcd,
                        ee_pos[t],
                        finger_pos_t,
                        ee_ori=ee_ori[t] if ee_ori is not None else None,
                        gripper_state=_ref_gripper_scalar,
                    )
                    for i, lbl in enumerate(self._pair_labels):
                        if lbl in dists:
                            traj[t, i] = dists[lbl]
                except Exception as exc:
                    logger.debug(f"[RolloutAbstractor] ref_traj step {t} failed: {exc}")

            # NaN carry-forward: propagate last valid value; fill leading NaNs with 0
            for t in range(1, T):
                nan_mask = np.isnan(traj[t])
                if nan_mask.any():
                    traj[t, nan_mask] = traj[t - 1, nan_mask]
            traj[0, np.isnan(traj[0])] = 0.0
            # Also NaN-carry-forward entity centroids (handles occlusion)
            for eid in _entity_ids:
                c = ref_entity_centroids[eid]
                for t in range(1, T):
                    if np.any(np.isnan(c[t])):
                        c[t] = c[t - 1]
                if np.any(np.isnan(c[0])):
                    c[0] = 0.0

            # Store reference data for dynamics chunk computation
            self._ref_actions = actions  # [T-1, action_dim]
            self._ref_entity_centroids = ref_entity_centroids

            # EE 2D projections for subtraj overlay. The camera is fixed for
            # the whole episode in every source in tree, so frame 0's matrices
            # stand in for all of them.
            self._ref_ee_pos = ee_pos
            self._ref_ee_ori = ee_ori
            self._ref_gripper = gripper_scalar
            try:
                K0 = np.asarray(episode.intrinsic(depth_frames[0]))
                E0 = np.asarray(episode.extrinsic(depth_frames[0]))
                pts_2d, valid = project_points_to_pixels(
                    ee_pos.astype(np.float64),
                    K0[:3, :3] if K0.shape == (4, 4) else K0,
                    E0,
                    H,
                    W,
                    preprocessed=preprocessed,
                )
                self._ref_ee_2d = pts_2d.astype(np.float32)
                self._ref_ee_valid = valid
            except Exception as exc:
                logger.debug(f"[RolloutAbstractor] EE 2D projection failed: {exc}")
                self._ref_ee_2d = None
                self._ref_ee_valid = None

            if self._subtraj_boundaries_override is not None:
                self._subtrajectories = list(self._subtraj_boundaries_override)
                logger.info(
                    "[RolloutAbstractor] Using verified sub-trajectory boundaries "
                    "from iterative analysis."
                )
            else:
                self._subtrajectories = compute_subtrajectories(
                    ee_pos, self._speed_threshold, self._subtraj_min_length
                )
            logger.info(
                f"[RolloutAbstractor] Sub-trajectories ({len(self._subtrajectories)}): "
                + str(self._subtrajectories)
            )

            return traj

        except Exception as exc:
            logger.error(f"[RolloutAbstractor] _precompute_ref_traj failed: {exc}")
            raise


# ---------------------------------------------------------------------------
# InteractionAbstractor
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class InteractionAbstractionResult:
    """Per-step output from :class:`InteractionAbstractor`.

    Contains the low-level alignment result for the retrieved interaction
    sub-trajectory.  Used alongside :class:`AbstractionResult` to build
    dual guidance signals with independent intensity scaling.
    """

    pairs: list[DistancePair]
    """Translated distance pairs relevant to the current interaction."""

    pair_labels: list[str]
    """Ordered pair labels (column order in ``ref_traj``)."""

    live_distances: dict[str, float]
    """``{pair_label: distance}`` from the current live frame (subset of high-level)."""

    aligned_t: int
    """Interaction-level MTA aligned reference index."""

    ref_distances: dict[str, float]
    """Reference distances at ``aligned_t`` from the interaction reference trajectory."""

    alignment_mse: float
    """MSE at the aligned reference frame.  Used for exponential-decay intensity."""

    mta_progress: float
    """Fractional completion ``aligned_t / (T_ref - 1)`` in ``[0, 1]``."""

    ref_traj: np.ndarray
    """``[T_int, n_ll_pairs]`` float32 reference trajectory for the interaction window.
    Passed to :func:`~policy_abstraction.evaluation.guidance_builder.build_distance_guidance_params`."""


class InteractionAbstractor:
    """Low-level abstractor that aligns live distances to a retrieved interaction.

    Reuses the translated distance pairs from :class:`RolloutAbstractor` but
    filters them to only those that involve the retrieved interaction's object.
    A separate :class:`MonotonicTrajectoryAligner` is maintained for the
    interaction-level alignment, whose MSE drives the low-level guidance intensity.

    Lifecycle
    ---------
    1. Construct with ``interaction_result`` (from
       :class:`~policy_abstraction.evaluation.interaction_retrieval.InteractionRetriever`),
       ``translated_pairs`` (from :class:`RolloutAbstractor`), and ``int_name_map``.
    2. Call :meth:`setup` once (after :class:`RolloutAbstractor` setup).
    3. Call :meth:`step` at each replanning boundary, passing the
       :class:`AbstractionResult` from the high-level abstractor.

    Parameters
    ----------
    interaction_result:
        Dict from :class:`~policy_abstraction.evaluation.interaction_retrieval.InteractionRetriever`
        with keys ``source_file``, ``demo_key``, ``start_timestep``, ``end_timestep``,
        ``object_name``.
    translated_pairs:
        Distance pairs already translated to the live scene's ID space by
        :class:`RolloutAbstractor`.
    int_name_map:
        ``{seg_id: name}`` for the live scene (used to match ``object_name``).
    feature:
        Distance feature for reference trajectory precomputation.
    mta_window:
        Lookahead window for the interaction-level MTA.
    erode_px:
        Erosion applied to segmentation masks during reference trajectory
        precomputation (matches :class:`RolloutAbstractor` settings).
    """

    def __init__(
        self,
        interaction_result: dict,
        translated_pairs: list[DistancePair],
        int_name_map: dict[int, str],
        *,
        primary_camera: CameraSpec,
        feature: DistanceFeature | None = None,
        mta_window: int | None = 10,
        erode_px: int = 3,
        dynamics_model: BaseDynamicsModel | None = None,
    ) -> None:
        self._interaction_result = interaction_result
        self._all_translated_pairs = translated_pairs
        self._int_name_map = int_name_map
        self._primary_camera: CameraSpec = primary_camera
        self._feature: DistanceFeature = feature or CentroidDistance()
        self._mta_window = mta_window
        self._erode_px = erode_px
        self._dynamics_model: BaseDynamicsModel = (
            dynamics_model or IntegrationDynamicsModel()
        )

        # Set after setup()
        self._pairs: list[DistancePair] = []
        self._pair_labels: list[str] = []
        self._ref_traj: np.ndarray | None = None
        self._ref_ee_ori: np.ndarray | None = None  # [T, 3] axis-angle
        self._ref_gripper: np.ndarray | None = None  # [T, 1] scalar
        self._aligner: MonotonicTrajectoryAligner | None = None
        self._ready: bool = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def pair_labels(self) -> list[str]:
        return self._pair_labels

    @property
    def ref_traj(self) -> np.ndarray | None:
        return self._ref_traj

    @property
    def ref_ee_ori(self) -> np.ndarray | None:
        return self._ref_ee_ori

    @property
    def ref_gripper(self) -> np.ndarray | None:
        return self._ref_gripper

    def setup(self) -> bool:
        """Identify interaction-relevant pairs and precompute the interaction reference trajectory.

        Returns
        -------
        bool
            ``True`` on success; ``False`` on any failure.
        """
        object_name: str = self._interaction_result.get("object_name", "")

        # Find seg_ids whose name matches the interaction object
        interaction_seg_ids: set[int] = {
            seg_id for seg_id, name in self._int_name_map.items() if name == object_name
        }

        # Filter translated pairs to those that involve the interaction object
        self._pairs = [
            pair
            for pair in self._all_translated_pairs
            if (
                pair.from_entity in interaction_seg_ids
                or pair.to_entity in interaction_seg_ids
            )
        ]

        if not self._pairs:
            logger.warning(
                f"[InteractionAbstractor] No translated pairs involve object '{object_name}'; "
                "low-level guidance disabled for this episode."
            )
            return False

        self._pair_labels = [p.label for p in self._pairs]

        # Precompute interaction-level reference trajectory
        ref_traj = self._precompute_interaction_ref_traj()
        if ref_traj is None:
            return False

        self._ref_traj = ref_traj
        self._aligner = MonotonicTrajectoryAligner(
            ref_traj,
            self._pair_labels,
            window=self._mta_window,
        )

        self._ready = True
        logger.info(
            f"[InteractionAbstractor] Ready: {len(self._pairs)} pairs for '{object_name}', "
            f"ref_traj shape {ref_traj.shape}"
        )
        return True

    def step(
        self, abstraction_result: AbstractionResult
    ) -> InteractionAbstractionResult | None:
        """Run interaction-level MTA using live distances from the high-level abstractor.

        Parameters
        ----------
        abstraction_result:
            Output from :class:`RolloutAbstractor.step` for the same timestep.

        Returns
        -------
        InteractionAbstractionResult or ``None``
            ``None`` when not ready or no relevant live distances are available.
        """
        if not self._ready or self._aligner is None or self._ref_traj is None:
            return None

        # Subset live distances to interaction pairs
        live_distances = {
            lbl: abstraction_result.live_distances[lbl]
            for lbl in self._pair_labels
            if lbl in abstraction_result.live_distances
        }

        if not live_distances:
            logger.warning(
                "[InteractionAbstractor] No live distances for interaction pairs; skipping."
            )
            return None

        live_vec = np.array(
            [live_distances.get(lbl, np.nan) for lbl in self._pair_labels],
            dtype=np.float32,
        )

        aligned_t = self._aligner.align(live_vec)
        ref_distances = self._aligner.ref_distances_at(aligned_t)
        mta_progress = self._aligner.progress()

        return InteractionAbstractionResult(
            pairs=self._pairs,
            pair_labels=self._pair_labels,
            live_distances=live_distances,
            aligned_t=aligned_t,
            ref_distances=ref_distances,
            alignment_mse=self._aligner.current_mse,
            mta_progress=mta_progress,
            ref_traj=self._ref_traj,
        )

    def _build_pcd(
        self,
        rgb: np.ndarray,
        dep: np.ndarray,
        seg: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
        *,
        preprocessed: bool = True,
    ) -> np.ndarray:
        if self._erode_px > 0:
            seg = erode_segmentation(seg, self._erode_px)
        return build_segmented_pointcloud(
            rgb, dep, seg, K, E, preprocessed=preprocessed
        )

    def _precompute_interaction_ref_traj(self) -> np.ndarray | None:
        """Load the retrieved interaction's frames and compute pair distances.

        Returns
        -------
        np.ndarray or ``None``
            ``[T_int, n_ll_pairs]`` float32 array, or ``None`` on failure.
        """
        source_file: str = self._interaction_result.get("source_file", "")
        demo_key: str = self._interaction_result.get("demo_key", "")
        start_t: int = int(self._interaction_result.get("start_timestep", 0))
        end_t: int = int(self._interaction_result.get("end_timestep", 0))

        if not source_file or not demo_key or end_t <= start_t:
            logger.warning(
                f"[InteractionAbstractor] Invalid interaction result: "
                f"source_file={source_file!r}, demo_key={demo_key!r}, "
                f"start={start_t}, end={end_t}"
            )
            return None

        n_pairs = len(self._pair_labels)
        try:
            # Same DemoEpisode contract the high-level reference uses, so this
            # is no longer a third private copy of the LIBERO HDF5 reader.
            # (The interaction *index* still only indexes HDF5 demos, so in
            # practice this path only ever sees one — but the reader no longer
            # assumes it.)
            episode = build_demo_episode(
                {
                    "source_kind": self._interaction_result.get(
                        "source_kind", "libero"
                    ),
                    "source_path": source_file,
                    "demo_key": demo_key,
                    "task": self._interaction_result.get("task", ""),
                }
            )
            window = list(range(start_t, min(end_t, len(episode))))
            T = len(window)
            if T == 0:
                raise RuntimeError(
                    f"interaction window [{start_t}, {end_t}) is empty for {demo_key!r}"
                )
            preprocessed = bool(getattr(episode, "frames_preprocessed", True))
            depth_frames = {
                t for t in episode.frames_with_depth() if start_t <= t < end_t
            }

            ee_pos_raw = np.asarray(episode.ee_positions())[start_t:end_t]
            _ee_ori_all = episode.ee_orientations()
            ee_ori_raw = (
                np.zeros_like(ee_pos_raw)
                if _ee_ori_all is None
                else np.asarray(_ee_ori_all)[start_t:end_t]
            )
            gripper_scalar_all = _gripper_aperture_scalar(episode.gripper_states())
            _all_actions = episode.actions()
            if _all_actions is None:
                raise RuntimeError(f"{demo_key!r} records no actions to integrate")
            # Actions span [start_t, end_t-1) since last obs has no action
            actions = np.asarray(_all_actions[start_t : end_t - 1])

            # Integrate actions via dynamics model to get EE positions:
            # states[0] is the achieved state, then one state per action.
            gripper_scalar_0 = (
                gripper_scalar_all[start_t]
                if gripper_scalar_all is not None
                else np.zeros(1, dtype=np.float32)
            )
            initial_state = np.concatenate(
                [
                    ee_pos_raw[0],
                    ee_ori_raw[0],
                    gripper_scalar_0,
                ]
            ).astype(np.float32)
            states = _states_after_actions(
                self._dynamics_model, actions, initial_state, dt=1.0 / 50
            )
            ee_pos = states[:T, :3].astype(np.float32)
            ee_ori = states[:T, 3:6].astype(np.float32)
            gripper_integrated = states[:T, 6:7].astype(np.float32)

            traj = np.full((T, n_pairs), np.nan, dtype=np.float32)
            for i_t, abs_t in enumerate(window):
                if abs_t not in depth_frames:
                    continue
                try:
                    dep_t = episode.depth_m(abs_t)
                    seg_t = episode.segmentation(abs_t)
                    K_t = episode.intrinsic(abs_t)
                    E_t = episode.extrinsic(abs_t)
                    if dep_t is None or seg_t is None or K_t is None or E_t is None:
                        continue
                    K_t = np.asarray(K_t)
                    pcd = self._build_pcd(
                        episode.rgb(abs_t),
                        dep_t,
                        seg_t,
                        K_t[:3, :3] if K_t.shape == (4, 4) else K_t,
                        np.asarray(E_t),
                        preprocessed=preprocessed,
                    )
                    if pcd.shape[0] == 0:
                        continue
                    dists = compute_distances(
                        self._pairs,
                        self._feature,
                        pcd,
                        ee_pos[i_t],
                        ee_ori=ee_ori[i_t],
                        # The roll-out keeps the gripper channel in the
                        # action frame; the distance wants metres, as the
                        # live side measures it.
                        gripper_state=float(
                            gripper_action_to_metric(
                                gripper_integrated[i_t, 0], episode.gripper_geometry()
                            )
                        ),
                    )
                    for i, lbl in enumerate(self._pair_labels):
                        if lbl in dists:
                            traj[i_t, i] = dists[lbl]
                except Exception as exc:
                    logger.debug(
                        f"[InteractionAbstractor] ref_traj step {abs_t} failed: {exc}"
                    )

            # NaN carry-forward
            for t in range(1, T):
                nan_mask = np.isnan(traj[t])
                if nan_mask.any():
                    traj[t, nan_mask] = traj[t - 1, nan_mask]
            traj[0, np.isnan(traj[0])] = 0.0

            self._ref_ee_ori = ee_ori
            self._ref_gripper = gripper_integrated

            return traj

        except Exception as exc:
            logger.error(
                f"[InteractionAbstractor] _precompute_interaction_ref_traj failed: {exc}"
            )
            raise
