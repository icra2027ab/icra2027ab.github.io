"""High-level analysis of a single demo rollout.

Extracts gripper-interacting objects, queries a VLM to identify informative
distance pairs, and returns structured results suitable for recording alongside
an evaluation episode.  DINO feature extraction is intentionally omitted here
(disabled for high-level object-distance analysis only).
"""

from __future__ import annotations

import colorsys
import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from policy_abstraction.analysis.strategies import (
    IterativeStrategyConfig,
    StrategyContext,
    StrategyKind,
    make_strategy,
)
from policy_abstraction.utils.pcd import project_points_to_pixels
from policy_abstraction.utils.segmentation import (
    draw_all_segment_ids,
    get_robot_segment_ids,
)
from policy_abstraction.utils.trajectory import compute_subtrajectories
from policy_abstraction.vlm.task_pair_selector import TaskPairSelector
from policy_abstraction.vlm.types import (
    ROBOT_ENTITIES,
    DistancePair,
    DistanceType,
    SubtrajConstraint,
    VLMConversation,
    VLMMessage,
)


def _distance_pair_to_dict(p: DistancePair) -> dict:
    return {
        "from_entity": p.from_entity,
        "to_entity": p.to_entity,
        "label": p.label,
        "distance_type": p.distance_type.value
        if isinstance(p.distance_type, DistanceType)
        else p.distance_type,
        "axis": p.axis,
        "plane": p.plane,
        "ee_vector": p.ee_vector,
        "subtraj_idx": p.subtraj_idx,
        "reasoning": p.reasoning,
    }


def _distance_pair_from_dict(d: dict) -> DistancePair:
    from_entity = d["from_entity"]
    if isinstance(from_entity, str) and from_entity not in ROBOT_ENTITIES:
        from_entity = int(from_entity)
    to_entity = d["to_entity"]
    if isinstance(to_entity, str) and to_entity not in ROBOT_ENTITIES:
        to_entity = int(to_entity)
    return DistancePair(
        from_entity=from_entity,
        to_entity=to_entity,
        label=d["label"],
        distance_type=DistanceType(d.get("distance_type", "centroid")),
        axis=d.get("axis"),
        plane=d.get("plane"),
        ee_vector=d.get("ee_vector"),
        subtraj_idx=d.get("subtraj_idx"),
        reasoning=d.get("reasoning"),
    )


def _subtraj_constraint_to_dict(c: SubtrajConstraint) -> dict:
    return {
        "kind": c.kind,
        "subtraj_idx": c.subtraj_idx,
        "label": c.label,
        "description": c.description,
        "expression": c.expression,
        "pairs_affected": c.pairs_affected,
        "reasoning": c.reasoning,
    }


def _subtraj_constraint_from_dict(d: dict) -> SubtrajConstraint:
    return SubtrajConstraint(
        kind=d["kind"],
        subtraj_idx=d["subtraj_idx"],
        label=d["label"],
        description=d.get("description", ""),
        expression=d["expression"],
        pairs_affected=list(d.get("pairs_affected", [])),
        reasoning=d.get("reasoning"),
    )


def _draw_subtraj_image(
    scene_image: np.ndarray,
    ee_pos: np.ndarray,
    subtrajs: list[tuple[int, int]],
    ref_K: np.ndarray,
    ref_E: np.ndarray,
    *,
    preprocessed: bool = True,
) -> np.ndarray:
    """Return a copy of *scene_image* with EE sub-trajectory paths overlaid.

    Each sub-trajectory is drawn as a thick polyline in a distinct bright colour
    and labelled ``ST0``, ``ST1``, … near its midpoint.

    *preprocessed* is the demo's own ``frames_preprocessed``: a simulator stores
    its frames 180°-rotated, a recording from the arm stores them upright, and
    projecting with the wrong one mirrors every path across the image.
    """
    H, W = scene_image.shape[:2]
    img = scene_image.copy()

    try:
        pts_2d, valid = project_points_to_pixels(
            ee_pos.astype(np.float64), ref_K, ref_E, H, W, preprocessed=preprocessed
        )
    except Exception:
        return img

    placed_labels: list[tuple[int, int, int, int]] = []  # (x1, y1, x2, y2) bboxes

    def _overlaps(x1: int, y1: int, x2: int, y2: int, pad: int = 4) -> bool:
        for px1, py1, px2, py2 in placed_labels:
            if x1 - pad < px2 and x2 + pad > px1 and y1 - pad < py2 and y2 + pad > py1:
                return True
        return False

    for i, (start, end) in enumerate(subtrajs):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        color = (int(r * 255), int(g * 255), int(b * 255))

        pts = pts_2d[start:end]
        vis = valid[start:end]
        prev_pt = None
        for j in range(len(pts)):
            if not vis[j]:
                prev_pt = None
                continue
            pt = (int(pts[j, 0]), int(pts[j, 1]))
            if prev_pt is not None:
                cv2.line(img, prev_pt, pt, color, 3, cv2.LINE_AA)
            prev_pt = pt

        # Label near the midpoint, nudging to avoid overlapping prior labels
        label = f"ST{i}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

        # Collect candidate anchor points: start from midpoint, then fan out
        mid_j = len(pts) // 2
        candidates: list[int] = []
        for offset in range(len(pts)):
            for sign in (1, -1):
                cj = mid_j + sign * offset
                if 0 <= cj < len(pts) and vis[cj]:
                    candidates.append(cj)
        # deduplicate while preserving order
        seen: set[int] = set()
        unique_candidates: list[int] = []
        for cj in candidates:
            if cj not in seen:
                seen.add(cj)
                unique_candidates.append(cj)

        placed = False
        for cj in unique_candidates:
            base_x, base_y = int(pts[cj, 0]) + 4, int(pts[cj, 1])
            # Try vertical offsets: no shift, then above, then below
            for dy in (0, -(th + 6), (th + 6)):
                lx, ly = base_x, base_y + dy
                bx1, by1 = lx - 1, ly - th - 1
                bx2, by2 = lx + tw + 1, ly + 1
                if (
                    bx1 >= 0
                    and by1 >= 0
                    and bx2 < W
                    and by2 < H
                    and not _overlaps(bx1, by1, bx2, by2)
                ):
                    cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                    cv2.putText(
                        img,
                        label,
                        (lx, ly),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                        cv2.LINE_AA,
                    )
                    placed_labels.append((bx1, by1, bx2, by2))
                    placed = True
                    break
            if placed:
                break

        # Fallback: draw at midpoint even if overlapping
        if not placed:
            for cj in unique_candidates[:1]:
                lx, ly = int(pts[cj, 0]) + 4, int(pts[cj, 1])
                bx1, by1 = lx - 1, ly - th - 1
                bx2, by2 = lx + tw + 1, ly + 1
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
                cv2.putText(
                    img,
                    label,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                placed_labels.append((bx1, by1, bx2, by2))

    return img


@dataclasses.dataclass
class DemoAnalysisResult:
    """Structured output from :class:`DemoAnalyzer.analyze`."""

    interacting_ids: set[int]
    """Segmentation IDs of objects that the gripper interacted with."""

    objects: dict[int, str]
    """Mapping from seg ID to name for the interacting objects."""

    distance_pairs: list[DistancePair]
    """VLM-selected entity pairs whose pairwise distance is task-informative."""

    constraints: list[SubtrajConstraint]
    """VLM-generated sub-goal and path constraints for each sub-trajectory."""

    vlm_prompt_text: str
    """Reconstructed user-facing text sent to the VLM (task + object list)."""

    vlm_response_json: str
    """JSON serialisation of the returned distance pairs (VLM answer)."""

    seg_index_image: np.ndarray | None
    """uint8 RGB image of the first agentview frame with all seg IDs labelled."""

    seg_mask: np.ndarray | None
    """Raw first-frame segmentation mask (upright, 180°-flipped); shape
    ``(H, W)`` or ``(H, W, 1)``.  Kept so downstream callers can re-render a
    filtered overlay scoped to a subset of seg IDs (e.g. only those referenced
    by ``distance_pairs``)."""

    scene_image: np.ndarray | None
    """uint8 RGB first agentview frame (upright, 180°-flipped)."""

    subtraj_image: np.ndarray | None
    """uint8 RGB first agentview frame with EE sub-trajectory paths overlaid."""

    pairs_selection_conversation: VLMConversation | None
    """Full VLM conversation used to select the distance pairs."""

    vlm_cache_file: str | None = None
    """Path to the cache file the VLM conversation was loaded from, or ``None`` if freshly queried."""

    iteration_log: list[dict] | None = None
    """Per-iteration proposal + score record for the iterative strategy.
    ``None`` when the single-turn strategy was used."""

    selected_subtraj_boundaries: list[tuple[int, int]] | None = None
    """Verified segment boundaries on the top-1 demo, when produced by the
    iterative strategy. ``None`` falls back to runtime speed-based
    segmentation in :class:`RolloutAbstractor`."""

    selected_iteration: int | None = None
    """Index into ``iteration_log`` of the chosen proposal (iterative only)."""


def _pair_seg_ids(pairs) -> set[int]:
    """Integer segmentation ids referenced by any of *pairs*.

    Robot entities (``"ee"``, ``"gripper_l"``, ``"gripper_r"``) are strings and
    are excluded: they are resolved from proprioception, not from a mask, so
    they stay valid however the scene was segmented.
    """
    return {
        int(e)
        for p in pairs
        for e in (p.from_entity, p.to_entity)
        if isinstance(e, int) or (isinstance(e, str) and e.lstrip("-").isdigit())
    }


def _pairs_match_segmentation(pairs, available_ids: set[int]) -> bool:
    """Whether every object id *pairs* names is present in this segmentation."""
    return _pair_seg_ids(pairs) <= set(available_ids)


class DemoAnalyzer:
    """Analyzes a single demo from an HDF5-style data group.

    Computes gripper–object interactions over the full rollout, then queries a
    VLM to select the most task-informative pairwise distances.  Results are
    returned as a :class:`DemoAnalysisResult` and are suitable for saving into
    an evaluation episode's metadata / recorder.

    DINO feature visualisation is intentionally **not** performed here; only
    the high-level object-distance (segmentation-ID pair) analysis is done.

    Parameters
    ----------
    vlm_model:
        OpenAI model name for :class:`TaskPairSelector`.
    """

    def __init__(
        self,
        vlm_model: str = "gpt-5.4",
        cache_dir: Path | str | None = None,
        gripper_speed_threshold: float = 0.002,
        subtraj_min_length: int = 20,
        strategy: StrategyKind = StrategyKind.SINGLE_TURN,
        iterative_config: IterativeStrategyConfig | None = None,
    ) -> None:
        self._vlm_model = vlm_model
        self._cache_dir: Path | None = (
            Path(cache_dir) if cache_dir is not None else None
        )
        self._gripper_speed_threshold = gripper_speed_threshold
        self._subtraj_min_length = subtraj_min_length
        self._strategy_kind = strategy
        self._iterative_config = iterative_config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _cache_path(self, cache_key: str) -> Path | None:
        if self._cache_dir is None:
            return None
        safe_key = cache_key.replace("/", "__").replace(" ", "_")
        # Suffix by strategy so single_turn and iterative caches don't collide.
        if self._strategy_kind != StrategyKind.SINGLE_TURN:
            safe_key = f"{safe_key}__{self._strategy_kind.value}"
        return self._cache_dir / f"{safe_key}.json"

    def _load_cache(self, cache_key: str) -> dict | None:
        path = self._cache_path(cache_key)
        if path is None or not path.exists():
            return None
        with path.open() as f:
            data = json.load(f)
        print(f"[DemoAnalyzer] Loaded cached distance pairs from {path}")
        pairs = [
            _distance_pair_from_dict(d)
            for d in data.get("pairs", data if isinstance(data, list) else [])
        ]
        constraints = [
            _subtraj_constraint_from_dict(d) for d in data.get("constraints", [])
        ]
        return {
            "pairs": pairs,
            "constraints": constraints,
            "assistant_text": data.get("assistant_text"),
            "iteration_log": data.get("iteration_log"),
            "selected_subtraj_boundaries": data.get("selected_subtraj_boundaries"),
            "selected_iteration": data.get("selected_iteration"),
        }

    def _save_cache(
        self,
        cache_key: str,
        distance_pairs: list[DistancePair],
        constraints: list[SubtrajConstraint],
        assistant_text: str = "",
        iteration_log: list[dict] | None = None,
        selected_subtraj_boundaries: list[tuple[int, int]] | None = None,
        selected_iteration: int | None = None,
    ) -> None:
        path = self._cache_path(cache_key)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "pairs": [_distance_pair_to_dict(p) for p in distance_pairs],
            "constraints": [_subtraj_constraint_to_dict(c) for c in constraints],
            "assistant_text": assistant_text,
        }
        if iteration_log is not None:
            payload["iteration_log"] = iteration_log
        if selected_subtraj_boundaries is not None:
            payload["selected_subtraj_boundaries"] = [
                list(b) for b in selected_subtraj_boundaries
            ]
        if selected_iteration is not None:
            payload["selected_iteration"] = selected_iteration
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"[DemoAnalyzer] Cached distance pairs to {path}")

    def analyze(
        self,
        episode,
        task_description: str,
        seg_name_map: dict[str, str] | None = None,
        cache_key: str | None = None,
        top_result: dict | None = None,
        other_results: list[dict] | None = None,
    ) -> DemoAnalysisResult:
        """Run high-level analysis on one demo.

        Parameters
        ----------
        episode:
            A :class:`~policy_abstraction.analysis.demo_episode.DemoEpisode`.
            Everything read here — upright RGB, segmentation, the camera
            matrices, the end-effector track — comes through that contract, so
            this method is the same for a simulator demo and a real recording.
        task_description:
            Natural-language task string (used as the VLM task prompt).
        seg_name_map:
            Optional ``{id: name}`` override. Defaults to the episode's own
            mapping (ground truth in a simulator, the segmenter's labels on a
            real recording).

        Returns
        -------
        DemoAnalysisResult
        """
        # ---- First frame scene image (already upright) ----------------------
        scene_image: np.ndarray = np.ascontiguousarray(episode.rgb(0))

        # ---- Segmentation index image (all IDs labelled) --------------------
        # Segment *before* reading the name map. A simulator's map is a file
        # attribute and is ready whenever asked, but a real recording has no
        # names until its segmenter has produced them -- and reading the map
        # first yields ``{}``, at which point every visible id counts as
        # unmapped and `get_robot_segment_ids` classifies the whole scene as
        # robot. The VLM then receives an empty object list and can only relate
        # the gripper to itself.
        seg_mask_0 = episode.segmentation(0)

        names = seg_name_map if seg_name_map else episode.seg_name_map()
        id_to_name: dict[int, str] = (
            {int(k): v for k, v in names.items()} if names else {}
        )

        if seg_mask_0 is None:
            raise RuntimeError(
                f"demo {getattr(episode, 'demo_key', '?')!r} "
                f"(source_kind={getattr(episode, 'source_kind', '?')!r}) has no "
                "segmentation. A simulator supplies ground-truth masks; a real "
                "recording needs a segmenter passed to the episode, since "
                "nothing in the recording labels its own pixels."
            )

        # ---- Identify robot/gripper segment IDs to exclude -----------------
        seg_2d = seg_mask_0[..., 0] if seg_mask_0.ndim == 3 else seg_mask_0
        all_seg_ids = {int(v) for v in np.unique(seg_2d) if v != 0}
        robot_ids = get_robot_segment_ids(id_to_name, all_seg_ids=all_seg_ids)
        all_seg_ids -= robot_ids
        seg_idx_bgr = draw_all_segment_ids(
            seg_mask_0, scene_image, exclude_ids=robot_ids
        )
        seg_index_image: np.ndarray = cv2.cvtColor(seg_idx_bgr, cv2.COLOR_BGR2RGB)

        # ---- Build objects dict for VLM -------------------------------------
        # Use the full set of non-robot segment IDs (same as the overlay image)
        objects: dict[int, str] = {
            sid: id_to_name.get(sid, str(sid)) for sid in all_seg_ids
        }

        # ---- Compute sub-trajectories and overlay image ---------------------
        subtrajectories: list[tuple[int, int]] | None = None
        subtraj_image: np.ndarray | None = None
        try:
            ee_pos = episode.ee_positions()
            _subtrajs = compute_subtrajectories(
                ee_pos, self._gripper_speed_threshold, self._subtraj_min_length
            )
            subtrajectories = _subtrajs
            ref_K = episode.intrinsic(0)
            ref_E = episode.extrinsic(0)
            if ref_K is None or ref_E is None:
                raise ValueError(
                    "no camera intrinsic/extrinsic for frame 0 (a real "
                    "recording needs --env.calibration-path at record time)"
                )
            subtraj_image = _draw_subtraj_image(
                scene_image,
                ee_pos,
                _subtrajs,
                np.asarray(ref_K),
                np.asarray(ref_E),
                preprocessed=bool(getattr(episode, "frames_preprocessed", True)),
            )
        except Exception as exc:
            import warnings

            warnings.warn(
                f"DemoAnalyzer: subtrajectory computation failed: {exc}",
                stacklevel=2,
            )

        # ---- Reconstruct the text prompt ------------------------------------
        objects_str = "\n".join(
            f" ({id_}: {name})" for id_, name in sorted(objects.items())
        )
        vlm_prompt_text = (
            f"Task: {task_description}\n\nAvailable objects (id: name):\n{objects_str}"
        )

        # ---- Query VLM (with optional cache) --------------------------------
        distance_pairs: list[DistancePair] = []
        subtraj_constraints: list[SubtrajConstraint] = []
        vlm_response_json: str = "[]"
        pairs_selection_conversation: VLMConversation | None = None
        vlm_cache_file: str | None = None
        iteration_log: list[dict] | None = None
        selected_subtraj_boundaries: list[tuple[int, int]] | None = None
        selected_iteration: int | None = None

        cached = self._load_cache(cache_key) if cache_key is not None else None
        if cached is not None and not _pairs_match_segmentation(
            cached["pairs"], set(objects)
        ):
            # The cached pairs name segment ids this demo's segmentation does
            # not produce. That happens whenever the masks are re-derived --
            # a re-segmented real recording numbers its instances afresh -- and
            # it is not recoverable downstream: equivalence mapping drops every
            # pair whose id it cannot find, leaving guidance with nothing that
            # references an object, and the run then skips every step with a
            # message about the *live* point cloud. Re-analysing costs one VLM
            # call; trusting the cache costs the experiment.
            logger.warning(
                f"[DemoAnalyzer] Discarding cached analysis for {cache_key!r}: it "
                f"references segment ids "
                f"{sorted(_pair_seg_ids(cached['pairs']) - set(objects))} that this "
                f"demo's segmentation does not contain (it has {sorted(objects)}). "
                "Re-running the VLM analysis."
            )
            cached = None
        if cached is not None:
            assert cache_key is not None
            vlm_cache_file = str(self._cache_path(cache_key))
            distance_pairs = cached["pairs"]
            subtraj_constraints = cached["constraints"]
            cached_assistant_text = cached.get("assistant_text")
            iteration_log = cached.get("iteration_log")
            cached_boundaries = cached.get("selected_subtraj_boundaries")
            if cached_boundaries is not None:
                selected_subtraj_boundaries = [
                    (int(s), int(e)) for s, e in cached_boundaries
                ]
            selected_iteration = cached.get("selected_iteration")
            # Use the stored assistant text if available; otherwise reconstruct from pairs.
            assistant_text_for_conv = cached_assistant_text or json.dumps(
                {
                    "subtrajectories": [
                        {
                            "subtraj_idx": p.subtraj_idx,
                            "reasoning": p.reasoning,
                            "pairs": [
                                {
                                    "from_entity": p.from_entity,
                                    "to_entity": p.to_entity,
                                    "label": p.label,
                                    "distance_type": p.distance_type.value,
                                    "axis": p.axis,
                                    "plane": p.plane,
                                    "ee_vector": p.ee_vector,
                                }
                            ],
                            "sub_goal_constraints": [],
                            "path_constraints": [],
                        }
                        for p in distance_pairs
                    ]
                }
            )
            user_images = [
                img
                for img in [scene_image, seg_index_image, subtraj_image]
                if img is not None
            ]
            pairs_selection_conversation = VLMConversation(
                messages=[
                    VLMMessage(role="system", text=TaskPairSelector._SYSTEM_PROMPT),
                    VLMMessage(role="user", text=vlm_prompt_text, images=user_images),
                    VLMMessage(role="assistant", text=assistant_text_for_conv),
                ],
                model=self._vlm_model,
            )
        else:
            try:
                strategy = make_strategy(
                    self._strategy_kind,
                    vlm_model=self._vlm_model,
                    iterative_config=self._iterative_config,
                )
                ctx = StrategyContext(
                    task=task_description,
                    objects=objects,
                    scene_image=scene_image,
                    seg_index_image=seg_index_image,
                    subtrajectories=subtrajectories,
                    subtraj_image=subtraj_image,
                    vlm_model=self._vlm_model,
                    top_result=top_result,
                    other_results=list(other_results) if other_results else [],
                    seg_name_map=seg_name_map or {},
                )
                result = strategy.propose(ctx)
                distance_pairs = result.pairs
                subtraj_constraints = result.constraints
                pairs_selection_conversation = result.conversation
                iteration_log = result.iteration_log if result.iteration_log else None
                selected_subtraj_boundaries = result.selected_subtraj_boundaries
                selected_iteration = result.selected_iteration
                if cache_key is not None:
                    raw_assistant = ""
                    if (
                        pairs_selection_conversation is not None
                        and len(pairs_selection_conversation.messages) > 2
                    ):
                        # Take the LAST assistant turn (covers iterative refinement).
                        for msg in reversed(pairs_selection_conversation.messages):
                            if msg.role == "assistant":
                                raw_assistant = msg.text
                                break
                    self._save_cache(
                        cache_key,
                        distance_pairs,
                        subtraj_constraints,
                        raw_assistant,
                        iteration_log=iteration_log,
                        selected_subtraj_boundaries=selected_subtraj_boundaries,
                        selected_iteration=selected_iteration,
                    )
            except Exception as exc:
                import warnings

                warnings.warn(f"DemoAnalyzer: VLM query failed: {exc}", stacklevel=2)

        vlm_response_json = json.dumps(
            [
                {
                    "from_entity": str(p.from_entity),
                    "to_entity": str(p.to_entity),
                    "label": p.label,
                }
                for p in distance_pairs
            ]
        )

        # ---- Derive interacting IDs from VLM distance pairs ----------------
        interacting_ids: set[int] = set()
        for p in distance_pairs:
            for entity in (p.from_entity, p.to_entity):
                if isinstance(entity, int):
                    interacting_ids.add(entity)

        return DemoAnalysisResult(
            interacting_ids=interacting_ids,
            objects=objects,
            distance_pairs=distance_pairs,
            constraints=subtraj_constraints,
            vlm_prompt_text=vlm_prompt_text,
            vlm_response_json=vlm_response_json,
            seg_index_image=seg_index_image,
            seg_mask=seg_mask_0,
            scene_image=scene_image,
            subtraj_image=subtraj_image,
            pairs_selection_conversation=pairs_selection_conversation,
            vlm_cache_file=vlm_cache_file,
            iteration_log=iteration_log,
            selected_subtraj_boundaries=selected_subtraj_boundaries,
            selected_iteration=selected_iteration,
        )
