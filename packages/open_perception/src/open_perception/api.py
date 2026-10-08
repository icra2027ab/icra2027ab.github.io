"""OpenPerception: public orchestrator class.

High-level API:

```python
from open_perception import OpenPerception

perc = OpenPerception()                                 # sensible defaults

# Single image
out = perc.detect(["mug", "plate"], rgb)
for det in out.detections["mug"]:
    ...  # det.mask, det.score, det.bbox

# Streaming video
for frame_out in perc.stream(["mug", "plate"], frames):
    ...  # frame_out.detections["mug"][i].mask for each instance
```

`detect(...)` runs the full pipeline (SAM3 → filter → disambiguate → assign)
and seeds the SAM3 video tracker. `track(frame)` propagates the anchored
masks forward one frame via SAM3's memory. `stream(queries, frames)` wraps
the detect-then-track loop so you don't have to write it yourself.

Any call can receive the richer `PerceptionInput` dataclass instead of the
positional convenience args.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Iterable, Iterator, Literal

import numpy as np
from loguru import logger

from .config import PerceptionConfig
from .context import DetectionContext
from .encoders.clip import ClipEncoder
from .encoders.dino import DinoEncoder
from .encoders.reference import ReferenceBank
from .models.lazy import LazyModel, resolve_device
from .models.sam2 import Sam2Wrapper
from .models.sam3 import Sam3Wrapper
from .pipeline import Pipeline
from .tracking.cotracker_stream import CoTrackerStream
from .tracking.sam2_tracker import Sam2VideoTracker
from .tracking.sam3_tracker import BaseVideoMaskTracker, Sam3VideoTracker, drift_ok
from .tracking.state import TrackerStateRegistry, TrackState
from .types import (
    CameraInfo,
    Detection,
    InferenceOptions,
    InferenceStats,
    NormalizedInput,
    PerceptionInput,
    PerceptionOutput,
    PipelineSnapshot,
    TrackedFrame,
    TrackHistory,
)
from .utils.geometry import mask_to_bbox
from .utils.iou import mask_iou
from .utils.timing import stage_timer


class _Services:
    """Backing services exposed to stage factories."""

    def __init__(self, parent: "OpenPerception") -> None:
        self._parent = parent

    @property
    def config(self) -> PerceptionConfig:
        return self._parent.config

    def get_sam3(self) -> Sam3Wrapper:
        return self._parent._sam3.get()

    def get_sam2(self) -> Sam2Wrapper:
        return self._parent._sam2.get()

    def get_clip(self) -> ClipEncoder:
        return self._parent._clip.get()

    def get_dino(self) -> DinoEncoder:
        return self._parent._dino.get()

    def get_vlm_client(self) -> Any:
        return self._parent._vlm_client

    def get_box_detector(self) -> Any:
        return self._parent._box_detector.get()

    def get_reference_bank(self, space: str) -> ReferenceBank | None:
        return self._parent._reference_banks.get(space)

    # Used only by the multi-view fusion stage.
    def get_tracker(self, view_idx: int) -> BaseVideoMaskTracker:
        return self._parent._get_tracker(view_idx)

    def attach_aligned_id(
        self,
        view_idx: int,
        query: str,
        obj_id: int,
        mask: np.ndarray,
    ) -> None:
        """Inject a TrackState carrying an externally-supplied id.

        The cross-view fusion stage uses this when seeding a secondary
        view with the main camera's ``instance_id``. The registry's
        next-id counter is bumped past ``obj_id`` so a future internal
        allocation in this view can't collide.
        """
        self._parent._attach_aligned_id(view_idx, query, obj_id, mask)

    def attach_obj_id_alias(
        self,
        view_idx: int,
        raw_obj_id: int,
        canonical_id: int,
        new_query: str,
    ) -> None:
        """Persist a secondary obj_id → (canonical_id, query) re-attribution.

        Subsequent track frames will resolve ``raw_obj_id`` through this
        alias, so the secondary keeps surfacing the canonical id even on
        frames where ``fuse_views`` can't reproject (e.g. main camera
        lost the object).
        """
        self._parent._attach_obj_id_alias(view_idx, raw_obj_id, canonical_id, new_query)


class OpenPerception:
    """Open-vocabulary multi-instance segmentation and tracking.

    Parameters
    ----------
    config:
        Pipeline configuration. Defaults to `PerceptionConfig()` which enables
        SAM3 segmentation + overlap/size filtering + CLIP disambiguation and
        a threshold-based multi-match assigner.
    vlm_client:
        Optional VLM rerank/pre-seg client (`VlmPreClient` or
        `VlmRerankClient` protocol). Only used when the corresponding VLM
        stage is enabled.
    box_detector:
        Optional box-proposal client (`BoxDetectClient` protocol) used by the
        `detect.gemini` stage. When omitted, a `GeminiBoxDetector` is built
        lazily from the `gemini_*` config fields. Pass an instance to inject a
        custom proposer (or a stub in tests).
    reference:
        Optional reference bank — HDF5 path, `{name: image | list[image]}`
        dict, or a prebuilt `ReferenceBank`. Convenience shortcut for
        immediately calling `register_reference()` after construction.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        vlm_client: Any | None = None,
        reference: str | dict[str, Any] | ReferenceBank | None = None,
        box_detector: Any | None = None,
    ) -> None:
        self.config = config or PerceptionConfig()
        self._device = resolve_device(self.config.device)
        self._vlm_client = vlm_client
        self._box_detector_override = box_detector

        self._sam3: LazyModel[Sam3Wrapper] = LazyModel(
            lambda: Sam3Wrapper(self.config.sam3_model, self._device), "sam3"
        )
        self._sam2: LazyModel[Sam2Wrapper] = LazyModel(
            lambda: Sam2Wrapper(self.config.sam2_model, self._device), "sam2"
        )
        self._clip: LazyModel[ClipEncoder] = LazyModel(
            lambda: ClipEncoder(self.config.clip_model, self._device), "clip"
        )
        self._dino: LazyModel[DinoEncoder] = LazyModel(
            lambda: DinoEncoder(self.config.dino_model, self._device), "dino"
        )
        # Box-proposal detector for the ``detect.gemini`` stage. Building it is
        # cheap (no genai import / API call), so eager preload is safe; the
        # genai client is created on first ``detect()``.
        self._box_detector: LazyModel[Any] = LazyModel(
            self._build_box_detector, "gemini_box_detector"
        )
        # One tracker session per view. The list grows on demand: when a
        # detect() call provides V views, slots 0..V-1 are populated lazily
        # from the same loader. Keeping them as separate LazyModel instances
        # means each view's session memory stays isolated — required because
        # SAM3's video session is single-view.
        self._trackers: list[LazyModel[BaseVideoMaskTracker]] = []
        # Optional per-view CoTracker streams for 2D point tracking. Only
        # instantiated when ``config.enable_point_tracking`` is True.
        self._point_trackers: list[LazyModel[CoTrackerStream]] = []
        self._point_track_rng = np.random.default_rng(0)

        self._reference_banks: dict[str, ReferenceBank] = {}
        self._services = _Services(self)
        self._pipeline = Pipeline(self.config, self._services)
        self._tracker_state = TrackerStateRegistry()
        self._last_stats: InferenceStats = InferenceStats()
        self._frame_idx = 0
        # Cross-frame accumulator — visualization helpers consume this
        # directly. Reset alongside tracker state in `reset()`.
        self.history: TrackHistory = TrackHistory()
        # Per-view mapping: obj_id surfaced by the view's tracker session
        # → query name. Index = view_idx.
        self._obj_id_to_query: list[dict[int, str]] = []
        # Per-view alias: tracker-emitted obj_id → canonical instance_id.
        # Populated by `fuse_views.primary_guided` when a secondary
        # detection is re-indexed to match the main camera. Persists across
        # frames so the secondary keeps reporting the canonical id even on
        # frames where the projection misses (e.g. object at the edge of
        # the main camera). Defaults to identity (no remap) per obj_id.
        self._obj_id_alias: list[dict[int, int]] = []
        # Queries from the most recent `detect()` call. Used by `track()` as
        # a fallback when the tracker state is empty — a detect that yielded
        # zero detections still tells us what the caller is looking for, so
        # subsequent track() calls retry the full pipeline rather than
        # raising.
        self._last_queries: list[str] = []

        if reference is not None:
            self.register_reference(reference)
        if self.config.eager_load:
            self._preload_pipeline_models()

    def _preload_pipeline_models(self) -> None:
        """Trigger every lazy load the configured pipeline + tracker will hit.

        Without this, the first `detect()` call's stage timings include model
        load latency, which makes a cold run's perf numbers misleading.
        """
        for stage in self._pipeline.build_stages():
            if not is_dataclass(stage):
                continue
            for f in fields(stage):
                if not f.name.endswith("_getter"):
                    continue
                getter = getattr(stage, f.name, None)
                if callable(getter):
                    getter()
        # Preload one tracker so single-view detect is hot. Additional
        # views' trackers are lazy-created on first multi-view call.
        self._get_tracker(0)

    def _build_box_detector(self) -> Any:
        """Return the injected box detector, or one built from the config.

        Selects the backend via ``config.box_detector_backend``. Construction
        never imports the SDK or touches an API key (that happens on first
        ``detect()``), so this is safe to call from eager preload even without
        the optional ``[gemini]`` / ``[vlm]`` extras installed.
        """
        if self._box_detector_override is not None:
            return self._box_detector_override

        cfg = self.config
        if cfg.box_detector_backend == "openrouter":
            from .models.openrouter import OpenRouterBoxDetector

            return OpenRouterBoxDetector(
                model_id=cfg.openrouter_model,
                base_url=cfg.openrouter_base_url,
                temperature=cfg.gemini_temperature,
                max_objects=cfg.gemini_max_objects,
            )
        from .models.gemini import GeminiBoxDetector

        return GeminiBoxDetector(
            model_id=cfg.gemini_model,
            temperature=cfg.gemini_temperature,
            max_objects=cfg.gemini_max_objects,
            thinking_budget=cfg.gemini_thinking_budget,
        )

    def _build_tracker(self) -> BaseVideoMaskTracker:
        backend = self.config.tracker_backend
        if backend == "sam2":
            return Sam2VideoTracker(self.config.sam2_model, self._device)
        if backend == "sam3":
            return Sam3VideoTracker(self.config.sam3_model, self._device)
        raise ValueError(
            f"unknown tracker_backend {backend!r}; expected 'sam3' or 'sam2'"
        )

    def _get_tracker(self, view_idx: int) -> BaseVideoMaskTracker:
        """Lazy-create the per-view tracker on first access."""
        while len(self._trackers) <= view_idx:
            i = len(self._trackers)
            self._trackers.append(
                LazyModel(
                    self._build_tracker,
                    f"{self.config.tracker_backend}_video_tracker[view{i}]",
                )
            )
        return self._trackers[view_idx].get()

    def _build_point_tracker(self) -> CoTrackerStream:
        return CoTrackerStream(
            device=self._device,
            model_name=self.config.cotracker_model,
        )

    def _get_point_tracker(self, view_idx: int) -> CoTrackerStream:
        """Lazy-create the per-view CoTracker stream on first access."""
        while len(self._point_trackers) <= view_idx:
            i = len(self._point_trackers)
            self._point_trackers.append(
                LazyModel(
                    self._build_point_tracker,
                    f"cotracker_stream[view{i}]",
                )
            )
        return self._point_trackers[view_idx].get()

    def _ensure_view_state(self, n_views: int) -> None:
        while len(self._obj_id_to_query) < n_views:
            self._obj_id_to_query.append({})
        while len(self._obj_id_alias) < n_views:
            self._obj_id_alias.append({})

    # ------------------------------------------------------------------
    # Public API

    @property
    def last_stats(self) -> InferenceStats:
        return self._last_stats

    def reset(self, queries: list[str] | None = None) -> None:
        """Clear tracker state. Pass `queries` to only clear specific names."""
        self._tracker_state.drop(queries)
        if queries is None:
            for lazy in self._trackers:
                if lazy.is_loaded():
                    lazy.get().reset()
            for lazy in self._point_trackers:
                if lazy.is_loaded():
                    lazy.get().reset()
            self._obj_id_to_query = []
            self._obj_id_alias = []
            self._frame_idx = 0
            self.history.reset()
            self._last_queries = []

    def register_reference(
        self,
        refs: str | dict[str, Any] | ReferenceBank,
        space: Literal["clip", "dino"] = "clip",
    ) -> None:
        """Attach a reference bank for image-based disambiguation.

        - HDF5 path → loads `scripts/extract_objects.py`-style references.
        - dict `{name: image | list[image]}` → encodes with CLIP/DINO.
        - `ReferenceBank` → stored as-is.
        """
        encoders: dict[str, Any] = {}
        if isinstance(refs, dict):
            # Only instantiate the encoder for the chosen space.
            encoders[space] = (self._clip if space == "clip" else self._dino).get()
        bank = ReferenceBank.from_any(refs, encoders=encoders, space=space)
        self._reference_banks[bank.space] = bank

    # --- Convenience overloads ----------------------------------------

    def detect(
        self,
        queries: str | list[str] | PerceptionInput,
        rgb: np.ndarray | list[np.ndarray] | None = None,
        *,
        depth: np.ndarray | list[np.ndarray] | None = None,
        camera: CameraInfo | list[CameraInfo] | None = None,
        reference: str | dict[str, Any] | ReferenceBank | None = None,
        options: InferenceOptions | None = None,
    ) -> PerceptionOutput:
        """Run the full pipeline from scratch. Initializes tracker state.

        Accepts either positional args (`detect("mug", rgb)`) or a fully-built
        `PerceptionInput` (`detect(PerceptionInput(...))`).
        """
        inputs = self._coerce_input(queries, rgb, depth, camera, reference, options)
        norm = inputs.normalized()
        if norm.reference is not None:
            self.register_reference(norm.reference)
        self._validate_input_fields(norm)
        self._last_queries = list(norm.queries)

        stats = InferenceStats()
        self._ensure_view_state(len(norm.rgbs))
        per_view_detections, history = self._run_all_views(norm, stats)
        # Seed both the lightweight anchor state and each view's SAM3 video
        # session so a subsequent `track(next_frame)` call has memory to
        # propagate from on every view.
        self._seed_tracker_state(per_view_detections, norm)
        self._initialize_video_trackers(norm, per_view_detections)
        frame_idx = self._frame_idx
        if self.config.enable_point_tracking:
            self._seed_point_trackers(norm, per_view_detections, frame_idx)
        self._frame_idx += 1
        out = self._finalize(per_view_detections, stats, history=history)
        self._record_frame(frame_idx, norm, out, history)
        return out

    def track(
        self,
        rgb: np.ndarray | list[np.ndarray] | PerceptionInput,
        *,
        depth: np.ndarray | list[np.ndarray] | None = None,
        camera: CameraInfo | list[CameraInfo] | None = None,
        options: InferenceOptions | None = None,
    ) -> PerceptionOutput:
        """Propagate tracked masks to a new frame.

        Accepts either a single image or a list of per-view images for
        multi-camera tracking — same shape contract as ``detect()``.
        Queries are inferred from the tracker state (everything seeded by
        the previous ``detect()`` call); if state is empty (e.g. detect
        produced no masks because filters dropped them all), the queries
        from the last ``detect()`` call are reused so the pipeline retries
        detection on this frame instead of failing. Raises only if
        ``detect()`` has never been called.
        """
        if isinstance(rgb, PerceptionInput):
            inputs = rgb
        else:
            if rgb is None:
                raise ValueError("track(rgb=...) cannot be None")
            if self._tracker_state.states:
                # Preserve the original detect() query order so per-query
                # color assignment in visualizations stays consistent across
                # detect/track frames. Append any tracker-state keys that
                # weren't in the last detect() call (shouldn't normally happen).
                state_keys = self._tracker_state.states.keys()
                queries = [q for q in self._last_queries if q in state_keys]
                queries += [q for q in state_keys if q not in self._last_queries]
            elif self._last_queries:
                # Detect ran but seeded nothing — keep retrying on subsequent
                # frames using the same query set (the per-view re-detection
                # path below handles `not in tracker_state` queries).
                queries = list(self._last_queries)
            else:
                raise RuntimeError(
                    "track() called before any detect() — "
                    "call detect() first, or use stream(queries, frames)"
                )
            inputs = PerceptionInput(
                queries=queries,
                rgb=rgb,
                depth=depth,
                cameras=camera,
                options=options,
            )
        return self._track_impl(inputs)

    def stream(
        self,
        queries: str | list[str],
        frames: Iterable[np.ndarray],
        *,
        reference: str | dict[str, Any] | ReferenceBank | None = None,
        options: InferenceOptions | None = None,
    ) -> Iterator[PerceptionOutput]:
        """Yield one `PerceptionOutput` per frame.

        The first frame runs `detect()`; subsequent frames run `track()`. The
        tracker auto-resets at the start so prior state doesn't leak.
        """
        self.reset()
        first = True
        for frame in frames:
            if first:
                yield self.detect(queries, frame, reference=reference, options=options)
                first = False
            else:
                yield self.track(frame, options=options)

    # ------------------------------------------------------------------
    # Internal helpers

    @staticmethod
    def _coerce_input(
        queries: str | list[str] | PerceptionInput,
        rgb: np.ndarray | list[np.ndarray] | None,
        depth: np.ndarray | list[np.ndarray] | None,
        camera: CameraInfo | list[CameraInfo] | None,
        reference: str | dict[str, Any] | ReferenceBank | None,
        options: InferenceOptions | None,
    ) -> PerceptionInput:
        if isinstance(queries, PerceptionInput):
            return queries
        if rgb is None:
            raise ValueError("rgb is required when queries is not a PerceptionInput")
        return PerceptionInput(
            queries=queries,
            rgb=rgb,
            depth=depth,
            cameras=camera,
            reference=reference,
            options=options,
        )

    def _track_impl(self, inputs: PerceptionInput) -> PerceptionOutput:
        norm = inputs.normalized()
        opts = norm.options
        self._validate_input_fields(norm)

        stats = InferenceStats()
        self._ensure_view_state(len(norm.rgbs))
        per_view_detections: list[dict[str, list[Detection]]] = []
        per_view_contexts: list[DetectionContext] = []

        for view_idx, rgb in enumerate(norm.rgbs):
            view_out: dict[str, list[Detection]] = {q: [] for q in norm.queries}

            # Step this view's video tracker once per frame. Timed under
            # `tracker.step.view{i}` so per-view inference cost is visible in
            # `stats.timings_ms` alongside the pipeline-stage timings.
            tracker = self._get_tracker(view_idx)
            propagated: dict[int, np.ndarray] = {}
            if tracker.is_initialized:
                with stage_timer(stats, f"tracker.step.view{view_idx}"):
                    propagated = tracker.step(rgb)
                stats.increment_model_call("sam3_video_step")

            view_obj_to_query = self._obj_id_to_query[view_idx]
            view_alias = self._obj_id_alias[view_idx]
            drifted_queries: set[str] = set()
            for obj_id, new_mask in propagated.items():
                query = view_obj_to_query.get(obj_id)
                if query is None or query not in set(norm.queries):
                    continue
                # Resolve the persistent re-attribution set by `fuse_views`
                # (default: identity). The canonical id is what callers see
                # on Detection.instance_id; obj_id stays the tracker key.
                canonical_id = view_alias.get(obj_id, obj_id)
                state = self._find_state(query, obj_id, view_idx)
                anchor = state.anchor_mask if state is not None else new_mask
                iou_to_anchor = mask_iou(new_mask, anchor)
                recovered = False
                if opts.drift_policy != "off" and not drift_ok(
                    new_mask, anchor, opts.drift_iou_threshold
                ):
                    if opts.drift_policy == "auto":
                        logger.info(
                            f"[track] drift detected for {query!r} "
                            f"(IoU={iou_to_anchor:.3f}); flagging for re-detect"
                        )
                        drifted_queries.add(query)
                        stats.recoveries.append(query)
                        recovered = True
                    else:  # "surface"
                        view_out[query].append(
                            Detection(
                                query=query,
                                mask=new_mask,
                                score=float(iou_to_anchor),
                                view_idx=view_idx,
                                bbox=mask_to_bbox(new_mask),
                                instance_id=canonical_id,
                                drifted=True,
                            )
                        )
                        if state is not None:
                            state.last_mask = new_mask
                            state.last_frame_idx = self._frame_idx
                        continue
                if recovered:
                    continue

                view_out[query].append(
                    Detection(
                        query=query,
                        mask=new_mask,
                        score=float(iou_to_anchor),
                        view_idx=view_idx,
                        bbox=mask_to_bbox(new_mask),
                        instance_id=canonical_id,
                    )
                )
                if state is not None:
                    state.last_mask = new_mask
                    state.last_frame_idx = self._frame_idx

            # Full re-detection for queries that drifted under "auto" policy,
            # plus any queries that didn't have tracker state yet.
            queries_needing_detect = list(
                drifted_queries
                | {q for q in norm.queries if q not in self._tracker_state}
            )
            if queries_needing_detect:
                partial_ctx = self._make_context(
                    rgb=rgb,
                    depth=norm.depths[view_idx],
                    camera=norm.cameras[view_idx],
                    queries=queries_needing_detect,
                    options=opts,
                    view_idx=view_idx,
                    stats=stats,
                )
                self._pipeline.run(partial_ctx)
                # Drop stale state for drifted queries, add fresh.
                for q in drifted_queries:
                    self._drop_obj_ids_for_query(view_idx, q)
                    self._tracker_state.drop([q])
                for q, lst in partial_ctx.detections.items():
                    view_out[q].extend(lst)
                    for det in lst:
                        self._add_track_state(q, det.mask, view_idx, det)
                # Re-anchor this view's video session with the refreshed masks.
                if partial_ctx.detections:
                    self._reseed_video_session_with_frame(
                        view_idx, rgb, partial_ctx.detections
                    )

            if self.config.enable_point_tracking:
                drift_recovered = bool(queries_needing_detect)
                if (
                    drift_recovered
                    and self.config.cotracker_drift_action == "reseed_view"
                ):
                    self._reseed_point_tracker(view_idx, rgb, view_out, norm)
                else:
                    self._step_point_tracker(view_idx, rgb, view_out)

            ctx = self._make_context(
                rgb=rgb,
                depth=norm.depths[view_idx],
                camera=norm.cameras[view_idx],
                queries=list(norm.queries),
                options=opts,
                view_idx=view_idx,
                stats=stats,
            )
            ctx.detections = view_out
            per_view_contexts.append(ctx)

        # Run cross-view (multi-view) stages on the propagated outputs so
        # fusion fires every frame, not just at detect time.
        per_view_contexts = self._pipeline.run_views(
            per_view_contexts, multiview_only=True
        )
        history: list[PipelineSnapshot] = []
        for ctx in per_view_contexts:
            per_view_detections.append(ctx.detections)
            history.extend(ctx.history)

        frame_idx = self._frame_idx
        self._frame_idx += 1
        out = self._finalize(per_view_detections, stats, history=history)
        self._record_frame(frame_idx, norm, out, history)
        return out

    def _make_context(
        self,
        *,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        camera,
        queries: list[str],
        options: InferenceOptions,
        view_idx: int,
        stats: InferenceStats,
    ) -> DetectionContext:
        return DetectionContext(
            queries=queries,
            rgb=rgb,
            depth=depth,
            camera=camera,
            options=options,
            view_idx=view_idx,
            frame_idx=self._frame_idx,
            stats=stats,
        )

    def _run_all_views(
        self, norm, stats: InferenceStats
    ) -> tuple[list[dict[str, list[Detection]]], list[PipelineSnapshot]]:
        ctxs: list[DetectionContext] = []
        for view_idx, rgb in enumerate(norm.rgbs):
            ctxs.append(
                self._make_context(
                    rgb=rgb,
                    depth=norm.depths[view_idx],
                    camera=norm.cameras[view_idx],
                    queries=list(norm.queries),
                    options=norm.options,
                    view_idx=view_idx,
                    stats=stats,
                )
            )
        ctxs = self._pipeline.run_views(ctxs)
        per_view: list[dict[str, list[Detection]]] = []
        history: list[PipelineSnapshot] = []
        for ctx in ctxs:
            per_view.append(ctx.detections)
            history.extend(ctx.history)
        return per_view, history

    def _seed_tracker_state(
        self, per_view: list[dict[str, list[Detection]]], norm
    ) -> None:
        """Persist per-Detection anchor masks so track() can follow them.

        Detections whose ``instance_id`` is already set (e.g. by the
        cross-view fusion stage assigning a main-camera id to a
        secondary detection) are kept verbatim and registered in the
        registry without minting a new id. Detections without an id
        get a freshly issued one.
        """
        self._tracker_state.drop(None)
        for d in self._obj_id_to_query:
            d.clear()
        for view_idx, view_dets in enumerate(per_view):
            for q, lst in view_dets.items():
                for det in lst:
                    if det.instance_id is not None:
                        # Pre-aligned by fusion — register without re-issuing.
                        self._attach_aligned_id(view_idx, q, det.instance_id, det.mask)
                    else:
                        self._add_track_state(q, det.mask, view_idx, det)

    def _add_track_state(
        self, query: str, mask: np.ndarray, view_idx: int, det: Detection
    ) -> None:
        instance_id = self._tracker_state.issue_instance_id()
        det.instance_id = instance_id
        self._obj_id_to_query[view_idx][instance_id] = query
        self._tracker_state.add(
            TrackState(
                query=query,
                instance_id=instance_id,
                view_idx=view_idx,
                anchor_mask=mask.copy(),
                last_mask=mask.copy(),
                last_frame_idx=self._frame_idx,
            )
        )

    def _attach_aligned_id(
        self, view_idx: int, query: str, obj_id: int, mask: np.ndarray
    ) -> None:
        """Inject a TrackState carrying an externally-supplied id."""
        self._ensure_view_state(view_idx + 1)
        self._tracker_state.register_external_id(obj_id)
        self._obj_id_to_query[view_idx][obj_id] = query
        # If a state with this id already exists for this view+query
        # (alignment hit on an existing detection), refresh its anchor
        # rather than appending a duplicate.
        existing = self._find_state(query, obj_id, view_idx)
        if existing is not None:
            existing.anchor_mask = mask.copy()
            existing.last_mask = mask.copy()
            existing.last_frame_idx = self._frame_idx
            return
        self._tracker_state.add(
            TrackState(
                query=query,
                instance_id=obj_id,
                view_idx=view_idx,
                anchor_mask=mask.copy(),
                last_mask=mask.copy(),
                last_frame_idx=self._frame_idx,
            )
        )

    def _attach_obj_id_alias(
        self,
        view_idx: int,
        raw_obj_id: int,
        canonical_id: int,
        new_query: str,
    ) -> None:
        """Persistently re-attribute a tracker obj_id to a different (id, query).

        Used by the cross-view fusion stage in ``primary_guided`` mode when
        it re-indexes a secondary detection to match the main camera's id.
        Without this persistence, frames where the projection misses (e.g.
        the main camera lost the object near the image border) would
        revert to surfacing the raw obj_id under its original query.
        """
        self._ensure_view_state(view_idx + 1)
        self._obj_id_to_query[view_idx][raw_obj_id] = new_query
        if raw_obj_id == canonical_id:
            self._obj_id_alias[view_idx].pop(raw_obj_id, None)
        else:
            self._obj_id_alias[view_idx][raw_obj_id] = canonical_id

    def _find_state(self, query: str, obj_id: int, view_idx: int) -> TrackState | None:
        for state in self._tracker_state.states.get(query, []):
            if state.instance_id == obj_id and state.view_idx == view_idx:
                return state
        return None

    def _drop_obj_ids_for_query(self, view_idx: int, query: str) -> None:
        view_obj_to_query = self._obj_id_to_query[view_idx]
        stale = [oid for oid, q in view_obj_to_query.items() if q == query]
        for oid in stale:
            view_obj_to_query.pop(oid, None)

    def _seed_point_trackers(
        self,
        norm: NormalizedInput,
        per_view: list[dict[str, list[Detection]]],
        frame_idx: int,
    ) -> None:
        """Seed each view's CoTracker stream from its detections + depth.

        Raises if any view has detections to seed but no depth/intrinsics —
        the contract is that point tracking depends on reconstructed-point
        pixels, which requires depth.
        """
        for view_idx, view_dets in enumerate(per_view):
            flat: list[Detection] = []
            for lst in view_dets.values():
                for det in lst:
                    if det.instance_id is not None:
                        flat.append(det)
            if not flat:
                continue
            depth = norm.depths[view_idx]
            cam = norm.cameras[view_idx]
            if depth is None or cam is None:
                raise RuntimeError(
                    f"point tracking enabled but view {view_idx} has no depth "
                    "or camera intrinsics; pass depth=... and camera=... to "
                    "detect() or set enable_point_tracking=False"
                )
            point_tracker = self._get_point_tracker(view_idx)
            point_tracker.seed(
                view_idx=view_idx,
                frame_idx=frame_idx,
                rgb=norm.rgbs[view_idx],
                detections=flat,
                depth=depth,
                intrinsics=cam.intrinsics,
                extrinsics=cam.extrinsics,
                max_points_per_det=self.config.cotracker_max_points_per_detection,
                min_points_per_det=self.config.cotracker_min_points_per_detection,
                erode_kernel_size=self.config.cotracker_erode_kernel_size,
                rng=self._point_track_rng,
            )

    def _step_point_tracker(
        self,
        view_idx: int,
        rgb: np.ndarray,
        view_out: dict[str, list[Detection]],
    ) -> None:
        """Step this view's CoTracker and attach the snapshot to detections."""
        if view_idx >= len(self._point_trackers):
            return
        lazy = self._point_trackers[view_idx]
        if not lazy.is_loaded():
            return
        point_tracker = lazy.get()
        if not point_tracker.is_seeded:
            return
        snapshot = point_tracker.step(rgb)
        if snapshot is None:
            return
        for dets in view_out.values():
            for det in dets:
                if (
                    det.instance_id is not None
                    and det.instance_id in snapshot.instance_slices
                ):
                    det.point_tracks = snapshot

    def _reseed_point_tracker(
        self,
        view_idx: int,
        rgb: np.ndarray,
        view_out: dict[str, list[Detection]],
        norm: NormalizedInput,
    ) -> None:
        """Reset and re-seed this view's CoTracker after a drift recovery.

        CoTracker's online API has no clean way to add queries mid-stream,
        so the entire view's session is restarted with all currently active
        detections. The view re-enters the ``2*step - 1`` warm-up window;
        until it ends, surfaced detections will have ``point_tracks=None``.
        """
        depth = norm.depths[view_idx]
        cam = norm.cameras[view_idx]
        if depth is None or cam is None:
            logger.warning(
                f"[cotracker] view {view_idx} drift recovery: cannot reseed "
                "without depth + camera; freezing existing session"
            )
            self._step_point_tracker(view_idx, rgb, view_out)
            return
        flat: list[Detection] = []
        for lst in view_out.values():
            for det in lst:
                if det.instance_id is not None:
                    flat.append(det)
        if not flat:
            return
        point_tracker = self._get_point_tracker(view_idx)
        point_tracker.reset()
        try:
            point_tracker.seed(
                view_idx=view_idx,
                frame_idx=self._frame_idx,
                rgb=rgb,
                detections=flat,
                depth=depth,
                intrinsics=cam.intrinsics,
                extrinsics=cam.extrinsics,
                max_points_per_det=self.config.cotracker_max_points_per_detection,
                min_points_per_det=self.config.cotracker_min_points_per_detection,
                erode_kernel_size=self.config.cotracker_erode_kernel_size,
                rng=self._point_track_rng,
            )
        except RuntimeError as exc:
            logger.warning(f"[cotracker] view {view_idx} drift reseed failed: {exc}")

    def _initialize_video_trackers(
        self, norm, per_view: list[dict[str, list[Detection]]]
    ) -> None:
        """Start one SAM3 video session per view, seeded with that view's masks.

        Each view's tracker session is independent; aligned ``instance_id``s
        already assigned by the fusion stage carry through naturally
        because the obj_ids registered here come straight from the
        Detections.
        """
        for view_idx, view_dets in enumerate(per_view):
            masks_by_obj_id: dict[int, np.ndarray] = {}
            for dets in view_dets.values():
                for det in dets:
                    if det.instance_id is not None:
                        masks_by_obj_id[det.instance_id] = det.mask
            if not masks_by_obj_id:
                continue
            tracker = self._get_tracker(view_idx)
            tracker.reset()
            try:
                tracker.initialize(norm.rgbs[view_idx], masks_by_obj_id)
            except Exception as exc:
                logger.warning(
                    f"[track] failed to initialize SAM3 video tracker for view "
                    f"{view_idx}: {exc}; subsequent track() calls will fall back "
                    "to per-frame detect"
                )

    def _reseed_video_session_with_frame(
        self,
        view_idx: int,
        rgb: np.ndarray,
        partial_dets: dict[str, list[Detection]],
    ) -> None:
        """After drift-recovery, restart this view's session anchored at the current frame.

        Without this, the tracker would keep carrying the stale memory from a
        failed propagation.
        """
        masks_by_obj_id: dict[int, np.ndarray] = {}
        for lst in partial_dets.values():
            for det in lst:
                if det.instance_id is not None:
                    masks_by_obj_id[det.instance_id] = det.mask
        if not masks_by_obj_id:
            return
        tracker = self._get_tracker(view_idx)
        tracker.reset()
        try:
            tracker.initialize(rgb, masks_by_obj_id)
        except Exception as exc:
            logger.warning(
                f"[track] re-initialize SAM3 video failed for view {view_idx}: {exc}"
            )

    def _validate_input_fields(self, norm: NormalizedInput) -> None:
        """Re-run config validation with the actual fields present at call time.

        At config-build time we don't know whether the caller will supply
        depth or camera intrinsics; running validate() here surfaces
        warnings like "reconstruct in pipeline but no depth supplied"
        with the real input shape.
        """
        from .stages.base import REGISTRY

        fields_present: set[str] = {"rgb"}
        if any(d is not None for d in norm.depths):
            fields_present.add("depth")
        if any(c is not None for c in norm.cameras):
            fields_present.add("camera.intrinsics")
            if any(c is not None and c.extrinsics is not None for c in norm.cameras):
                fields_present.add("camera.extrinsics")
        self.config.validate(REGISTRY.names(), input_fields=fields_present)

    def _record_frame(
        self,
        frame_idx: int,
        norm: NormalizedInput,
        out: PerceptionOutput,
        history: list[PipelineSnapshot],
    ) -> None:
        """Append a TrackedFrame to ``self.history`` if recording is enabled."""
        if not norm.options.record_track_history:
            return
        self.history.append(
            TrackedFrame(
                frame_idx=frame_idx,
                queries=list(norm.queries),
                rgbs=list(norm.rgbs),
                depths=list(norm.depths),
                cameras=list(norm.cameras),
                detections=out.detections,
                stage_history=list(history),
                stats=out.stats,
            )
        )

    def _finalize(
        self,
        per_view: list[dict[str, list[Detection]]],
        stats: InferenceStats,
        *,
        history: list[PipelineSnapshot] | None = None,
    ) -> PerceptionOutput:
        merged: dict[str, list[Detection]] = {}
        for view_dets in per_view:
            for q, lst in view_dets.items():
                merged.setdefault(q, []).extend(lst)
                stats.n_detections_per_query[q] = stats.n_detections_per_query.get(
                    q, 0
                ) + len(lst)
        self._last_stats = stats
        return PerceptionOutput(detections=merged, stats=stats, history=history or [])
