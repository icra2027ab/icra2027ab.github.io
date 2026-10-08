"""Pipeline runner — build stages from the config, execute in order per view."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from loguru import logger

from .config import PerceptionConfig
from .context import DetectionContext
from .stages import REGISTRY
from .stages.base import MultiViewStage, Stage
from .types import PipelineSnapshot
from .utils.timing import stage_timer

AnyStage = Stage | MultiViewStage


@dataclass
class Pipeline:
    config: PerceptionConfig
    services: object

    def __post_init__(self) -> None:
        self.config.validate(REGISTRY.names())

    def build_stages(self, stage_names: set[str] | None = None) -> list[AnyStage]:
        """Instantiate every stage in ``config._order``.

        Strings are resolved via the registry with ``config._params[name]``.
        ``Stage`` instances are used directly (bypass registry + params).
        ``stage_names`` restricts the build to a subset — used by the tracking
        code-path to build only the disambiguate/assign stages when
        re-detecting a drifted query without re-running SAM3.
        """
        out: list[AnyStage] = []
        for item in self.config._order:
            if isinstance(item, str):
                name = item
                if stage_names is not None and name not in stage_names:
                    continue
                factory = REGISTRY.get(name)
                params = self.config._params.get(name, {})
                stage = factory(self.services, **params)
                if not hasattr(stage, "name") or stage.name != name:
                    # Factories may return a stage that already sets its `name`;
                    # force-match the declared name for clean stats.
                    try:
                        object.__setattr__(stage, "name", name)
                    except Exception:
                        pass
            else:
                stage = item
                name = getattr(stage, "name", stage.__class__.__name__)
                if stage_names is not None and name not in stage_names:
                    continue
            out.append(stage)
        return out

    def run(
        self, ctx: DetectionContext, stages: list[AnyStage] | None = None
    ) -> DetectionContext:
        """Single-view convenience: wraps ``run_views([ctx])``."""
        return self.run_views([ctx], stages=stages)[0]

    def run_views(
        self,
        ctxs: list[DetectionContext],
        *,
        stages: list[AnyStage] | None = None,
        multiview_only: bool = False,
    ) -> list[DetectionContext]:
        """Execute the pipeline across one or more views.

        Per-view stages (``Stage``) run once per ``ctx``; multi-view
        stages (``getattr(stage, "multiview", False)``) run once with
        the full list. ``multiview_only=True`` skips per-view stages —
        used by the ``track()`` path so propagated masks still flow
        through cross-view fusion without re-running SAM3 detection.
        Per-view stages that opt in via ``runs_on_track=True`` (e.g.
        ``reconstruct``) still execute in this mode so post-detection
        work like depth unprojection refreshes every frame.
        """
        if not ctxs:
            return ctxs
        if stages is None:
            stages = self.build_stages()

        display_names = _make_display_names(stages)
        record = bool(getattr(ctxs[0].options, "record_history", False))

        for stage, display in zip(stages, display_names, strict=True):
            is_multiview = getattr(stage, "multiview", False)
            runs_on_track = getattr(stage, "runs_on_track", False)
            if multiview_only and not is_multiview and not runs_on_track:
                continue
            if is_multiview:
                mv_stage: MultiViewStage = stage  # type: ignore[assignment]
                with stage_timer(ctxs[0].stats, display):
                    try:
                        ctxs = mv_stage(ctxs)
                    except Exception:
                        logger.exception(f"[pipeline] stage {display!r} failed")
                        raise
                for ctx in ctxs:
                    ctx.stats.stages_run.append(display)
                    if record:
                        ctx.history.append(_snapshot(ctx, display))
            else:
                pv_stage: Stage = stage  # type: ignore[assignment]
                for ctx in ctxs:
                    with stage_timer(ctx.stats, display):
                        try:
                            ctx = pv_stage(ctx)
                        except Exception:
                            logger.exception(f"[pipeline] stage {display!r} failed")
                            raise
                    ctx.stats.stages_run.append(display)
                    if record:
                        ctx.history.append(_snapshot(ctx, display))
        return ctxs


def _make_display_names(stages: list[AnyStage]) -> list[str]:
    """Suffix duplicate stage names with ``#index`` so stats keys stay unique."""
    names = [getattr(s, "name", s.__class__.__name__) for s in stages]
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    seen: dict[str, int] = {}
    display_names: list[str] = []
    for n in names:
        if counts[n] > 1:
            idx = seen.get(n, 0)
            seen[n] = idx + 1
            display_names.append(f"{n}#{idx}")
        else:
            display_names.append(n)
    return display_names


def _snapshot(ctx: DetectionContext, stage: str) -> PipelineSnapshot:
    """Capture a deep-ish copy of the mask state for later inspection.

    Detection objects are shallow-copied so later in-place mutations by
    downstream stages (e.g. ``det.mask = ...`` in filter stages) don't
    overwrite this snapshot's references. Underlying mask ndarrays are not
    copied — stages always create new arrays rather than mutating in place.
    """
    detections = {
        q: [copy.copy(det) for det in lst] for q, lst in ctx.detections.items()
    }
    instances = ctx.instances.copy() if ctx.instances is not None else None
    mask_scores = ctx.mask_scores.copy() if ctx.mask_scores is not None else None
    scores = {k: v.copy() for k, v in ctx.scores.items()}
    projected_refs = list(ctx.projected_refs)
    box_prompts = (
        {k: np.asarray(v).copy() for k, v in ctx.box_prompts.items()}
        if ctx.box_prompts
        else {}
    )
    point_prompts = (
        {k: np.asarray(v).copy() for k, v in ctx.point_prompts.items()}
        if ctx.point_prompts
        else {}
    )
    return PipelineSnapshot(
        stage=stage,
        view_idx=ctx.view_idx,
        frame_idx=ctx.frame_idx,
        queries=list(ctx.queries),
        detections=detections,
        instances=instances,
        mask_scores=mask_scores,
        scores=scores,
        projected_refs=projected_refs,
        box_prompts=box_prompts,
        point_prompts=point_prompts,
    )
