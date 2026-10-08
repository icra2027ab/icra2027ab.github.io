"""Cross-view fusion stage.

Operates on every view's :class:`DetectionContext` at once
(``MultiViewStage``) — the runner dispatches it once per pipeline run
rather than once per view. Two modes:

- ``"independent"`` (default): no-op pass-through. Useful as an explicit
  declaration that the user wants per-camera ids without any alignment.
- ``"primary_guided"``: one camera (``main_camera``) is authoritative. For
  each main-camera detection with a populated ``Detection.pointcloud``,
  project the points into each secondary camera (with occlusion culling
  against the secondary's depth map). Then either re-attribute a matching
  secondary detection's ``instance_id`` to the main camera's, or seed a
  brand-new tracked instance in the secondary's SAM3 video session via
  point prompts at the main camera's id.

``reconstruct`` must run earlier in ``cfg.order([...])`` for
``primary_guided`` mode — the projection step needs ``Detection.pointcloud``
populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, ClassVar, Literal

import numpy as np
from loguru import logger

from ..context import DetectionContext
from ..tracking.sam3_tracker import BaseVideoMaskTracker
from ..types import Detection, ProjectedReference
from ..utils.geometry import mask_to_bbox
from ..utils.projection import (
    projected_points_to_mask,
    world_to_image,
)
from .base import REGISTRY

TrackerGetter = Callable[[int], BaseVideoMaskTracker]
AttachIdFn = Callable[[int, str, int, np.ndarray], None]
AttachAliasFn = Callable[[int, int, int, str], None]


@dataclass
class FuseViewsStage:
    name: str = "fuse_views"
    multiview: bool = True
    mode: Literal["independent", "primary_guided"] = "independent"
    main_camera: int = 0
    iou_align_threshold: float = 0.3
    occlusion_tolerance: float = 0.05
    min_projected_points: int = 5
    max_seed_points: int = 16
    footprint_dilate: int = 2
    flip_u: bool = False
    get_tracker_fn: TrackerGetter | None = None
    attach_id_fn: AttachIdFn | None = None
    attach_alias_fn: AttachAliasFn | None = None
    seeded_instances: set[tuple[int, int]] = field(default_factory=set)
    provides: ClassVar[tuple[str, ...]] = ()

    @property
    def requires(self) -> tuple[str, ...]:
        # `independent` is a no-op pass-through; `primary_guided` projects
        # the main camera's pointclouds into the others, so it needs them.
        if self.mode == "primary_guided":
            return ("detections.pointcloud", "camera.intrinsics")
        return ()

    def __call__(self, ctxs: list[DetectionContext]) -> list[DetectionContext]:
        if self.mode == "independent" or len(ctxs) <= 1:
            return ctxs
        return self._fuse_primary_guided(ctxs)

    def _fuse_primary_guided(
        self, ctxs: list[DetectionContext]
    ) -> list[DetectionContext]:
        if self.main_camera < 0 or self.main_camera >= len(ctxs):
            logger.warning(
                f"[fuse_views] main_camera={self.main_camera} out of range "
                f"for {len(ctxs)} views; skipping"
            )
            return ctxs

        main_ctx = ctxs[self.main_camera]
        for query, main_dets in main_ctx.detections.items():
            for det in main_dets:
                main_id = det.instance_id
                pc = det.pointcloud
                if main_id is None or pc is None or pc.shape[0] == 0:
                    continue
                for s_idx, sec_ctx in enumerate(ctxs):
                    if s_idx == self.main_camera:
                        continue
                    self._align_or_seed(
                        det, main_id, pc, query, main_ctx, sec_ctx, s_idx
                    )
        return ctxs

    def _align_or_seed(
        self,
        main_det: Detection,
        main_id: int,
        main_pointcloud: np.ndarray,
        query: str,
        main_ctx: DetectionContext,
        sec_ctx: DetectionContext,
        s_idx: int,
    ) -> None:
        if sec_ctx.camera is None:
            return
        H, W = sec_ctx.image_hw
        proj = world_to_image(
            main_pointcloud,
            sec_ctx.camera,
            image_hw=(H, W),
            depth_map=sec_ctx.depth,
            occlusion_tolerance=self.occlusion_tolerance,
            flip_u=self.flip_u,
        )
        if proj.uv.shape[0] < self.min_projected_points:
            return
        footprint = projected_points_to_mask(
            proj.uv, image_hw=(H, W), dilate=self.footprint_dilate
        )

        # Make sure the target query bucket exists so we can append a
        # newly-seeded detection in case (a).
        sec_ctx.detections.setdefault(query, [])

        # Three-way fusion policy based on what the projected reference
        # *points* hit in the secondary view:
        #   (c) the points fall inside a sec_det that already carries main_id
        #       → skip, the views are already aligned for this object.
        #   (b) the points fall inside a sec_det with a *different* id
        #       (regardless of which query bucket it lives in) → re-index
        #       that sec_det's instance_id to main_id (cheap, no model call);
        #       if the secondary had it under the wrong query, also move it
        #       into the right query bucket.
        #   (a) the points don't fall inside any sec_det meaningfully → seed
        #       a fresh instance via point prompts on the secondary tracker.
        #
        # Coverage is point-in-mask fraction (intersection / |projected pts|),
        # not symmetric IoU — the dilated footprint is sparse and a dense
        # secondary mask would always lose under IoU even when the points
        # clearly land inside it.
        n_pts = int(proj.uv.shape[0])
        us = proj.uv[:, 0]
        vs = proj.uv[:, 1]
        same_id_cov = 0.0
        diff_q: str | None = None
        diff_idx = -1
        diff_cov = 0.0
        for q_iter, dets in sec_ctx.detections.items():
            for i, e in enumerate(dets):
                if e.mask is None:
                    continue
                inside = int(e.mask[vs, us].sum())
                cov = inside / max(n_pts, 1)
                if e.instance_id == main_id:
                    if cov > same_id_cov:
                        same_id_cov = cov
                else:
                    if cov > diff_cov:
                        diff_cov = cov
                        diff_q = q_iter
                        diff_idx = i

        already_aligned = same_id_cov >= self.iou_align_threshold
        diff_id_match = (
            not already_aligned
            and diff_idx >= 0
            and diff_cov >= self.iou_align_threshold
        )
        # Diagnostic ProjectedReference: best coverage against *any* sec_det,
        # `matched=True` whenever the footprint already lines up with some
        # sec_det (same- or different-id) — i.e. no point-prompt seeding will
        # fire on this call.
        sec_ctx.projected_refs.append(
            ProjectedReference(
                query=query,
                instance_id=main_id,
                source_view_idx=main_ctx.view_idx,
                uv=proj.uv.copy(),
                footprint=footprint,
                iou=float(max(same_id_cov, diff_cov)),
                matched=bool(already_aligned or diff_id_match),
            )
        )

        if already_aligned:
            return

        if diff_id_match:
            assert diff_q is not None
            target = sec_ctx.detections[diff_q][diff_idx]
            old_obj_id = target.instance_id
            target.instance_id = main_id
            # If the secondary had segmented this physical object under a
            # different query (mis-classification by SAM3), move it into
            # the main camera's query bucket so downstream consumers see
            # consistent labelling across views.
            if diff_q != query:
                target.query = query
                sec_ctx.detections[diff_q].pop(diff_idx)
                sec_ctx.detections.setdefault(query, []).append(target)
            # Persist the re-attribution so the api keeps surfacing this
            # tracker obj_id under the canonical (id, query) on future
            # frames — even when the projection misses.
            if self.attach_alias_fn is not None and old_obj_id is not None:
                self.attach_alias_fn(s_idx, int(old_obj_id), main_id, query)
            if self.attach_id_fn is not None:
                self.attach_id_fn(s_idx, query, main_id, target.mask)
            return

        # Case (a) — no useful overlap. Condition the secondary tracker via
        # point prompts. `seeded_instances` is a safety guard against
        # pathological tracker failures that'd otherwise re-seed every frame.
        if (s_idx, main_id) in self.seeded_instances:
            return
        if self.get_tracker_fn is None or self.attach_id_fn is None:
            return

        points_uv = _subsample_uv(proj.uv, self.max_seed_points)
        labels = np.ones((points_uv.shape[0],), dtype=np.int32)

        try:
            tracker = self.get_tracker_fn(s_idx)
        except Exception as exc:
            logger.warning(
                f"[fuse_views] failed to get tracker for view {s_idx}: {exc}"
            )
            return
        if not tracker.is_initialized:
            # Detect-time first call — the per-view tracker hasn't been
            # seeded yet (that happens after this stage runs). Skip
            # point-prompt seeding here; once track() has stepped and
            # propagated, future frames will route through this branch.
            return

        try:
            new_mask = tracker.add_object_with_points(
                frame_rgb=sec_ctx.rgb,
                obj_id=main_id,
                points_uv=points_uv,
                point_labels=labels,
            )
        except Exception as exc:
            logger.warning(
                f"[fuse_views] add_object_with_points failed for view "
                f"{s_idx} obj {main_id}: {exc}"
            )
            return

        if not new_mask.any():
            return

        new_det = Detection(
            query=query,
            mask=new_mask,
            score=float(diff_cov),
            view_idx=s_idx,
            bbox=mask_to_bbox(new_mask),
            instance_id=main_id,
        )
        sec_ctx.detections.setdefault(query, []).append(new_det)
        self.attach_id_fn(s_idx, query, main_id, new_mask)
        self.seeded_instances.add((s_idx, main_id))


def _subsample_uv(uv: np.ndarray, max_points: int) -> np.ndarray:
    """Return at most ``max_points`` from ``uv`` via even-stride subsampling.

    Sort by Y first so the subsample spans the projected footprint top
    to bottom rather than clustering at the start of the array.
    """
    n = uv.shape[0]
    if n <= max_points:
        return uv.astype(np.float32)
    order = np.argsort(uv[:, 1])
    stride = max(1, n // max_points)
    picks = order[::stride][:max_points]
    return uv[picks].astype(np.float32)


@REGISTRY.register("fuse_views")
def _build_fuse_views(services, **params) -> FuseViewsStage:
    return FuseViewsStage(
        mode=params.get("mode", "independent"),
        main_camera=params.get("main_camera", 0),
        iou_align_threshold=params.get("iou_align_threshold", 0.3),
        occlusion_tolerance=params.get("occlusion_tolerance", 0.05),
        min_projected_points=params.get("min_projected_points", 5),
        max_seed_points=params.get("max_seed_points", 16),
        footprint_dilate=params.get("footprint_dilate", 2),
        flip_u=params.get("flip_u", False),
        get_tracker_fn=getattr(services, "get_tracker", None),
        attach_id_fn=getattr(services, "attach_aligned_id", None),
        attach_alias_fn=getattr(services, "attach_obj_id_alias", None),
    )
