"""Multi-instance, multi-view perception ablation via ``open_perception``.

Mirrors the production config (``OpenPerceptionClient._build_config``) but
parameterised so a sweep can vary every detection knob. Per-frame stateless
inference — no tracker state is touched, no ``detect()`` / ``track()`` calls.
The pipeline runs purely as ``Pipeline.run_views(ctxs)`` over fresh
``DetectionContext``s built from the LIBERO HDF5's first-frame multi-view
RGBD.

Scored against the panoptic-IoU metric in :mod:`_common`: per canonical name
the score is ``sum(matched IoUs) / max(|preds|, |GTs|)``, so excess masks
*and* missed instances both reduce the score. ``count_error`` and
``f1@0.5`` are reported as auxiliary diagnostics.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro
from _common import (
    BaseArgs,
    ViewFrame,
    ViewMask,
    run_pipeline_multi,
)
from loguru import logger

# Make the in-repo open_perception package importable without installing it
# into the root workspace (it ships with its own venv).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OP_SRC = _REPO_ROOT / "packages" / "open_perception" / "src"
if str(_OP_SRC) not in sys.path:
    sys.path.insert(0, str(_OP_SRC))

from open_perception import (  # noqa: E402
    CameraInfo,
    DetectionContext,
    InferenceOptions,
    InferenceStats,
    OpenPerception,
    PerceptionConfig,
    ReferenceBank,
)


@dataclasses.dataclass
class Args(BaseArgs):
    out_dir: str = "tmp/op_multi_instance_eval"
    sam3_model: str = "facebook/sam3"
    clip_model: str = "openai/clip-vit-base-patch16"
    dino_model: str = "facebook/dinov2-base"
    reference_hdf5: str = "datasets/all_objects.hdf5"

    # Segmentation backend. ``sam3`` is text-conditioned (uses prompts);
    # ``sam2`` runs auto mask generation (class-agnostic) and relies on the
    # downstream disambiguator + assign to bucket masks against queries.
    detector: Literal["sam3", "sam2"] = "sam3"

    # SAM3 thresholds (ignored when ``detector == "sam2"``).
    score_threshold: float = 0.05
    mask_threshold: float = 0.4

    # Filter / assign knobs.
    filter_score_threshold: float = 0.2
    assign_top_k: int = -1
    """``-1`` allows multiple detections per query (multi-instance)."""

    # Score weights for assign / filter.score / filter.overlap.
    dino_weight: float = 1.0
    clip_weight: float = 0.0

    # Disambiguator behaviour.
    dino_fallback_to_text: bool = True
    """Per request: production has it ``False`` but ``True`` is needed when
    prompts are missing from the reference bank (e.g. unseen objects)."""
    disambig_agg_mode: Literal["mean", "max"] = "max"

    # Stage on/off knobs (sweep dimensions).
    enable_filter_overlap: bool = True
    enable_filter_score: bool = True
    enable_filter_erode: bool = True
    enable_reconstruct: bool = True
    enable_fuse_views: bool = True

    # Multi-view layout.
    primary_view: int = 0


def _score_weights(args: Args) -> dict[str, float]:
    weights: dict[str, float] = {}
    if args.dino_weight > 0:
        weights["dino"] = args.dino_weight
    if args.clip_weight > 0:
        weights["clip"] = args.clip_weight
    if not weights:
        # Defensive: assign requires at least one positive weight.
        weights["dino"] = 1.0
    return weights


def _build_config(args: Args) -> PerceptionConfig:
    """Mirror ``OpenPerceptionClient._build_config`` but parameterised.

    ``tracker_backend`` is hardcoded to ``"sam2"`` to match production but
    has zero effect on this eval — we only run detection stages, never
    invoke ``detect()`` / ``track()``.
    """
    score_weights = _score_weights(args)

    order: list[str] = [args.detector, "filter.size", "disambiguate.dino", "assign"]
    if args.enable_filter_overlap:
        order.append("filter.overlap")
    if args.enable_filter_score:
        order.append("filter.score")
    if args.enable_filter_erode:
        order.append("filter.erode")
    if args.enable_reconstruct:
        order.append("reconstruct")
    if args.enable_fuse_views:
        order.append("fuse_views")

    cfg = PerceptionConfig(
        device=args.device,
        sam3_model=args.sam3_model,
        clip_model=args.clip_model,
        dino_model=args.dino_model,
        eager_load=False,
    )
    cfg.tracker_backend = "sam2"
    cfg.order(order)

    # LIBERO RGB/depth come 180°-rotated and the loader already
    # `[::-1, ::-1]`s them, so reconstruct + fuse_views need flip_u to keep
    # the secondary projection on the right pixels.
    if args.enable_reconstruct:
        cfg.update("reconstruct", flip_u=True)
    if args.enable_fuse_views:
        cfg.update(
            "fuse_views",
            mode="primary_guided",
            main_camera=args.primary_view,
            flip_u=True,
        )
    if args.detector == "sam3":
        cfg.update(
            "sam3",
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            prompts=["<object_names>"],
        )
    cfg.update(
        "disambiguate.dino",
        agg_mode=args.disambig_agg_mode,
        fallback_to_text=args.dino_fallback_to_text,
    )
    cfg.update("assign", top_k=args.assign_top_k, weights=score_weights)
    if args.enable_filter_score:
        cfg.update(
            "filter.score",
            score_weights=score_weights,
            threshold=args.filter_score_threshold,
        )
    if args.enable_filter_overlap:
        cfg.update("filter.overlap", score_weights=score_weights)
    return cfg


class OpMultiInstancePredictor:
    """Adapt :class:`OpenPerception` to :class:`MultiInstancePredictor`.

    Constructs the perception engine once; ``predict()`` builds fresh
    ``DetectionContext``s per call and runs ``_pipeline.run_views`` so no
    tracker state is ever touched. Returns every detection across every
    view (caller filters to the primary view for the metric).
    """

    name = "op_multi_instance"

    def __init__(self, args: Args) -> None:
        self.args = args
        self.missing_prompts: set[str] = set()

        self.cfg = _build_config(args)
        ref_path = Path(args.reference_hdf5)
        bank: ReferenceBank | None = None
        if ref_path.exists():
            bank = ReferenceBank.from_hdf5(
                str(ref_path), embedding_key="dino_embedding", space="dino"
            )
        else:
            logger.warning(
                f"reference bank not found at {ref_path}; running with text fallback only"
            )
        self.perc = OpenPerception(self.cfg, reference=bank)
        # Stages built once — predict() runs many times.
        self._stages = self.perc._pipeline.build_stages()
        self._reference_bank = bank
        self._opts = InferenceOptions(
            top_k_per_query=None,
            match_threshold=0.0,
            record_history=False,
            record_track_history=False,
        )

    def predict(
        self, views: list[ViewFrame], prompts: list[str]
    ) -> dict[str, list[ViewMask]]:
        bank = self._reference_bank
        if bank is not None:
            for p in prompts:
                if p not in bank and p not in self.missing_prompts:
                    if self.args.dino_fallback_to_text:
                        logger.info(
                            f"prompt '{p}' missing from reference bank — falling back to CLIP text"
                        )
                    else:
                        logger.warning(
                            f"prompt '{p}' missing from reference bank — zero vector"
                        )
                    self.missing_prompts.add(p)

        ctxs: list[DetectionContext] = []
        for v in views:
            cam = CameraInfo(
                intrinsics=v.intrinsic, extrinsics=v.extrinsic, name=v.name
            )
            ctxs.append(
                DetectionContext(
                    queries=list(prompts),
                    rgb=v.rgb,
                    depth=v.depth,
                    camera=cam,
                    options=self._opts,
                    stats=InferenceStats(),
                    view_idx=v.view_idx,
                )
            )

        self.perc._pipeline.run_views(ctxs, stages=self._stages)

        # Assemble per-prompt detection lists across all views.
        out: dict[str, list[ViewMask]] = {p: [] for p in prompts}
        primary_idx = self.args.primary_view
        # First pass: collect all detections + their (view_idx, instance_id, query).
        all_dets: list[tuple[str, int, int | None, np.ndarray, float]] = []
        primary_ids_by_query: dict[str, set[int]] = {p: set() for p in prompts}
        for ctx in ctxs:
            for query, dets in ctx.detections.items():
                if query not in out:
                    # disambiguate.dino can rewrite query strings via fallback;
                    # if the resolved key isn't a known prompt, skip.
                    continue
                for det in dets:
                    if det.mask is None:
                        continue
                    iid = det.instance_id
                    all_dets.append(
                        (
                            query,
                            ctx.view_idx,
                            iid,
                            det.mask.astype(bool),
                            float(det.score),
                        )
                    )
                    if ctx.view_idx == primary_idx and iid is not None:
                        primary_ids_by_query[query].add(iid)

        # Second pass: drop secondary-view detections whose instance_id doesn't
        # appear in the primary view (mirrors production OpenPerceptionClient).
        for query, view_idx, iid, mask, score in all_dets:
            primary_ids = primary_ids_by_query.get(query, set())
            if (
                view_idx != primary_idx
                and primary_ids
                and (iid is None or iid not in primary_ids)
            ):
                continue
            out[query].append(ViewMask(view_idx=view_idx, mask=mask, score=score))
        return out


def main(args: Args) -> None:
    predictor = OpMultiInstancePredictor(args)
    logger.info(predictor.cfg.summary)

    bank = predictor._reference_bank
    n_reference_objects = len(bank.embeddings) if bank is not None else 0

    tag = (
        f"op_multi_s{args.score_threshold:.2f}_m{args.mask_threshold:.2f}"
        f"_t{args.filter_score_threshold:.2f}_k{args.assign_top_k}"
    )
    extra_meta: dict[str, Any] = {
        "backend": "open_perception",
        "detector": args.detector,
        "tracker_backend": "sam2",
        "sam3_model": args.sam3_model,
        "clip_model": args.clip_model,
        "dino_model": args.dino_model,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "filter_score_threshold": args.filter_score_threshold,
        "assign_top_k": args.assign_top_k,
        "dino_weight": args.dino_weight,
        "clip_weight": args.clip_weight,
        "dino_fallback_to_text": args.dino_fallback_to_text,
        "disambig_agg_mode": args.disambig_agg_mode,
        "enable_filter_overlap": args.enable_filter_overlap,
        "enable_filter_score": args.enable_filter_score,
        "enable_filter_erode": args.enable_filter_erode,
        "enable_reconstruct": args.enable_reconstruct,
        "enable_fuse_views": args.enable_fuse_views,
        "reference_hdf5": args.reference_hdf5,
        "n_reference_objects": n_reference_objects,
        "primary_view": args.primary_view,
    }
    run_pipeline_multi(
        args,
        predictor,
        tag=tag,
        extra_meta=extra_meta,
        primary_view=args.primary_view,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
