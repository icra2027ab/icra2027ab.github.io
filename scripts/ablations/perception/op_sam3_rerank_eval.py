"""SAM3 + CLIP/DINO re-rank ablation via the ``open_perception`` package.

Functionally equivalent to ``sam3_clip_rerank_eval.py`` — same SAM3 prompts,
thresholds, reference HDF5, Hungarian 1-to-1 assignment, CSV/plot schema —
but routes every frame through the reusable ``open_perception`` pipeline so
the package itself is exercised end-to-end on real data.

Exported schema (``tag``, summary JSON, failure figures) matches the legacy
eval so results are directly comparable against the non-package baseline.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tyro
from _common import BaseArgs, run_pipeline
from loguru import logger

# Make the in-repo open_perception package importable without installing it
# into the root workspace (it ships with its own venv).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OP_SRC = _REPO_ROOT / "packages" / "open_perception" / "src"
if str(_OP_SRC) not in sys.path:
    sys.path.insert(0, str(_OP_SRC))

from open_perception import (  # noqa: E402
    DetectionContext,
    InferenceOptions,
    InferenceStats,
    OpenPerception,
    PerceptionConfig,
    ReferenceBank,
)

# Reuse the failure-sample figure renderer — same output layout as the legacy eval.
from sam3_clip_rerank_eval import _save_clip_failure_figure  # noqa: E402


@dataclasses.dataclass
class Args(BaseArgs):
    out_dir: str = "tmp/op_sam3_rerank_eval"
    sam3_model: str = "facebook/sam3"
    clip_model: str = "openai/clip-vit-base-patch16"
    dino_model: str = "facebook/dinov2-base"
    sam3_prompt: list[str] = dataclasses.field(
        default_factory=lambda: ["object", "<object_name>"]
    )
    score_threshold: float = 0.15
    mask_threshold: float = 0.4
    min_mask_area: int = 10
    max_instances: int = 40
    bbox_pad: int = 4
    clip_mask_bg: bool = True
    rerank_mode: Literal["image", "text", "dino"] = "image"
    reference_hdf5: str = "datasets/all_objects.hdf5"
    fallback_to_text: bool = True
    agg_mode: Literal["mean", "max"] = "max"
    filter_overlap: bool = False
    """Enable open_perception's mask-IoU NMS. Off by default to match the
    original ``sam3_clip_rerank_eval.py`` which only does area/top-K filtering."""


def _build_config(args: Args) -> PerceptionConfig:
    """Declare the pipeline shape via the new ``order`` + ``update`` API.

    Legacy eval behaviour:
      sam3 → (filter.overlap)? → filter.size → disambiguate.{clip|dino} → assign
    Hungarian 1-to-1 with raw-cosine combination (no per-column L2 rescaling).
    """
    cfg = PerceptionConfig(
        device=args.device,
        sam3_model=args.sam3_model,
        clip_model=args.clip_model,
        dino_model=args.dino_model,
        sam3_prompts=list(args.sam3_prompt),
        sam3_score_threshold=args.score_threshold,
        sam3_mask_threshold=args.mask_threshold,
        default_min_area=args.min_mask_area,
        default_max_instances=args.max_instances,
        default_bbox_pad=args.bbox_pad,
        default_clip_mask_bg=args.clip_mask_bg,
        # The ablation never tracks — skip eager-loading the SAM3 video tracker.
        eager_load=False,
    )

    disambig = (
        "disambiguate.dino" if args.rerank_mode == "dino" else "disambiguate.clip"
    )
    order: list[str] = ["sam3"]
    if args.filter_overlap:
        order.append("filter.overlap")
    order += ["filter.size", disambig, "assign"]
    cfg.order(order)

    cfg.update(disambig, agg_mode=args.agg_mode, fallback_to_text=args.fallback_to_text)
    cfg.update("assign", top_1_hungarian=True, normalize_sources=False)
    return cfg


def _build_reference(args: Args) -> ReferenceBank | None:
    if args.rerank_mode == "image":
        return ReferenceBank.from_hdf5(
            args.reference_hdf5, embedding_key="clip_embedding", space="clip"
        )
    if args.rerank_mode == "dino":
        return ReferenceBank.from_hdf5(
            args.reference_hdf5, embedding_key="dino_embedding", space="dino"
        )
    return None


class OpSam3RerankPredictor:
    """Adapt ``OpenPerception`` to the ``_common.Predictor`` protocol.

    Each ``predict()`` builds a fresh ``DetectionContext`` and runs the
    configured stages via the underlying ``OpenPerception`` instance. Debug
    fields mirror ``Sam3ClipRerankPredictor`` so the shared failure-sample
    figure renderer works unchanged.
    """

    def __init__(self, args: Args) -> None:
        self.args = args
        self.name = (
            "op_sam3_dino_rerank"
            if args.rerank_mode == "dino"
            else "op_sam3_clip_rerank"
        )
        self.missing_prompts: set[str] = set()
        self.last_debug: dict[str, Any] = {}

        self.cfg = _build_config(args)
        self.perc = OpenPerception(self.cfg, reference=_build_reference(args))
        # Stages built once — predict() runs thousands of times per eval.
        self._stages = self.perc._pipeline.build_stages()

    @property
    def _reference_bank(self) -> ReferenceBank | None:
        space = "dino" if self.args.rerank_mode == "dino" else "clip"
        return self.perc._reference_banks.get(space)

    @property
    def reference_images(self) -> dict[str, list[np.ndarray]]:
        bank = self._reference_bank
        return bank.images if bank is not None else {}

    def predict(self, rgb: np.ndarray, prompts: list[str]) -> dict[str, np.ndarray]:
        bank = self._reference_bank
        for p in prompts:
            if bank is not None and p not in bank and p not in self.missing_prompts:
                if self.args.fallback_to_text:
                    logger.info(
                        f"prompt '{p}' missing from reference bank — falling back to CLIP text"
                    )
                else:
                    logger.warning(
                        f"prompt '{p}' missing from reference bank — zero vector"
                    )
                self.missing_prompts.add(p)

        ctx = DetectionContext(
            queries=list(prompts),
            rgb=rgb,
            depth=None,
            camera=None,
            # match_threshold/top_k irrelevant under Hungarian 1-to-1;
            # record_history=False skips per-stage snapshots we don't use.
            options=InferenceOptions(match_threshold=0.0, record_history=False),
            stats=InferenceStats(),
        )
        self.perc._pipeline.run(ctx, stages=self._stages)

        H, W = rgb.shape[:2]
        out = {p: np.zeros((H, W), dtype=bool) for p in prompts}
        for p, dets in ctx.detections.items():
            if dets:
                out[p] = dets[0].mask

        self._snapshot_debug(ctx, rgb, prompts)
        return out

    def _snapshot_debug(
        self, ctx: DetectionContext, rgb: np.ndarray, prompts: list[str]
    ) -> None:
        """Populate ``last_debug`` in the schema the shared failure-figure expects.

        The primary similarity matrix is the sum of every non-zero score source
        (dino + clip_text_fallback stitched, or clip alone) — which is what
        Hungarian saw in ``AssignStage`` with ``normalize_sources=False``.
        """
        H, W = rgb.shape[:2]
        N = ctx.instances.shape[0] if ctx.instances is not None else 0
        P = len(prompts)

        if ctx.scores:
            sim = np.zeros((N, P), dtype=np.float32)
            for mat in ctx.scores.values():
                if mat.shape == sim.shape:
                    sim = sim + mat
        else:
            sim = np.zeros((N, P), dtype=np.float32)

        bank = self._reference_bank
        sources: list[str] = []
        for p in prompts:
            if bank is not None and p in bank:
                sources.append("dino" if self.args.rerank_mode == "dino" else "image")
            elif self.args.rerank_mode == "text" or self.args.fallback_to_text:
                sources.append("text")
            else:
                sources.append("missing")

        assignment: dict[str, int] = {}
        for p, dets in ctx.detections.items():
            if dets and dets[0].source_index is not None:
                assignment[p] = int(dets[0].source_index)

        instances = (
            ctx.instances.copy()
            if ctx.instances is not None
            else np.zeros((0, H, W), dtype=bool)
        )
        self.last_debug = {
            "rgb": rgb,
            "instances": instances,
            "crops": [c.copy() for c in ctx.crops],
            "sim": sim,
            "prompts": list(prompts),
            "prompt_sources": sources,
            "assignment": assignment,
        }


def _failure_extra(predictor: OpSam3RerankPredictor, prompt: str) -> dict[str, Any]:
    """Snapshot per-prompt debug state — matches the legacy eval's schema."""
    d = predictor.last_debug
    prompts = d.get("prompts", [])
    if prompt not in prompts:
        return {}
    col = prompts.index(prompt)
    sim = d["sim"]
    scores_for_prompt = (
        sim[:, col].copy() if sim.size else np.zeros((0,), dtype=np.float32)
    )
    sources = d.get("prompt_sources", ["text"] * len(prompts))
    ref_imgs = predictor.reference_images.get(prompt, [])
    return {
        "instances": d["instances"].copy(),
        "crops": [c.copy() for c in d["crops"]],
        "scores_for_prompt": scores_for_prompt,
        "chosen_idx": d["assignment"].get(prompt),
        "query_source": sources[col],
        "rerank_mode": predictor.args.rerank_mode,
        "reference_images": [img.copy() for img in ref_imgs],
    }


def main(args: Args) -> None:
    predictor = OpSam3RerankPredictor(args)
    logger.info(predictor.cfg.summary)

    bank = predictor._reference_bank
    n_reference_objects = len(bank.embeddings) if bank is not None else 0

    tag = (
        f"op_{args.rerank_mode}_s{args.score_threshold:.2f}_m{args.mask_threshold:.2f}"
    )
    extra_meta: dict[str, Any] = {
        "backend": "open_perception",
        "sam3_model": args.sam3_model,
        "clip_model": args.clip_model,
        "dino_model": args.dino_model,
        "sam3_prompt": args.sam3_prompt,
        "score_threshold": args.score_threshold,
        "mask_threshold": args.mask_threshold,
        "min_mask_area": args.min_mask_area,
        "max_instances": args.max_instances,
        "bbox_pad": args.bbox_pad,
        "clip_mask_bg": args.clip_mask_bg,
        "rerank_mode": args.rerank_mode,
        "fallback_to_text": args.fallback_to_text,
        "agg_mode": args.agg_mode,
        "reference_hdf5": args.reference_hdf5,
        "n_reference_objects": n_reference_objects,
        "filter_overlap": args.filter_overlap,
    }
    run_pipeline(
        args,
        predictor,
        tag=tag,
        extra_meta=extra_meta,
        failure_figure_fn=_save_clip_failure_figure,
        failure_extra_fn=_failure_extra,
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
