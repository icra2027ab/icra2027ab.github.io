"""SAM3 (2-D) segmentation quality vs GT masks on LIBERO HDF5 datasets.

For each rollout's first frame, query ``facebook/sam3`` with the GT object names
and compare against the GT segmentation. Reports per-object IoU, IoU@50,
IoU@75, and an inter-class confusion matrix (leakage of predicted mask over
other GT objects).

See ``PerceptionMode.REAL_PERCEPTION`` in
``src/policy_abstraction/evaluation/abstraction.py`` for where a SAM3 backend
would plug in once numbers look good.
"""

from __future__ import annotations

import csv
import dataclasses
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from loguru import logger
from rich.console import Console
from rich.table import Table

from policy_abstraction.utils.segmentation import get_robot_segment_ids


@dataclasses.dataclass
class Args:
    dataset_globs: list[str] = dataclasses.field(
        default_factory=lambda: [
            "datasets/libero_object_new/*.hdf5",
            "datasets/libero_10_new/*.hdf5",
        ]
    )
    """Glob patterns for HDF5 files. Pass multiple to cover several suites."""
    out_dir: str = "tmp/sam3_seg_eval"
    max_demos_per_file: int | None = None
    max_files: int | None = None
    model_id: str = "facebook/sam3"
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    device: str = "cuda"
    normalize_names: bool = True
    """Strip trailing ``_\\d+`` and replace ``_`` with space before prompting SAM3."""
    sweep_score_thresholds: list[float] = dataclasses.field(default_factory=list)
    """If non-empty, evaluate every combination; overrides ``score_threshold``."""
    sweep_mask_thresholds: list[float] = dataclasses.field(default_factory=list)
    """If non-empty, evaluate every combination; overrides ``mask_threshold``."""
    save_failure_samples: bool = False
    """Save side-by-side pred/GT images for low-IoU prompts. Single-combo only."""
    failure_iou_threshold: float = 0.3
    """Only collect as failure samples prompts with IoU below this."""
    n_samples_per_object: int = 5
    """Max samples saved per object."""
    n_worst_objects: int = 15
    """Keep samples for this many worst-IoU objects (by mean IoU)."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _suite_name(hdf5_path: str) -> str:
    """Use the immediate parent directory as the dataset-suite label."""
    return Path(hdf5_path).parent.name


def _expand_globs(globs: list[str]) -> list[tuple[str, str]]:
    """Return ``[(suite, path), ...]`` deduped and sorted, ``.hdf5`` only."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for pattern in globs:
        for path in sorted(glob.glob(pattern)):
            if not path.endswith(".hdf5") or path in seen:
                continue
            seen.add(path)
            out.append((_suite_name(path), path))
    return out


_DIGIT_RE = re.compile(r"\d+")


def _normalize_name(name: str) -> str:
    """Strip all digits and underscores — see ``_common.normalize_name``."""
    return " ".join(_DIGIT_RE.sub("", name).replace("_", " ").split())


def _load_first_frame(
    hdf5_path: str, demo_key: str, id_to_name: dict[int, str]
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, str]] | None:
    """Load RGB + per-object binary masks for the first frame of one demo.

    Returns ``(rgb[H,W,3] uint8, {seg_id: [H,W] bool}, {seg_id: name})`` or
    ``None`` if the demo has no non-robot objects in the first frame.
    """
    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo_key]["obs"]
        rgb = np.ascontiguousarray(obs["agentview_rgb"][0][::-1, ::-1, :])
        seg = np.asarray(obs["agentview_segmentation"][0][::-1, ::-1])
    if seg.ndim == 3:
        seg = seg[..., 0]

    all_ids = {int(v) for v in np.unique(seg) if v != 0}
    robot_ids = get_robot_segment_ids(id_to_name, all_seg_ids=all_ids)
    object_ids = sorted(all_ids - robot_ids)
    if not object_ids:
        return None

    masks = {sid: (seg == sid) for sid in object_ids}
    names = {sid: id_to_name[sid] for sid in object_ids if sid in id_to_name}
    if not names:
        return None
    return rgb, masks, names


def _iter_frames(hdf5_path: str, max_demos: int | None):
    """Yield ``(demo_key, rgb, gt_masks, id_to_name)`` for each demo's first frame."""
    with h5py.File(hdf5_path, "r") as f:
        seg_map_json = f["data"].attrs.get("segmentation_name_mapping", None)
        demo_keys = list(f["data"].keys())
    id_to_name: dict[int, str] = {}
    if seg_map_json:
        id_to_name = {int(k): v for k, v in json.loads(seg_map_json).items()}

    if max_demos is not None:
        demo_keys = demo_keys[:max_demos]

    for dk in demo_keys:
        result = _load_first_frame(hdf5_path, dk, id_to_name)
        if result is None:
            logger.debug(f"{hdf5_path}:{dk} has no objects, skipping")
            continue
        rgb, gt_masks, names = result
        yield dk, rgb, gt_masks, names


# ---------------------------------------------------------------------------
# SAM3 wrapper
# ---------------------------------------------------------------------------


class Sam3Predictor:
    def __init__(self, model_id: str, device: str):
        import torch
        from transformers import Sam3Model, Sam3Processor

        self._torch = torch
        self.device = (
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu"
        )
        if self.device != device:
            logger.warning(
                f"requested device={device} unavailable, falling back to {self.device}"
            )
        logger.info(f"loading {model_id} on {self.device}")
        self.processor = Sam3Processor.from_pretrained(model_id)
        self.model = Sam3Model.from_pretrained(model_id).to(self.device).eval()

    def run(
        self, rgb: np.ndarray, prompts: list[str]
    ) -> tuple[list[Any], list[list[int]], int, int]:
        """Forward SAM3 once per prompt. Returns raw outputs for later post-proc.

        ``outputs_per_prompt[i]`` is the transformers model output for ``prompts[i]``;
        it can be post-processed at any ``(score_threshold, mask_threshold)``.
        """
        from PIL import Image

        torch = self._torch
        H, W = rgb.shape[:2]
        image = Image.fromarray(rgb)
        img_inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_embeds = self.model.get_vision_features(
                pixel_values=img_inputs.pixel_values
            )
        target_sizes = img_inputs.get("original_sizes")
        target_sizes = target_sizes.tolist() if target_sizes is not None else [[H, W]]

        outs: list[Any] = []
        for prompt in prompts:
            text_inputs = self.processor(text=prompt, return_tensors="pt").to(
                self.device
            )
            with torch.no_grad():
                outputs = self.model(vision_embeds=vision_embeds, **text_inputs)
            outs.append(outputs)
        return outs, target_sizes, H, W

    def finalize(
        self,
        outputs: Any,
        target_sizes: list[list[int]],
        H: int,
        W: int,
        score_threshold: float,
        mask_threshold: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Post-process raw outputs.

        Returns ``(merged[H,W] bool, instances[N,H,W] bool, scores[N] float)``.
        ``merged`` is the single highest-scoring instance; ``instances`` and
        ``scores`` are sorted by descending score (so ``instances[0]`` is the
        same mask as ``merged``).
        """
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )[0]
        masks = results.get("masks")
        scores = results.get("scores")
        if masks is None or len(masks) == 0:
            return (
                np.zeros((H, W), dtype=bool),
                np.zeros((0, H, W), dtype=bool),
                np.zeros((0,), dtype=np.float32),
            )
        if hasattr(masks, "cpu"):
            inst = masks.cpu().numpy().astype(bool)
        else:
            inst = np.stack([np.asarray(m).astype(bool) for m in masks], axis=0)
        if scores is not None and hasattr(scores, "cpu"):
            score_arr = scores.cpu().numpy().astype(np.float32)
        elif scores is not None:
            score_arr = np.asarray(scores, dtype=np.float32)
        else:
            score_arr = np.zeros((inst.shape[0],), dtype=np.float32)
        order = np.argsort(-score_arr)
        inst = inst[order]
        score_arr = score_arr[order]
        return inst[0].copy(), inst, score_arr


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    return float(inter / union) if union else float("nan")


@dataclasses.dataclass
class FrameRecord:
    suite: str
    hdf5: str
    demo: str
    # canonical name -> iou vs its own gt
    per_object_iou: dict[str, float]
    # (pred_name, gt_name) -> intersection / gt_area (only objects present in this frame)
    confusion_cells: dict[tuple[str, str], float]
    gt_names: list[str]
    empty_predictions: list[str]


@dataclasses.dataclass
class FailureSample:
    """One frame's worth of failure context for a single prompt."""

    suite: str
    demo: str
    prompt: str
    gt_name: str
    iou: float
    n_blobs: int
    rgb: np.ndarray
    instances: np.ndarray  # [N, H, W] bool
    scores: np.ndarray  # [N] float
    gt_mask: np.ndarray  # [H, W] bool


def _overlay_masks(
    rgb: np.ndarray, masks: np.ndarray, alpha: float = 0.55
) -> np.ndarray:
    """Tint each boolean mask with a distinct color from tab10."""
    if masks.shape[0] == 0:
        return rgb.copy()
    cmap = plt.get_cmap("tab10")
    out = rgb.astype(np.float32).copy()
    for i in range(masks.shape[0]):
        color = np.array(cmap(i % 10)[:3]) * 255.0
        m = masks[i]
        out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def _save_failure_figure(out_path: Path, s: FailureSample) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6))
    pred_overlay = _overlay_masks(s.rgb, s.instances)
    axes[0].imshow(pred_overlay)
    score_txt = ", ".join(f"{v:.2f}" for v in s.scores.tolist()) if s.n_blobs else "—"
    axes[0].set_title(
        f"pred '{s.prompt}'  IoU={s.iou:.3f}  blobs={s.n_blobs} (top-1 used)\n"
        f"scores=[{score_txt}]",
        fontsize=9,
    )
    axes[0].axis("off")

    gt_rgb = _overlay_masks(s.rgb, s.gt_mask[None, ...])
    axes[1].imshow(gt_rgb)
    gt_area = int(s.gt_mask.sum())
    axes[1].set_title(f"GT '{s.gt_name}'  area={gt_area}px", fontsize=9)
    axes[1].axis("off")
    fig.suptitle(f"{s.suite} :: {s.demo}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def evaluate_frame(
    pred_masks: dict[str, np.ndarray],  # keyed by normalized prompt
    gt_masks: dict[int, np.ndarray],
    id_to_name: dict[int, str],
    prompt_to_canonical: dict[str, str],  # normalized prompt -> canonical name
    id_to_canonical: dict[int, str],
) -> tuple[dict[str, float], dict[tuple[str, str], float], list[str]]:
    per_iou: dict[str, float] = {}
    empty: list[str] = []

    # Diagonal IoU
    for prompt, pred in pred_masks.items():
        canon = prompt_to_canonical[prompt]
        # find matching gt_id
        gt_id = next((sid for sid, n in id_to_canonical.items() if n == canon), None)
        if gt_id is None:
            continue
        gt = gt_masks[gt_id]
        if pred.sum() == 0:
            empty.append(canon)
        per_iou[canon] = iou(pred, gt)

    # Confusion: intersection(pred_i, gt_j) / gt_j.sum()
    confusion: dict[tuple[str, str], float] = {}
    gt_areas = {sid: int(m.sum()) for sid, m in gt_masks.items()}
    for prompt, pred in pred_masks.items():
        pred_canon = prompt_to_canonical[prompt]
        for gt_id, gt in gt_masks.items():
            gt_canon = id_to_canonical[gt_id]
            area = gt_areas[gt_id]
            if area == 0:
                continue
            inter = int(np.logical_and(pred, gt).sum())
            confusion[(pred_canon, gt_canon)] = inter / area
    return per_iou, confusion, empty


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _aggregate(records: list[FrameRecord]) -> dict[str, Any]:
    """Aggregate metrics across records (overall, per-suite, per-object)."""
    ious_by_obj: dict[str, list[float]] = defaultdict(list)
    confusion_sum: dict[tuple[str, str], float] = defaultdict(float)
    confusion_n: dict[tuple[str, str], int] = defaultdict(int)

    for r in records:
        for name, v in r.per_object_iou.items():
            if not np.isnan(v):
                ious_by_obj[name].append(v)
        for key, v in r.confusion_cells.items():
            confusion_sum[key] += v
            confusion_n[key] += 1

    all_ious = [v for lst in ious_by_obj.values() for v in lst]
    overall = {
        "n_frames": len(records),
        "n_evaluations": len(all_ious),
        "mean_iou": float(np.mean(all_ious)) if all_ious else float("nan"),
        "iou@50": float(np.mean([v >= 0.5 for v in all_ious]))
        if all_ious
        else float("nan"),
        "iou@75": float(np.mean([v >= 0.75 for v in all_ious]))
        if all_ious
        else float("nan"),
    }
    per_object = {
        name: {
            "mean_iou": float(np.mean(vs)),
            "iou@50": float(np.mean([v >= 0.5 for v in vs])),
            "iou@75": float(np.mean([v >= 0.75 for v in vs])),
            "n": len(vs),
        }
        for name, vs in sorted(ious_by_obj.items())
    }
    confusion_mean = {k: (confusion_sum[k] / confusion_n[k]) for k in confusion_sum}
    return {
        "overall": overall,
        "per_object": per_object,
        "confusion_mean": confusion_mean,
    }


def _plot_confusion(
    confusion_mean: dict[tuple[str, str], float],
    out_path: Path,
    title: str,
) -> None:
    if not confusion_mean:
        return
    preds = sorted({p for p, _ in confusion_mean})
    gts = sorted({g for _, g in confusion_mean})
    M = np.zeros((len(preds), len(gts)), dtype=np.float32)
    for (p, g), v in confusion_mean.items():
        M[preds.index(p), gts.index(g)] = v

    fig, ax = plt.subplots(
        figsize=(max(6, 0.45 * len(gts) + 4), max(5, 0.45 * len(preds) + 3))
    )
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(gts)))
    ax.set_xticklabels(gts, rotation=45, ha="right")
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels(preds)
    ax.set_xlabel("GT object")
    ax.set_ylabel("Predicted prompt")
    ax.set_title(title)
    for i in range(len(preds)):
        for j in range(len(gts)):
            v = M[i, j]
            if v >= 0.01:
                ax.text(
                    j,
                    i,
                    f"{v:.2f}",
                    ha="center",
                    va="center",
                    color="white" if v < 0.6 else "black",
                    fontsize=7,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="inter(pred, gt) / gt_area")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"wrote {out_path}")


def _write_csv(per_object: dict[str, dict[str, float]], out_path: Path) -> None:
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["object", "mean_iou", "iou@50", "iou@75", "n_frames"])
        for name, m in per_object.items():
            w.writerow(
                [
                    name,
                    f"{m['mean_iou']:.4f}",
                    f"{m['iou@50']:.4f}",
                    f"{m['iou@75']:.4f}",
                    m["n"],
                ]
            )
    logger.info(f"wrote {out_path}")


def _print_tables(console: Console, agg: dict[str, Any], title: str) -> None:
    t = Table(title=f"{title} — overall", show_header=True, header_style="bold")
    for col in ["metric", "value"]:
        t.add_column(col)
    o = agg["overall"]
    for k in ["n_frames", "n_evaluations", "mean_iou", "iou@50", "iou@75"]:
        v = o[k]
        t.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(t)

    po = agg["per_object"]
    if po:
        t2 = Table(
            title=f"{title} — per object (sorted by mean IoU)",
            show_header=True,
            header_style="bold",
        )
        for col in ["object", "mean_iou", "iou@50", "iou@75", "n"]:
            t2.add_column(col)
        for name, m in sorted(po.items(), key=lambda kv: -kv[1]["mean_iou"]):
            t2.add_row(
                name,
                f"{m['mean_iou']:.3f}",
                f"{m['iou@50']:.2f}",
                f"{m['iou@75']:.2f}",
                str(m["n"]),
            )
        console.print(t2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _conf_to_json(d: dict[tuple[str, str], float]) -> dict[str, float]:
    return {f"{p}||{g}": v for (p, g), v in d.items()}


def _write_combo_outputs(
    out_dir: Path,
    tag: str,
    all_records: list[FrameRecord],
    records_by_suite: dict[str, list[FrameRecord]],
    console: Console,
    meta: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate + dump outputs for one threshold combo (or the single-combo default)."""
    summary: dict[str, Any] = {"overall": {}, "per_suite": {}}
    overall_agg = _aggregate(all_records)
    summary["overall"] = overall_agg
    _print_tables(console, overall_agg, f"ALL [{tag}]")
    _plot_confusion(
        overall_agg["confusion_mean"],
        out_dir / f"confusion_overall_{tag}.png",
        f"Confusion — overall [{tag}]",
    )
    _write_csv(overall_agg["per_object"], out_dir / f"per_object_iou_overall_{tag}.csv")

    for suite, recs in records_by_suite.items():
        agg = _aggregate(recs)
        summary["per_suite"][suite] = agg
        _print_tables(console, agg, f"{suite} [{tag}]")
        _plot_confusion(
            agg["confusion_mean"],
            out_dir / f"confusion_{suite}_{tag}.png",
            f"Confusion — {suite} [{tag}]",
        )
        _write_csv(agg["per_object"], out_dir / f"per_object_iou_{suite}_{tag}.csv")

    summary["meta"] = meta
    summary["overall"]["confusion_mean"] = _conf_to_json(overall_agg["confusion_mean"])
    for suite in summary["per_suite"]:
        summary["per_suite"][suite]["confusion_mean"] = _conf_to_json(
            summary["per_suite"][suite]["confusion_mean"]
        )
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote {out_dir / f'summary_{tag}.json'}")
    return summary


def _tag(score_thr: float, mask_thr: float) -> str:
    return f"s{score_thr:.2f}_m{mask_thr:.2f}"


def main(args: Args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _expand_globs(args.dataset_globs)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        logger.error(f"no HDF5 files matched {args.dataset_globs}")
        return
    logger.info(f"found {len(files)} files across {len({s for s, _ in files})} suites")

    score_grid = args.sweep_score_thresholds or [args.score_threshold]
    mask_grid = args.sweep_mask_thresholds or [args.mask_threshold]
    combos = [(s, m) for s in score_grid for m in mask_grid]
    logger.info(f"sweep: {len(combos)} combos score={score_grid} mask={mask_grid}")

    predictor = Sam3Predictor(args.model_id, args.device)

    # Per-combo accumulators
    combo_records: dict[tuple[float, float], list[FrameRecord]] = {
        c: [] for c in combos
    }
    combo_records_by_suite: dict[tuple[float, float], dict[str, list[FrameRecord]]] = {
        c: defaultdict(list) for c in combos
    }
    n_preds = 0
    combo_empty: dict[tuple[float, float], int] = dict.fromkeys(combos, 0)

    is_sweep = len(combos) > 1
    collect_samples = args.save_failure_samples and not is_sweep
    if args.save_failure_samples and is_sweep:
        logger.warning(
            "--save-failure-samples ignored when sweeping; run single-combo to save."
        )
    # Sample collection uses the single active combo (when not sweeping).
    sample_combo = combos[0] if collect_samples else None
    samples_by_object: dict[str, list[FailureSample]] = defaultdict(list)

    n_frames_processed = 0
    for suite, hdf5_path in files:
        logger.info(f"[{suite}] {os.path.basename(hdf5_path)}")
        for demo_key, rgb, gt_masks, id_to_name in _iter_frames(
            hdf5_path, args.max_demos_per_file
        ):
            id_to_canonical = (
                {sid: _normalize_name(n) for sid, n in id_to_name.items()}
                if args.normalize_names
                else {sid: n for sid, n in id_to_name.items()}
            )
            prompt_to_canonical = {canon: canon for canon in id_to_canonical.values()}
            prompts = sorted(prompt_to_canonical)

            # One SAM3 forward per prompt, reused across all threshold combos.
            raw_outs, target_sizes, H, W = predictor.run(rgb, prompts)
            n_preds += len(prompts)
            n_frames_processed += 1

            for score_thr, mask_thr in combos:
                pred_masks: dict[str, np.ndarray] = {}
                instances_per_prompt: dict[str, np.ndarray] = {}
                scores_per_prompt: dict[str, np.ndarray] = {}
                for prompt, outputs in zip(prompts, raw_outs, strict=True):
                    merged, inst, sc = predictor.finalize(
                        outputs, target_sizes, H, W, score_thr, mask_thr
                    )
                    pred_masks[prompt] = merged
                    instances_per_prompt[prompt] = inst
                    scores_per_prompt[prompt] = sc
                per_iou, conf, empty = evaluate_frame(
                    pred_masks,
                    gt_masks,
                    id_to_name,
                    prompt_to_canonical,
                    id_to_canonical,
                )
                combo_empty[(score_thr, mask_thr)] += len(empty)
                rec = FrameRecord(
                    suite=suite,
                    hdf5=hdf5_path,
                    demo=demo_key,
                    per_object_iou=per_iou,
                    confusion_cells=conf,
                    gt_names=sorted(set(id_to_canonical.values())),
                    empty_predictions=empty,
                )
                combo_records[(score_thr, mask_thr)].append(rec)
                combo_records_by_suite[(score_thr, mask_thr)][suite].append(rec)

                # Sample collection — only for the designated combo.
                if sample_combo is not None and (score_thr, mask_thr) == sample_combo:
                    for prompt, iv in per_iou.items():
                        if np.isnan(iv) or iv >= args.failure_iou_threshold:
                            continue
                        if len(samples_by_object[prompt]) >= args.n_samples_per_object:
                            continue
                        gt_id = next(
                            (sid for sid, n in id_to_canonical.items() if n == prompt),
                            None,
                        )
                        if gt_id is None:
                            continue
                        samples_by_object[prompt].append(
                            FailureSample(
                                suite=suite,
                                demo=demo_key,
                                prompt=prompt,
                                gt_name=id_to_name[gt_id],
                                iou=iv,
                                n_blobs=int(instances_per_prompt[prompt].shape[0]),
                                rgb=rgb.copy(),
                                instances=instances_per_prompt[prompt].copy(),
                                scores=scores_per_prompt[prompt].copy(),
                                gt_mask=gt_masks[gt_id].copy(),
                            )
                        )

    if n_frames_processed == 0:
        logger.error("no frames evaluated")
        return

    logger.info(f"done: {n_frames_processed} frames, {n_preds} prompts per combo")

    console = Console()

    # Sweep-summary ranking table (even in the single-combo case, this is a nice header).
    combo_summaries: dict[tuple[float, float], dict[str, float]] = {}
    for (s, m), recs in combo_records.items():
        agg = _aggregate(recs)["overall"]
        combo_summaries[(s, m)] = {
            "mean_iou": agg["mean_iou"],
            "iou@50": agg["iou@50"],
            "iou@75": agg["iou@75"],
            "empty_frac": combo_empty[(s, m)] / max(n_preds, 1),
        }

    ranking = sorted(
        combo_summaries.items(),
        key=lambda kv: (
            -(kv[1]["mean_iou"] if not np.isnan(kv[1]["mean_iou"]) else -1),
            -kv[1]["iou@50"],
        ),
    )

    if is_sweep:
        rank_table = Table(
            title="Threshold sweep (ranked by mean IoU)",
            show_header=True,
            header_style="bold",
        )
        for col in [
            "rank",
            "score_thr",
            "mask_thr",
            "mean_iou",
            "iou@50",
            "iou@75",
            "empty_frac",
        ]:
            rank_table.add_column(col)
        for i, ((s, m), metrics) in enumerate(ranking, start=1):
            rank_table.add_row(
                str(i),
                f"{s:.2f}",
                f"{m:.2f}",
                f"{metrics['mean_iou']:.4f}",
                f"{metrics['iou@50']:.3f}",
                f"{metrics['iou@75']:.3f}",
                f"{metrics['empty_frac']:.3f}",
            )
        console.print(rank_table)

    # Detailed outputs for each combo
    meta_base: dict[str, Any] = {
        "dataset_globs": args.dataset_globs,
        "model_id": args.model_id,
        "normalize_names": args.normalize_names,
        "n_files": len(files),
        "n_frames": n_frames_processed,
        "n_prompts_per_combo": n_preds,
    }
    all_combo_summaries: dict[str, Any] = {}
    for s, m in combos:
        tag = _tag(s, m)
        meta = {
            **meta_base,
            "score_threshold": s,
            "mask_threshold": m,
            "n_empty": combo_empty[(s, m)],
        }
        summary = _write_combo_outputs(
            out_dir,
            tag,
            combo_records[(s, m)],
            combo_records_by_suite[(s, m)],
            console,
            meta,
        )
        all_combo_summaries[tag] = {
            "score_threshold": s,
            "mask_threshold": m,
            "overall": summary["overall"],
            "per_suite_overall": {
                suite: summary["per_suite"][suite]["overall"]
                for suite in summary["per_suite"]
            },
        }

    (out_dir / "sweep_index.json").write_text(
        json.dumps(
            {
                "combos": [
                    {"score_threshold": s, "mask_threshold": m} for s, m in combos
                ],
                "ranking": [
                    {
                        "score_threshold": s,
                        "mask_threshold": m,
                        **metrics,
                    }
                    for (s, m), metrics in ranking
                ],
                "best": {
                    "score_threshold": ranking[0][0][0],
                    "mask_threshold": ranking[0][0][1],
                    **ranking[0][1],
                },
                "summaries": all_combo_summaries,
                "meta": meta_base,
            },
            indent=2,
        )
    )
    logger.info(f"wrote {out_dir / 'sweep_index.json'}")
    if is_sweep:
        best = ranking[0]
        logger.info(
            f"BEST: score={best[0][0]:.2f} mask={best[0][1]:.2f} "
            f"mean_iou={best[1]['mean_iou']:.4f} iou@50={best[1]['iou@50']:.3f}"
        )

    # Dump failure samples for the worst-K objects (single-combo only).
    if collect_samples and samples_by_object:
        assert sample_combo is not None
        active_tag = _tag(*sample_combo)
        per_object = _aggregate(combo_records[sample_combo])["per_object"]
        # Rank objects with collected samples by ascending mean IoU.
        candidate_objs = [
            name
            for name in sorted(
                samples_by_object.keys(),
                key=lambda n: per_object.get(n, {}).get("mean_iou", 0.0),
            )
        ]
        worst = candidate_objs[: args.n_worst_objects]
        failures_root = out_dir / "failures"
        failures_root.mkdir(exist_ok=True)
        blob_stats: dict[str, dict[str, float]] = {}
        for obj in worst:
            obj_dir = failures_root / obj.replace(" ", "_")
            obj_dir.mkdir(exist_ok=True)
            for idx, s in enumerate(samples_by_object[obj]):
                _save_failure_figure(obj_dir / f"{idx:02d}_{s.suite}_{s.demo}.png", s)
            blobs = [s.n_blobs for s in samples_by_object[obj]]
            blob_stats[obj] = {
                "mean_iou": per_object.get(obj, {}).get("mean_iou", float("nan")),
                "n_samples": len(samples_by_object[obj]),
                "mean_blobs": float(np.mean(blobs)) if blobs else 0.0,
                "empty_frac": float(np.mean([b == 0 for b in blobs])) if blobs else 0.0,
            }
        (failures_root / "blob_stats.json").write_text(
            json.dumps(
                {"combo": active_tag, "objects": blob_stats},
                indent=2,
            )
        )
        # Terminal table: worst objects, blob statistics.
        t = Table(
            title=f"Failure samples [{active_tag}] — per-object blob count",
            show_header=True,
            header_style="bold",
        )
        for col in ["object", "mean_iou", "n_samples", "mean_blobs", "empty_frac"]:
            t.add_column(col)
        for obj in worst:
            st = blob_stats[obj]
            t.add_row(
                obj,
                f"{st['mean_iou']:.3f}",
                str(st["n_samples"]),
                f"{st['mean_blobs']:.2f}",
                f"{st['empty_frac']:.2f}",
            )
        console.print(t)
        logger.info(
            f"wrote {sum(len(samples_by_object[o]) for o in worst)} sample figures "
            f"for {len(worst)} worst objects under {failures_root}/"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
