"""Shared dataset loading, metrics, and reporting for perception ablations.

The original SAM3-text baseline lives in ``sam3_segmentation_eval.py`` and stays
unchanged. New variants (``sam3_clip_rerank_eval.py``,
``dino_sam3_box_eval.py``) share this module so their numbers are directly
comparable to the baseline — same frames, same IoU definition, same CSV/plot
schema.
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
from typing import Any, Protocol

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment

from policy_abstraction.utils.depth import mujoco_depth_to_metric
from policy_abstraction.utils.segmentation import get_robot_segment_ids

# ---------------------------------------------------------------------------
# Shared Args — variant scripts add their own extra fields on top of this.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BaseArgs:
    dataset_globs: list[str] = dataclasses.field(
        default_factory=lambda: [
            "datasets/libero_object_new/*.hdf5",
            "datasets/libero_10_new/*.hdf5",
            "datasets/libero_spatial_new/*.hdf5",
            "datasets/libero_goal_new/*.hdf5",
        ]
    )
    """Glob patterns for HDF5 files. Pass multiple to cover several suites."""
    out_dir: str = "tmp/perception_ablation"
    max_demos_per_file: int | None = None
    max_files: int | None = None
    device: str = "cuda"
    normalize_names: bool = True
    """Strip trailing ``_\\d+`` and replace ``_`` with space before prompting."""
    save_failure_samples: bool = True
    """Dump pred-vs-GT figures for low-IoU prompts under ``<out_dir>/failures/``."""
    failure_iou_threshold: float = 0.3
    """Only collect prompts with per-frame IoU strictly below this."""
    n_samples_per_object: int = 5
    """Max failure figures saved per worst-performing object."""
    n_worst_objects: int = 15
    """Number of worst-mean-IoU objects to save failures for."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _suite_name(hdf5_path: str) -> str:
    return Path(hdf5_path).parent.name


def expand_globs(globs: list[str]) -> list[tuple[str, str]]:
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


def normalize_name(name: str) -> str:
    """Strip all digits and underscores, matching ``scripts/ablations/extraction/extract_objects.py``.

    Must stay in lock-step with ``sanitize_object_name`` in that script so the
    reference bank keys and eval-time prompts agree on canonical names (e.g.
    ``chefmate_8_frypan`` → ``"chefmate frypan"`` on both sides).
    """
    return " ".join(_DIGIT_RE.sub("", name).replace("_", " ").split())


def _load_first_frame(
    hdf5_path: str, demo_key: str, id_to_name: dict[int, str]
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, str]] | None:
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


def iter_frames(hdf5_path: str, max_demos: int | None):
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
# Predictor protocol — each variant implements this.
# ---------------------------------------------------------------------------


class Predictor(Protocol):
    name: str

    def predict(self, rgb: np.ndarray, prompts: list[str]) -> dict[str, np.ndarray]:
        """Return ``{prompt: mask[H,W] bool}`` for each requested normalised name."""
        ...


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
    per_object_iou: dict[str, float]
    confusion_cells: dict[tuple[str, str], float]
    gt_names: list[str]
    empty_predictions: list[str]


def evaluate_frame(
    pred_masks: dict[str, np.ndarray],
    gt_masks: dict[int, np.ndarray],
    prompt_to_canonical: dict[str, str],
    id_to_canonical: dict[int, str],
) -> tuple[dict[str, float], dict[tuple[str, str], float], list[str]]:
    per_iou: dict[str, float] = {}
    empty: list[str] = []

    # Multiple GT seg ids can share a canonical name (e.g. ``moka_pot_1`` and
    # ``moka_pot_2`` both collapse to ``"moka pot"``). The predictor returns a
    # single mask per canonical name, so we credit the best-matching instance.
    gt_ids_by_canon: dict[str, list[int]] = defaultdict(list)
    for sid, canon in id_to_canonical.items():
        gt_ids_by_canon[canon].append(sid)

    for prompt, pred in pred_masks.items():
        canon = prompt_to_canonical[prompt]
        gt_ids = gt_ids_by_canon.get(canon, [])
        if not gt_ids:
            continue
        if pred.sum() == 0:
            empty.append(canon)
        per_iou[canon] = max(iou(pred, gt_masks[sid]) for sid in gt_ids)

    confusion: dict[tuple[str, str], float] = {}
    for prompt, pred in pred_masks.items():
        pred_canon = prompt_to_canonical[prompt]
        for gt_canon, gt_ids in gt_ids_by_canon.items():
            best = 0.0
            any_union = False
            for sid in gt_ids:
                gt = gt_masks[sid]
                union = int(np.logical_or(pred, gt).sum())
                if union == 0:
                    continue
                any_union = True
                inter = int(np.logical_and(pred, gt).sum())
                best = max(best, inter / union)
            if any_union:
                confusion[(pred_canon, gt_canon)] = best
    return per_iou, confusion, empty


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _aggregate(records: list[FrameRecord]) -> dict[str, Any]:
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


_MISSING_LABEL_COLOR = "#8b0000"  # dark red


# LIBERO task filenames like ``pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo``
# carry the task description verbatim. ``_STOPWORDS`` keeps grammatical filler
# from making every object "task-relevant" via words like ``the`` or ``it``.
_STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "to",
    "in",
    "on",
    "at",
    "it",
    "and",
    "or",
    "for",
    "from",
    "with",
    "into",
    "onto",
    "is",
    "be",
    "demo",
}


def _task_name_from_path(hdf5_path: str) -> str:
    """Extract a human-readable task description from a LIBERO HDF5 filename."""
    stem = Path(hdf5_path).stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return " ".join(stem.replace("_", " ").split()).lower()


def _task_relevant_objects(objects: set[str], task_names: list[str]) -> set[str]:
    """Return the subset of ``objects`` whose (sanitised) name shares at least
    one non-stopword with any task name in ``task_names``.
    """
    task_words: set[str] = set()
    for t in task_names:
        task_words.update(w for w in t.split() if w and w not in _STOPWORDS)
    relevant: set[str] = set()
    for obj in objects:
        obj_words = {w for w in obj.lower().split() if w and w not in _STOPWORDS}
        if obj_words & task_words:
            relevant.add(obj)
    return relevant


def _filter_agg_to_objects(agg: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    """Restrict an aggregate dict to the subset of objects in ``keep``.

    Recomputes the overall summary from the kept per-object IoUs (using each
    object's mean as one sample) — the raw per-frame IoUs are no longer
    available at this stage, so ``overall`` here is the macro-average over
    kept objects rather than the micro-average over kept frames. Good enough
    for a quick "task-only" view.
    """
    per_obj_kept = {n: m for n, m in agg["per_object"].items() if n in keep}
    conf_kept = {
        (p, g): v
        for (p, g), v in agg["confusion_mean"].items()
        if p in keep and g in keep
    }
    ious = [m["mean_iou"] for m in per_obj_kept.values()]
    overall = {
        "n_objects_kept": len(per_obj_kept),
        "mean_iou": float(np.mean(ious)) if ious else float("nan"),
        "iou@50": float(np.mean([v >= 0.5 for v in ious])) if ious else float("nan"),
        "iou@75": float(np.mean([v >= 0.75 for v in ious])) if ious else float("nan"),
    }
    return {
        "overall": overall,
        "per_object": per_obj_kept,
        "confusion_mean": conf_kept,
    }


def _plot_confusion(
    confusion_mean: dict[tuple[str, str], float],
    out_path: Path,
    title: str,
    missing_prompts: set[str] | None = None,
) -> None:
    """Render the confusion matrix. Prompts in ``missing_prompts`` (those that
    weren't found in the reference bank and fell back to text) get a dark-red
    tick label on both axes so they're easy to spot."""
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
    missing = missing_prompts or set()
    ax.set_xticks(range(len(gts)))
    ax.set_xticklabels(gts, rotation=45, ha="right")
    ax.set_yticks(range(len(preds)))
    ax.set_yticklabels(preds)
    for tick, name in zip(ax.get_xticklabels(), gts, strict=True):
        if name in missing:
            tick.set_color(_MISSING_LABEL_COLOR)
    for tick, name in zip(ax.get_yticklabels(), preds, strict=True):
        if name in missing:
            tick.set_color(_MISSING_LABEL_COLOR)
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
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="IoU(pred, gt)")
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


def _conf_to_json(d: dict[tuple[str, str], float]) -> dict[str, float]:
    return {f"{p}||{g}": v for (p, g), v in d.items()}


def _write_outputs(
    out_dir: Path,
    tag: str,
    all_records: list[FrameRecord],
    records_by_suite: dict[str, list[FrameRecord]],
    console: Console,
    meta: dict[str, Any],
    missing_prompts: set[str] | None = None,
    task_names: list[str] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "per_suite": {}}
    overall_agg = _aggregate(all_records)
    summary["overall"] = overall_agg
    _print_tables(console, overall_agg, f"ALL [{tag}]")
    _plot_confusion(
        overall_agg["confusion_mean"],
        out_dir / f"confusion_overall_{tag}.png",
        f"Confusion — overall [{tag}]",
        missing_prompts=missing_prompts,
    )
    _write_csv(overall_agg["per_object"], out_dir / f"per_object_iou_overall_{tag}.csv")

    # Task-only subset: objects whose names share a word with at least one
    # task filename. Renders a parallel confusion matrix + CSV + summary block.
    task_relevant: set[str] = set()
    if task_names:
        task_relevant = _task_relevant_objects(
            set(overall_agg["per_object"].keys()), task_names
        )
        logger.info(
            f"task-relevant objects ({len(task_relevant)}/{len(overall_agg['per_object'])}): "
            f"{sorted(task_relevant)}"
        )
        if task_relevant:
            task_overall = _filter_agg_to_objects(overall_agg, task_relevant)
            summary["overall_tasks"] = task_overall
            _plot_confusion(
                task_overall["confusion_mean"],
                out_dir / f"confusion_overall_tasks_{tag}.png",
                f"Confusion — overall task-objects [{tag}]",
                missing_prompts=missing_prompts,
            )
            _write_csv(
                task_overall["per_object"],
                out_dir / f"per_object_iou_overall_tasks_{tag}.csv",
            )
            tt = Table(
                title=f"ALL task-objects [{tag}]",
                show_header=True,
                header_style="bold",
            )
            for col in ["metric", "value"]:
                tt.add_column(col)
            for k, v in task_overall["overall"].items():
                tt.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            console.print(tt)

    for suite, recs in records_by_suite.items():
        agg = _aggregate(recs)
        summary["per_suite"][suite] = agg
        _print_tables(console, agg, f"{suite} [{tag}]")
        _plot_confusion(
            agg["confusion_mean"],
            out_dir / f"confusion_{suite}_{tag}.png",
            f"Confusion — {suite} [{tag}]",
            missing_prompts=missing_prompts,
        )
        _write_csv(agg["per_object"], out_dir / f"per_object_iou_{suite}_{tag}.csv")
        if task_relevant:
            suite_task = _filter_agg_to_objects(agg, task_relevant)
            summary["per_suite"][f"{suite}_tasks"] = suite_task
            if suite_task["per_object"]:
                _plot_confusion(
                    suite_task["confusion_mean"],
                    out_dir / f"confusion_{suite}_tasks_{tag}.png",
                    f"Confusion — {suite} task-objects [{tag}]",
                    missing_prompts=missing_prompts,
                )
                _write_csv(
                    suite_task["per_object"],
                    out_dir / f"per_object_iou_{suite}_tasks_{tag}.csv",
                )

    summary["meta"] = meta
    summary["overall"]["confusion_mean"] = _conf_to_json(overall_agg["confusion_mean"])
    if "overall_tasks" in summary:
        summary["overall_tasks"]["confusion_mean"] = _conf_to_json(
            summary["overall_tasks"]["confusion_mean"]
        )
    for suite in summary["per_suite"]:
        summary["per_suite"][suite]["confusion_mean"] = _conf_to_json(
            summary["per_suite"][suite]["confusion_mean"]
        )
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote {out_dir / f'summary_{tag}.json'}")
    return summary


# ---------------------------------------------------------------------------
# Failure-sample collection
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FailureSample:
    """One (frame, prompt) pair with its predicted mask and GT mask.

    ``extra`` carries variant-specific visualisation data (e.g. SAM3 candidate
    instances + CLIP scores for the ``sam3_clip_rerank`` variant). The default
    figure writer ignores it; variants can pass a custom ``failure_figure_fn``
    to ``run_pipeline`` that consumes it.
    """

    suite: str
    demo: str
    prompt: str
    iou: float
    rgb: np.ndarray  # (H, W, 3) uint8
    pred_mask: np.ndarray  # (H, W) bool
    gt_mask: np.ndarray  # (H, W) bool
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


def _overlay_masks(
    rgb: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.55,
    colors: np.ndarray | None = None,
) -> np.ndarray:
    """Tint each ``masks[i]`` with a distinct colour. ``masks`` is ``(N, H, W)`` bool."""
    if masks.shape[0] == 0:
        return rgb.copy()
    cmap = plt.get_cmap("tab10")
    out = rgb.astype(np.float32).copy()
    for i in range(masks.shape[0]):
        color = colors[i] if colors is not None else np.array(cmap(i % 10)[:3]) * 255.0
        m = masks[i]
        out[m] = (1.0 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def default_failure_figure(out_path: Path, s: FailureSample) -> None:
    """Side-by-side pred vs GT overlay on the RGB frame."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))
    pred_overlay = _overlay_masks(
        s.rgb, s.pred_mask[None, ...], colors=np.array([[230, 60, 60]])
    )
    axes[0].imshow(pred_overlay)
    axes[0].set_title(
        f"pred '{s.prompt}'  IoU={s.iou:.3f}  area={int(s.pred_mask.sum())}px",
        fontsize=9,
    )
    axes[0].axis("off")

    gt_overlay = _overlay_masks(
        s.rgb, s.gt_mask[None, ...], colors=np.array([[60, 200, 90]])
    )
    axes[1].imshow(gt_overlay)
    axes[1].set_title(f"GT '{s.prompt}'  area={int(s.gt_mask.sum())}px", fontsize=9)
    axes[1].axis("off")
    fig.suptitle(f"{s.suite} :: {s.demo}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _dump_failure_samples(
    samples_by_object: dict[str, list[FailureSample]],
    per_object_agg: dict[str, dict[str, float]],
    out_dir: Path,
    tag: str,
    n_worst_objects: int,
    figure_fn,
    console: Console,
) -> None:
    """Write figures for the ``n_worst_objects`` objects with lowest mean IoU."""
    if not samples_by_object:
        return
    # Rank by ascending mean IoU among objects that actually have collected failures.
    candidates = sorted(
        samples_by_object.keys(),
        key=lambda n: per_object_agg.get(n, {}).get("mean_iou", 0.0),
    )
    worst = candidates[:n_worst_objects]
    failures_root = out_dir / "failures"
    failures_root.mkdir(exist_ok=True)

    stats: dict[str, dict[str, float]] = {}
    total = 0
    for obj in worst:
        obj_dir = failures_root / obj.replace(" ", "_")
        obj_dir.mkdir(exist_ok=True)
        samples = samples_by_object[obj]
        for idx, s in enumerate(samples):
            figure_fn(obj_dir / f"{idx:02d}_{s.suite}_{s.demo}.png", s)
        total += len(samples)
        stats[obj] = {
            "mean_iou": per_object_agg.get(obj, {}).get("mean_iou", float("nan")),
            "n_samples": len(samples),
            "mean_pred_area": float(np.mean([int(s.pred_mask.sum()) for s in samples])),
            "empty_frac": float(np.mean([s.pred_mask.sum() == 0 for s in samples])),
        }
    (failures_root / "blob_stats.json").write_text(
        json.dumps({"tag": tag, "objects": stats}, indent=2)
    )
    t = Table(
        title=f"Failure samples [{tag}] — worst objects",
        show_header=True,
        header_style="bold",
    )
    for col in ["object", "mean_iou", "n_samples", "mean_pred_area", "empty_frac"]:
        t.add_column(col)
    for obj in worst:
        st = stats[obj]
        t.add_row(
            obj,
            f"{st['mean_iou']:.3f}",
            str(st["n_samples"]),
            f"{st['mean_pred_area']:.0f}",
            f"{st['empty_frac']:.2f}",
        )
    console.print(t)
    logger.info(
        f"wrote {total} failure figures for {len(worst)} worst objects under "
        f"{failures_root}/"
    )


# ---------------------------------------------------------------------------
# Driver — call from each variant's ``main``.
# ---------------------------------------------------------------------------


def run_pipeline(
    args: BaseArgs,
    predictor: Predictor,
    tag: str,
    extra_meta: dict[str, Any] | None = None,
    failure_figure_fn=None,
    failure_extra_fn=None,
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = expand_globs(args.dataset_globs)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        logger.error(f"no HDF5 files matched {args.dataset_globs}")
        return
    logger.info(f"found {len(files)} files across {len({s for s, _ in files})} suites")

    task_names = [_task_name_from_path(p) for _, p in files]

    all_records: list[FrameRecord] = []
    records_by_suite: dict[str, list[FrameRecord]] = defaultdict(list)
    n_frames = 0
    n_preds = 0
    n_empty = 0
    samples_by_object: dict[str, list[FailureSample]] = defaultdict(list)

    for suite, hdf5_path in files:
        logger.info(f"[{suite}] {os.path.basename(hdf5_path)}")
        for demo_key, rgb, gt_masks, id_to_name in iter_frames(
            hdf5_path, args.max_demos_per_file
        ):
            id_to_canonical = (
                {sid: normalize_name(n) for sid, n in id_to_name.items()}
                if args.normalize_names
                else dict(id_to_name)
            )
            prompt_to_canonical = {canon: canon for canon in id_to_canonical.values()}
            prompts = sorted(prompt_to_canonical)

            pred_masks = predictor.predict(rgb, prompts)

            per_iou, conf, empty = evaluate_frame(
                pred_masks,
                gt_masks,
                prompt_to_canonical,
                id_to_canonical,
            )
            rec = FrameRecord(
                suite=suite,
                hdf5=hdf5_path,
                demo=demo_key,
                per_object_iou=per_iou,
                confusion_cells=conf,
                gt_names=sorted(set(id_to_canonical.values())),
                empty_predictions=empty,
            )
            all_records.append(rec)
            records_by_suite[suite].append(rec)
            n_frames += 1
            n_preds += len(prompts)
            n_empty += len(empty)

            if args.save_failure_samples:
                gt_ids_by_canon: dict[str, list[int]] = defaultdict(list)
                for sid, canon in id_to_canonical.items():
                    gt_ids_by_canon[canon].append(sid)
                for prompt, iv in per_iou.items():
                    if np.isnan(iv) or iv >= args.failure_iou_threshold:
                        continue
                    if len(samples_by_object[prompt]) >= args.n_samples_per_object:
                        continue
                    gt_ids = gt_ids_by_canon.get(prompt, [])
                    if not gt_ids:
                        continue
                    # Show the GT instance the score was computed against.
                    pred = pred_masks[prompt]
                    best_gt_id = max(gt_ids, key=lambda sid: iou(pred, gt_masks[sid]))
                    samples_by_object[prompt].append(
                        FailureSample(
                            suite=suite,
                            demo=demo_key,
                            prompt=prompt,
                            iou=iv,
                            rgb=rgb.copy(),
                            pred_mask=pred_masks[prompt].copy(),
                            gt_mask=gt_masks[best_gt_id].copy(),
                            extra=(
                                failure_extra_fn(predictor, prompt)
                                if failure_extra_fn is not None
                                else {}
                            ),
                        )
                    )

    if n_frames == 0:
        logger.error("no frames evaluated")
        return

    logger.info(f"done: {n_frames} frames, {n_preds} prompts, {n_empty} empty preds")

    console = Console()
    meta: dict[str, Any] = {
        "predictor": predictor.name,
        "dataset_globs": args.dataset_globs,
        "normalize_names": args.normalize_names,
        "n_files": len(files),
        "n_frames": n_frames,
        "n_prompts": n_preds,
        "n_empty": n_empty,
        **(extra_meta or {}),
    }
    missing_prompts = getattr(predictor, "missing_prompts", None)
    summary = _write_outputs(
        out_dir,
        tag,
        all_records,
        records_by_suite,
        console,
        meta,
        missing_prompts=missing_prompts,
        task_names=task_names,
    )

    if args.save_failure_samples and samples_by_object:
        _dump_failure_samples(
            samples_by_object,
            summary["overall"]["per_object"],
            out_dir,
            tag,
            args.n_worst_objects,
            failure_figure_fn or default_failure_figure,
            console,
        )


# ===========================================================================
# Multi-instance evaluation (panoptic-IoU style).
# ---------------------------------------------------------------------------
# The single-instance path above credits the best-matching GT instance per
# canonical name. That silently rewards under-prediction (extra GT instances
# are ignored) and ignores over-prediction (predictors that emit multiple
# masks per query only get the first one read).
#
# The multi-instance path:
#   * loads RGB + depth + camera intrinsics/extrinsics for both views
#     (so multi-view stages like ``reconstruct``/``fuse_views`` run end-to-end),
#   * accepts predictors that return ``list[ViewMask]`` per canonical name,
#   * scores each canonical via Hungarian assignment + panoptic-IoU
#     (``sum(matched IoUs) / max(|P|, |G|)``) so both excess and missing
#     masks reduce the score.
#
# Frame-level evaluation only — no temporal tracking. First frame per demo,
# stateless inference, identical to the single-instance driver above.
# ===========================================================================


@dataclasses.dataclass
class ViewFrame:
    """One camera view of a single timestep — RGB, metric depth, K, E."""

    view_idx: int
    name: str
    rgb: np.ndarray  # (H, W, 3) uint8
    depth: np.ndarray  # (H, W) float32, metres
    intrinsic: np.ndarray  # (3, 3) float64
    extrinsic: np.ndarray  # (4, 4) float64


@dataclasses.dataclass
class ViewMask:
    """One predicted mask, scoped to the view it was produced in."""

    view_idx: int
    mask: np.ndarray  # (H, W) bool
    score: float


# Layout in the views list returned by ``iter_frames_multi``:
#   index 0 — primary (agentview)
#   index 1 — secondary (eye_in_hand)
_PRIMARY_VIEW_NAME = "agentview"
_SECONDARY_VIEW_NAME = "eye_in_hand"


def _load_first_frame_multi(
    hdf5_path: str,
    demo_key: str,
    id_to_name: dict[int, str],
    depth_near: float,
    depth_far: float,
) -> tuple[list[ViewFrame], list[dict[int, np.ndarray]], dict[int, str]] | None:
    """Read first-frame multi-view RGBD + GT segmentation for one demo.

    Mirrors :func:`_load_first_frame` (same ``[::-1, ::-1]`` rotation, same
    object-id filtering) but returns both views and converts the MuJoCo
    normalised depth to metric metres so :mod:`open_perception`'s
    ``reconstruct`` stage gets the units it expects.
    """
    views: list[ViewFrame] = []
    gt_by_view: list[dict[int, np.ndarray]] = []
    primary_seg: np.ndarray | None = None

    with h5py.File(hdf5_path, "r") as f:
        obs = f["data"][demo_key]["obs"]
        for v_idx, name in enumerate((_PRIMARY_VIEW_NAME, _SECONDARY_VIEW_NAME)):
            rgb_key = f"{name}_rgb"
            dep_key = f"{name}_depth"
            seg_key = f"{name}_segmentation"
            int_key = f"{name}_intrinsic"
            ext_key = f"{name}_extrinsic"
            if any(k not in obs for k in (rgb_key, dep_key, seg_key, int_key, ext_key)):
                # Demo missing one of the required keys — fall back to single-view.
                if v_idx == 0:
                    return None
                break
            rgb = np.ascontiguousarray(np.asarray(obs[rgb_key][0])[::-1, ::-1, :])
            dep = np.asarray(obs[dep_key][0])
            if dep.ndim == 3 and dep.shape[-1] == 1:
                dep = dep[..., 0]
            dep = np.ascontiguousarray(dep[::-1, ::-1]).astype(np.float32)
            if dep.max() <= 1.0:
                dep = mujoco_depth_to_metric(dep, near=depth_near, far=depth_far)
            seg = np.asarray(obs[seg_key][0])
            if seg.ndim == 3:
                seg = seg[..., 0]
            seg = np.ascontiguousarray(seg[::-1, ::-1])
            K = np.asarray(obs[int_key][0])
            E = np.asarray(obs[ext_key][0])

            views.append(
                ViewFrame(
                    view_idx=v_idx,
                    name=name,
                    rgb=rgb,
                    depth=dep,
                    intrinsic=K[:3, :3] if K.shape == (4, 4) else K,
                    extrinsic=E,
                )
            )
            if v_idx == 0:
                primary_seg = seg
            gt_by_view.append({})
            for sid in (int(v) for v in np.unique(seg) if v != 0):
                gt_by_view[v_idx][sid] = seg == sid

    if primary_seg is None or not views:
        return None

    all_ids = {int(v) for v in np.unique(primary_seg) if v != 0}
    robot_ids = get_robot_segment_ids(id_to_name, all_seg_ids=all_ids)
    object_ids = sorted(all_ids - robot_ids)
    if not object_ids:
        return None

    # Strip robot/background ids from every view's GT dict so consumers only see
    # candidate-object masks (matches the single-view loader's contract).
    object_ids_set = set(object_ids)
    gt_by_view = [
        {sid: m for sid, m in d.items() if sid in object_ids_set} for d in gt_by_view
    ]
    names = {sid: id_to_name[sid] for sid in object_ids if sid in id_to_name}
    if not names:
        return None
    return views, gt_by_view, names


def iter_frames_multi(hdf5_path: str, max_demos: int | None):
    """Yield ``(demo_key, views, gt_masks_by_view, id_to_name)`` per demo.

    Multi-view counterpart of :func:`iter_frames` — reads both ``agentview``
    and ``eye_in_hand`` RGB/depth/segmentation/intrinsics/extrinsics for the
    first frame of each demo. Depth is converted to metres using the demo's
    ``depth_near`` / ``depth_far`` HDF5 attrs.
    """
    with h5py.File(hdf5_path, "r") as f:
        seg_map_json = f["data"].attrs.get("segmentation_name_mapping", None)
        depth_near = float(f["data"].attrs.get("depth_near", 0.01))
        depth_far = float(f["data"].attrs.get("depth_far", 10.0))
        demo_keys = list(f["data"].keys())
    id_to_name: dict[int, str] = {}
    if seg_map_json:
        id_to_name = {int(k): v for k, v in json.loads(seg_map_json).items()}

    if max_demos is not None:
        demo_keys = demo_keys[:max_demos]

    for dk in demo_keys:
        result = _load_first_frame_multi(
            hdf5_path, dk, id_to_name, depth_near, depth_far
        )
        if result is None:
            logger.debug(f"{hdf5_path}:{dk} has no objects, skipping")
            continue
        views, gt_by_view, names = result
        yield dk, views, gt_by_view, names


# ---------------------------------------------------------------------------
# Multi-instance predictor protocol
# ---------------------------------------------------------------------------


class MultiInstancePredictor(Protocol):
    name: str

    def predict(
        self, views: list[ViewFrame], prompts: list[str]
    ) -> dict[str, list[ViewMask]]:
        """Return ``{prompt: [ViewMask, ...]}`` — every detection across views.

        Empty list for a prompt means the predictor produced nothing (counts
        against panoptic-IoU as a missed instance for any GT with that name).
        """
        ...


# ---------------------------------------------------------------------------
# Panoptic-IoU metric
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FrameRecordMulti:
    suite: str
    hdf5: str
    demo: str
    per_object_panoptic: dict[str, float]
    per_object_count_error: dict[str, int]
    per_object_recall50: dict[str, float]
    per_object_precision50: dict[str, float]
    per_object_f1_50: dict[str, float]
    confusion_cells: dict[tuple[str, str], float]
    gt_names: list[str]
    n_hallucinations: int
    """Predictions for canonical names with no GT instance in the frame."""


def _panoptic_for_canonical(
    preds: list[np.ndarray], gts: list[np.ndarray]
) -> tuple[float, float, float, float]:
    """Return ``(panoptic_iou, recall@50, precision@50, f1@50)``.

    Hungarian over IoU. ``panoptic_iou = sum(matched IoUs) / max(|P|, |G|)``.
    ``recall/precision`` count matches with IoU >= 0.5.
    """
    P, G = len(preds), len(gts)
    if P == 0 and G == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    if P == 0:
        return 0.0, 0.0, float("nan"), 0.0
    if G == 0:
        # Hallucinations — caller tracks separately. Return 0 across the board
        # so they don't accidentally rank as "perfect" anywhere downstream.
        return 0.0, float("nan"), 0.0, 0.0

    M = np.zeros((P, G), dtype=np.float32)
    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            M[i, j] = iou(p, g)
    row_idx, col_idx = linear_sum_assignment(-M)

    matched_ious = M[row_idx, col_idx]
    panoptic = float(matched_ious.sum() / max(P, G))
    n_match50 = int((matched_ious >= 0.5).sum())
    recall = n_match50 / G
    precision = n_match50 / P
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return panoptic, recall, precision, f1


def evaluate_frame_multi(
    pred_lists: dict[str, list[ViewMask]],
    gt_masks: dict[int, np.ndarray],
    prompt_to_canonical: dict[str, str],
    id_to_canonical: dict[int, str],
    primary_view: int = 0,
) -> tuple[
    dict[str, float],  # panoptic per canonical (only canonicals with GT)
    dict[str, int],  # count_error per canonical
    dict[str, float],  # recall@50
    dict[str, float],  # precision@50
    dict[str, float],  # f1@50
    dict[tuple[str, str], float],  # confusion (best primary-view IoU)
    int,  # n_hallucinations
]:
    """Score one frame's multi-instance predictions against the GT.

    ``pred_lists`` keys are caller prompts; ``prompt_to_canonical`` maps them
    to canonical names. Only ``view_idx == primary_view`` masks are scored —
    secondary-view detections are exposed by the predictor for debugging but
    don't affect the metric.
    """
    # Collapse predictions to {canonical: [primary-view masks]}.
    preds_by_canon: dict[str, list[np.ndarray]] = defaultdict(list)
    for prompt, items in pred_lists.items():
        canon = prompt_to_canonical.get(prompt, prompt)
        for vm in items:
            if vm.view_idx == primary_view and vm.mask is not None and vm.mask.any():
                preds_by_canon[canon].append(vm.mask)

    gts_by_canon: dict[str, list[np.ndarray]] = defaultdict(list)
    for sid, canon in id_to_canonical.items():
        if sid in gt_masks:
            gts_by_canon[canon].append(gt_masks[sid])

    panoptic: dict[str, float] = {}
    count_error: dict[str, int] = {}
    recall: dict[str, float] = {}
    precision: dict[str, float] = {}
    f1: dict[str, float] = {}
    n_hallucinations = 0

    # Canonicals with GT — these drive the primary metric.
    for canon, gts in gts_by_canon.items():
        preds = preds_by_canon.get(canon, [])
        pi, rec, prec, f1v = _panoptic_for_canonical(preds, gts)
        panoptic[canon] = pi
        count_error[canon] = len(preds) - len(gts)
        recall[canon] = rec
        precision[canon] = prec
        f1[canon] = f1v

    # Canonicals predicted but absent from GT — pure hallucinations.
    for canon, preds in preds_by_canon.items():
        if canon not in gts_by_canon and preds:
            n_hallucinations += len(preds)

    # Confusion: best IoU between any primary-view pred mask of pred_canon
    # and any GT mask of gt_canon. Same shape as the single-instance path so
    # ``_plot_confusion`` works unchanged.
    confusion: dict[tuple[str, str], float] = {}
    for pred_canon, preds in preds_by_canon.items():
        for gt_canon, gts in gts_by_canon.items():
            best = 0.0
            any_union = False
            for p in preds:
                for g in gts:
                    union = int(np.logical_or(p, g).sum())
                    if union == 0:
                        continue
                    any_union = True
                    inter = int(np.logical_and(p, g).sum())
                    best = max(best, inter / union)
            if any_union:
                confusion[(pred_canon, gt_canon)] = best

    return panoptic, count_error, recall, precision, f1, confusion, n_hallucinations


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_multi(records: list[FrameRecordMulti]) -> dict[str, Any]:
    """Aggregate frame-level multi-instance records into per-object + overall."""
    panoptic_by_obj: dict[str, list[float]] = defaultdict(list)
    count_err_by_obj: dict[str, list[int]] = defaultdict(list)
    recall_by_obj: dict[str, list[float]] = defaultdict(list)
    precision_by_obj: dict[str, list[float]] = defaultdict(list)
    f1_by_obj: dict[str, list[float]] = defaultdict(list)
    confusion_sum: dict[tuple[str, str], float] = defaultdict(float)
    confusion_n: dict[tuple[str, str], int] = defaultdict(int)
    total_hallucinations = 0

    for r in records:
        for name, v in r.per_object_panoptic.items():
            if not np.isnan(v):
                panoptic_by_obj[name].append(v)
        for name, v in r.per_object_count_error.items():
            count_err_by_obj[name].append(v)
        for name, v in r.per_object_recall50.items():
            if not np.isnan(v):
                recall_by_obj[name].append(v)
        for name, v in r.per_object_precision50.items():
            if not np.isnan(v):
                precision_by_obj[name].append(v)
        for name, v in r.per_object_f1_50.items():
            if not np.isnan(v):
                f1_by_obj[name].append(v)
        for key, v in r.confusion_cells.items():
            confusion_sum[key] += v
            confusion_n[key] += 1
        total_hallucinations += r.n_hallucinations

    all_panoptic = [v for lst in panoptic_by_obj.values() for v in lst]
    all_count_err = [v for lst in count_err_by_obj.values() for v in lst]

    overall = {
        "n_frames": len(records),
        "n_evaluations": len(all_panoptic),
        "panoptic_iou": float(np.mean(all_panoptic)) if all_panoptic else float("nan"),
        "iou@50": float(np.mean([v >= 0.5 for v in all_panoptic]))
        if all_panoptic
        else float("nan"),
        "iou@75": float(np.mean([v >= 0.75 for v in all_panoptic]))
        if all_panoptic
        else float("nan"),
        "mean_count_error": float(np.mean(all_count_err))
        if all_count_err
        else float("nan"),
        "mean_abs_count_error": float(np.mean(np.abs(all_count_err)))
        if all_count_err
        else float("nan"),
        "recall@50": float(np.mean([v for lst in recall_by_obj.values() for v in lst]))
        if recall_by_obj
        else float("nan"),
        "precision@50": float(
            np.mean([v for lst in precision_by_obj.values() for v in lst])
        )
        if precision_by_obj
        else float("nan"),
        "f1@50": float(np.mean([v for lst in f1_by_obj.values() for v in lst]))
        if f1_by_obj
        else float("nan"),
        "n_hallucinations": int(total_hallucinations),
    }

    per_object: dict[str, dict[str, float]] = {}
    for name in sorted(panoptic_by_obj.keys()):
        pis = panoptic_by_obj[name]
        ces = count_err_by_obj.get(name, [])
        per_object[name] = {
            "panoptic_iou": float(np.mean(pis)),
            "iou@50": float(np.mean([v >= 0.5 for v in pis])),
            "iou@75": float(np.mean([v >= 0.75 for v in pis])),
            "mean_count_error": float(np.mean(ces)) if ces else float("nan"),
            "recall@50": float(np.mean(recall_by_obj.get(name, [])))
            if recall_by_obj.get(name)
            else float("nan"),
            "precision@50": float(np.mean(precision_by_obj.get(name, [])))
            if precision_by_obj.get(name)
            else float("nan"),
            "f1@50": float(np.mean(f1_by_obj.get(name, [])))
            if f1_by_obj.get(name)
            else float("nan"),
            "n": len(pis),
        }

    confusion_mean = {k: (confusion_sum[k] / confusion_n[k]) for k in confusion_sum}
    return {
        "overall": overall,
        "per_object": per_object,
        "confusion_mean": confusion_mean,
    }


def _filter_agg_to_objects_multi(agg: dict[str, Any], keep: set[str]) -> dict[str, Any]:
    """Task-relevant subset of a multi-instance aggregate (parallel to the
    single-instance ``_filter_agg_to_objects``)."""
    per_obj_kept = {n: m for n, m in agg["per_object"].items() if n in keep}
    conf_kept = {
        (p, g): v
        for (p, g), v in agg["confusion_mean"].items()
        if p in keep and g in keep
    }
    pis = [m["panoptic_iou"] for m in per_obj_kept.values()]
    ces = [
        m["mean_count_error"]
        for m in per_obj_kept.values()
        if not np.isnan(m["mean_count_error"])
    ]
    f1s = [m["f1@50"] for m in per_obj_kept.values() if not np.isnan(m["f1@50"])]
    overall = {
        "n_objects_kept": len(per_obj_kept),
        "panoptic_iou": float(np.mean(pis)) if pis else float("nan"),
        "iou@50": float(np.mean([v >= 0.5 for v in pis])) if pis else float("nan"),
        "iou@75": float(np.mean([v >= 0.75 for v in pis])) if pis else float("nan"),
        "mean_count_error": float(np.mean(ces)) if ces else float("nan"),
        "f1@50": float(np.mean(f1s)) if f1s else float("nan"),
    }
    return {
        "overall": overall,
        "per_object": per_obj_kept,
        "confusion_mean": conf_kept,
    }


# ---------------------------------------------------------------------------
# Reporting (multi-instance)
# ---------------------------------------------------------------------------


def _print_tables_multi(console: Console, agg: dict[str, Any], title: str) -> None:
    t = Table(title=f"{title} — overall", show_header=True, header_style="bold")
    t.add_column("metric")
    t.add_column("value")
    for k, v in agg["overall"].items():
        t.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
    console.print(t)

    po = agg["per_object"]
    if po:
        t2 = Table(
            title=f"{title} — per object (sorted by panoptic IoU)",
            show_header=True,
            header_style="bold",
        )
        for col in ["object", "panoptic", "iou@50", "f1@50", "Δcount", "n"]:
            t2.add_column(col)
        for name, m in sorted(po.items(), key=lambda kv: -kv[1]["panoptic_iou"]):
            t2.add_row(
                name,
                f"{m['panoptic_iou']:.3f}",
                f"{m['iou@50']:.2f}",
                f"{m['f1@50']:.2f}" if not np.isnan(m["f1@50"]) else "n/a",
                f"{m['mean_count_error']:+.1f}"
                if not np.isnan(m["mean_count_error"])
                else "n/a",
                str(m["n"]),
            )
        console.print(t2)


def _write_csv_multi(per_object: dict[str, dict[str, float]], out_path: Path) -> None:
    with out_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "object",
                "panoptic_iou",
                "iou@50",
                "iou@75",
                "f1@50",
                "recall@50",
                "precision@50",
                "mean_count_error",
                "n_frames",
            ]
        )
        for name, m in per_object.items():
            w.writerow(
                [
                    name,
                    f"{m['panoptic_iou']:.4f}",
                    f"{m['iou@50']:.4f}",
                    f"{m['iou@75']:.4f}",
                    f"{m['f1@50']:.4f}" if not np.isnan(m["f1@50"]) else "",
                    f"{m['recall@50']:.4f}" if not np.isnan(m["recall@50"]) else "",
                    f"{m['precision@50']:.4f}"
                    if not np.isnan(m["precision@50"])
                    else "",
                    f"{m['mean_count_error']:.4f}"
                    if not np.isnan(m["mean_count_error"])
                    else "",
                    m["n"],
                ]
            )
    logger.info(f"wrote {out_path}")


def _write_outputs_multi(
    out_dir: Path,
    tag: str,
    all_records: list[FrameRecordMulti],
    records_by_suite: dict[str, list[FrameRecordMulti]],
    console: Console,
    meta: dict[str, Any],
    missing_prompts: set[str] | None = None,
    task_names: list[str] | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "per_suite": {}}
    overall_agg = _aggregate_multi(all_records)
    summary["overall"] = overall_agg
    _print_tables_multi(console, overall_agg, f"ALL [{tag}]")
    _plot_confusion(
        overall_agg["confusion_mean"],
        out_dir / f"confusion_overall_{tag}.png",
        f"Confusion — overall [{tag}]",
        missing_prompts=missing_prompts,
    )
    _write_csv_multi(
        overall_agg["per_object"], out_dir / f"per_object_iou_overall_{tag}.csv"
    )

    task_relevant: set[str] = set()
    if task_names:
        task_relevant = _task_relevant_objects(
            set(overall_agg["per_object"].keys()), task_names
        )
        logger.info(
            f"task-relevant objects ({len(task_relevant)}/{len(overall_agg['per_object'])}): "
            f"{sorted(task_relevant)}"
        )
        if task_relevant:
            task_overall = _filter_agg_to_objects_multi(overall_agg, task_relevant)
            summary["overall_tasks"] = task_overall
            _plot_confusion(
                task_overall["confusion_mean"],
                out_dir / f"confusion_overall_tasks_{tag}.png",
                f"Confusion — overall task-objects [{tag}]",
                missing_prompts=missing_prompts,
            )
            _write_csv_multi(
                task_overall["per_object"],
                out_dir / f"per_object_iou_overall_tasks_{tag}.csv",
            )
            tt = Table(
                title=f"ALL task-objects [{tag}]",
                show_header=True,
                header_style="bold",
            )
            for col in ["metric", "value"]:
                tt.add_column(col)
            for k, v in task_overall["overall"].items():
                tt.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
            console.print(tt)

    for suite, recs in records_by_suite.items():
        agg = _aggregate_multi(recs)
        summary["per_suite"][suite] = agg
        _print_tables_multi(console, agg, f"{suite} [{tag}]")
        _plot_confusion(
            agg["confusion_mean"],
            out_dir / f"confusion_{suite}_{tag}.png",
            f"Confusion — {suite} [{tag}]",
            missing_prompts=missing_prompts,
        )
        _write_csv_multi(
            agg["per_object"], out_dir / f"per_object_iou_{suite}_{tag}.csv"
        )
        if task_relevant:
            suite_task = _filter_agg_to_objects_multi(agg, task_relevant)
            summary["per_suite"][f"{suite}_tasks"] = suite_task
            if suite_task["per_object"]:
                _plot_confusion(
                    suite_task["confusion_mean"],
                    out_dir / f"confusion_{suite}_tasks_{tag}.png",
                    f"Confusion — {suite} task-objects [{tag}]",
                    missing_prompts=missing_prompts,
                )
                _write_csv_multi(
                    suite_task["per_object"],
                    out_dir / f"per_object_iou_{suite}_tasks_{tag}.csv",
                )

    summary["meta"] = meta
    summary["overall"]["confusion_mean"] = _conf_to_json(overall_agg["confusion_mean"])
    if "overall_tasks" in summary:
        summary["overall_tasks"]["confusion_mean"] = _conf_to_json(
            summary["overall_tasks"]["confusion_mean"]
        )
    for suite in summary["per_suite"]:
        summary["per_suite"][suite]["confusion_mean"] = _conf_to_json(
            summary["per_suite"][suite]["confusion_mean"]
        )
    (out_dir / f"summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"wrote {out_dir / f'summary_{tag}.json'}")
    return summary


# ---------------------------------------------------------------------------
# Failure samples (multi-instance)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class FailureSampleMulti:
    suite: str
    demo: str
    prompt: str
    panoptic_iou: float
    rgb: np.ndarray  # (H, W, 3) uint8 — primary view
    pred_masks: list[np.ndarray]  # primary-view predicted masks for this canonical
    gt_masks: list[np.ndarray]  # GT masks for this canonical
    count_error: int
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


def default_failure_figure_multi(out_path: Path, s: FailureSampleMulti) -> None:
    """Side-by-side overlay: all preds (left) vs all GTs (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.4))

    if s.pred_masks:
        pred_arr = np.stack(s.pred_masks, axis=0).astype(bool)
    else:
        pred_arr = np.zeros((0, *s.rgb.shape[:2]), dtype=bool)
    if s.gt_masks:
        gt_arr = np.stack(s.gt_masks, axis=0).astype(bool)
    else:
        gt_arr = np.zeros((0, *s.rgb.shape[:2]), dtype=bool)

    pred_overlay = _overlay_masks(s.rgb, pred_arr)
    axes[0].imshow(pred_overlay)
    axes[0].set_title(
        f"pred '{s.prompt}'  P={len(s.pred_masks)}  panoptic={s.panoptic_iou:.3f}",
        fontsize=9,
    )
    axes[0].axis("off")

    gt_overlay = _overlay_masks(s.rgb, gt_arr)
    axes[1].imshow(gt_overlay)
    axes[1].set_title(
        f"GT '{s.prompt}'  G={len(s.gt_masks)}  Δcount={s.count_error:+d}",
        fontsize=9,
    )
    axes[1].axis("off")
    fig.suptitle(f"{s.suite} :: {s.demo}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _dump_failure_samples_multi(
    samples_by_object: dict[str, list[FailureSampleMulti]],
    per_object_agg: dict[str, dict[str, float]],
    out_dir: Path,
    tag: str,
    n_worst_objects: int,
    figure_fn,
    console: Console,
) -> None:
    """Multi-instance counterpart of :func:`_dump_failure_samples`."""
    if not samples_by_object:
        return
    candidates = sorted(
        samples_by_object.keys(),
        key=lambda n: per_object_agg.get(n, {}).get("panoptic_iou", 0.0),
    )
    worst = candidates[:n_worst_objects]
    failures_root = out_dir / "failures"
    failures_root.mkdir(exist_ok=True)

    stats: dict[str, dict[str, float]] = {}
    total = 0
    for obj in worst:
        obj_dir = failures_root / obj.replace(" ", "_")
        obj_dir.mkdir(exist_ok=True)
        samples = samples_by_object[obj]
        for idx, s in enumerate(samples):
            figure_fn(obj_dir / f"{idx:02d}_{s.suite}_{s.demo}.png", s)
        total += len(samples)
        stats[obj] = {
            "panoptic_iou": per_object_agg.get(obj, {}).get(
                "panoptic_iou", float("nan")
            ),
            "n_samples": len(samples),
            "mean_count_error": float(np.mean([s.count_error for s in samples])),
        }
    (failures_root / "blob_stats.json").write_text(
        json.dumps({"tag": tag, "objects": stats}, indent=2)
    )
    t = Table(
        title=f"Failure samples [{tag}] — worst objects",
        show_header=True,
        header_style="bold",
    )
    for col in ["object", "panoptic_iou", "n_samples", "mean_count_error"]:
        t.add_column(col)
    for obj in worst:
        st = stats[obj]
        t.add_row(
            obj,
            f"{st['panoptic_iou']:.3f}",
            str(st["n_samples"]),
            f"{st['mean_count_error']:+.1f}",
        )
    console.print(t)
    logger.info(
        f"wrote {total} failure figures for {len(worst)} worst objects under "
        f"{failures_root}/"
    )


# ---------------------------------------------------------------------------
# Driver — multi-instance
# ---------------------------------------------------------------------------


def run_pipeline_multi(
    args: BaseArgs,
    predictor: MultiInstancePredictor,
    tag: str,
    extra_meta: dict[str, Any] | None = None,
    failure_figure_fn=None,
    failure_extra_fn=None,
    primary_view: int = 0,
) -> None:
    """Multi-view, multi-instance counterpart of :func:`run_pipeline`."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = expand_globs(args.dataset_globs)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        logger.error(f"no HDF5 files matched {args.dataset_globs}")
        return
    logger.info(f"found {len(files)} files across {len({s for s, _ in files})} suites")

    task_names = [_task_name_from_path(p) for _, p in files]

    all_records: list[FrameRecordMulti] = []
    records_by_suite: dict[str, list[FrameRecordMulti]] = defaultdict(list)
    n_frames = 0
    n_preds = 0
    n_empty = 0
    samples_by_object: dict[str, list[FailureSampleMulti]] = defaultdict(list)

    for suite, hdf5_path in files:
        logger.info(f"[{suite}] {os.path.basename(hdf5_path)}")
        for demo_key, views, gt_by_view, id_to_name in iter_frames_multi(
            hdf5_path, args.max_demos_per_file
        ):
            id_to_canonical = (
                {sid: normalize_name(n) for sid, n in id_to_name.items()}
                if args.normalize_names
                else dict(id_to_name)
            )
            prompt_to_canonical = {canon: canon for canon in id_to_canonical.values()}
            prompts = sorted(prompt_to_canonical)

            pred_lists = predictor.predict(views, prompts)

            (
                panoptic,
                count_error,
                recall,
                precision,
                f1,
                conf,
                n_hall,
            ) = evaluate_frame_multi(
                pred_lists,
                gt_by_view[primary_view],
                prompt_to_canonical,
                id_to_canonical,
                primary_view=primary_view,
            )
            rec = FrameRecordMulti(
                suite=suite,
                hdf5=hdf5_path,
                demo=demo_key,
                per_object_panoptic=panoptic,
                per_object_count_error=count_error,
                per_object_recall50=recall,
                per_object_precision50=precision,
                per_object_f1_50=f1,
                confusion_cells=conf,
                gt_names=sorted(set(id_to_canonical.values())),
                n_hallucinations=n_hall,
            )
            all_records.append(rec)
            records_by_suite[suite].append(rec)
            n_frames += 1
            n_preds += len(prompts)
            for _prompt_canon, items in pred_lists.items():
                primary_n = sum(
                    1 for vm in items if vm.view_idx == primary_view and vm.mask.any()
                )
                if primary_n == 0:
                    n_empty += 1

            if args.save_failure_samples:
                gt_ids_by_canon: dict[str, list[int]] = defaultdict(list)
                for sid, canon in id_to_canonical.items():
                    gt_ids_by_canon[canon].append(sid)
                for canon, pi in panoptic.items():
                    if np.isnan(pi) or pi >= args.failure_iou_threshold:
                        continue
                    if len(samples_by_object[canon]) >= args.n_samples_per_object:
                        continue
                    pred_masks = [
                        vm.mask
                        for vm in pred_lists.get(canon, [])
                        if vm.view_idx == primary_view and vm.mask.any()
                    ]
                    gt_ids = gt_ids_by_canon.get(canon, [])
                    gt_masks_list = [
                        gt_by_view[primary_view][sid]
                        for sid in gt_ids
                        if sid in gt_by_view[primary_view]
                    ]
                    samples_by_object[canon].append(
                        FailureSampleMulti(
                            suite=suite,
                            demo=demo_key,
                            prompt=canon,
                            panoptic_iou=pi,
                            rgb=views[primary_view].rgb.copy(),
                            pred_masks=[m.copy() for m in pred_masks],
                            gt_masks=[m.copy() for m in gt_masks_list],
                            count_error=count_error.get(canon, 0),
                            extra=(
                                failure_extra_fn(predictor, canon)
                                if failure_extra_fn is not None
                                else {}
                            ),
                        )
                    )

    if n_frames == 0:
        logger.error("no frames evaluated")
        return

    logger.info(f"done: {n_frames} frames, {n_preds} prompts, {n_empty} empty preds")

    console = Console()
    meta: dict[str, Any] = {
        "predictor": predictor.name,
        "dataset_globs": args.dataset_globs,
        "normalize_names": args.normalize_names,
        "n_files": len(files),
        "n_frames": n_frames,
        "n_prompts": n_preds,
        "n_empty": n_empty,
        "primary_view": primary_view,
        **(extra_meta or {}),
    }
    missing_prompts = getattr(predictor, "missing_prompts", None)
    summary = _write_outputs_multi(
        out_dir,
        tag,
        all_records,
        records_by_suite,
        console,
        meta,
        missing_prompts=missing_prompts,
        task_names=task_names,
    )

    if args.save_failure_samples and samples_by_object:
        _dump_failure_samples_multi(
            samples_by_object,
            summary["overall"]["per_object"],
            out_dir,
            tag,
            args.n_worst_objects,
            failure_figure_fn or default_failure_figure_multi,
            console,
        )
