"""Staged hyperparameter sweep over ``op_multi_instance_eval.py``.

Drives the multi-instance, multi-view perception eval as subprocesses across
multiple GPUs (one process per GPU) to find the configuration that maximises
``overall_tasks.overall.panoptic_iou`` (panoptic IoU on task-relevant
objects only — accounts for both excess and missed masks).

Strategy — greedy staged search:
  1. ``1_pipeline_shape``  cartesian over the optional pipeline stages
                          (filter.overlap × filter.score × filter.erode ×
                          fuse_views) at ``--coarse-demos-per-file`` demos.
  2. ``2_thresholds``      score x mask x filter_score thresholds around the
                          stage-1 winner at ``--detailed-demos-per-file``.
  3. ``3_assign``          assign top_k x score-weights around the stage-2
                          winner at ``--detailed-demos-per-file``.
  4. ``4_final``           top-K configs from stages 2+3 re-evaluated at
                          ``--final-demos-per-file`` (None = all demos)
                          with failure samples enabled.

Outputs in ``--out-root``:
  * ``<stage>/<config_id>/`` -- per-config eval outputs (summary_*.json, etc.)
  * ``leaderboard.csv``      -- updated incrementally as configs complete
  * ``best_config.json``     -- params + metrics of the stage-4 winner

This sweep does **not** test temporal tracking — every demo's first frame is
evaluated independently with no tracker state seeded. ``tracker_backend``
is hardcoded to ``"sam2"`` in the eval to match production but is never
exercised here.
"""

from __future__ import annotations

import csv
import dataclasses
import itertools
import json
import os
import queue
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import tyro

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = "scripts/ablations/perception/op_multi_instance_eval.py"


@dataclasses.dataclass
class SweepArgs:
    out_root: str = "tmp/perception/multi_sweep"
    gpus: list[int] = dataclasses.field(default_factory=lambda: [2, 3])
    coarse_demos_per_file: int = 2
    detailed_demos_per_file: int = 10
    final_demos_per_file: int | None = None
    """``None`` (default) means evaluate every demo per file."""
    n_top_for_final: int = 3
    skip_existing: bool = True
    """Re-use any directory that already has a ``summary_*.json``."""
    per_run_timeout_s: int = 60 * 60


# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------


def _config_id(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for k in sorted(params):
        v = params[k]
        if isinstance(v, bool):
            parts.append(f"{k}={int(v)}")
        elif isinstance(v, float):
            parts.append(f"{k}={v:g}")
        else:
            parts.append(f"{k}={v}")
    return "_".join(parts)


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _build_cmd(
    params: dict[str, Any],
    out_dir: Path,
    max_demos: int | None,
    save_failures: bool,
) -> list[str]:
    cmd = ["uv", "run", "python", EVAL_SCRIPT, "--out-dir", str(out_dir)]
    cmd.append(
        "--save-failure-samples" if save_failures else "--no-save-failure-samples"
    )
    if max_demos is not None:
        cmd += ["--max-demos-per-file", str(max_demos)]
    for k, v in params.items():
        if isinstance(v, bool):
            cmd.append(_flag(k) if v else _flag(f"no_{k}"))
        else:
            cmd += [_flag(k), str(v)]
    return cmd


def _parse_summary(out_dir: Path) -> dict[str, float | None]:
    """Pull metrics out of the eval's ``summary_*.json``.

    Optimises for ``overall_tasks.overall.panoptic_iou``; reports ``f1@50``
    and ``mean_count_error`` as secondary diagnostics.
    """
    files = sorted(out_dir.glob("summary_*.json"))
    if not files:
        return {
            "task_panoptic_iou": None,
            "task_iou50": None,
            "task_f1_50": None,
            "task_mean_count_error": None,
            "overall_panoptic_iou": None,
        }
    with files[-1].open() as f:
        data = json.load(f)

    def _get(d: Any, *keys: str) -> float | None:
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return float(cur) if isinstance(cur, (int, float)) else None

    return {
        "task_panoptic_iou": _get(data, "overall_tasks", "overall", "panoptic_iou"),
        "task_iou50": _get(data, "overall_tasks", "overall", "iou@50"),
        "task_f1_50": _get(data, "overall_tasks", "overall", "f1@50"),
        "task_mean_count_error": _get(
            data, "overall_tasks", "overall", "mean_count_error"
        ),
        "overall_panoptic_iou": _get(data, "overall", "overall", "panoptic_iou"),
    }


def _run_config(
    stage: str,
    params: dict[str, Any],
    out_dir: Path,
    max_demos: int | None,
    gpu: int,
    save_failures: bool,
    skip_existing: bool,
    timeout: int,
) -> dict[str, Any]:
    summary_files = list(out_dir.glob("summary_*.json")) if out_dir.exists() else []
    elapsed = 0.0
    rc = 0
    if skip_existing and summary_files:
        print(f"[gpu{gpu}|{stage}] SKIP {out_dir.name} (existing)", flush=True)
    else:
        # Wipe leftover state from a previous interrupted/crashed run so the
        # eval starts from a clean slate.
        if out_dir.exists() and not summary_files:
            shutil.rmtree(out_dir)
            print(f"[gpu{gpu}|{stage}] CLEAN {out_dir.name} (partial)", flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = _build_cmd(params, out_dir, max_demos, save_failures)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = out_dir / "run.log"
        (out_dir / "cmd.txt").write_text(" ".join(cmd) + "\n")
        print(f"[gpu{gpu}|{stage}] START {out_dir.name}", flush=True)
        t0 = time.time()
        try:
            with log_path.open("w") as logf:
                proc = subprocess.run(
                    cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            print(f"[gpu{gpu}|{stage}] TIMEOUT {out_dir.name}", flush=True)
            rc = -1
        elapsed = time.time() - t0
        print(
            f"[gpu{gpu}|{stage}] DONE  {out_dir.name} rc={rc} ({elapsed:.0f}s)",
            flush=True,
        )

    metrics = _parse_summary(out_dir)
    pi = metrics["task_panoptic_iou"]
    pi_str = f"{pi:.4f}" if pi is not None else "n/a"
    print(
        f"[gpu{gpu}|{stage}] METRIC {out_dir.name} task_panoptic_iou={pi_str}",
        flush=True,
    )
    return {
        "stage": stage,
        "params": dict(params),
        "out_dir": str(out_dir),
        "rc": rc,
        "elapsed_s": elapsed,
        **metrics,
    }


# ---------------------------------------------------------------------------
# Stage runner + leaderboard
# ---------------------------------------------------------------------------


def _write_leaderboard(path: Path, results: list[dict[str, Any]]) -> None:
    if not results:
        return
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r["params"].keys())
    fieldnames = [
        "stage",
        "task_panoptic_iou",
        "task_iou50",
        "task_f1_50",
        "task_mean_count_error",
        "overall_panoptic_iou",
        "elapsed_s",
        "rc",
        "out_dir",
    ] + sorted(all_keys)

    valid = [r for r in results if r["task_panoptic_iou"] is not None]
    invalid = [r for r in results if r["task_panoptic_iou"] is None]
    valid.sort(
        key=lambda r: (
            -(r["task_panoptic_iou"] or 0.0),
            -(r["task_f1_50"] or 0.0),
        )
    )
    rows = valid + invalid

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row: dict[str, Any] = {
                "stage": r["stage"],
                "task_panoptic_iou": r["task_panoptic_iou"],
                "task_iou50": r["task_iou50"],
                "task_f1_50": r["task_f1_50"],
                "task_mean_count_error": r["task_mean_count_error"],
                "overall_panoptic_iou": r["overall_panoptic_iou"],
                "elapsed_s": f"{r['elapsed_s']:.0f}",
                "rc": r["rc"],
                "out_dir": r["out_dir"],
            }
            for k in all_keys:
                row[k] = r["params"].get(k, "")
            w.writerow(row)


def _run_stage(
    stage: str,
    configs: list[dict[str, Any]],
    max_demos: int | None,
    gpus: list[int],
    out_root: Path,
    save_failures: bool,
    skip_existing: bool,
    timeout: int,
    leaderboard_path: Path,
    prior_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    print(
        f"\n=== STAGE {stage}: {len(configs)} configs, max_demos_per_file={max_demos} ===",
        flush=True,
    )
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    results: list[dict[str, Any]] = []

    def _task(params: dict[str, Any]) -> dict[str, Any]:
        gpu = gpu_q.get()
        try:
            cid = _config_id(params)
            out_dir = out_root / stage / cid
            return _run_config(
                stage=stage,
                params=params,
                out_dir=out_dir,
                max_demos=max_demos,
                gpu=gpu,
                save_failures=save_failures,
                skip_existing=skip_existing,
                timeout=timeout,
            )
        finally:
            gpu_q.put(gpu)

    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        futures = [ex.submit(_task, c) for c in configs]
        for fut in futures:
            try:
                r = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"[{stage}] task crashed: {exc}", file=sys.stderr, flush=True)
                continue
            results.append(r)
            _write_leaderboard(leaderboard_path, prior_results + results)

    return results


def _select_best(results: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    valid = [r for r in results if r["task_panoptic_iou"] is not None]
    valid.sort(
        key=lambda r: (
            -(r["task_panoptic_iou"] or 0.0),
            -(r["task_f1_50"] or 0.0),
        )
    )
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in valid:
        key = json.dumps(r["params"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# Search spaces per stage
# ---------------------------------------------------------------------------


def _stage1_configs() -> list[dict[str, Any]]:
    """Pipeline-shape grid: which optional stages help the multi-instance metric?

    ``reconstruct`` is forced on whenever ``fuse_views`` is on (fuse needs
    point clouds to align views) and forced off otherwise (it would just be
    dead weight without fusion). That collapses the 5-axis grid to 4.
    """
    space = {
        "enable_filter_overlap": [False, True],
        "enable_filter_score": [False, True],
        "enable_filter_erode": [False, True],
        "enable_fuse_views": [False, True],
    }
    keys = list(space)
    out: list[dict[str, Any]] = []
    for vals in itertools.product(*space.values()):
        cfg = dict(zip(keys, vals, strict=False))
        cfg["enable_reconstruct"] = cfg["enable_fuse_views"]
        out.append(cfg)
    return out


def _stage2_configs(base: dict[str, Any]) -> list[dict[str, Any]]:
    space = {
        "score_threshold": [0.03, 0.05, 0.10, 0.15],
        "mask_threshold": [0.30, 0.40, 0.50],
        "filter_score_threshold": [0.10, 0.20, 0.30],
    }
    keys = list(space)
    out: list[dict[str, Any]] = []
    for vals in itertools.product(*space.values()):
        cfg = dict(base)
        cfg.update(dict(zip(keys, vals, strict=False)))
        out.append(cfg)
    return out


def _stage3_configs(base: dict[str, Any]) -> list[dict[str, Any]]:
    space = {
        "assign_top_k": [-1, 1, 3, 5],
        # (dino_weight, clip_weight)
        "weights": [(1.0, 0.0), (1.0, 0.5), (0.5, 1.0)],
    }
    out: list[dict[str, Any]] = []
    for top_k, (dw, cw) in itertools.product(space["assign_top_k"], space["weights"]):
        cfg = dict(base)
        cfg["assign_top_k"] = top_k
        cfg["dino_weight"] = dw
        cfg["clip_weight"] = cw
        out.append(cfg)
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main(args: SweepArgs) -> None:
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    leaderboard = out_root / "leaderboard.csv"
    gpus = list(args.gpus)
    if not gpus:
        print("at least one GPU required (--gpus)", file=sys.stderr)
        sys.exit(2)

    sweep_t0 = time.time()
    all_results: list[dict[str, Any]] = []

    # ---- Stage 1: pipeline shape ---------------------------------------------
    s1 = _run_stage(
        "1_pipeline_shape",
        _stage1_configs(),
        args.coarse_demos_per_file,
        gpus,
        out_root,
        save_failures=False,
        skip_existing=args.skip_existing,
        timeout=args.per_run_timeout_s,
        leaderboard_path=leaderboard,
        prior_results=all_results,
    )
    all_results.extend(s1)
    s1_top = _select_best(s1, 1)
    if not s1_top:
        print("Stage 1 produced no valid results; aborting", file=sys.stderr)
        sys.exit(1)
    best1 = s1_top[0]
    print(
        f"\n>>> Stage 1 winner: {best1['params']} task_panoptic_iou={best1['task_panoptic_iou']:.4f}\n",
        flush=True,
    )

    # ---- Stage 2: thresholds around stage-1 winner ---------------------------
    s2 = _run_stage(
        "2_thresholds",
        _stage2_configs(best1["params"]),
        args.detailed_demos_per_file,
        gpus,
        out_root,
        save_failures=False,
        skip_existing=args.skip_existing,
        timeout=args.per_run_timeout_s,
        leaderboard_path=leaderboard,
        prior_results=all_results,
    )
    all_results.extend(s2)
    s2_top = _select_best(s2, 1)
    if not s2_top:
        print("Stage 2 produced no valid results; aborting", file=sys.stderr)
        sys.exit(1)
    best2 = s2_top[0]
    print(
        f"\n>>> Stage 2 winner: {best2['params']} task_panoptic_iou={best2['task_panoptic_iou']:.4f}\n",
        flush=True,
    )

    # ---- Stage 3: assign tuning around stage-2 winner ------------------------
    s3 = _run_stage(
        "3_assign",
        _stage3_configs(best2["params"]),
        args.detailed_demos_per_file,
        gpus,
        out_root,
        save_failures=False,
        skip_existing=args.skip_existing,
        timeout=args.per_run_timeout_s,
        leaderboard_path=leaderboard,
        prior_results=all_results,
    )
    all_results.extend(s3)

    # ---- Stage 4: top-K full eval --------------------------------------------
    pool = s2 + s3
    final_picks = _select_best(pool, args.n_top_for_final)
    if not final_picks:
        print("No valid configs to promote to final stage; aborting", file=sys.stderr)
        sys.exit(1)
    final_configs = [r["params"] for r in final_picks]
    print(
        f"\n>>> Promoting {len(final_configs)} configs to full eval:",
        flush=True,
    )
    for i, cfg in enumerate(final_configs, 1):
        prev = final_picks[i - 1]["task_panoptic_iou"]
        print(f"    #{i}  prev_task_panoptic_iou={prev:.4f}  {cfg}")

    s4 = _run_stage(
        "4_final",
        final_configs,
        args.final_demos_per_file,
        gpus,
        out_root,
        save_failures=True,
        skip_existing=args.skip_existing,
        timeout=args.per_run_timeout_s,
        leaderboard_path=leaderboard,
        prior_results=all_results,
    )
    all_results.extend(s4)

    # ---- Final report --------------------------------------------------------
    final_ranked = _select_best(s4, len(s4))
    elapsed = time.time() - sweep_t0
    print(f"\n=== FINAL LEADERBOARD ({elapsed / 60:.1f} min total) ===", flush=True)
    for i, r in enumerate(final_ranked, 1):
        pi = r["task_panoptic_iou"] or 0.0
        i50 = r["task_iou50"] or 0.0
        f1 = r["task_f1_50"] or 0.0
        ce = r["task_mean_count_error"] or 0.0
        ov = r["overall_panoptic_iou"] or 0.0
        print(
            f"  #{i}  task_panoptic_iou={pi:.4f}  iou@50={i50:.4f}  "
            f"f1@50={f1:.4f}  Δcount={ce:+.2f}  overall_panoptic_iou={ov:.4f}"
        )
        print(f"      params:  {r['params']}")
        print(f"      out_dir: {r['out_dir']}")

    if final_ranked:
        winner = final_ranked[0]
        (out_root / "best_config.json").write_text(
            json.dumps(
                {
                    "task_panoptic_iou": winner["task_panoptic_iou"],
                    "task_iou50": winner["task_iou50"],
                    "task_f1_50": winner["task_f1_50"],
                    "task_mean_count_error": winner["task_mean_count_error"],
                    "overall_panoptic_iou": winner["overall_panoptic_iou"],
                    "params": winner["params"],
                    "out_dir": winner["out_dir"],
                    "command": " ".join(
                        _build_cmd(
                            winner["params"],
                            Path(winner["out_dir"]),
                            args.final_demos_per_file,
                            save_failures=True,
                        )
                    ),
                },
                indent=2,
            )
        )
        print(f"\nWrote {out_root}/best_config.json and {out_root}/leaderboard.csv")


if __name__ == "__main__":
    main(tyro.cli(SweepArgs))
