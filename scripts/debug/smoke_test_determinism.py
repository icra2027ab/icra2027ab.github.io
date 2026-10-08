"""Pipeline determinism smoke test.

Runs the full LIBERO-PRO evaluation pipeline multiple times with the same
seed on 2 tasks x 1 episode each, then diffs the recorded per-step
outputs of each phase (retrieval -> abstraction -> alignment -> guidance
-> policy action correction) across two comparison runs.

The first `num_warmup_runs` run(s) populate the shared VLM cache so that
`run_a` and `run_b` hit the cache deterministically.

Usage
-----
    python scripts/smoke_test_determinism.py \
        --task-suite-name libero_object --seed 7

Exit code is 0 when every phase is bit-identical (or within
``compare_tolerance``) across ``run_a`` and ``run_b``; 1 otherwise.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Any, Optional

import numpy as np
import tyro
from loguru import logger

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from evaluate import (  # noqa: E402
    Args,
    DemoAnalysisArgs,
    EnvArgs,
    GuidanceArgs,
    OutputArgs,
    PolicyArgs,
    TaskIterArgs,
    VisualizerArgs,
    eval_libero,
)

from policy_abstraction import PROJECT_ROOT  # noqa: E402
from policy_abstraction.data.loader import VLALoader  # noqa: E402


@dataclasses.dataclass
class SmokeArgs:
    """CLI for the determinism smoke test."""

    seed: int = 7
    task_suite_name: str = "libero_object"
    run_prefix: str = "smoke"  # under data/<run_prefix>/...
    vlm_cache_dir: Optional[str] = (
        None  # shared across runs; defaults to data/<run_prefix>/vlm_cache
    )
    num_warmup_runs: int = 1
    compare_tolerance: float = 0.0  # atol for float diffs; 0 = bit-identity
    use_distance_guidance: bool = True

    # Compute / server passthroughs (forwarded to Args)
    instance: int = 0
    policy_host: str = "0.0.0.0"
    gpus: Optional[str] = None


# Phase -> (source, keys). Order matters: upstream-first so a divergence
# short-circuits downstream "MISMATCH" reports to "SKIP".
_PHASES: list[tuple[str, str, list[str]]] = [
    (
        "Retrieval",
        "metadata",
        [
            "abstraction_pair_labels",
            "abstraction_ref_traj",
            "equivalence_mapping",
            "translated_pairs",
        ],
    ),
    (
        "Abstraction",
        "per_step",
        [
            "abstraction/live_distances",
            "abstraction/ref_distances",
            "abstraction/distance_errors",
        ],
    ),
    (
        "Alignment",
        "per_step",
        ["abstraction/aligned_t", "abstraction/mta_progress"],
    ),
    (
        "Guidance",
        "per_step",
        [
            "guidance/weights",
            "guidance/factor",
            "guidance/relevances",
            "guidance/ori_weight",
            "guidance/gripper_weight",
        ],
    ),
    (
        "Policy correction",
        "per_step",
        ["policy/action_chunks", "action"],
    ),
]


def _build_args(
    smoke: SmokeArgs,
    run_name: str,
    vlm_cache_dir: str,
    base_out_dir: pathlib.Path,
) -> Args:
    """Clone a smoke-test Args for a single evaluation run."""
    run_dir = base_out_dir / run_name
    return Args(
        instance=smoke.instance,
        seed=smoke.seed,
        policy=PolicyArgs(host=smoke.policy_host, gpus=smoke.gpus),
        env=EnvArgs(task_suite_name=smoke.task_suite_name),
        iter=TaskIterArgs(
            num_trials_per_task=1,
            max_tasks=2,
            start_task_idx=0,
            start_trial_idx=0,
        ),
        viz=VisualizerArgs(enable=False),
        guidance=GuidanceArgs(enable=smoke.use_distance_guidance),
        demo_analysis=DemoAnalysisArgs(vlm_cache_dir=vlm_cache_dir),
        output=OutputArgs(
            run=f"{smoke.run_prefix}/{run_name}",
            save_episodes=True,
            episode_data_path=str(run_dir / "episodes"),
            video_out_path=str(run_dir / "videos"),
            summary_out_path=str(run_dir / "summary.json"),
            log_output_path=str(run_dir / "evaluation.log"),
        ),
    )


def _load_episodes(episode_dir: pathlib.Path) -> dict[str, VLALoader]:
    loaders: dict[str, VLALoader] = {}
    for vla_path in sorted(episode_dir.glob("*.vla")):
        loaders[vla_path.stem] = VLALoader(str(vla_path)).load()
    return loaders


def _diff_array(a: np.ndarray, b: np.ndarray, tol: float) -> Optional[dict]:
    """None if arrays match, else a structured diff summary."""
    if a.shape != b.shape:
        return {"reason": f"shape mismatch {a.shape} vs {b.shape}"}
    if a.dtype.kind in ("f", "c") and b.dtype.kind in ("f", "c"):
        if np.allclose(a, b, atol=tol, rtol=0.0, equal_nan=True):
            return None
        diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
        # Locate the first diverging step along axis 0 if it exists.
        first_step: Optional[int] = None
        if diff.ndim >= 1 and diff.shape[0] > 0:
            per_step = diff.reshape(diff.shape[0], -1).max(axis=1)
            above = np.where(per_step > tol)[0]
            if above.size:
                first_step = int(above[0])
        return {
            "max_abs_diff": float(np.nanmax(diff)),
            "first_step": first_step,
        }
    if np.array_equal(a, b):
        return None
    # Integer / bool arrays.
    mismatches = np.where(a != b)[0] if a.ndim >= 1 else None
    first_step = (
        int(mismatches[0]) if mismatches is not None and mismatches.size else None
    )
    return {"reason": "non-float arrays differ", "first_step": first_step}


def _diff_metadata(a: Any, b: Any) -> Optional[str]:
    if a == b:
        return None
    return "metadata differs"


def _compare_phase(
    kind: str,
    keys: list[str],
    loader_a: VLALoader,
    loader_b: VLALoader,
    tol: float,
) -> tuple[bool, str]:
    details: list[str] = []
    ok = True
    for key in keys:
        if kind == "metadata":
            va = loader_a.metadata.get(key)
            vb = loader_b.metadata.get(key)
            if va is None and vb is None:
                continue
            problem = _diff_metadata(va, vb)
            if problem is None:
                details.append(f"{key}=eq")
            else:
                ok = False
                details.append(f"{key}: {problem}")
            continue

        # per_step arrays live on the loader's private `_data` dict.
        data_a = loader_a._data or {}
        data_b = loader_b._data or {}
        arr_a = data_a.get(key)
        arr_b = data_b.get(key)
        if arr_a is None and arr_b is None:
            continue
        if arr_a is None or arr_b is None:
            ok = False
            details.append(f"{key}: present in only one run")
            continue
        diff = _diff_array(np.asarray(arr_a), np.asarray(arr_b), tol)
        if diff is None:
            n = len(arr_a) if hasattr(arr_a, "__len__") else 1
            details.append(f"{key} ok over {n} steps")
        else:
            ok = False
            details.append(f"{key}: {diff}")
    return ok, "; ".join(details) if details else "(no data)"


def _compare_episode(
    loader_a: VLALoader,
    loader_b: VLALoader,
    tol: float,
) -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []
    upstream_diverged = False
    for phase_name, kind, keys in _PHASES:
        if upstream_diverged:
            results.append(("SKIP", phase_name, "upstream phase diverged"))
            continue
        ok, detail = _compare_phase(kind, keys, loader_a, loader_b, tol)
        if ok:
            results.append(("OK", phase_name, detail))
        else:
            results.append(("MISMATCH", phase_name, detail))
            upstream_diverged = True
    return results


_STATUS_TAG = {"OK": "[OK]      ", "MISMATCH": "[MISMATCH]", "SKIP": "[SKIP]    "}


def compare_runs(
    run_a_dir: pathlib.Path,
    run_b_dir: pathlib.Path,
    tol: float,
) -> int:
    eps_a = _load_episodes(run_a_dir)
    eps_b = _load_episodes(run_b_dir)
    shared = sorted(set(eps_a) & set(eps_b))
    only_a = sorted(set(eps_a) - set(eps_b))
    only_b = sorted(set(eps_b) - set(eps_a))
    for m in only_a:
        logger.warning(f"Episode only in run_a: {m}")
    for m in only_b:
        logger.warning(f"Episode only in run_b: {m}")

    total_mismatches = 0
    if not shared:
        logger.error("No shared episodes between run_a and run_b; cannot compare.")
        return 1

    for stem in shared:
        logger.info(f"\n=== {stem} ===")
        results = _compare_episode(eps_a[stem], eps_b[stem], tol)
        for status, phase, detail in results:
            tag = _STATUS_TAG[status]
            logger.info(f"  {tag} {phase:<20s} {detail}")
            if status == "MISMATCH":
                total_mismatches += 1
    return total_mismatches


def main() -> int:
    smoke = tyro.cli(SmokeArgs)
    base_out_dir = PROJECT_ROOT / "data" / smoke.run_prefix
    base_out_dir.mkdir(parents=True, exist_ok=True)

    vlm_cache_dir = smoke.vlm_cache_dir or str(base_out_dir / "vlm_cache")
    pathlib.Path(vlm_cache_dir).mkdir(parents=True, exist_ok=True)

    run_names: list[str] = [f"warmup_{i}" for i in range(smoke.num_warmup_runs)]
    run_names.extend(["run_a", "run_b"])

    for name in run_names:
        logger.info(f"\n########## Evaluation run: {name} ##########")
        args = _build_args(smoke, name, vlm_cache_dir, base_out_dir)
        eval_libero(args)

    run_a_dir = base_out_dir / "run_a" / "episodes"
    run_b_dir = base_out_dir / "run_b" / "episodes"
    logger.info("\n########## Determinism comparison ##########")
    mismatches = compare_runs(run_a_dir, run_b_dir, smoke.compare_tolerance)
    if mismatches == 0:
        logger.info("\nAll phases bit-identical across run_a and run_b.")
        return 0
    logger.error(f"\n{mismatches} phase-level mismatch(es) detected.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
