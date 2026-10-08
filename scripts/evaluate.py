"""Evaluation entry point for LIBERO-PRO / RoboCasa / CALVIN / real UR5e tasks.

Env and policy backends are selected independently via tyro subcommands, e.g.
``env:libero policy:openpi`` or ``env:ur5 policy:lerobot``. See ``docs/ur5.md``
for the real-robot path.

The argument tree and the server/client lifecycle both live in the library
(:mod:`policy_abstraction.evaluation.config` and
:mod:`policy_abstraction.evaluation.backends`) rather than here, because the
operator-driven collection GUI in ``scripts/experiments/collect_experiment.py``
runs episodes against the same backends with the same flags. This file is the
batch driver: it loops over a task suite and writes the suite-level summary.
Every ``Args`` name is re-exported below, so ``import evaluate; evaluate.Args``
still works.
"""

import os

# Must be set before torch is imported. Reduces CUDA allocator fragmentation so
# the SAM3 tracker can reclaim memory between episodes via empty_cache().
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import dataclasses
import json
import logging
import pathlib
import time

import tqdm
import tyro
from loguru import logger

from policy_abstraction.data.recorder import RobodmRecorder
from policy_abstraction.evaluation.backends import (
    EvalBackends,
    build_runner_components,
    cleanup_servers,
    dynamics_model_for,
    resolve_flip_u,
    start_backends,
    start_env_server,
)
from policy_abstraction.evaluation.config import (
    Args,
    CosmosPolicyArgs,
    GrootPolicyArgs,
    LerobotPolicyArgs,
    LiberoEnvArgs,
    OpenpiPolicyArgs,
    env_kind,
    eval_type_for,
)
from policy_abstraction.utils.provenance import collect_provenance, write_provenance
from policy_abstraction.utils.seeding import seed_everything
from policy_abstraction.visualization.profiling_plot import save_profiling_plot
from policy_abstraction.visualization.rerun import require_viewer, viewer_url

# Backwards-compatible aliases: these were module-private helpers here before
# they moved into the library, and sweeps/tests referenced a few of them.
_env_kind = env_kind
_eval_type_for = eval_type_for
_resolve_flip_u = resolve_flip_u
_dynamics_model_for = dynamics_model_for
_build_runner_components = build_runner_components
_start_env_server = start_env_server
_cleanup_servers = cleanup_servers

# Silence per-request HTTP/HF chatter from in-process model loaders (SAM2/SAM3/
# DINO via OpenPerceptionClient). Runs at import time, before any HF call.
for _name in (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "filelock",
):
    logging.getLogger(_name).setLevel(logging.WARNING)


def _aggregate_task_profiling(episode_profiles: list[dict]) -> dict:
    """Aggregate per-episode profiling dicts into task-level stats (in seconds)."""
    if not episode_profiles:
        return {}
    # Collect all keys across episodes
    all_keys: set[str] = set()
    for ep in episode_profiles:
        all_keys.update(ep.keys())

    aggregated: dict = {"per_episode": episode_profiles}
    for key in sorted(all_keys):
        means = [ep[key]["mean"] for ep in episode_profiles if key in ep]
        totals = [ep[key]["total"] for ep in episode_profiles if key in ep]
        if means:
            aggregated[key] = {
                "mean_per_step": float(sum(means) / len(means)),
                "total": float(sum(totals)),
            }
    return aggregated


def _save_json(path: str, data: dict) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Episode / task evaluation
# ---------------------------------------------------------------------------


def run_task_evaluation(task_id, task_language, backends: EvalBackends, max_steps):
    """Run all episodes for one task.

    Returns ``(episodes, successes, inferences, profiling)``.
    """
    args = backends.args
    env = backends.env
    env.make_env(
        task_id=task_id,
        camera_heights=args.camera.height,
        camera_widths=args.camera.width,
        seed=args.seed,
    )

    initial_states = env.get_task_init_states(task_id=task_id)
    if isinstance(args.env, LiberoEnvArgs):
        perturb_active = any(
            getattr(args.env.perturb, f.name)
            for f in dataclasses.fields(args.env.perturb)
        )
        task_description = env.language_instruction if perturb_active else task_language
    else:
        task_description = env.language_instruction or task_language

    assert args.output.episode_data_path is not None
    assert args.output.video_out_path is not None
    episodes_path = pathlib.Path(args.output.episode_data_path)
    episodes_path.mkdir(parents=True, exist_ok=True)

    episodes = successes = inferences = 0
    all_profiling: list[dict] = []

    for episode_idx in tqdm.tqdm(
        range(args.iter.start_trial_idx, args.iter.num_trials_per_task)
    ):
        task_name = task_description.replace(" ", "_")
        episode_data_file = (
            str(
                episodes_path
                / f"task_{task_id:02d}_{task_name}_episode_{episode_idx:03d}.vla"
            )
            if args.output.save_episodes
            else None
        )

        runner = backends.build_runner(
            task_description=task_description,
            initial_state=initial_states[episode_idx],
            max_steps=max_steps,
            episode_data_path=episode_data_file,
            episode_idx=episode_idx,
            recorder=RobodmRecorder() if args.output.save_episodes else None,
        )

        success, num_inf = runner.run()
        runner.save_episode(success=bool(success))
        runner.save_video(args.output.video_out_path, episode_idx, success)

        successes += int(success)
        episodes += 1
        inferences += num_inf
        all_profiling.append(runner.get_profiling_summary())
        logger.info(f"Episode {episode_idx + 1} — success: {success}")

    return episodes, successes, inferences, all_profiling


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_eval(args: Args) -> None:
    """Main evaluation loop, shared by every env and policy backend."""
    assert args.output.log_output_path is not None
    assert args.output.summary_out_path is not None
    assert args.output.video_out_path is not None
    assert args.output.episode_data_path is not None
    assert args.policy.port is not None
    assert args.policy.gpus is not None
    assert args.env.port is not None
    logger.add(
        args.output.log_output_path, rotation="10 MB", level="DEBUG", enqueue=True
    )

    seed_everything(args.seed)

    run_dir = pathlib.Path(args.output.summary_out_path).parent
    policy_checkpoints: list[str] = []
    if isinstance(args.policy, OpenpiPolicyArgs) and args.policy.dynamics_model_path:
        policy_checkpoints.append(args.policy.dynamics_model_path)
    elif isinstance(args.policy, GrootPolicyArgs) and args.policy.model_path:
        policy_checkpoints.append(args.policy.model_path)
    elif isinstance(args.policy, CosmosPolicyArgs):
        policy_checkpoints.append(args.policy.model_path)
    elif isinstance(args.policy, LerobotPolicyArgs) and args.policy.model_path:
        policy_checkpoints.append(args.policy.model_path)
    extra_dataset_paths: list[str] = []
    if isinstance(args.env, LiberoEnvArgs):
        extra_dataset_paths.append(args.env.evaluation_config_path)
    provenance = collect_provenance(
        args,
        checkpoint_paths=policy_checkpoints,
        dataset_paths=[
            p
            for p in [args.perception.reference_hdf5]
            + extra_dataset_paths
            + (args.retrieval.dataset_dirs or [])
            if p
        ],
    )
    write_provenance(run_dir, provenance)

    # Before anything expensive: a viewer that was never started would drop
    # every logged frame silently, and you would only find out after the
    # policy server's ~3 minute load.
    if (
        args.viz.enable
        and args.viz.connect
        and args.viz.require_viewer
        and not args.viz.serve
    ):
        require_viewer(viewer_url(args.viz.connect))

    backends = start_backends(args, run_dir)
    suite_info = backends.suite_info
    num_tasks: int = backends.num_tasks
    if args.iter.max_tasks is not None:
        num_tasks = min(num_tasks, args.iter.start_task_idx + args.iter.max_tasks)
    task_languages = backends.task_languages
    evaluation_cfg = suite_info["evaluation_cfg"]
    task_max_steps = backends.task_max_steps
    suite_name = suite_info.get("task_suite_name", env_kind(args.env))
    logger.info(
        f"Task suite: {suite_name} ({num_tasks} tasks; "
        f"max_steps min={min(task_max_steps)} max={max(task_max_steps)})"
    )

    pathlib.Path(args.output.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.output.episode_data_path).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # --- Resume support: load existing summary only when skip_completed is on ---
    existing_summary: dict | None = None
    summary_path = pathlib.Path(args.output.summary_out_path)
    if args.output.skip_completed and summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_summary = json.load(f)
            logger.info(
                f"Found existing summary with "
                f"{len(existing_summary.get('per_task', {}))} completed task(s)"
            )
        except (json.JSONDecodeError, KeyError):
            logger.warning("Existing summary file is corrupted — starting fresh")
            existing_summary = None
    elif summary_path.exists():
        logger.info(
            "Existing summary present but --output.skip-completed is off — overwriting."
        )

    summary = {
        "per_task": existing_summary.get("per_task", {}) if existing_summary else {},
        "total": {},
        "metadata": {
            "args": dataclasses.asdict(args),
            "evaluation_cfg": evaluation_cfg,
            "suite_info": suite_info,
            "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time)),
        },
    }
    _save_json(args.output.summary_out_path, summary)

    # Accumulate totals from previously completed tasks
    total_ep = total_succ = total_inf = 0
    completed_task_ids: set[int] = set()
    for task_key, task_data in summary["per_task"].items():
        completed_task_ids.add(int(task_key))
        total_ep += task_data["episodes"]
        total_succ += task_data["successes"]
        total_inf += task_data["inferences"]

    if completed_task_ids:
        logger.info(
            f"Resuming — skipping {len(completed_task_ids)} completed task(s): "
            f"{sorted(completed_task_ids)}  "
            f"(accumulated {total_succ}/{total_ep} successes so far)"
        )

    try:
        for task_id in tqdm.tqdm(
            range(args.iter.start_task_idx, num_tasks), desc="Tasks"
        ):
            if task_id in completed_task_ids:
                logger.info(f"Task {task_id} already completed — skipping")
                continue

            t0 = time.time()
            ep, succ, inf, task_profiling = run_task_evaluation(
                task_id,
                task_languages[task_id],
                backends,
                task_max_steps[task_id],
            )
            total_ep += ep
            total_succ += succ
            total_inf += inf

            rate = succ / ep * 100
            logger.info(
                f"Task {task_id} — success rate: {rate:.1f}%  avg_inf: {inf / ep:.1f}"
            )
            summary["per_task"][task_id] = {
                "task_description": task_languages[task_id],
                "episodes": ep,
                "successes": succ,
                "inferences": inf,
                "success_rate": rate,
                "avg_inferences_per_episode": inf / ep,
                "time": time.time() - t0,
                "profiling": _aggregate_task_profiling(task_profiling),
            }
            _save_json(args.output.summary_out_path, summary)
    finally:
        backends.shutdown()

    success_rate = total_succ / total_ep * 100
    logger.info(
        f"\n{'=' * 60}\nFINAL: {total_succ}/{total_ep} ({success_rate:.1f}%)\n{'=' * 60}"
    )

    summary["total"] = {
        "episodes": total_ep,
        "successes": total_succ,
        "inferences": total_inf,
        "success_rate": success_rate,
        "avg_inferences_per_episode": total_inf / total_ep,
        "time": time.time() - start_time,
    }
    _save_json(args.output.summary_out_path, summary)
    logger.info(f"Saved summary to {args.output.summary_out_path}")

    profiling_path = pathlib.Path(args.output.summary_out_path).with_name(
        "profiling.png"
    )
    save_profiling_plot(summary, profiling_path)
    logger.info(f"Saved profiling plot to {profiling_path}")


if __name__ == "__main__":
    run_eval(tyro.cli(Args))
