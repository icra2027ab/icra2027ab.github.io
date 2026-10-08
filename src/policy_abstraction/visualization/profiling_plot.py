"""Generate a profiling dashboard from evaluation summary data."""

import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# Phases to display (order matters for stacking / legend).
_STEP_PHASES = [
    "policy_infer",
    "perception",
    "abstraction",
    "guidance_build",
    "frame_bundle",
    "alignment_plot",
    "visualizer",
]

_SETUP_PHASES = [
    "setup/retrieval_and_analysis",
    "setup/abstractor_setup",
    "setup/viz_logging",
    "setup/interaction_retrieval",
]

_PHASE_LABELS = {
    "policy_infer": "Policy inference",
    "perception": "Perception",
    "abstraction": "Abstraction",
    "guidance_build": "Guidance build",
    "frame_bundle": "Frame bundle",
    "alignment_plot": "Alignment plot",
    "visualizer": "Visualizer",
    "retrieval_setup": "Retrieval + setup",
    "setup/retrieval_and_analysis": "Retrieval & VLM analysis",
    "setup/abstractor_setup": "Abstractor setup",
    "setup/viz_logging": "Viz logging",
    "setup/interaction_retrieval": "Interaction retrieval",
}

_PHASE_COLORS = {
    "policy_infer": "#4C72B0",
    "perception": "#CCB974",
    "abstraction": "#DD8452",
    "guidance_build": "#55A868",
    "frame_bundle": "#C44E52",
    "alignment_plot": "#8172B3",
    "visualizer": "#937860",
    "retrieval_setup": "#DA8BC3",
    "setup/retrieval_and_analysis": "#E07B39",
    "setup/abstractor_setup": "#76B7B2",
    "setup/viz_logging": "#B07AA1",
    "setup/interaction_retrieval": "#FF9DA7",
}


def _build_subtitle(summary: dict) -> str:
    """Build a subtitle string from the summary totals."""
    parts: list[str] = []
    total = summary.get("total", {})
    if (wt := total.get("time")) is not None:
        parts.append(f"Wall time: {wt:.1f}s")
    if (n := total.get("episodes")) is not None:
        parts.append(f"Episodes: {n}")
    if (sr := total.get("success_rate")) is not None:
        parts.append(f"Success rate: {sr:.1f}%")
    return f"  ({', '.join(parts)})" if parts else ""


def _collect_task_data(per_task: dict, task_ids: list):
    """Extract per-step means, retrieval totals, and setup breakdown from per_task."""
    task_labels = [
        per_task[tid].get("task_description", f"Task {tid}") for tid in task_ids
    ]
    task_labels = [lbl[:40] + "\u2026" if len(lbl) > 40 else lbl for lbl in task_labels]

    step_means: dict[str, list[float]] = {p: [] for p in _STEP_PHASES}
    retrieval_totals: list[float] = []
    setup_totals: dict[str, list[float]] = {p: [] for p in _SETUP_PHASES}

    for tid in task_ids:
        prof = per_task[tid].get("profiling", {})
        for phase in _STEP_PHASES:
            step_means[phase].append(
                prof.get(phase, {}).get("mean_per_step", 0.0) * 1e3
            )
        retrieval_totals.append(prof.get("retrieval_setup", {}).get("total", 0.0))
        for phase in _SETUP_PHASES:
            setup_totals[phase].append(prof.get(phase, {}).get("total", 0.0))

    # Global totals across all tasks
    global_totals: dict[str, float] = {}
    for phase in _STEP_PHASES:
        global_totals[phase] = sum(
            per_task[tid].get("profiling", {}).get(phase, {}).get("total", 0.0)
            for tid in task_ids
        )
    global_totals["retrieval_setup"] = sum(retrieval_totals)

    global_setup: dict[str, float] = {}
    for phase in _SETUP_PHASES:
        global_setup[phase] = sum(setup_totals[phase])

    return task_labels, step_means, global_totals, setup_totals, global_setup


def _has_setup_data(global_setup: dict[str, float]) -> bool:
    """Return True if any setup sub-phase has recorded time."""
    return any(v > 0 for v in global_setup.values())


def save_profiling_plot(summary: dict, output_path: str | pathlib.Path) -> None:
    """Save a profiling dashboard PNG from an evaluation summary dict.

    Args:
        summary: The full summary dict (with ``per_task`` and ``total`` keys).
        output_path: Destination file path (e.g. ``data/run/profiling.png``).
    """
    per_task: dict = summary.get("per_task", {})
    if not per_task:
        return

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_ids = sorted(per_task.keys(), key=lambda k: int(k))
    task_labels, step_means, global_totals, setup_totals, global_setup = (
        _collect_task_data(per_task, task_ids)
    )
    has_setup = _has_setup_data(global_setup)

    # --- Layout --------------------------------------------------------------
    n_rows = 2 if has_setup else 1
    fig_h = max(4, len(task_ids) * 0.6 + 1.5)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(14, fig_h * n_rows),
        gridspec_kw={
            "width_ratios": [3, 1.2],
            "height_ratios": [1.2, 1] if has_setup else [1],
        },
    )
    if n_rows == 1:
        axes = np.array([axes])
    ax_bar, ax_pie = axes[0]

    # --- Top-left: horizontal stacked bar chart (mean per-step, ms) ----------
    y_pos = np.arange(len(task_ids))
    left = np.zeros(len(task_ids))
    for phase in _STEP_PHASES:
        vals = np.array(step_means[phase])
        ax_bar.barh(
            y_pos,
            vals,
            left=left,
            height=0.6,
            label=_PHASE_LABELS[phase],
            color=_PHASE_COLORS[phase],
        )
        left += vals

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(task_labels, fontsize=8)
    ax_bar.set_xlabel("Mean time per replan step (ms)")
    ax_bar.set_title("Per-step time breakdown by task")
    ax_bar.invert_yaxis()
    ax_bar.legend(fontsize=7, loc="lower right")

    # --- Top-right: pie chart (total time share) -----------------------------
    all_phases = _STEP_PHASES + ["retrieval_setup"]
    sizes = [global_totals[p] for p in all_phases]
    colors = [_PHASE_COLORS[p] for p in all_phases]
    labels = [_PHASE_LABELS[p] for p in all_phases]

    total = sum(sizes)
    if total > 0:
        filtered = [
            (s, c, lbl)
            for s, c, lbl in zip(sizes, colors, labels, strict=True)
            if s / total > 0.005
        ]
        if filtered:
            sizes, colors, labels = zip(*filtered, strict=True)

        ax_pie.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            textprops={"fontsize": 7},
            pctdistance=0.75,
            startangle=90,
        )
        ax_pie.set_title("Total time share")
    else:
        ax_pie.text(0.5, 0.5, "No profiling data", ha="center", va="center")
        ax_pie.set_axis_off()

    # --- Bottom row: retrieval + setup breakdown -----------------------------
    if has_setup:
        ax_setup_bar, ax_setup_pie = axes[1]

        # Bottom-left: horizontal stacked bar per task (seconds)
        left = np.zeros(len(task_ids))
        for phase in _SETUP_PHASES:
            vals = np.array(setup_totals[phase])
            ax_setup_bar.barh(
                y_pos,
                vals,
                left=left,
                height=0.6,
                label=_PHASE_LABELS[phase],
                color=_PHASE_COLORS[phase],
            )
            left += vals

        ax_setup_bar.set_yticks(y_pos)
        ax_setup_bar.set_yticklabels(task_labels, fontsize=8)
        ax_setup_bar.set_xlabel("Total time per task (s)")
        ax_setup_bar.set_title("Retrieval + setup breakdown by task")
        ax_setup_bar.invert_yaxis()
        ax_setup_bar.legend(fontsize=7, loc="lower right")

        # Bottom-right: pie chart of setup sub-phases
        setup_sizes = [global_setup[p] for p in _SETUP_PHASES]
        setup_colors = [_PHASE_COLORS[p] for p in _SETUP_PHASES]
        setup_labels = [_PHASE_LABELS[p] for p in _SETUP_PHASES]

        setup_total = sum(setup_sizes)
        if setup_total > 0:
            filtered = [
                (s, c, lbl)
                for s, c, lbl in zip(
                    setup_sizes, setup_colors, setup_labels, strict=True
                )
                if s / setup_total > 0.005
            ]
            if filtered:
                setup_sizes, setup_colors, setup_labels = zip(*filtered, strict=True)

            ax_setup_pie.pie(
                setup_sizes,
                labels=setup_labels,
                colors=setup_colors,
                autopct="%1.1f%%",
                textprops={"fontsize": 7},
                pctdistance=0.75,
                startangle=90,
            )
            ax_setup_pie.set_title("Setup time share")
        else:
            ax_setup_pie.text(0.5, 0.5, "No setup data", ha="center", va="center")
            ax_setup_pie.set_axis_off()

    # --- Finalize ------------------------------------------------------------
    fig.suptitle(
        "Evaluation Profiling" + _build_subtitle(summary),
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
