"""Draw a paper-ready failure breakdown, baseline against guided.

The per-experiment ``report/failure_modes.png`` draws one policy on its own
and counts a failure under every mode the operator ticked. For the paper the
two policies of one experiment are drawn on one axis, and every failure is
filed under a single *primary* cause, so the bars of a policy add up to its
rollouts. The primary cause is the most specific mode among those ticked::

    wrong target > collision > dropped > grasp miss > other

*Wrong target* is taken from the record, not only from the tick: a failure
whose "which object did it go for" answer names something other than the
prompted object is a wrong target whatever else was ticked, so the figure
agrees with the report's confusion matrix. ``timeout`` is never primary: on this cell it is the budget running out
*after* the arm missed, dropped or stalled, so it is the consequence of the
mode it was ticked with. A failure ticked ``timeout`` alone or with ``other``
lands in *Other*. ``--counting any`` reproduces the report's overlapping
count instead.

Each figure is sized for half a column of a double-column paper, so the two
sit side by side in one ``\\columnwidth`` figure::

    uv run python scripts/experiments/plot_failure_comparison.py

writes ``failure_modes_disambiguation.{pdf,png}`` and
``failure_modes_position.{pdf,png}`` under ``figures/`` plus a
``failure_modes_preview.png`` that shows the pair at the width they will be
printed at.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Literal, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.patches import Patch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policy_abstraction.experiments.outcomes import (  # noqa: E402
    NO_OBJECT,
    FailureMode,
    RolloutRecord,
)
from policy_abstraction.experiments.report import _acted_on  # noqa: E402
from policy_abstraction.experiments.store import (  # noqa: E402
    ExperimentStore,
    experiment_root,
)

# Most specific first. The first of these a failure carries is its primary
# cause; ``timeout`` and ``other`` are the fallback row.
PRIMARY_ORDER: tuple[FailureMode, ...] = (
    FailureMode.WRONG_TARGET,
    FailureMode.CATASTROPHIC,
    FailureMode.COLLISION,
    FailureMode.DROPPED,
    FailureMode.GRASP_MISS,
    FailureMode.NO_MOTION,
)
OTHER = "other"
SUCCESS = "success"

# The rows of the chart, top to bottom: the outcome, then the causes in the
# order the task unfolds — pick the target, grasp it, carry it, keep it safe.
ROW_ORDER: tuple[str, ...] = (
    SUCCESS,
    FailureMode.WRONG_TARGET.value,
    FailureMode.GRASP_MISS.value,
    FailureMode.DROPPED.value,
    FailureMode.COLLISION.value,
    FailureMode.CATASTROPHIC.value,
    FailureMode.NO_MOTION.value,
    FailureMode.TIMEOUT.value,
    OTHER,
)
ROW_LABELS = {
    SUCCESS: "Success",
    FailureMode.WRONG_TARGET.value: "Wrong target",
    FailureMode.GRASP_MISS.value: "Grasp miss",
    FailureMode.DROPPED.value: "Dropped",
    FailureMode.COLLISION.value: "Collision",
    FailureMode.CATASTROPHIC.value: "E-stop",
    FailureMode.NO_MOTION.value: "No motion",
    FailureMode.TIMEOUT.value: "Timeout",
    OTHER: "Other",
}

# Baseline in a neutral, guided in the accent, so the pair survives every
# colour deficiency and a greyscale print (the baseline is hatched too).
BASELINE_COLOR = "#9a9a9a"
GUIDED_COLOR = "#2a78d6"
INK = "#1a1a20"
MUTED_INK = "#5c5c63"
GRID = "#e4e4e8"


@dataclasses.dataclass(frozen=True)
class Comparison:
    key: str
    """File stem suffix."""
    title: str
    baseline: str
    guided: str


DEFAULT_COMPARISONS = (
    Comparison(
        "disambiguation",
        "Object disambiguation",
        "cable_disambiguation_baseline_v3",
        "cable_disambiguation_guided_v3",
    ),
    Comparison(
        "position",
        "Cable position grid",
        "cable_position_baseline_v3_merged",
        "cable_position_guided_v3_merged",
    ),
)


@dataclasses.dataclass
class Args:
    run: str = "ur5_experiments"
    base: str = "data"
    out: str = "figures"
    """Directory the figures go in, relative to the project root unless
    absolute."""
    counting: Literal["primary", "any"] = "primary"
    """``primary`` files each failure under one cause (see module doc);
    ``any`` counts it under every mode it was given, as the report does."""
    width_in: float = 1.68
    """Width of one figure in inches: just under half of a 3.5 in column,
    leaving room for the gap between subfigures."""
    height_in: float = 1.55
    font_pt: float = 7.0
    """Body font size. IEEE asks figure text to stay readable at print size;
    7 pt on a 1.7 in figure is the floor."""
    disambiguation: Optional[tuple[str, str]] = None
    """Override the (baseline, guided) experiment names for the
    disambiguation figure."""
    position: Optional[tuple[str, str]] = None
    """Override the (baseline, guided) experiment names for the position
    figure."""


@dataclasses.dataclass(frozen=True)
class Loaded:
    records: list[RolloutRecord]
    target_factor: Optional[str]
    """Name of the factor that picks the prompted object, when there is one."""

    def __len__(self) -> int:
        return len(self.records)


def scored_records(root: pathlib.Path) -> Loaded:
    store = ExperimentStore(root)
    factor = store.load_config().target_factor()
    return Loaded(
        [r for r in store.results() if r.outcome.counts_toward_rate],
        factor.name if factor is not None else None,
    )


def went_for_wrong_object(record: RolloutRecord, target_factor: Optional[str]) -> bool:
    """The record names an object it went for, and it is not the prompted one."""
    if target_factor is None:
        return False
    intended = record.assignments.get(target_factor)
    reached = [t for t in _acted_on(record) if t and t != NO_OBJECT]
    return bool(reached) and intended not in reached


def primary_cause(record: RolloutRecord, target_factor: Optional[str]) -> str:
    modes = set(record.failure_modes)
    if went_for_wrong_object(record, target_factor):
        modes.add(FailureMode.WRONG_TARGET)
    for mode in PRIMARY_ORDER:
        if mode in modes:
            return mode.value
    return OTHER


def tally(loaded: Loaded, counting: str) -> dict[str, int]:
    """Rollouts per row. Under ``primary`` the values sum to ``len(loaded)``."""
    counts: dict[str, int] = {}
    for r in loaded.records:
        if r.succeeded:
            counts[SUCCESS] = counts.get(SUCCESS, 0) + 1
        elif counting == "primary":
            key = primary_cause(r, loaded.target_factor)
            counts[key] = counts.get(key, 0) + 1
        else:
            for mode in r.failure_modes or [FailureMode.OTHER]:
                counts[mode.value] = counts.get(mode.value, 0) + 1
    return counts


def style(font_pt: float) -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Nimbus Sans",
                "Liberation Sans",
                "Helvetica",
                "Arial",
                "DejaVu Sans",
            ],
            "font.size": font_pt,
            "axes.labelsize": font_pt,
            "axes.titlesize": font_pt + 0.5,
            "xtick.labelsize": font_pt - 0.5,
            "ytick.labelsize": font_pt,
            "legend.fontsize": font_pt - 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.color": INK,
            "axes.labelcolor": INK,
            "xtick.color": MUTED_INK,
            "ytick.color": INK,
            "axes.edgecolor": MUTED_INK,
            "hatch.linewidth": 0.5,
        }
    )


def draw(
    ax: plt.Axes,
    title: str | None,
    baseline: Loaded,
    guided: Loaded,
    counting: str,
    font_pt: float,
) -> None:
    b_counts, g_counts = tally(baseline, counting), tally(guided, counting)
    rows = [r for r in ROW_ORDER if b_counts.get(r) or g_counts.get(r)]
    n_b, n_g = len(baseline), len(guided)
    b_pct = np.array([100 * b_counts.get(r, 0) / n_b for r in rows])
    g_pct = np.array([100 * g_counts.get(r, 0) / n_g for r in rows])

    y = np.arange(len(rows))[::-1].astype(float)
    # Push the failure rows down a little so the success row reads as its
    # own block, separated from the causes below it.
    y[1:] -= 0.35
    h = 0.36
    ax.barh(
        y + h / 2,
        b_pct,
        height=h,
        color=BASELINE_COLOR,
        hatch="////",
        edgecolor="white",
        linewidth=0,
        zorder=3,
    )
    ax.barh(y - h / 2, g_pct, height=h, color=GUIDED_COLOR, linewidth=0, zorder=3)

    # Counts at the bar ends: the sample sizes behind each percentage.
    for yy, pct, count in (
        *zip(y + h / 2, b_pct, [b_counts.get(r, 0) for r in rows], strict=True),
        *zip(y - h / 2, g_pct, [g_counts.get(r, 0) for r in rows], strict=True),
    ):
        ax.text(
            pct + 1.5,
            yy,
            f"{count}",
            va="center",
            ha="left",
            fontsize=font_pt - 1,
            color=MUTED_INK,
            zorder=4,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([ROW_LABELS[r] for r in rows])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", length=2, pad=1)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)

    xmax = max(b_pct.max(), g_pct.max())
    ax.set_xlim(0, min(100, xmax + 14))
    ticks = [t for t in (0, 25, 50, 75, 100) if t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}%" for t in ticks])
    n = f"{n_b}" if n_b == n_g else f"{n_b} / {n_g}"
    ax.set_xlabel(f"% of rollouts (n = {n})", labelpad=1.5)
    ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6)

    # A hairline under the success row: outcome above, causes below.
    ax.axhline(y[0] - 0.62, color=GRID, linewidth=0.6, zorder=1)

    if title:
        ax.set_title(title, pad=3, fontweight="semibold")

    # The short rows at the bottom leave the lower right empty: the legend
    # lives there rather than costing the figure width or height.
    ax.legend(
        handles=legend_handles(),
        loc="lower right",
        frameon=False,
        handlelength=1.3,
        handleheight=0.8,
        borderaxespad=0.1,
        handletextpad=0.4,
        labelspacing=0.3,
    )


def legend_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=BASELINE_COLOR,
            hatch="////",
            edgecolor="white",
            linewidth=0,
            label="Baseline",
        ),
        Patch(facecolor=GUIDED_COLOR, linewidth=0, label="Guided"),
    ]


def render(
    comparison: Comparison,
    baseline: Loaded,
    guided: Loaded,
    args: Args,
    out_dir: pathlib.Path,
) -> list[pathlib.Path]:
    fig, ax = plt.subplots(
        figsize=(args.width_in, args.height_in), layout="constrained"
    )
    draw(ax, None, baseline, guided, args.counting, args.font_pt)
    # Centred over the whole figure, not the axes, so a title wider than the
    # plot area still fits inside the 1.7 in.
    fig.suptitle(comparison.title, fontweight="semibold", fontsize=args.font_pt + 0.5)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, hspace=0, wspace=0)
    paths = []
    for ext in ("pdf", "png"):
        path = out_dir / f"failure_modes_{comparison.key}.{ext}"
        fig.savefig(
            path,
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
            pad_inches=0.01,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def render_preview(
    comparisons: Sequence[tuple[Comparison, Loaded, Loaded]],
    args: Args,
    path: pathlib.Path,
) -> pathlib.Path:
    """Both figures on one 3.5 in canvas, as the column will show them."""
    fig, axes = plt.subplots(
        1, len(comparisons), figsize=(3.5, args.height_in), layout="constrained"
    )
    for ax, (comparison, baseline, guided) in zip(axes, comparisons, strict=True):
        draw(ax, comparison.title, baseline, guided, args.counting, args.font_pt)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.08)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return path


def main(args: Args) -> int:
    style(args.font_pt)
    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = {"disambiguation": args.disambiguation, "position": args.position}
    loaded = []
    for comparison in DEFAULT_COMPARISONS:
        names = overrides.get(comparison.key) or (
            comparison.baseline,
            comparison.guided,
        )
        comparison = dataclasses.replace(comparison, baseline=names[0], guided=names[1])
        baseline = scored_records(
            experiment_root(args.run, comparison.baseline, base=args.base)
        )
        guided = scored_records(
            experiment_root(args.run, comparison.guided, base=args.base)
        )
        loaded.append((comparison, baseline, guided))
        for path in render(comparison, baseline, guided, args, out_dir):
            print(path)
        for name, records in (
            (comparison.baseline, baseline),
            (comparison.guided, guided),
        ):
            counts = tally(records, args.counting)
            print(
                f"  {name}: n={len(records)} "
                + ", ".join(
                    f"{ROW_LABELS[k]}={counts[k]}" for k in ROW_ORDER if k in counts
                )
            )
    print(render_preview(loaded, args, out_dir / "failure_modes_preview.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
