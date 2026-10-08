"""Plots for a finished (or in-progress) experiment.

The headline figure is the grid map: every cell coloured by its success rate,
green through red, annotated with the raw counts. That is the picture a
positional distribution shift makes — a policy that works near the base and
fails at reach shows up as a colour gradient, not a number.

Everything is matplotlib on the Agg backend and writes PNGs into the store's
``report/`` directory. Nothing here touches the robot or the env.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from policy_abstraction.experiments.config import ExperimentConfig  # noqa: E402
from policy_abstraction.experiments.factors import (  # noqa: E402
    GridPositionFactor,
    ObjectLayoutFactor,
    TargetObjectFactor,
)
from policy_abstraction.experiments.outcomes import (  # noqa: E402
    FailureMode,
    RolloutRecord,
)
from policy_abstraction.experiments.report import (  # noqa: E402
    confusion_matrix,
    outcomes_by_condition,
)
from policy_abstraction.experiments.store import ExperimentStore  # noqa: E402

# Red at 0, amber at the midpoint, green at 1 — the requested reading, with a
# midpoint that keeps a 50% cell from being mistaken for either extreme.
SUCCESS_CMAP = LinearSegmentedColormap.from_list(
    "success", ["#c0392b", "#e0a30c", "#27923f"]
)
NO_DATA_COLOR = "#d9d9de"


def _cell_stats(
    records: Sequence[RolloutRecord], factor: GridPositionFactor
) -> dict[str, tuple[int, int]]:
    """``{level_id: (successes, n)}`` over scored records."""
    stats: dict[str, tuple[int, int]] = {}
    for level in factor.levels:
        at = [
            r
            for r in records
            if r.outcome.counts_toward_rate
            and r.assignments.get(factor.name) == level.id
        ]
        stats[level.id] = (sum(r.succeeded for r in at), len(at))
    return stats


def plot_grid_success(
    records: Sequence[RolloutRecord],
    factor: GridPositionFactor,
    path: pathlib.Path,
    *,
    title: str = "",
) -> pathlib.Path:
    """Success rate per grid cell, laid out the way the grid is laid out."""
    grid = factor.grid
    stats = _cell_stats(records, factor)

    rates = np.full((grid.rows, grid.cols), np.nan)
    for row, col in factor.cells:
        successes, n = stats.get(grid.label((row, col)), (0, 0))
        if n:
            rates[row, col] = successes / n

    fig, ax = plt.subplots(figsize=(1.5 * grid.cols + 2.5, 1.4 * grid.rows + 2.0))
    masked = np.ma.masked_invalid(rates)
    cmap = SUCCESS_CMAP.copy()
    cmap.set_bad(NO_DATA_COLOR)
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")

    for row in range(grid.rows):
        for col in range(grid.cols):
            label = grid.label((row, col))
            successes, n = stats.get(label, (0, 0))
            if not n:
                ax.text(
                    col,
                    row,
                    f"{label}\nno data",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#55555c",
                )
                continue
            rate = successes / n
            # White on the dark ends of the ramp, near-black in the amber middle.
            colour = "white" if rate < 0.35 or rate > 0.75 else "#101014"
            ax.text(
                col,
                row,
                f"{label}\n{rate:.0%}\n{successes}/{n}",
                ha="center",
                va="center",
                fontsize=9,
                color=colour,
                fontweight="bold",
            )

    ax.set_xticks(range(grid.cols))
    ax.set_xticklabels([str(c + grid.first_col) for c in range(grid.cols)])
    ax.set_yticks(range(grid.rows))
    ax.set_yticklabels([grid.label((r, 0))[:-1] for r in range(grid.rows)])
    ax.set_xlabel(f"column  ->  {grid.cell_size_u * 100:.0f} cm cells along u")
    ax.set_ylabel(f"row  ->  {grid.cell_size_v * 100:.0f} cm cells along v")
    ax.set_title(title or f"Success rate by {factor.name}")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.04, label="success rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_grid_outcomes(
    records: Sequence[RolloutRecord],
    factor: GridPositionFactor,
    path: pathlib.Path,
) -> pathlib.Path:
    """Every individual rollout as a dot on the grid — green pass, red fail.

    The rate map answers "how often"; this one answers "in what order and with
    what spread", which is where a drifting setup or a single catastrophic
    outlier becomes visible.
    """
    grid = factor.grid
    fig, ax = plt.subplots(figsize=(1.5 * grid.cols + 2.5, 1.4 * grid.rows + 2.0))

    for row in range(grid.rows + 1):
        ax.plot([-0.5, grid.cols - 0.5], [row - 0.5, row - 0.5], color="#c8c8ce", lw=1)
    for col in range(grid.cols + 1):
        ax.plot([col - 0.5, col - 0.5], [-0.5, grid.rows - 0.5], color="#c8c8ce", lw=1)

    by_cell: dict[str, list[RolloutRecord]] = {}
    for r in records:
        if r.outcome.counts_toward_rate:
            by_cell.setdefault(r.assignments.get(factor.name, ""), []).append(r)

    for level_id, at in by_cell.items():
        try:
            row, col = factor.cell(level_id)
        except KeyError:
            continue
        n = len(at)
        # Lay the attempts out on a small arc inside the cell so overlapping
        # repeats stay countable.
        # np.zeros(n), not [0.0]: a cell with no rollouts must
        # produce no angles, or the zip below is unbalanced.
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False) if n > 1 else np.zeros(n)
        radius = 0.0 if n == 1 else 0.22
        for rec, angle in zip(at, angles, strict=True):
            ax.scatter(
                col + radius * np.cos(angle),
                row + radius * np.sin(angle),
                s=110,
                c="#27923f" if rec.succeeded else "#c0392b",
                marker="o" if rec.succeeded else "X",
                edgecolors="white",
                linewidths=1.0,
                zorder=3,
            )

    ax.set_xlim(-0.6, grid.cols - 0.4)
    ax.set_ylim(grid.rows - 0.4, -0.6)
    ax.set_xticks(range(grid.cols))
    ax.set_xticklabels([str(c + grid.first_col) for c in range(grid.cols)])
    ax.set_yticks(range(grid.rows))
    ax.set_yticklabels([grid.label((r, 0))[:-1] for r in range(grid.rows)])
    ax.set_aspect("equal")
    ax.set_title(f"Individual rollouts by {factor.name}")
    ax.scatter([], [], c="#27923f", marker="o", label="success")
    ax.scatter([], [], c="#c0392b", marker="X", label="failure")
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _rate_rgb(rate: float) -> tuple[int, int, int]:
    """Success rate to an RGB triple off the shared red-amber-green ramp."""
    r, g, b, _ = SUCCESS_CMAP(float(rate))
    return (int(r * 255), int(g * 255), int(b * 255))


CELL_FAILED_RGB = (225, 70, 60)
CELL_GREEN_LOW_RGB = (190, 238, 160)
CELL_GREEN_HIGH_RGB = (0, 100, 30)


def _cell_rgb(successes: int, n: int) -> tuple[int, int, int]:
    """Cell tint for the camera view: red for none, a green ramp for any.

    A cell that ever succeeded is a different finding from one that never
    did, so the two are not on one ramp: any success is green, from pale at
    one-in-three to deep at three-in-three, and only a clean sweep of
    failures is red.
    """
    if not n or not successes:
        return CELL_FAILED_RGB
    t = successes / n
    return tuple(
        int(round(lo + (hi - lo) * t))
        for lo, hi in zip(CELL_GREEN_LOW_RGB, CELL_GREEN_HIGH_RGB, strict=True)
    )  # type: ignore[return-value]


def plot_grid_success_projected(
    records: Sequence[RolloutRecord],
    factor: GridPositionFactor,
    store: ExperimentStore,
    path: pathlib.Path,
    *,
    scale: float = 3.0,
) -> pathlib.Path | None:
    """The success map drawn onto the actual camera view of the workcell.

    The abstract heatmap answers "which cell", but reading it means
    translating ``B3`` back into a place on the table. This draws the same
    numbers where the object actually was: each cell tinted by its rate,
    and one dot per rollout — filled green for a success, hollow red for a
    failure. A rollout whose object position was **measured** from its
    recording (``observed_positions``, see
    :mod:`policy_abstraction.experiments.measure`) gets its dot exactly
    there; the rest are laid on a small ring around the cell centre so
    repeats stay countable.

    Returns ``None`` when the experiment has no saved reference frame or
    the camera was uncalibrated, since there is then nothing to project
    onto. The abstract map is always written regardless.
    """
    image, camera = store.load_reference_view()
    if image is None:
        return None
    intrinsic = camera.get("intrinsic")
    extrinsic = camera.get("extrinsic")
    if intrinsic is None or extrinsic is None:
        return None

    from policy_abstraction.experiments.overlay import OverlayCanvas

    canvas = OverlayCanvas(
        np.asarray(image)[..., :3],
        intrinsic=np.asarray(intrinsic),
        extrinsic=np.asarray(extrinsic),
        preprocessed=bool(camera.get("preprocessed", True)),
        scale=scale,
    )
    grid = factor.grid
    stats = _cell_stats(records, factor)

    for cell in factor.cells:
        label = grid.label(cell)
        successes, n = stats.get(label, (0, 0))
        if n:
            canvas.shade_cell(grid, cell, _cell_rgb(successes, n), alpha=0.45)
        else:
            canvas.shade_cell(grid, cell, (150, 150, 158), alpha=0.18)
    canvas.draw_grid(grid, labels=False, cells=factor.cells)

    by_cell: dict[str, list[RolloutRecord]] = {}
    for record in records:
        if record.outcome.counts_toward_rate:
            by_cell.setdefault(record.assignments.get(factor.name, ""), []).append(
                record
            )
    n_measured = 0
    n_scored = sum(len(by_cell.get(grid.label(c), [])) for c in factor.cells)

    for cell in factor.cells:
        label = grid.label(cell)
        centre = canvas.cell_center_px(grid, cell)
        if centre is None:
            continue
        at = by_cell.get(label, [])
        # Size the dots and the label off the cell's own projected
        # box, so both stay inside the cell under perspective — the near
        # row's cells are a good deal larger on screen than the far row's,
        # and a single shared offset either overflows the far cells or
        # looks lost in the near ones.
        corners, ok = canvas.project(grid.cell_corners(cell))
        span = float(np.linalg.norm(corners[0] - corners[2])) if ok.all() else 40.0
        radius = 0.0 if len(at) <= 1 else 0.20 * span
        # np.zeros(len(at)), not [0.0]: a cell with no rollouts must
        # produce no angles, or the zip below is unbalanced.
        angles = (
            np.linspace(0, 2 * np.pi, len(at), endpoint=False)
            if len(at) > 1
            else np.zeros(len(at))
        )
        dot = max(3, int(round(0.035 * span)))
        for record, angle in zip(at, angles, strict=True):
            px = (
                int(centre[0] + radius * np.cos(angle)),
                int(centre[1] + radius * np.sin(angle)),
            )
            measured = record.observed_positions.get(factor.name)
            if measured is not None:
                mpx, ok = canvas.project(np.asarray([measured], dtype=np.float64))
                if ok[0]:
                    px = (int(mpx[0][0]), int(mpx[0][1]))
                    n_measured += 1
            if record.succeeded:
                cv2.circle(canvas.image, px, dot, (60, 235, 90), -1, cv2.LINE_AA)
                cv2.circle(canvas.image, px, dot, (12, 60, 20), 1, cv2.LINE_AA)
            else:
                # Hollow, so a failure reads as a miss even in greyscale
                # and even where it sits on top of a green cell.
                cv2.circle(
                    canvas.image,
                    px,
                    dot,
                    (235, 70, 60),
                    max(2, dot // 2),
                    cv2.LINE_AA,
                )

        successes, n = stats.get(label, (0, 0))
        # Just the fraction, small, to the right of the centre and kept
        # inside the cell: the cell letters are on the grid itself, and a
        # big label over the centre hid the very dots it was counting.
        text = f"{successes}/{n}" if n else "-"
        text_scale = 0.7
        tw, th = canvas.text_size(text, scale=text_scale)
        x = int(centre[0] + 0.16 * span)
        if ok.all():
            right_edge = int(corners[:, 0].max()) - max(2, int(0.04 * span))
            x = min(x, right_edge - tw)
        canvas._text(
            text,
            (x, int(centre[1] + th / 2)),
            (255, 255, 255),
            scale=text_scale,
            outline=True,
        )

    canvas.draw_badge(f"{factor.name}: success by cell", line=0)
    canvas.draw_badge("filled = success   ring = failure", line=1)
    line = 2
    if n_measured:
        canvas.draw_badge(
            f"dots at measured object positions ({n_measured}/{n_scored})",
            line=line,
        )
        line += 1

    cv2.imwrite(str(path), canvas.image[:, :, ::-1])
    return path


STAGING_BANNER_RGB = (43, 108, 176)
"""The GUI's staging-state banner colour, so a saved staging view matches it."""


def render_reference_staging(
    store: ExperimentStore,
    config: ExperimentConfig,
    record: RolloutRecord,
    path: pathlib.Path,
    *,
    scale: float = 3.0,
) -> pathlib.Path | None:
    """The reference frame annotated the way the GUI showed *record*'s condition.

    Every factor draws its target for the rollout's assignments — the grid
    and its highlighted cell, or the layout's region with each staged object
    and the named target picked out — under the same banner the operator saw.
    It is the picture to put next to the results: what was asked for, on the
    frame the results are drawn on.

    Returns ``None`` when there is no reference frame to draw on.
    """
    from policy_abstraction.experiments.overlay import render_staging_view

    image, camera = store.load_reference_view()
    if image is None:
        return None
    intrinsic = camera.get("intrinsic")
    extrinsic = camera.get("extrinsic")
    total = len(store.load_plan())
    by_name = {f.name: f for f in config.factors}
    parts = []
    for name, level_id in record.assignments.items():
        factor = by_name.get(name)
        label = level_id
        if factor is not None and factor.has_level(level_id):
            label = factor.level(level_id).label
        parts.append(f"{name}={label}")
    banner = f"STAGE rollout {record.index + 1}/{total} - {', '.join(parts)}"
    annotated = render_staging_view(
        np.asarray(image)[..., :3],
        config.factors,
        record.assignments,
        intrinsic=None if intrinsic is None else np.asarray(intrinsic),
        extrinsic=None if extrinsic is None else np.asarray(extrinsic),
        scale=scale,
        banner=banner,
        banner_color=STAGING_BANNER_RGB,
        preprocessed=bool(camera.get("preprocessed", True)),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), annotated[:, :, ::-1])
    return path


def plot_measured_positions(
    records: Sequence[RolloutRecord],
    factor: GridPositionFactor,
    path: pathlib.Path,
) -> pathlib.Path | None:
    """Where the object really was, in the grid's own plane, metric.

    Every measured rollout as a dot at its measured ``(u, v)`` on the table,
    over the cell outlines — filled green for a success, hollow red for a
    failure — with a line back to the cell centre it was staged at. The
    camera view shows the same thing in perspective; this is the version
    with a ruler on it, and the one to check that the staging error is
    small against the cell size before reading anything into a per-cell
    rate.

    Returns ``None`` when no rollout has a measured position.
    """
    grid = factor.grid
    origin = np.asarray(grid.origin)
    u_axis, v_axis = np.asarray(grid.u_axis), np.asarray(grid.v_axis)
    points = []
    for record in records:
        measured = record.observed_positions.get(factor.name)
        if measured is None or not record.outcome.counts_toward_rate:
            continue
        level = record.assignments.get(factor.name)
        if level is None or not factor.has_level(level):
            # A cell the factor no longer sweeps: left off, so a factor
            # restricted to a region plots only that region's rollouts.
            continue
        rel = np.asarray(measured, dtype=np.float64) - origin
        cell = factor.cell(level)
        points.append(
            (
                float(rel @ u_axis),
                float(rel @ v_axis),
                (cell[1] + 0.5) * grid.cell_size_u,
                (cell[0] + 0.5) * grid.cell_size_v,
                record.succeeded,
            )
        )
    if not points:
        return None

    # The plot covers the cells the factor sweeps, not the whole grid: a
    # factor restricted to a region (or to the rows within reach) should
    # not be drawn with empty cells around it.
    row_lo = min(r for r, _ in factor.cells)
    row_hi = max(r for r, _ in factor.cells) + 1
    col_lo = min(c for _, c in factor.cells)
    col_hi = max(c for _, c in factor.cells) + 1
    n_rows, n_cols = row_hi - row_lo, col_hi - col_lo
    fig, ax = plt.subplots(figsize=(1.6 * n_cols + 2.5, 1.6 * n_rows + 2.0))
    for r in range(row_lo, row_hi + 1):
        ax.axhline(r * grid.cell_size_v, color="#9a9aa2", lw=0.8)
    for c in range(col_lo, col_hi + 1):
        ax.axvline(c * grid.cell_size_u, color="#9a9aa2", lw=0.8)
    for cell in factor.cells:
        ax.text(
            (cell[1] + 0.5) * grid.cell_size_u,
            (cell[0] + 0.5) * grid.cell_size_v,
            grid.label(cell),
            ha="center",
            va="center",
            fontsize=9,
            color="#8a8a92",
        )
    for u, v, cu, cv_, ok in points:
        ax.plot([cu, u], [cv_, v], color="#c9c9d0", lw=0.8, zorder=1)
        if ok:
            ax.scatter(u, v, s=42, color="#27923f", edgecolor="#0c3c14", zorder=3)
        else:
            ax.scatter(
                u, v, s=42, facecolor="none", edgecolor="#c0392b", linewidth=2, zorder=3
            )
    offsets = [((u - cu) ** 2 + (v - cv_) ** 2) ** 0.5 for u, v, cu, cv_, _ in points]
    ax.set_xlim(col_lo * grid.cell_size_u - 0.05, col_hi * grid.cell_size_u + 0.05)
    ax.set_ylim(row_hi * grid.cell_size_v + 0.05, row_lo * grid.cell_size_v - 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("u along the grid (m)")
    ax.set_ylabel("v along the grid (m)")
    ax.set_title(
        f"Measured {factor.object} positions ({len(points)} rollouts)\n"
        f"offset from cell centre: median {np.median(offsets) * 100:.1f} cm, "
        f"max {max(offsets) * 100:.1f} cm"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_layouts_projected(
    records: Sequence[RolloutRecord],
    layout_factor: ObjectLayoutFactor,
    target_factor,
    store: ExperimentStore,
    path: pathlib.Path,
    *,
    scale: float = 3.0,
) -> pathlib.Path | None:
    """Every staged object drawn on the real camera view, scored as a target.

    The bar chart says *which object* the policy fails on; this says *where*.
    Each layout's objects are drawn at the places the operator was told to put
    them, and the ones that were the named target that rollout are scored —
    filled green for a success, hollow red for a failure. A policy that is
    really reading the prompt fails evenly across the region; one that reaches
    for whatever is nearest fails in a band.

    Returns ``None`` when there is no saved reference frame or the camera was
    uncalibrated, since there is then nothing to project onto.
    """
    image, camera = store.load_reference_view()
    if image is None:
        return None
    intrinsic = camera.get("intrinsic")
    extrinsic = camera.get("extrinsic")
    if intrinsic is None or extrinsic is None:
        return None

    from policy_abstraction.experiments.overlay import (
        COLOR_REGION,
        OverlayCanvas,
    )

    canvas = OverlayCanvas(
        np.asarray(image)[..., :3],
        intrinsic=np.asarray(intrinsic),
        extrinsic=np.asarray(extrinsic),
        preprocessed=bool(camera.get("preprocessed", True)),
        scale=scale,
    )
    canvas.draw_region(layout_factor.region, color=COLOR_REGION)

    scored = [r for r in records if r.outcome.counts_toward_rate]
    target_name = target_factor.name if target_factor is not None else None
    for level in layout_factor.levels:
        positions = layout_factor.positions(level.id)
        at_layout = [
            r for r in scored if r.assignments.get(layout_factor.name) == level.id
        ]
        for object_id, position in positions.items():
            px, valid = canvas.project(np.asarray(position)[None, :])
            if not valid[0]:
                continue
            centre = (int(px[0][0]), int(px[0][1]))
            # Faint dot for every staged position, so the layouts are visible
            # even where nothing has been collected yet.
            cv2.circle(canvas.image, centre, 2, (200, 200, 210), -1, cv2.LINE_AA)
            if target_name is None:
                continue
            as_target = [
                r for r in at_layout if r.assignments.get(target_name) == object_id
            ]
            dot = max(4, int(3.0 * canvas.scale))
            for i, record in enumerate(as_target):
                # Fan repeats out so two rollouts at one place stay countable.
                offset = int(i * 2.2 * dot)
                at = (centre[0] + offset, centre[1])
                if record.succeeded:
                    cv2.circle(canvas.image, at, dot, (60, 235, 90), -1, cv2.LINE_AA)
                    cv2.circle(canvas.image, at, dot, (12, 60, 20), 1, cv2.LINE_AA)
                else:
                    cv2.circle(
                        canvas.image,
                        at,
                        dot,
                        (235, 70, 60),
                        max(2, dot // 2),
                        cv2.LINE_AA,
                    )

    canvas.draw_badge(f"{layout_factor.name}: outcome where the target stood", line=0)
    canvas.draw_badge("filled = success   ring = failure   dot = staged", line=1)
    cv2.imwrite(str(path), canvas.image[:, :, ::-1])
    return path


def plot_factor_rates(
    records: Sequence[RolloutRecord],
    factor,
    path: pathlib.Path,
) -> pathlib.Path:
    """Success rate per level with Wilson error bars — for non-grid factors."""
    from policy_abstraction.experiments.report import _wilson_interval

    labels, rates, lows, highs, ns = [], [], [], [], []
    for level in factor.levels:
        at = [
            r
            for r in records
            if r.outcome.counts_toward_rate
            and r.assignments.get(factor.name) == level.id
        ]
        n = len(at)
        s = sum(r.succeeded for r in at)
        rate = s / n if n else 0.0
        low, high = _wilson_interval(s, n)
        labels.append(level.label)
        rates.append(rate)
        lows.append(max(0.0, rate - low))
        highs.append(max(0.0, high - rate))
        ns.append(n)

    fig, ax = plt.subplots(figsize=(max(4.0, 1.1 * len(labels) + 2.0), 4.0))
    x = np.arange(len(labels))
    ax.bar(x, rates, color=[SUCCESS_CMAP(r) for r in rates], edgecolor="#33333a")
    ax.errorbar(
        x, rates, yerr=[lows, highs], fmt="none", ecolor="#33333a", capsize=4, lw=1.2
    )
    for xi, (rate, n) in enumerate(zip(rates, ns, strict=True)):
        ax.text(xi, min(1.0, rate + 0.04), f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if len(labels) <= 6 else 40, ha="center")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("success rate")
    ax.set_title(f"Success rate by {factor.name}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_failure_modes(
    records: Sequence[RolloutRecord], path: pathlib.Path
) -> pathlib.Path | None:
    """How the failures break down. ``None`` when nothing failed.

    A failure given several modes appears under each of them, so the bars can
    sum to more than the failure count in the title.
    """
    failures = [r for r in records if r.outcome.counts_toward_rate and not r.succeeded]
    if not failures:
        return None
    counts: dict[str, int] = {}
    for r in failures:
        for mode in r.failure_modes or [FailureMode.OTHER]:
            counts[mode.value] = counts.get(mode.value, 0) + 1
    order = sorted(counts.items(), key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(6.0, max(2.4, 0.5 * len(order) + 1.4)))
    ax.barh(
        [FailureMode(m).label for m, _ in order],
        [c for _, c in order],
        color="#c0392b",
        edgecolor="#33333a",
    )
    for i, (_, c) in enumerate(order):
        ax.text(c, i, f" {c}", va="center", fontsize=9)
    ax.set_xlabel("rollouts")
    ax.set_title(f"Failure modes ({len(failures)} failures)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_progress(
    records: Sequence[RolloutRecord], path: pathlib.Path
) -> pathlib.Path | None:
    """Cumulative success rate over the session — a drift check.

    A rate that trends within a single session usually means something outside
    the experiment changed: the operator's staging got sloppier, the lighting
    moved, the gripper pads wore. Worth seeing before drawing conclusions.
    """
    scored = [r for r in records if r.outcome.counts_toward_rate]
    if len(scored) < 3:
        return None
    outcomes = np.array([float(r.succeeded) for r in scored])
    cumulative = np.cumsum(outcomes) / np.arange(1, len(outcomes) + 1)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(np.arange(1, len(outcomes) + 1), cumulative, color="#2b6cb0", lw=1.8)
    ax.scatter(
        np.arange(1, len(outcomes) + 1),
        outcomes,
        s=26,
        c=["#27923f" if o else "#c0392b" for o in outcomes],
        zorder=3,
        alpha=0.75,
    )
    ax.axhline(cumulative[-1], ls="--", lw=1, color="#8a8a92")
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("rollout (collection order)")
    ax.set_ylabel("success")
    ax.set_title(f"Cumulative success rate ({cumulative[-1]:.0%} final)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_completion_times(
    records: Sequence[RolloutRecord], path: pathlib.Path
) -> pathlib.Path | None:
    """How long rollouts took, successes against failures.

    Two strips rather than one histogram: with a few dozen rollouts a histogram
    is mostly empty bins, and the question is anyway not "what is the
    distribution" but "do the ones that work finish sooner". Every rollout is a
    point, so a single 120 s outlier cannot masquerade as a mode.
    """
    scored = [
        r
        for r in records
        # Warm-up rollouts are dropped rather than drawn: one point three times
        # further right than every other sets the x scale and squashes the
        # comparison the plot exists for.
        if r.outcome.counts_toward_rate and r.duration_s > 0 and not r.warmup
    ]
    if len(scored) < 2:
        return None
    excluded = sum(1 for r in records if r.warmup and r.duration_s > 0)
    groups = [
        ("success", [r.duration_s for r in scored if r.succeeded], "#27923f"),
        ("failure", [r.duration_s for r in scored if not r.succeeded], "#c0392b"),
    ]
    groups = [g for g in groups if g[1]]

    fig, ax = plt.subplots(figsize=(7.0, 1.3 * len(groups) + 1.8))
    rng = np.random.default_rng(0)
    # Successes on top: the strips read in the order the legend below names
    # them, and matplotlib counts y upwards from the bottom.
    for i, (_label, values, colour) in enumerate(groups):
        y = len(groups) - 1 - i
        # A little vertical jitter so identical times do not hide each other.
        ax.scatter(
            values,
            y + rng.uniform(-0.13, 0.13, size=len(values)),
            s=42,
            color=colour,
            alpha=0.75,
            edgecolor="#33333a",
            linewidth=0.6,
            zorder=3,
        )
        median = float(np.median(values))
        ax.plot([median, median], [y - 0.28, y + 0.28], color="#101014", lw=2.0)
        ax.text(
            median,
            y + 0.34,
            f"median {median:.0f}s  (n={len(values)})",
            ha="center",
            fontsize=8,
            color="#33333a",
        )
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([label for label, _, _ in reversed(groups)])
    ax.set_ylim(-0.6, len(groups) - 0.25)
    ax.set_xlim(left=0)
    ax.set_xlabel("completion time (s)")
    ax.set_title(
        "Completion time per rollout"
        + (f"  ({excluded} warm-up excluded)" if excluded else "")
    )
    ax.grid(axis="x", color="#e2e2e6", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _target_factor_for(config: ExperimentConfig, layout: ObjectLayoutFactor):
    """The ``target_object`` factor bound to *layout*, if the experiment has one.

    ``None`` for a layout factor nothing names — a legitimate configuration
    (stage the scene, always the same prompt), and one where there is no target
    to score positions against.
    """
    for factor in config.factors:
        if isinstance(factor, TargetObjectFactor) and factor.layout == layout.name:
            return factor
    return None


def plot_matrix(
    matrix: dict,
    path: pathlib.Path,
    *,
    title: str,
    row_label: str,
    column_label: str,
    diagonal: bool = False,
) -> "pathlib.Path | None":
    """A cross-tab as an annotated heatmap.

    Shaded by the share of each *row*, not by the raw count, so a condition run
    three times and one run thirty are read on the same scale — the question a
    confusion matrix answers is "where did this one go", and that is a
    proportion. The counts are printed in the cells so the sample size behind
    each shade stays visible.

    ``None`` when nothing has been collected, so a caller can leave the plot
    out of the report rather than publishing an empty grid.
    """
    rows = matrix.get("rows") or {}
    columns = list(matrix.get("columns") or [])
    row_ids = [r for r in rows if any(rows[r].values())]
    if not row_ids or not columns:
        return None
    labels = matrix.get("labels", {})

    counts = np.array(
        [[rows[r].get(c, 0) for c in columns] for r in row_ids], dtype=float
    )
    totals = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)

    fig, ax = plt.subplots(
        figsize=(1.15 * len(columns) + 3.2, 0.75 * len(row_ids) + 2.6)
    )
    # Sequential, not the red-amber-green of the rate plots: on a cross-tab
    # only the diagonal has a direction ("more is better"), and every cell off
    # it means the opposite — so a scale that colours low as bad would paint
    # the empty cells, which are the good news, in alarm red. Intensity here
    # reads as "how much of this row landed in this cell", nothing more; the
    # ring marks the cells that mean resolved.
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        [c.replace("_", " ") for c in columns], rotation=30, ha="right", fontsize=9
    )
    ax.set_yticks(range(len(row_ids)))
    ax.set_yticklabels([labels.get(r, r) for r in row_ids], fontsize=9)
    ax.set_xlabel(column_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)

    for i in range(len(row_ids)):
        for j in range(len(columns)):
            count = int(counts[i, j])
            if not count:
                continue
            # White on the dark end of either colour map, black on the light.
            ax.text(
                j,
                i,
                f"{count}\n{shares[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if shares[i, j] > 0.6 else "#1a1a20",
            )
    if diagonal:
        # Ring the cells that mean "resolved correctly", so the reading does
        # not depend on the row and column orders happening to line up.
        for i, row_id in enumerate(row_ids):
            if row_id in columns:
                ax.add_patch(
                    Rectangle(
                        (columns.index(row_id) - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#c0392b",
                        linewidth=2,
                    )
                )
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    matrix: dict,
    path: pathlib.Path,
    *,
    title: str = "Which object the policy went for",
) -> "pathlib.Path | None":
    """The disambiguation result as an objects-by-objects heatmap.

    Only the objects are drawn — an N x N square, the diagonal ringed — and
    each row carries its **task success rate** down the right-hand side. The
    two are different questions: the cells say which object the policy went
    for, the rate says whether it completed the task on the object the
    prompt named. A row can be all-diagonal and still fail every time by
    dropping what it picked up.

    Rollouts that reached no object, or whose reached object was never
    written down, are not cells here; they still count in each row's total
    (so the percentages stay honest) and are named in a line under the
    axes when there are any.

    ``None`` when nothing has been collected.
    """
    rows = matrix.get("rows") or {}
    objects = list(
        matrix.get("objects") or [c for c in matrix.get("columns", ()) if c in rows]
    )
    row_ids = [r for r in objects if any(rows.get(r, {}).values())]
    if not row_ids or not objects:
        return None
    labels = matrix.get("labels", {})
    success = matrix.get("success") or {}

    counts = np.array(
        [[rows[r].get(c, 0) for c in objects] for r in row_ids], dtype=float
    )
    totals = np.array([[sum(rows[r].values())] for r in row_ids], dtype=float)
    shares = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    hidden = {
        c: int(sum(rows[r].get(c, 0) for r in row_ids))
        for c in matrix.get("columns", ())
        if c not in objects
    }
    hidden = {c: n for c, n in hidden.items() if n}

    n = len(objects)
    fig, ax = plt.subplots(figsize=(1.9 * n + 4.6, 1.9 * len(row_ids) + 2.8))
    ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels([labels.get(c, c) for c in objects], fontsize=19)
    ax.set_yticks(range(len(row_ids)))
    ax.set_yticklabels([labels.get(r, r) for r in row_ids], fontsize=19)
    ax.set_xlabel("it went for", fontsize=19, labelpad=8)
    ax.set_ylabel("the prompt named", fontsize=19, labelpad=8)
    ax.set_title(title, fontsize=21, pad=14)
    ax.tick_params(length=0)

    for i in range(len(row_ids)):
        for j in range(n):
            count = int(counts[i, j])
            if not count:
                continue
            ax.text(
                j,
                i,
                f"{count}\n{shares[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=30,
                fontweight="bold",
                color="white" if shares[i, j] > 0.6 else "#1a1a20",
            )
    for i, row_id in enumerate(row_ids):
        if row_id in objects:
            ax.add_patch(
                Rectangle(
                    (objects.index(row_id) - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="#c0392b",
                    linewidth=2.5,
                )
            )

    # Task success per row, as a second set of row labels on the right —
    # the fraction over its percentage — headed "Task SR" above the column.
    # A secondary axis shares the main axis' transform, so the labels stay
    # on their rows when the square aspect shrinks the axes box.
    right = ax.secondary_yaxis("right")
    right.set_yticks(range(len(row_ids)))
    right.set_yticklabels(
        [
            (
                f"{success[r]['successes']}/{success[r]['n']}\n{success[r]['rate']:.0%}"
                if success.get(r, {}).get("n")
                else "—"
            )
            for r in row_ids
        ],
        fontsize=27,
        fontweight="bold",
        color="#1f6f2f",
        ha="center",
    )
    right.tick_params(length=0, pad=48)
    ax.annotate(
        "Task\nSR",
        xy=(1.0, 1.0),
        xycoords="axes fraction",
        xytext=(48, 6),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=19,
        fontweight="bold",
        color="#1f6f2f",
        annotation_clip=False,
    )

    if hidden:
        note = "; ".join(f"{v} reached {k}" for k, v in hidden.items())
        ax.annotate(
            f"not shown: {note} (still counted in the row totals)",
            xy=(0.0, 0.0),
            xycoords="axes fraction",
            xytext=(0, -78),
            textcoords="offset points",
            fontsize=13,
            color="#55555c",
            annotation_clip=False,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_plots(
    store: ExperimentStore,
    config: ExperimentConfig,
    records: Sequence[RolloutRecord],
) -> list[pathlib.Path]:
    """Render every plot the experiment's factors support."""
    out: list[pathlib.Path] = []
    store.report_dir.mkdir(parents=True, exist_ok=True)
    for factor in config.factors:
        stem = f"{factor.name}"
        if isinstance(factor, GridPositionFactor):
            # The projected map goes first: it is the one an operator
            # reads, so it should head the report's plot list.
            projected = plot_grid_success_projected(
                records, factor, store, store.report_dir / f"{stem}_on_camera.png"
            )
            if projected is not None:
                out.append(projected)
            measured = plot_measured_positions(
                records, factor, store.report_dir / f"{stem}_measured.png"
            )
            if measured is not None:
                out.append(measured)
            out.append(
                plot_grid_success(
                    records, factor, store.report_dir / f"{stem}_grid_success.png"
                )
            )
            out.append(
                plot_grid_outcomes(
                    records, factor, store.report_dir / f"{stem}_grid_rollouts.png"
                )
            )
        elif isinstance(factor, ObjectLayoutFactor):
            projected = plot_layouts_projected(
                records,
                factor,
                _target_factor_for(config, factor),
                store,
                store.report_dir / f"{stem}_on_camera.png",
            )
            if projected is not None:
                out.append(projected)
            out.append(
                plot_factor_rates(
                    records, factor, store.report_dir / f"{stem}_rates.png"
                )
            )
        else:
            out.append(
                plot_factor_rates(
                    records, factor, store.report_dir / f"{stem}_rates.png"
                )
            )
    confusion = confusion_matrix(records, config)
    if confusion is not None:
        drawn = plot_confusion_matrix(
            confusion, store.report_dir / "confusion_matrix.png"
        )
        if drawn is not None:
            out.append(drawn)
    for factor in config.factors:
        if not factor.levels:
            continue
        drawn = plot_matrix(
            outcomes_by_condition(records, factor),
            store.report_dir / f"{factor.name}_outcomes.png",
            title=f"How each {factor.name} ended",
            row_label=factor.name,
            column_label="outcome",
        )
        if drawn is not None:
            out.append(drawn)
    modes = plot_failure_modes(records, store.report_dir / "failure_modes.png")
    if modes is not None:
        out.append(modes)
    progress = plot_progress(records, store.report_dir / "progress.png")
    if progress is not None:
        out.append(progress)
    timing = plot_completion_times(records, store.report_dir / "completion_time.png")
    if timing is not None:
        out.append(timing)
    return out


__all__ = [
    "SUCCESS_CMAP",
    "plot_completion_times",
    "plot_factor_rates",
    "plot_failure_modes",
    "plot_grid_outcomes",
    "plot_grid_success",
    "plot_grid_success_projected",
    "plot_layouts_projected",
    "plot_matrix",
    "plot_progress",
    "write_plots",
]
