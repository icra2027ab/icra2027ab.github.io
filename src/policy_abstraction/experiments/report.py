"""Turning a pile of rollout records into an answer.

The question the positional experiment asks is not "what is the success rate"
but "does the success rate depend on where the object was, and does guidance
change that dependence". So the summary is computed per factor level, not only
in aggregate, and failures are broken down by cause — a policy that fails by
missing the grasp at the far cells is a different finding from one that fails by
dropping the cable everywhere.

Everything here reads :class:`~policy_abstraction.experiments.store.ExperimentStore`
and writes into its ``report/`` directory. It is pure post-processing: safe to
re-run at any time, including on a half-finished experiment, and safe to run on
someone else's results directory.
"""

from __future__ import annotations

import collections
import json
import pathlib
import time
from typing import Iterable, Optional, Sequence

import numpy as np
from loguru import logger

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.outcomes import (
    NO_OBJECT,
    FailureMode,
    Outcome,
    RolloutRecord,
)
from policy_abstraction.experiments.store import ExperimentStore


def _rate(successes: int, total: int) -> float:
    return successes / total if total else 0.0


def _wilson_interval(
    successes: int, total: int, z: float = 1.96
) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion.

    Wilson rather than the normal approximation because these cells hold very
    few samples — often three — where the normal interval runs past 0 and 1 and
    collapses to zero width at a clean sweep. Wilson stays inside [0, 1] and
    stays wide when n is small, which is the honest picture of "3 for 3".
    """
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z / denom * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def duration_stats(values: Sequence[float]) -> dict:
    """Spread of a set of rollout completion times, in seconds.

    The mean alone hides the shape that matters here: a policy that finishes in
    40 s when it works and grinds to the 120 s timeout when it does not has a
    bimodal distribution, and its mean describes neither mode. So the median and
    the quartiles are reported too, and every consumer gets ``n`` so it can tell
    "90 s typical" from "90 s, once".
    """
    clean = sorted(float(v) for v in values if v is not None and float(v) > 0)
    if not clean:
        return {
            "n": 0,
            "mean_s": 0.0,
            "median_s": 0.0,
            "std_s": 0.0,
            "min_s": 0.0,
            "max_s": 0.0,
            "p25_s": 0.0,
            "p75_s": 0.0,
            "p95_s": 0.0,
            "total_s": 0.0,
        }
    n = len(clean)
    mean = sum(clean) / n
    variance = sum((v - mean) ** 2 for v in clean) / n
    return {
        "n": n,
        "mean_s": mean,
        "median_s": _quantile(clean, 0.5),
        "std_s": variance**0.5,
        "min_s": clean[0],
        "max_s": clean[-1],
        "p25_s": _quantile(clean, 0.25),
        "p75_s": _quantile(clean, 0.75),
        "p95_s": _quantile(clean, 0.95),
        "total_s": sum(clean),
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted, non-empty sequence."""
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return float(sorted_values[low] * (1 - frac) + sorted_values[high] * frac)


def timed(records: Sequence[RolloutRecord]) -> list[RolloutRecord]:
    """The records whose completion time means what it says.

    Drops the warm-up rollouts — the first of each session to reach the policy,
    which paid for compiling a freshly loaded checkpoint. On the real cell that
    is minutes against a ~90 s rollout, so one of them in a plan of thirty-six
    moves the mean further than the effect being measured.
    """
    return [r for r in records if not r.warmup]


def _completion_times(records: Sequence[RolloutRecord]) -> dict:
    """Completion time overall and split by how the rollout ended.

    The split is the useful part. A success time is "how long the policy took to
    do the task"; a failure time is mostly "how long the operator waited before
    calling it", and a timeout is just the cap read back. Pooling the three
    produces a number that means nothing.

    Warm-up rollouts are excluded from every figure here except ``warmup``
    itself, which reports them so the cost is visible rather than hidden.
    """
    scored = [r for r in records if r.outcome.counts_toward_rate]
    warm = timed(scored)
    successes = [r for r in warm if r.succeeded]
    failures = [r for r in warm if not r.succeeded]
    timeouts = [r for r in failures if FailureMode.TIMEOUT in r.failure_modes]
    return {
        "all": duration_stats([r.duration_s for r in warm]),
        "successes": duration_stats([r.duration_s for r in successes]),
        "failures": duration_stats([r.duration_s for r in failures]),
        "timeouts": duration_stats([r.duration_s for r in timeouts]),
        "aborted": duration_stats(
            [r.duration_s for r in timed(records) if r.outcome is Outcome.ABORTED]
        ),
        "warmup": duration_stats([r.duration_s for r in records if r.warmup]),
        "with_warmup": duration_stats([r.duration_s for r in scored]),
    }


NOT_RESOLVED = "no object"
"""Column for a rollout that never committed to any of the named objects."""
UNRECORDED = "unrecorded"
"""Column for a rollout whose acted-on object nobody wrote down."""


def outcomes_by_condition(records: Sequence[RolloutRecord], factor) -> dict:
    """How every rollout at each level of *factor* ended.

    Rows are the factor's levels, columns ``success`` and each failure mode
    that actually occurred — a cross-tab of what was staged against what
    happened. A failure counts once under every mode it was given, so a row's
    columns can sum to more than its rollouts when the operator ticked several.
    Derived entirely from the verdict, so it works on every experiment and on
    data collected before any of this existed.
    """
    modes: list[str] = []
    for record in records:
        if record.succeeded:
            continue
        for mode in _modes_of(record):
            if mode not in modes:
                modes.append(mode)
    columns = ["success"] + sorted(modes)
    rows: dict[str, dict[str, int]] = {}
    for level in factor.levels:
        at_level = [r for r in records if r.assignments.get(factor.name) == level.id]
        counts = {c: 0 for c in columns}
        for record in at_level:
            keys = ["success"] if record.succeeded else _modes_of(record)
            for key in keys:
                counts[key] = counts.get(key, 0) + 1
        rows[level.id] = counts
    return {
        "factor": factor.name,
        "columns": columns,
        "rows": rows,
        "labels": {level.id: level.label for level in factor.levels},
    }


def confusion_matrix(records: Sequence[RolloutRecord], config) -> Optional[dict]:
    """Which object the policy went for against the one the prompt named.

    The disambiguation experiment is a classification task — the prompt names
    one of N objects on the table — so the diagonal is the reference resolved
    and everything off it is the specific mistake made. That is a strictly
    sharper result than a success rate: "it fails on the white cable" and "it
    fails on the white cable *by going to the black one every time*" call for
    different next experiments.

    ``None`` when the experiment names no target — a positional experiment
    stages one object and has nothing to confuse it with.

    Rollouts whose acted-on object was never written down land in an
    ``unrecorded`` column rather than being dropped or assumed onto the
    diagonal: they were collected before the question was asked, and quietly
    counting them as correct would flatter the policy.

    A rollout that went for several objects is counted once under each of
    them, like a failure given several modes — so a row can add up to more
    than its rollouts. ``scored``, ``resolved`` and ``correct`` count
    rollouts, not cells: ``correct`` is how many went for the named object at
    some point, and ``multiple`` how many rows carry more than one object.

    ``success`` carries each row's **task** success — how many rollouts that
    named the object actually completed the task — which is not the
    diagonal: a rollout can reach the right object and still drop it.
    """
    factor = config.target_factor()
    if factor is None:
        return None
    ids = [level.id for level in factor.levels]
    labels = {level.id: level.label for level in factor.levels}
    columns = ids + [NOT_RESOLVED, UNRECORDED]
    rows = {i: {c: 0 for c in columns} for i in ids}
    success = {i: {"n": 0, "successes": 0, "rate": 0.0} for i in ids}
    scored = resolved = correct = multiple = 0
    for record in records:
        intended = record.assignments.get(factor.name)
        if intended not in rows or not record.outcome.counts_toward_rate:
            # An aborted rollout never really ran; it is neither a
            # confusion nor a task attempt.
            continue
        success[intended]["n"] += 1
        success[intended]["successes"] += int(record.succeeded)
        if record.succeeded:
            # A success went for the object it was asked for; that is what the
            # verdict asserts, whatever else was or was not written down.
            reached = [intended]
        else:
            reached = [t for t in _acted_on(record) if t in rows]
            if not reached and record.actual_target == NO_OBJECT:
                reached = [NOT_RESOLVED]
        if not reached:
            # Empty, or an id that is no longer in the object list because the
            # config was edited. Either way nobody can say what it went for.
            reached = [UNRECORDED]
        for column in reached:
            rows[intended][column] += 1
        scored += 1
        resolved += reached != [UNRECORDED]
        correct += intended in reached
        multiple += len(reached) > 1
    for entry in success.values():
        entry["rate"] = _rate(entry["successes"], entry["n"])
    return {
        "factor": factor.name,
        "columns": columns,
        "objects": ids,
        "rows": rows,
        "labels": labels,
        "success": success,
        "correct": correct,
        "scored": scored,
        "resolved": resolved,
        "multiple": multiple,
        "accuracy": _rate(correct, resolved),
    }


def _acted_on(record: RolloutRecord) -> tuple[str, ...]:
    """Every object the record says it went for, oldest column as fallback.

    Reads the list through ``getattr`` rather than the attribute so this
    module also runs against a record type loaded before the list existed —
    a collector that started on the older code and imports the report lazily
    at the end of a session.
    """
    targets = getattr(record, "actual_targets", None)
    if targets:
        return tuple(targets)
    return (record.actual_target,) if record.actual_target else ()


def measured_position_stats(
    records: Sequence[RolloutRecord], config: Optional[ExperimentConfig] = None
) -> dict:
    """How far the staged objects sat from their nominal positions.

    Reads ``observed_positions`` against ``target_positions`` on every record
    that has both. The offsets are the staging error the operator introduced
    — worth knowing before attributing a cell's rate to its position.

    For a position on a grid the offset is measured **in the grid's plane**:
    the nominal point is lifted off the table by the config's marker height
    while the measured centroid sits wherever the object's surface is, and
    that difference is the object's thickness, not a staging error.
    """
    planes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if config is not None:
        for factor in config.factors:
            grid = getattr(factor, "grid", None)
            if grid is not None:
                planes[factor.name] = (np.asarray(grid.u_axis), np.asarray(grid.v_axis))

    offsets: list[float] = []
    measured = 0
    for r in records:
        if not r.observed_positions:
            continue
        measured += 1
        for key, pos in r.observed_positions.items():
            target = (r.target_positions or {}).get(key)
            if target is None:
                continue
            delta = np.asarray(pos, dtype=np.float64) - np.asarray(
                target, dtype=np.float64
            )
            plane = planes.get(key)
            if plane is not None:
                u, v = plane
                delta = float(delta @ u) * u + float(delta @ v) * v
            offsets.append(float(np.linalg.norm(delta)))
    if not offsets:
        return {"measured": measured, "total": len(records)}
    ordered = sorted(offsets)
    return {
        "measured": measured,
        "total": len(records),
        "mean_offset_m": sum(offsets) / len(offsets),
        "median_offset_m": ordered[len(ordered) // 2],
        "max_offset_m": ordered[-1],
    }


def summarize(
    records: Sequence[RolloutRecord],
    config: ExperimentConfig,
    *,
    total_planned: Optional[int] = None,
) -> dict:
    """Aggregate *records* into the summary the report and plots both read.

    Pass :meth:`~policy_abstraction.experiments.store.ExperimentStore.results`
    plus the aborted rows, not the raw log: a rollout whose classification was
    corrected has two rows, and both would be counted here.
    """
    scored = [r for r in records if r.outcome.counts_toward_rate]
    aborted = [r for r in records if r.outcome is Outcome.ABORTED]
    successes = sum(r.succeeded for r in scored)
    measured = measured_position_stats(scored, config)

    # Timing figures come from the rollouts whose clock means something; the
    # counts still come from every scored rollout.
    durations = [r.duration_s for r in timed(scored) if r.duration_s > 0]
    steps = [r.steps for r in scored if r.steps > 0]

    per_factor: dict[str, dict] = {}
    for factor in config.factors:
        by_level: dict[str, dict] = {}
        for level in factor.levels:
            at_level = [r for r in scored if r.assignments.get(factor.name) == level.id]
            n = len(at_level)
            s = sum(r.succeeded for r in at_level)
            low, high = _wilson_interval(s, n)
            by_level[level.id] = {
                "label": level.label,
                "n": n,
                "successes": s,
                "failures": n - s,
                "success_rate": _rate(s, n),
                "ci95": [low, high],
                "failure_modes": _count_failures(at_level),
                "mean_duration_s": (
                    sum(r.duration_s for r in at_level) / n if n else 0.0
                ),
                "duration": duration_stats([r.duration_s for r in timed(at_level)]),
                "duration_successes": duration_stats(
                    [r.duration_s for r in timed(at_level) if r.succeeded]
                ),
                "mean_steps": (sum(r.steps for r in at_level) / n if n else 0.0),
            }
        rates = [v["success_rate"] for v in by_level.values() if v["n"]]
        per_factor[factor.name] = {
            "kind": factor.kind,
            "levels": by_level,
            # The headline number for a distribution-shift experiment: how much
            # the success rate moves as this factor moves. A large spread is the
            # shift being visible; a small one means the policy is indifferent
            # to this axis over the range tested.
            "rate_spread": (max(rates) - min(rates)) if rates else 0.0,
            "worst_level": (
                min(
                    (k for k, v in by_level.items() if v["n"]),
                    key=lambda k: by_level[k]["success_rate"],
                    default=None,
                )
            ),
            "best_level": (
                max(
                    (k for k, v in by_level.items() if v["n"]),
                    key=lambda k: by_level[k]["success_rate"],
                    default=None,
                )
            ),
            "levels_without_data": [k for k, v in by_level.items() if not v["n"]],
        }

    low, high = _wilson_interval(successes, len(scored))
    return {
        "experiment": config.name,
        "task": config.task,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "totals": {
            "planned": total_planned if total_planned is not None else len(scored),
            "collected": len(scored),
            "successes": successes,
            "failures": len(scored) - successes,
            "aborted": len(aborted),
            "success_rate": _rate(successes, len(scored)),
            "ci95": [low, high],
            "mean_duration_s": sum(durations) / len(durations) if durations else 0.0,
            "median_duration_s": _quantile(sorted(durations), 0.5)
            if durations
            else 0.0,
            "mean_steps": sum(steps) / len(steps) if steps else 0.0,
            "total_wall_clock_s": sum(r.duration_s for r in records),
            "warmup_rollouts": sum(1 for r in records if r.warmup),
        },
        "completion_time": _completion_times(records),
        "failure_modes": _count_failures(scored),
        "measured_positions": measured,
        "per_factor": per_factor,
        "per_policy_setting": _by_policy_setting(scored),
        "outcomes_by_condition": [
            outcomes_by_condition(scored, f) for f in config.factors if f.levels
        ],
        "confusion_matrix": confusion_matrix(scored, config),
    }


def _modes_of(record: RolloutRecord) -> list[str]:
    """The failure modes a record carries, or ``other`` for a failure that
    somehow has none, so every failure lands in some column."""
    if record.failure_modes:
        return [m.value for m in record.failure_modes]
    return [FailureMode.OTHER.value]


def _count_failures(records: Iterable[RolloutRecord]) -> dict[str, int]:
    """How many failures carry each mode.

    A failure with several modes is counted under each, so the values can sum
    to more than the number of failures. Report the share of *failures* that
    show a mode, not a share of this total, or the percentages stop meaning
    anything.
    """
    counts = collections.Counter(
        mode for r in records if not r.succeeded for mode in _modes_of(r)
    )
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def _by_policy_setting(records: Sequence[RolloutRecord]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for setting in sorted({r.policy_setting for r in records}):
        subset = [r for r in records if r.policy_setting == setting]
        s = sum(r.succeeded for r in subset)
        low, high = _wilson_interval(s, len(subset))
        out[setting or "unlabelled"] = {
            "n": len(subset),
            "successes": s,
            "success_rate": _rate(s, len(subset)),
            "ci95": [low, high],
        }
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _bar(rate: float, width: int = 12) -> str:
    filled = int(round(rate * width))
    return "#" * filled + "." * (width - filled)


def _clock(seconds: float) -> str:
    """Seconds as something an operator reads at a glance."""
    if float(seconds) <= 0:
        return "—"
    # Rounded before the comparison, so 59.6 s does not print as "60s" beside a
    # 60.0 s "1m00s". The cut is at a minute rather than somewhere more
    # forgiving so a range like "84s – 1m52s" cannot mix two formats.
    total = int(round(float(seconds)))
    if total < 60:
        return f"{total}s"
    minutes, rest = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m{rest:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _duration_row(label: str, stats: dict) -> str:
    if not stats["n"]:
        return f"| {label} | 0 | — | — | — | — | — |"
    return (
        f"| {label} | {stats['n']} | {_clock(stats['median_s'])} "
        f"| {_clock(stats['mean_s'])} | {_clock(stats['std_s'])} "
        f"| {_clock(stats['min_s'])} – {_clock(stats['max_s'])} "
        f"| {_clock(stats['p95_s'])} |"
    )


def render_markdown(
    summary: dict, config: ExperimentConfig, plots: Sequence[str]
) -> str:
    """A report that answers the experiment's question in the first screenful."""
    t = summary["totals"]
    lines: list[str] = [
        f"# {summary['experiment']}",
        "",
        f"*{summary['generated_at']}*",
        "",
    ]
    if summary.get("task"):
        lines += [f"**Task** — {summary['task']}", ""]

    lines += [
        "## Overall",
        "",
        f"- **{t['successes']}/{t['collected']} succeeded "
        f"({t['success_rate']:.0%})**, 95% CI "
        f"[{t['ci95'][0]:.0%}, {t['ci95'][1]:.0%}]",
        f"- {t['collected']}/{t['planned']} planned rollouts collected"
        + (f", {t['aborted']} aborted" if t["aborted"] else ""),
        f"- Median {_clock(t.get('median_duration_s', 0.0))} "
        f"(mean {_clock(t['mean_duration_s'])}) and "
        f"{t['mean_steps']:.0f} steps per rollout"
        + (", warm-up excluded" if t.get("warmup_rollouts") else ""),
        f"- {_clock(t['total_wall_clock_s'])} of robot time in total",
        "",
    ]

    lines += _completion_time_section(summary.get("completion_time") or {})

    measured = summary.get("measured_positions") or {}
    if measured.get("measured"):
        lines += ["### Measured object positions", ""]
        lines.append(
            f"- {measured['measured']}/{measured['total']} rollouts have the "
            "staged object's position measured from the recording's first "
            "frame by the perception pipeline; the camera-view plots draw "
            "those instead of the nominal cell centres."
        )
        if "mean_offset_m" in measured:
            lines.append(
                f"- In-plane offset from the nominal position: median "
                f"{measured['median_offset_m'] * 100:.1f} cm, mean "
                f"{measured['mean_offset_m'] * 100:.1f} cm, "
                f"max {measured['max_offset_m'] * 100:.1f} cm"
            )
        lines.append("")

    if summary["failure_modes"]:
        lines += ["### Failure modes", ""]
        # Share of failures showing the mode: a failure ticked with two modes
        # is one failure, so dividing by the sum of the counts would be wrong.
        total_failures = summary["totals"].get("failures") or sum(
            summary["failure_modes"].values()
        )
        for mode, count in summary["failure_modes"].items():
            label = FailureMode(mode).label
            lines.append(
                f"- **{label}** — {count} ({count / total_failures:.0%} of failures)"
            )
        if sum(summary["failure_modes"].values()) > total_failures:
            lines.append(
                "- _Some failures were given more than one mode, so the shares "
                "add up to more than 100%._"
            )
        lines.append("")

    settings = summary.get("per_policy_setting") or {}
    if len(settings) > 1:
        lines += [
            "## By policy setting",
            "",
            "| Setting | n | Success | 95% CI |",
            "|---|---:|---:|---|",
        ]
        for name, s in settings.items():
            lines.append(
                f"| {name} | {s['n']} | {s['success_rate']:.0%} "
                f"| {s['ci95'][0]:.0%} – {s['ci95'][1]:.0%} |"
            )
        lines.append("")

    for factor_name, data in summary["per_factor"].items():
        lines += [
            f"## {factor_name}",
            "",
            f"Success rate ranges over **{data['rate_spread']:.0%}** across this "
            f"factor's levels"
            + (
                f" (best {data['best_level']}, worst {data['worst_level']})."
                if data["best_level"]
                else "."
            ),
            "",
            "| Level | n | Success | 95% CI | Median time | Median when successful | Failures |",
            "|---|---:|---|---|---:|---:|---|",
        ]
        for level_id, lv in data["levels"].items():
            if not lv["n"]:
                lines.append(f"| {level_id} | 0 | — | — | — | — | — |")
                continue
            modes = ", ".join(
                f"{FailureMode(m).label} x{c}" for m, c in lv["failure_modes"].items()
            )
            timing = lv.get("duration") or {}
            won = lv.get("duration_successes") or {}
            lines.append(
                f"| {level_id} | {lv['n']} "
                f"| `{_bar(lv['success_rate'])}` {lv['success_rate']:.0%} "
                f"| {lv['ci95'][0]:.0%} – {lv['ci95'][1]:.0%} "
                f"| {_clock(timing.get('median_s', lv['mean_duration_s']))} "
                f"| {_clock(won.get('median_s', 0.0))} "
                f"| {modes or '—'} |"
            )
        if data["levels_without_data"]:
            lines += [
                "",
                f"Not yet collected: {', '.join(data['levels_without_data'])}",
            ]
        lines.append("")

    lines += _confusion_section(summary.get("confusion_matrix"))
    lines += _outcome_matrix_section(summary.get("outcomes_by_condition") or [])

    if plots:
        lines += ["## Plots", ""]
        lines += [f"![{pathlib.Path(p).stem}]({p})" for p in plots]
        lines.append("")

    lines += [
        "## Provenance",
        "",
        "```yaml",
        *(f"{k}: {v}" for k, v in config.to_dict().items() if k != "factors"),
        "```",
        "",
    ]
    return "\n".join(lines)


def _matrix_table(
    matrix: dict, corner: str, *, success: Optional[dict] = None
) -> list[str]:
    """One cross-tab as a markdown table, zeros left blank.

    Blank rather than ``0`` because the eye needs to find the few cells that
    hold anything: a 3x5 table of mostly zeros reads as noise, and the pattern
    of *where* the counts fall is the whole point of showing a matrix instead
    of a list. With *success*, a final column carries each row's task
    success rate.
    """
    columns = matrix["columns"]
    labels = matrix.get("labels", {})
    head = [corner] + [_column_title(c) for c in columns]
    if success:
        head.append("task success")
    lines = [
        "| " + " | ".join(head) + " |",
        "|" + "---|" + "---:|" * (len(columns) + (1 if success else 0)),
    ]
    for row_id, counts in matrix["rows"].items():
        cells = [str(counts.get(c, 0) or "") for c in columns]
        if success:
            s = success.get(row_id) or {}
            cells.append(
                f"{s.get('successes', 0)}/{s.get('n', 0)} ({s.get('rate', 0.0):.0%})"
                if s.get("n")
                else "—"
            )
        lines.append("| " + " | ".join([labels.get(row_id, row_id)] + cells) + " |")
    return lines


def _column_title(column: str) -> str:
    if column in (NOT_RESOLVED, UNRECORDED):
        return f"*{column}*"
    return column.replace("_", " ")


def _confusion_section(matrix: Optional[dict]) -> list[str]:
    """Named target against the object actually reached."""
    if not matrix or not matrix.get("scored"):
        return []
    lines = [
        "## Confusion matrix",
        "",
        "Rows are the object the prompt named; columns are the object the "
        "policy actually went for. The diagonal is the reference resolved.",
        "",
    ]
    lines += _matrix_table(matrix, "named \\ went for", success=matrix.get("success"))
    lines.append("")
    resolved, correct = matrix["resolved"], matrix["correct"]
    if resolved:
        lines.append(
            f"**{correct}/{resolved}** went to the object named "
            f"({matrix['accuracy']:.0%})"
            + (
                f" — {matrix['scored'] - resolved} rollout"
                f"{'' if matrix['scored'] - resolved == 1 else 's'} left out, "
                "with no record of what was reached."
                if matrix["scored"] != resolved
                else "."
            )
        )
        multiple = matrix.get("multiple", 0)
        if multiple:
            lines.append(
                f"{multiple} rollout{'' if multiple == 1 else 's'} went for more "
                "than one object and count once under each, so a row can add "
                "up to more than its rollouts."
            )
    else:
        lines.append(
            "No rollout has a record of which object was reached, so there is "
            "nothing to resolve here yet."
        )
    lines.append("")
    return lines


def _outcome_matrix_section(matrices: Sequence[dict]) -> list[str]:
    """What was staged against how it ended, one table per factor."""
    usable = [m for m in matrices if any(any(r.values()) for r in m["rows"].values())]
    if not usable:
        return []
    lines = ["## Outcomes by condition", ""]
    for matrix in usable:
        lines += [f"### {matrix['factor']}", ""]
        lines += _matrix_table(matrix, matrix["factor"])
        lines.append("")
    return lines


def _completion_time_section(timing: dict) -> list[str]:
    """The completion-time table, split by how the rollout ended."""
    overall = timing.get("all") or {}
    if not overall.get("n"):
        return []
    lines = [
        "## Completion time",
        "",
        "Wall clock per rollout, from the operator pressing start to the "
        "verdict being recorded.",
        "",
        "| Rollouts | n | Median | Mean | Std | Range | p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _duration_row("All scored", overall),
    ]
    for key, label in (
        ("successes", "Successes"),
        ("failures", "Failures"),
        ("timeouts", "— of which timed out"),
        ("aborted", "Aborted (excluded from rates)"),
        ("warmup", "Warm-up, excluded above"),
    ):
        stats = timing.get(key) or {}
        if stats.get("n"):
            lines.append(_duration_row(label, stats))
    lines.append("")

    warmup = timing.get("warmup") or {}
    if warmup.get("n"):
        lines += [
            f"The first rollout of each session — {warmup['n']} here, "
            f"{_clock(warmup['median_s'])} at the median — is left out of "
            "every figure above. It pays for the policy server compiling a "
            "freshly loaded checkpoint on its first inference, which is not a "
            "cost the task incurs again.",
            "",
        ]

    won, lost = timing.get("successes") or {}, timing.get("failures") or {}
    if won.get("n") and lost.get("n"):
        delta = lost["median_s"] - won["median_s"]
        lines += [
            f"Successful rollouts finish {_clock(abs(delta))} "
            f"{'faster' if delta > 0 else 'slower'} than failed ones at the "
            f"median ({_clock(won['median_s'])} vs {_clock(lost['median_s'])}).",
            "",
        ]
    return lines


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_report(
    store: ExperimentStore,
    config: Optional[ExperimentConfig] = None,
    *,
    plots: bool = True,
) -> dict:
    """Write ``report/summary.json``, ``report/report.md`` and the plots.

    Returns the summary dict, so a caller can print the headline without
    re-reading the file.
    """
    config = config or store.load_config()
    # Deduped results plus the aborted rows: the rates come from the former,
    # the "how many did we throw away" count from the latter.
    records = store.results() + [
        r for r in store.records() if r.outcome is Outcome.ABORTED
    ]
    store.attach_observed_positions(records)
    plan = store.load_plan()
    summary = summarize(records, config, total_planned=len(plan))

    store.report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = store.report_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    plot_names: list[str] = []
    if plots:
        # Imported here so a headless caller that only wants the numbers never
        # pays for matplotlib, and a broken plotting backend cannot take the
        # summary down with it.
        try:
            from policy_abstraction.experiments.viz import write_plots

            plot_names = [p.name for p in write_plots(store, config, records)]
        except Exception as exc:
            logger.warning(f"Could not render report plots: {exc}")

    report_path = store.report_dir / "report.md"
    report_path.write_text(
        render_markdown(summary, config, plot_names), encoding="utf-8"
    )
    logger.info(f"Wrote {report_path} and {summary_path}")
    return summary


__all__ = [
    "build_report",
    "confusion_matrix",
    "duration_stats",
    "outcomes_by_condition",
    "render_markdown",
    "summarize",
    "timed",
]
