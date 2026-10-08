"""Summarize the real_perception_HL perturbation sweep as a markdown table.

One row, one column per (perturbation, task_suite), plus an avg per perturbation,
plus a TOTAL block. Pulls `total.success_rate` from each run's summary.json;
falls back to counting `episodes/*_S.json` vs `*_F.json` for in-progress runs
(those cells are highlighted yellow in the terminal and marked with `*` in LaTeX).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import tyro

DEFAULT_RUN_NAME = "pi05_no_guide"

YELLOW = "\033[33m"
RESET = "\033[0m"


@dataclass
class Args:
    """Render a sweep's success-rate table for a given run folder."""

    run_name: str = DEFAULT_RUN_NAME
    """Sweep --output.run value (folder under data/)."""

    data_root: Path | None = None
    """Explicit path to the run folder. Overrides --run-name when set."""

    no_color: bool = False
    """Disable ANSI color in the terminal output."""


CANONICAL_SUITES = ["libero_goal", "libero_spatial", "libero_10", "libero_object"]
SUITE_LABELS = {
    "libero_goal": "Goal",
    "libero_spatial": "Spatial",
    "libero_10": "10",
    "libero_object": "Object",
}
CANONICAL_PERTURBATIONS = ["task", "swap", "object", "language"]
PERT_LABELS = {
    "task": "Task Perturbation",
    "swap": "Pos Perturbation",
    "object": "Object Perturbation",
    "language": "Language Perturbation",
}


def suite_label(suite: str) -> str:
    return SUITE_LABELS.get(suite, suite.removeprefix("libero_").title())


def pert_label(pert: str) -> str:
    return PERT_LABELS.get(pert, pert)


def discover_suites(root: Path) -> list[str]:
    if not root.exists():
        return list(CANONICAL_SUITES)
    found = {
        d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("libero_")
    }
    ordered = [s for s in CANONICAL_SUITES if s in found]
    extras = sorted(found - set(CANONICAL_SUITES))
    return ordered + extras


def discover_perturbations(root: Path, suites: list[str]) -> list[str]:
    found: set[str] = set()
    for s in suites:
        suite_dir = root / s
        if not suite_dir.exists():
            continue
        for d in suite_dir.iterdir():
            if d.is_dir():
                found.add(d.name)
    ordered = [p for p in CANONICAL_PERTURBATIONS if p in found]
    extras = sorted(found - set(CANONICAL_PERTURBATIONS))
    return ordered + extras


def load_success_rate(
    root: Path, suite: str, pert: str
) -> tuple[float | None, bool, tuple[int, int] | None]:
    """Return (success_rate, is_partial, counts).

    Partial means computed from episode files; counts is (successes, total) for
    partial runs, else None.
    """
    summary_path = root / suite / pert / "summary.json"
    if summary_path.exists():
        try:
            data = json.loads(summary_path.read_text())
            rate = data.get("total", {}).get("success_rate")
            if isinstance(rate, (int, float)):
                return float(rate), False, None
        except json.JSONDecodeError:
            pass

    episodes_dir = root / suite / pert / "episodes"
    if episodes_dir.exists():
        s_count = sum(1 for p in episodes_dir.glob("*_S.json"))
        f_count = sum(1 for p in episodes_dir.glob("*_F.json"))
        total = s_count + f_count
        if total > 0:
            return 100.0 * s_count / total, True, (s_count, total)
    return None, False, None


def fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:.1f}"


def _root_has_suites(root: Path) -> bool:
    """True when ``libero_*`` suite dirs sit directly under *root*."""
    return root.exists() and any(
        d.is_dir() and d.name.startswith("libero_") for d in root.iterdir()
    )


def _policy_subdirs(root: Path) -> list[Path]:
    """Per-policy subdirs that themselves hold ``libero_*`` suites.

    A sweep with a policy axis (``cell_run_template: "{output_run}/{policy}"``)
    nests suites one level deeper: ``<run>/<policy>/<suite>/<pert>/``. Returns
    those policy dirs (e.g. ``openpi``, ``groot``), skipping non-suite dirs like
    ``perturbation_sweep_logs``.
    """
    if not root.exists():
        return []
    return [d for d in sorted(root.iterdir()) if d.is_dir() and _root_has_suites(d)]


def main(args: Args) -> None:
    root = (
        args.data_root.resolve()
        if args.data_root is not None
        else (Path(__file__).resolve().parents[2] / "data" / args.run_name)
    )
    # Policy-nested sweep (no suites directly under the run dir, but subdirs
    # hold them) -> render one table per policy.
    if not _root_has_suites(root) and (policies := _policy_subdirs(root)):
        for i, pdir in enumerate(policies):
            if i:
                print()
            summarize_root(
                pdir, f"{args.run_name}/{pdir.name}", use_color=not args.no_color
            )
        return
    summarize_root(root, args.run_name, use_color=not args.no_color)


def summarize_root(root: Path, run_name: str, use_color: bool) -> None:
    print(f"% run_name: {run_name}")
    print(f"% data_root: {root}")

    suites = discover_suites(root)
    perturbations = (
        discover_perturbations(root, suites)
        if root.exists()
        else list(CANONICAL_PERTURBATIONS)
    )
    if not perturbations:
        perturbations = list(CANONICAL_PERTURBATIONS)

    cells: dict[tuple[str, str], float | None] = {}
    partial: dict[tuple[str, str], bool] = {}
    counts: dict[tuple[str, str], tuple[int, int] | None] = {}
    for suite in suites:
        for pert in perturbations:
            rate, is_partial, cell_counts = load_success_rate(root, suite, pert)
            cells[(suite, pert)] = rate
            partial[(suite, pert)] = is_partial
            counts[(suite, pert)] = cell_counts

    pert_avgs: dict[str, float | None] = {}
    pert_avg_partial: dict[str, bool] = {}
    for pert in perturbations:
        vals = [v for s in suites if (v := cells[(s, pert)]) is not None]
        pert_avgs[pert] = mean(vals) if vals else None
        pert_avg_partial[pert] = any(
            partial[(s, pert)] for s in suites if cells[(s, pert)] is not None
        )

    total_per_suite: dict[str, float | None] = {}
    total_per_suite_partial: dict[str, bool] = {}
    for suite in suites:
        vals = [v for p in perturbations if (v := cells[(suite, p)]) is not None]
        total_per_suite[suite] = mean(vals) if vals else None
        total_per_suite_partial[suite] = any(
            partial[(suite, p)] for p in perturbations if cells[(suite, p)] is not None
        )
    total_avg_vals = [v for v in pert_avgs.values() if v is not None]
    total_avg = mean(total_avg_vals) if total_avg_vals else None
    total_avg_partial = any(pert_avg_partial.values())

    sub_cols = [suite_label(s) for s in suites] + ["Avg."]
    blocks = [pert_label(p) for p in perturbations] + ["TOTAL"]

    # (text, is_partial) pairs, in the order the table renders them
    entries: list[tuple[str, bool]] = []
    for pert in perturbations:
        for suite in suites:
            entries.append((fmt(cells[(suite, pert)]), partial[(suite, pert)]))
        entries.append((fmt(pert_avgs[pert]), pert_avg_partial[pert]))
    for suite in suites:
        entries.append((fmt(total_per_suite[suite]), total_per_suite_partial[suite]))
    entries.append((fmt(total_avg), total_avg_partial))

    print_latex(blocks, sub_cols, entries)
    print()
    print_terminal(blocks, sub_cols, entries, use_color=use_color)
    print_partial_counts(suites, perturbations, counts, use_color=use_color)


def print_partial_counts(
    suites: list[str],
    perturbations: list[str],
    counts: dict[tuple[str, str], tuple[int, int] | None],
    use_color: bool = True,
) -> None:
    """List unfinished runs and their (successes/total) below the table."""
    rows = [
        (pert, suite, c)
        for pert in perturbations
        for suite in suites
        if (c := counts[(suite, pert)]) is not None
    ]
    if not rows:
        return
    head, tail = (YELLOW, RESET) if use_color else ("", "")
    print()
    print(f"{head}unfinished runs (successes/total):{tail}")
    for pert, suite, (s, total) in rows:
        print(f"{head}  {pert_label(pert)} / {suite_label(suite)}: {s}/{total}{tail}")


def print_latex(
    blocks: list[str], sub_cols: list[str], entries: list[tuple[str, bool]]
) -> None:
    n_sub = len(sub_cols)
    col_spec = "l" + ("|" + "c" * n_sub) * len(blocks)
    has_partial = any(p for _, p in entries)
    print("% LaTeX version (requires \\usepackage{booktabs} or just tabular)")
    if has_partial:
        print(
            "% cells marked with * are partial (computed from episode files, run still in progress)"
        )
    print(r"\begin{tabular}{" + col_spec + "}")
    print(r"\toprule")
    underscore_repl = "\\_"
    top = [""] + [
        rf"\multicolumn{{{n_sub}}}{{c}}{{\textbf{{{b.replace('_', underscore_repl)}}}}}"
        for b in blocks
    ]
    print(" & ".join(top) + r" \\")
    cmid_parts = []
    for i in range(len(blocks)):
        start = 2 + i * n_sub
        end = start + n_sub - 1
        cmid_parts.append(rf"\cmidrule(lr){{{start}-{end}}}")
    print("".join(cmid_parts))
    bot = [""] + [c for _ in blocks for c in sub_cols]
    print(" & ".join(bot) + r" \\")
    print(r"\midrule")
    cells = [(f"{v}*" if p and v != "-" else v) for v, p in entries]
    row = ["success rate (\\%)"] + cells
    print(" & ".join(row) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def print_terminal(
    blocks: list[str],
    sub_cols: list[str],
    entries: list[tuple[str, bool]],
    use_color: bool = True,
) -> None:
    n_sub = len(sub_cols)
    metric_col_w = len("success rate (%)")
    cell_w = max(
        max(len(c) for c in sub_cols),
        max(len(v) for v, _ in entries),
        len("100.0"),
    )
    block_w = n_sub * (cell_w + 3) - 3

    horiz = "+" + "-" * (metric_col_w + 2) + "+"
    for _ in blocks:
        horiz += "-" * (block_w + 2) + "+"

    sub_horiz = "+" + "-" * (metric_col_w + 2) + "+"
    for _ in blocks:
        for _ in range(n_sub):
            sub_horiz += "-" * (cell_w + 2) + "+"

    def block_label(name: str) -> str:
        return name.center(block_w)

    def cell(text: str, partial: bool) -> str:
        padded = text.rjust(cell_w)
        if partial and use_color:
            return f"{YELLOW}{padded}{RESET}"
        return padded

    print(horiz)
    line = "| " + "".ljust(metric_col_w) + " |"
    for b in blocks:
        line += " " + block_label(b) + " |"
    print(line)
    print(sub_horiz)
    line = "| " + "metric".ljust(metric_col_w) + " |"
    for _ in blocks:
        for sc in sub_cols:
            line += " " + sc.rjust(cell_w) + " |"
    print(line)
    print(sub_horiz)
    line = "| " + "success rate (%)".ljust(metric_col_w) + " |"
    idx = 0
    for _ in blocks:
        for _ in range(n_sub):
            text, partial = entries[idx]
            line += " " + cell(text, partial) + " |"
            idx += 1
    print(line)
    print(sub_horiz)
    if any(p for _, p in entries) and use_color:
        print(
            f"{YELLOW}yellow = partial (run in progress, computed from episode files){RESET}"
        )


if __name__ == "__main__":
    main(tyro.cli(Args))
