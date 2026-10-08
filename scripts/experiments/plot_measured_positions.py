"""Re-draw the positional plots for part of the grid.

The report already holds ``<factor>_measured.png`` and
``<factor>_on_camera.png`` for the whole grid. This draws the same plots for
a subset of cells — the rows within the arm's reach, say, or everything but
the row the operator could not stage well — without touching the
experiment's config or report. ::

    uv run python scripts/experiments/plot_measured_positions.py \\
        --name cable_position_baseline_v3 --run ur5_experiments \\
        --exclude-rows D --out cable_position_measured_3x3.png \\
        --camera-out cable_position_on_camera_3x3.png
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Optional

import tyro

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policy_abstraction.experiments.factors import GridPositionFactor  # noqa: E402
from policy_abstraction.experiments.store import (  # noqa: E402
    ExperimentStore,
    experiment_root,
)
from policy_abstraction.experiments.viz import (  # noqa: E402
    plot_grid_success_projected,
    plot_measured_positions,
)


@dataclasses.dataclass
class Args:
    name: str
    """The experiment, as given to ``--experiment.experiment-name``."""
    run: str = "ur5_experiments"
    base: str = "data"
    factor: Optional[str] = None
    """Which grid factor to plot. Default: the experiment's only one."""
    cells: Optional[list[str]] = None
    """Keep only these cells, e.g. ``A1 A2 B1``. Default: every cell."""
    exclude_rows: Optional[list[str]] = None
    """Drop whole rows by letter, e.g. ``D``."""
    exclude_cols: Optional[list[int]] = None
    """Drop whole columns by 1-based number, e.g. ``4``."""
    out: Optional[str] = "cable_position_measured_subset.png"
    """File name for the metric plot, under the experiment's ``report/``
    directory unless given as a path. ``None`` skips it."""
    camera_out: Optional[str] = None
    """File name for the camera-view success map of the same cells. Skipped
    unless given."""


def main(args: Args) -> int:
    store = ExperimentStore(experiment_root(args.run, args.name, base=args.base))
    if not store.exists:
        raise SystemExit(f"No experiment at {store.root}. Check --name and --run.")
    config = store.load_config()
    grids = [f for f in config.factors if isinstance(f, GridPositionFactor)]
    if args.factor:
        grids = [f for f in grids if f.name == args.factor]
    if len(grids) != 1:
        raise SystemExit(
            f"Pass --factor: grid factors here are {[f.name for f in grids]}"
        )
    factor = grids[0]

    keep = list(factor.cells)
    if args.cells:
        wanted = {factor.grid.parse_label(c) for c in args.cells}
        keep = [c for c in keep if c in wanted]
    if args.exclude_rows:
        rows = {
            factor.grid.parse_label(f"{r.strip()}{factor.grid.first_col}")[0]
            for r in args.exclude_rows
        }
        keep = [c for c in keep if c[0] not in rows]
    if args.exclude_cols:
        cols = {int(c) - factor.grid.first_col for c in args.exclude_cols}
        keep = [c for c in keep if c[1] not in cols]
    if not keep:
        raise SystemExit("No cells left after the filters.")
    subset = GridPositionFactor(
        factor.name, factor.grid, cells=keep, object=factor.object
    )

    records = store.attach_observed_positions(store.results())

    def _resolve(name: str) -> pathlib.Path:
        out = pathlib.Path(name)
        if not out.is_absolute() and out.parent == pathlib.Path("."):
            out = store.report_dir / out
        return out

    if args.out:
        written = plot_measured_positions(records, subset, _resolve(args.out))
        if written is None:
            raise SystemExit(
                "No measured positions in those cells. Run measure_positions.py first."
            )
        print(f"Wrote {written} ({len(keep)} cells)")
    if args.camera_out:
        written = plot_grid_success_projected(
            records, subset, store, _resolve(args.camera_out)
        )
        if written is None:
            raise SystemExit("No reference frame or camera calibration to draw on.")
        print(f"Wrote {written} ({len(keep)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
