"""Choose which recorded frame the report draws its results on.

The report projects every cell and dot onto ``reference_frame.png`` — the
staging view saved when the session started. That frame is sometimes the
wrong picture: the object was still in the operator's hand, or sat where it
hides the grid. Any rollout's first recorded frame carries the same
calibrated camera, so it can stand in. Pick one by the level it staged or by
its plan index; the frame it replaces is kept as
``reference_frame.original.png`` and the report is rebuilt. ::

    uv run python scripts/experiments/set_reference_frame.py \\
        --name cable_position_baseline_v3 --run ur5_experiments --level B3

    # the arm already on its way to the cable
    uv run python scripts/experiments/set_reference_frame.py \\
        --name cable_position_baseline_v3 --run ur5_experiments \\
        --level B3 --timestep 40
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

from policy_abstraction.experiments.measure import set_reference_view  # noqa: E402
from policy_abstraction.experiments.report import build_report  # noqa: E402
from policy_abstraction.experiments.store import (  # noqa: E402
    ExperimentStore,
    experiment_root,
)
from policy_abstraction.experiments.viz import render_reference_staging  # noqa: E402


@dataclasses.dataclass
class Args:
    name: str
    """The experiment, as given to ``--experiment.experiment-name``."""
    run: str = "ur5_experiments"
    base: str = "data"
    level: Optional[str] = None
    """A level id such as ``B3``: the first scored rollout that staged it is
    used. Give ``factor=level`` when the experiment has several factors."""
    index: Optional[int] = None
    """A plan index, instead of ``--level``."""
    timestep: int = 0
    """Which frame of the recording. 0 is the staged scene before the arm
    moves; a few dozen steps in shows the arm on its way to the object."""
    report: bool = True
    """Rebuild the report on the new frame."""
    staging_out: Optional[str] = "reference_staging_view.png"
    """Also write the frame annotated as the GUI showed that rollout's
    condition — region, staged objects, highlighted target — under
    ``report/``. ``None`` skips it."""


def _pick(results, level: Optional[str], index: Optional[int]):
    if index is not None:
        for r in results:
            if r.index == index:
                return r
        raise SystemExit(f"No scored rollout at index {index}.")
    if not level:
        raise SystemExit("Pass --level <id> (e.g. B3) or --index <n>.")
    factor, _, wanted = level.rpartition("=")
    for r in results:
        for name, value in r.assignments.items():
            if value == wanted and (not factor or name == factor):
                return r
    raise SystemExit(f"No scored rollout staged {level!r}.")


def main(args: Args) -> int:
    store = ExperimentStore(experiment_root(args.run, args.name, base=args.base))
    if not store.exists:
        raise SystemExit(f"No experiment at {store.root}. Check --name and --run.")
    record = _pick(store.results(), args.level, args.index)
    frame = set_reference_view(store, record, timestep=args.timestep)
    print(
        f"Reference view: rollout {record.index} ({record.assignments}), "
        f"step {frame.timestep} of {record.episode_path}"
    )
    if args.staging_out:
        out = pathlib.Path(args.staging_out)
        if not out.is_absolute() and out.parent == pathlib.Path("."):
            out = store.report_dir / out
        written = render_reference_staging(store, store.load_config(), record, out)
        if written is not None:
            print(f"Staging view: {written}")
    if args.report:
        build_report(store)
        print(f"Report: {store.report_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
