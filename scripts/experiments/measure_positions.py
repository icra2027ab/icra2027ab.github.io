"""Measure where each staged object really was, from an experiment's recordings.

The positional experiment records the *nominal* position of the object — the
centre of the cell the operator was shown. This reads each rollout's first
recorded frame, runs the open-perception pipeline on it (the same one the
guided policy uses), and writes the measured base-frame position beside the
experiment as ``observed_positions.json``. The report is rebuilt afterwards,
so its camera-view dots move to where the object actually was.

No robot, no servers: only the recordings, the perception models and the
detector's API key (``OPENROUTER_API_KEY`` or ``GEMINI_API_KEY``, as for a
guided run). ::

    uv run python scripts/experiments/measure_positions.py \\
        --name cable_position_baseline_v3 --run ur5_experiments

    # a different noun for the detector, a subset, and redo what is there
    uv run python scripts/experiments/measure_positions.py \\
        --name cable_position_baseline_v3 --run ur5_experiments \\
        --query "cable_position=orange usb cable" --indices 0 1 2 --force
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Literal, Optional

import tyro
from loguru import logger

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policy_abstraction.experiments.measure import (  # noqa: E402
    measure_experiment,
    open_perception_measurer,
)
from policy_abstraction.experiments.report import build_report  # noqa: E402
from policy_abstraction.experiments.store import (  # noqa: E402
    ExperimentStore,
    experiment_root,
)


@dataclasses.dataclass
class Args:
    name: str
    """The experiment, as given to ``--experiment.experiment-name`` (or the
    launcher's ``--name``)."""
    run: str = "ur5_experiments"
    """The evaluation run the experiment lives under: ``data/<run>/experiments/``."""
    base: str = "data"

    indices: Optional[list[int]] = None
    """Only these plan indices. Default: every scored rollout."""
    force: bool = False
    """Re-measure rollouts that already have an entry."""
    query: Optional[list[str]] = None
    """``key=object`` overrides for what the detector is asked for, keyed like
    the records' ``target_positions`` (e.g. ``cable_position=orange cable``).
    Default: each positional factor's ``object:``."""
    report: bool = True
    """Rebuild the report once measured, so the plots pick the positions up."""
    debug_images: bool = True
    """Write one annotated frame per rollout to ``report/measured/``."""

    device: Optional[str] = None
    """Torch device for the perception models. Default: the pipeline's (CUDA)."""
    detector_backend: Literal["gemini", "openrouter"] = "openrouter"
    detector_model: Optional[str] = None
    vlm_model: Optional[str] = None
    pixels_mirrored: Literal["auto", "true", "false"] = "auto"
    """Whether the recordings are in the simulator's flipped convention.
    ``auto`` reads it from the recording's own camera spec — false for the
    UR5e."""


def _parse_queries(items: Optional[list[str]]) -> Optional[dict[str, str]]:
    if not items:
        return None
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--query expects key=object, got {item!r}")
        key, obj = item.split("=", 1)
        out[key.strip()] = obj.strip()
    return out


def main(args: Args) -> int:
    store = ExperimentStore(experiment_root(args.run, args.name, base=args.base))
    if not store.exists:
        raise SystemExit(f"No experiment at {store.root}. Check --name and --run.")
    config = store.load_config()
    results = store.results()
    if not results:
        raise SystemExit(f"{store.root} has no scored rollouts to measure.")
    queries = _parse_queries(args.query)
    asked = queries or config.measurement_queries(results[0].assignments)
    if not asked:
        raise SystemExit(
            "This experiment stages nothing positional, so there is nothing to "
            "measure. Pass --query key=object to measure something anyway."
        )
    logger.info(
        f"Measuring {len(results)} rollouts of {store.root.name}; asking the "
        f"detector for {sorted(set(asked.values()))}"
    )

    mirrored = {"auto": None, "true": True, "false": False}[args.pixels_mirrored]
    if mirrored is None:
        mirrored = _recorded_pixels_mirrored(store, results)
    perceive = open_perception_measurer(
        device=args.device,
        detector_backend=args.detector_backend,
        detector_model=args.detector_model,
        vlm_model=args.vlm_model,
        pixels_mirrored=mirrored,
    )

    def progress(record, found):
        if not found:
            print(f"  rollout {record.index:4d}  {record.assignments}: not measured")
            return
        for key, m in found.items():
            off = f"{m.offset_m * 100:5.1f} cm off" if m.offset_m is not None else ""
            print(
                f"  rollout {record.index:4d}  {key}={record.assignments.get(key, '?')}"
                f"  -> [{m.position[0]:+.3f}, {m.position[1]:+.3f}, "
                f"{m.position[2]:+.3f}]  {m.points:5d} pts  {off}"
            )

    data = measure_experiment(
        store,
        perceive,
        config=config,
        indices=args.indices,
        force=args.force,
        queries=queries,
        debug_dir=(store.report_dir / "measured") if args.debug_images else None,
        on_measured=progress,
    )
    entries = data.get("rollouts") or {}
    measured = sum(
        1 for v in entries.values() if any(k not in ("error", "missing") for k in v)
    )
    missed = sorted(int(k) for k, v in entries.items() if "missing" in v)
    failed = sorted(int(k) for k, v in entries.items() if "error" in v)
    print(
        f"\n{measured}/{len(entries)} rollouts measured -> {store.observed_positions_path}"
    )
    if missed:
        print(
            f"Detector found nothing in rollouts {missed}. See report/measured/ for "
            "what it was shown; try --query <key>=<a more specific noun> "
            f"--force --indices {' '.join(str(i) for i in missed)}"
        )
    if failed:
        print(f"Recordings that could not be read: {failed}")

    if args.report:
        summary = build_report(store, config)
        stats = summary.get("measured_positions") or {}
        if "median_offset_m" in stats:
            print(
                f"Offset from the nominal cell centre: median "
                f"{stats['median_offset_m'] * 100:.1f} cm, "
                f"max {stats['max_offset_m'] * 100:.1f} cm"
            )
        print(f"Report: {store.report_dir / 'report.md'}")
    return 0


def _recorded_pixels_mirrored(store: ExperimentStore, results) -> bool:
    """Read the convention off the first readable recording's camera spec."""
    from policy_abstraction.experiments.measure import first_frame

    for record in results:
        if not record.episode_path:
            continue
        episode = store.root / record.episode_path
        if not episode.exists():
            continue
        try:
            return first_frame(episode).pixels_mirrored
        except Exception as exc:
            logger.debug(f"{episode.name}: could not read ({exc})")
    return False


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
