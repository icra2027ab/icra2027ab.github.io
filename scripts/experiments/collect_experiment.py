"""Operator-driven rollout collection for a real-world experiment.

Runs many rollouts against a policy that is loaded **once**. The operator stages
the scene at the condition the overlay shows, presses SPACE, watches, and calls
the result with a hotkey. The policy server stays up across the whole session —
which is the point, since loading and compiling a pi05 checkpoint costs about
three minutes and would otherwise be paid per rollout.

The env and policy backends are selected exactly as in ``scripts/evaluate.py``
(the same tyro subcommands, the same flags), because they are the same code:
:class:`~policy_abstraction.evaluation.backends.EvalBackends`.

Typical use — the positional experiment on the real arm, baseline then guided::

    # 1. define the grid once (edit the YAML afterwards to taste).
    #    --no-guide because the UR5e publishes no GT segmentation, and
    #    calibration runs no perception anyway.
    python scripts/experiments/collect_experiment.py \\
        --experiment.experiment-config configs/experiments/cable_in_box_position.yaml \\
        --experiment.calibrate --no-guide \\
      env:ur5 --env.calibration-path <calib.json> policy:openpi

    # 2. collect the baseline
    python scripts/experiments/collect_experiment.py \\
        --experiment.experiment-config configs/experiments/cable_in_box_position.yaml \\
        --experiment.experiment-name cable_position_baseline \\
        --experiment.policy-setting baseline --no-guide \\
      env:ur5 --env.no-launch-server \\
      policy:lerobot --policy.model-path <checkpoint> --policy.no-launch-server

    # 3. same plan, guidance on
    python scripts/experiments/collect_experiment.py \\
        --experiment.experiment-config configs/experiments/cable_in_box_position.yaml \\
        --experiment.experiment-name cable_position_guided \\
        --experiment.policy-setting guided --perception.mode REAL_PERCEPTION \\
      env:ur5 ... policy:lerobot ...

    # 3'. or take the plan from the baseline's own directory rather than
    #     from the YAML: the identical settings and sequence of positions
    #     the baseline collected under, GUI edits included.
    python scripts/experiments/collect_experiment.py \\
        --experiment.reference-experiment cable_position_baseline \\
        --experiment.experiment-name cable_position_guided \\
        --experiment.policy-setting guided --perception.mode REAL_PERCEPTION \\
      env:ur5 ... policy:lerobot ...

The disambiguation experiment is the same command with a different config: the
scene holds several named objects, the task is a template, and the prompt names
one of them per rollout::

    python scripts/experiments/collect_experiment.py \\
        --experiment.experiment-config configs/experiments/object_disambiguation.yaml \\
        --experiment.experiment-name object_disambiguation_baseline \\
        --experiment.policy-setting baseline --no-guide \\
      env:ur5 ... policy:lerobot ...

``--experiment.objects "black cable,white cable,red cable,blue cable"`` swaps the
object list — and the object *count* — without editing the YAML.

Interrupted at rollout 19 of 36? Re-run the same command. The plan is frozen on
disk, so collection resumes at 19 with the same conditions in the same order.
An existing experiment name loads the settings stored with that experiment —
including anything edited from the GUI — so the YAML can even be left off::

    python scripts/experiments/collect_experiment.py \
        --experiment.experiment-name cable_position_baseline \
        --experiment.policy-setting baseline --no-guide \
      env:ur5 ... policy:lerobot ...

Report only, no robot::

    python scripts/experiments/collect_experiment.py \\
        --experiment.report-only \\
        --experiment.experiment-name cable_position_baseline

Hardware-free rehearsal, to check a grid before going near the arm::

    MUJOCO_GL=egl python scripts/experiments/collect_experiment.py \\
        --experiment.experiment-config configs/experiments/libero_rehearsal.yaml \\
        --experiment.experiment-name dry_run \\
        --guidance.debug --retrieval.no-enable --demo-analysis.no-enable \\
      env:libero --env.perturb.no-swap policy:openpi
"""

from __future__ import annotations

import os

# Must precede any torch import; mirrors scripts/evaluate.py.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import dataclasses  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from typing import Optional  # noqa: E402

import tyro  # noqa: E402
from loguru import logger  # noqa: E402

from policy_abstraction.evaluation.abstraction import PerceptionMode  # noqa: E402
from policy_abstraction.evaluation.backends import start_backends  # noqa: E402
from policy_abstraction.evaluation.config import Args as EvalArgs  # noqa: E402
from policy_abstraction.evaluation.config import Ur5EnvArgs  # noqa: E402
from policy_abstraction.experiments.config import ExperimentConfig  # noqa: E402
from policy_abstraction.experiments.reference import ReferenceFrames  # noqa: E402
from policy_abstraction.experiments.report import build_report  # noqa: E402
from policy_abstraction.experiments.revise import (  # noqa: E402
    AppliedRevision,
    ExperimentConflict,
    ExperimentReviser,
)
from policy_abstraction.experiments.session import (  # noqa: E402
    ExperimentSession,
    fetch_camera_params,
    frame_from_obs,
    resolve_staging_camera,
)
from policy_abstraction.experiments.store import (  # noqa: E402
    ExperimentExistsError,
    ExperimentStore,
    experiment_root,
)
from policy_abstraction.utils.provenance import (  # noqa: E402
    collect_provenance,
    write_provenance,
)
from policy_abstraction.utils.seeding import seed_everything  # noqa: E402


@dataclasses.dataclass
class ExperimentArgs:
    """Experiment-specific flags, on top of the shared evaluation ones."""

    experiment_config: Optional[str] = None
    """Path to the experiment YAML (grid, factors, repeats). Required to
    *create* an experiment; an existing one resumes under the settings stored
    with it, so this may be omitted when ``--experiment-name`` names an
    experiment that already exists (or ``--report-only`` is set)."""
    reference_experiment: Optional[str] = None
    """An existing experiment to create this one from: its name under
    ``data/<run>/experiments/``, or the path to its directory. Replaces
    ``--experiment-config``. The new experiment takes the reference's *stored*
    settings (``experiment.yaml``, GUI edits and any ``--objects`` override
    included), so it stages the identical sequence of positions and prompts;
    the plan those settings generate is checked against the reference's frozen
    ``plan.json`` and against the positions it recorded before anything is
    collected. This is how the second arm of a comparison is run against
    exactly the plan the first arm collected. Like the YAML, it only ever
    *creates*: an experiment that already exists under ``--experiment-name``
    resumes under its own settings (``--overwrite`` recreates it from the
    reference)."""
    experiment_name: Optional[str] = None
    """Directory name under ``data/<run>/experiments/``. Defaults to the
    config's own ``name``. Give each policy setting its own name — that is what
    keeps two arms of a comparison in separate, individually resumable
    directories. If the name already exists, the experiment's own stored
    configuration is loaded and the YAML is not consulted (pass
    ``--overwrite`` to start over from the YAML instead)."""
    policy_setting: str = ""
    """Label recorded on every rollout, e.g. ``baseline`` or ``guided``. Shows
    up in the report, and lets several experiments be pooled later without
    losing which arm each rollout came from."""
    task_id: int = 0
    """Which of the env server's tasks to run."""

    overwrite: bool = False
    """Start over from the YAML rather than resuming under the stored settings.
    The existing results are renamed aside, not deleted."""
    calibrate: bool = False
    """Write a starting experiment config by projecting a candidate grid onto a
    live camera frame, then exit. Nothing is collected and the robot is not
    commanded."""
    report_only: bool = False
    """Rebuild the report from what is already on disk and exit. No servers are
    started and no hardware is touched."""
    headless: bool = False
    """Run without the GUI, driving rollouts from stdin. For a machine with no
    display; the same hotkey letters are typed at a prompt."""
    objects: Optional[str] = None
    """Comma-separated object names, replacing the ones an ``object_layout``
    factor declares — how the disambiguation experiment is run with a different
    set, or a different *number*, of objects without editing the YAML. The
    layouts are re-sampled for the new count. This changes the plan, so it is
    ignored when resuming an existing experiment; start over with
    ``--overwrite`` to apply it."""
    refresh_hz: float = 15.0
    """Camera view refresh rate."""


@dataclasses.dataclass
class Args(EvalArgs):
    """Everything ``scripts/evaluate.py`` takes, plus the experiment flags.

    Subclassing rather than nesting keeps every evaluation flag spelled
    exactly as it is for ``evaluate.py`` — ``--guidance.debug``,
    ``env:libero``, ``policy:lerobot --policy.model-path ...`` — so a
    command can be moved between the two scripts by changing only the
    script name. The experiment's own flags are under ``--experiment.``.
    """

    experiment: ExperimentArgs = dataclasses.field(default_factory=ExperimentArgs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_store(args: Args, config: Optional[ExperimentConfig]) -> ExperimentStore:
    name = args.experiment.experiment_name or (config.name if config else None)
    if not name:
        raise SystemExit(
            "Pass --experiment.experiment-name, or an --experiment.experiment-config "
            "whose `name:` field can supply one."
        )
    return ExperimentStore(experiment_root(args.output.run, name))


def _resolve_reference(reference: str, run: str) -> ExperimentStore:
    """The experiment ``--reference-experiment`` names.

    A path to an experiment directory is taken as given; anything else is a
    name under this run's ``experiments/``, the same place ``--experiment-name``
    resolves to.
    """
    path = pathlib.Path(reference)
    if ExperimentStore(path).exists:
        return ExperimentStore(path)
    store = ExperimentStore(experiment_root(run, reference))
    if not store.exists:
        raise SystemExit(
            f"No collected experiment at {store.root} (nor at {path}): "
            "--experiment.reference-experiment takes the name of an experiment "
            "under data/<run>/experiments/, or the path to its directory."
        )
    return store


def _load_reference_config(
    exp: ExperimentArgs, run: str
) -> tuple[ExperimentConfig, ExperimentStore]:
    """The settings a reference experiment collected under, checked against it.

    The positions are not written into the plan — a layout is re-sampled from
    the seed in the config every time it is loaded — so "the same sequence of
    positions as the reference" rests on the stored config generating the
    stored plan again. That is checked here rather than trusted: the plan the
    config generates has to match the reference's ``plan.json`` row for row,
    and the positions it puts each object at have to match what the reference
    recorded on every rollout it collected. A config edited by hand after the
    fact, or a sampler that has since changed, is refused before the
    checkpoint loads instead of quietly staging a different scene under the
    same level names.
    """
    assert exp.reference_experiment is not None
    ref = _resolve_reference(exp.reference_experiment, run)
    try:
        config = ref.load_config()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"Could not read the reference's settings at {ref.config_path}: {exc}"
        ) from None
    stored = [p.to_dict() for p in ref.load_plan()]
    rebuilt = [p.to_dict() for p in config.build_plan()]
    if stored != rebuilt:
        raise SystemExit(
            f"The settings stored with {ref.root.name} ({ref.config_path}) "
            f"generate {len(rebuilt)} rollouts that do not match the "
            f"{len(stored)} in its {ref.plan_path.name}, so this experiment "
            "could not be guaranteed the reference's sequence of positions. "
            "Restore that file from the reference's era."
        )
    drifted = _drifted_positions(config, ref)
    if drifted:
        shown = ", ".join(f"rollout {i} ({key})" for i, key in drifted[:5])
        raise SystemExit(
            f"The settings stored with {ref.root.name} no longer put the objects "
            f"where {ref.root.name} recorded them: {shown}"
            f"{' …' if len(drifted) > 5 else ''}. The layout sampler, or "
            f"{ref.config_path.name}, has changed since that experiment was "
            "collected, so its positions cannot be reproduced from its settings."
        )
    logger.info(
        f"Taking the settings and the plan ({len(stored)} rollouts) from {ref.root}"
    )
    return config, ref


def _drifted_positions(
    config: ExperimentConfig, ref: ExperimentStore, *, tolerance: float = 1e-6
) -> list[tuple[int, str]]:
    """``(rollout index, position key)`` pairs *config* would stage elsewhere.

    Compared against what the reference recorded as the target on each
    rollout, which is the position the operator actually staged to.
    """
    drifted: list[tuple[int, str]] = []
    for record in ref.records():
        if not record.target_positions:
            continue
        expected = config.target_positions(record.assignments)
        for key, recorded in record.target_positions.items():
            planned = expected.get(key)
            if planned is None or len(planned) != len(recorded):
                drifted.append((record.index, key))
                continue
            if (
                max(abs(a - b) for a, b in zip(planned, recorded, strict=True))
                > tolerance
            ):
                drifted.append((record.index, key))
    return drifted


def _load_config(exp: ExperimentArgs) -> ExperimentConfig:
    """The experiment YAML, with any command-line overrides applied."""
    config = ExperimentConfig.load(str(exp.experiment_config))
    if exp.objects:
        names = [name.strip() for name in exp.objects.split(",") if name.strip()]
        if not names:
            raise SystemExit("--experiment.objects was given but named nothing.")
        config = config.with_objects(names)
        logger.info(f"Staging {len(names)} objects instead of the config's: {names}")
    return config


# Key under which an experiment remembers the YAML it was created from, as
# parsed. Provenance only: it records what the operator launched with, so a
# later reader can tell a GUI edit from the file as it was.
SOURCE_SNAPSHOT = "source_config_snapshot"


def _differing_keys(a: dict, b: dict) -> list[str]:
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def _effective_config(
    store: ExperimentStore,
    source: Optional[ExperimentConfig],
    *,
    source_label: str = "the YAML",
) -> tuple[Optional[ExperimentConfig], bool]:
    """The config to actually run, and whether it came from the experiment.

    An experiment that already exists is resumed under its **own** stored
    ``experiment.yaml`` — the settings it has been collecting under, including
    anything edited from the GUI since it was created. The operator's YAML is
    only what a *new* experiment is created from; on resume it is not
    consulted, and ``--overwrite`` is the way to start over from it.

    That is what makes an experiment name sufficient to resume: the file may
    since have been edited for the other arm of the comparison, or not be to
    hand at all, and neither changes what this experiment runs.

    Returns ``(None, False)`` when there is nothing to resume and no YAML to
    create from; the caller turns that into a usage error.
    """
    if not store.exists:
        return source, False
    try:
        stored = store.load_config()
    except (FileNotFoundError, ValueError) as exc:
        if source is None:
            raise SystemExit(
                f"Could not read {store.config_path} ({exc}), and no "
                "--experiment.experiment-config was given to fall back on."
            ) from None
        logger.warning(f"Could not read {store.config_path} ({exc}); using the YAML")
        return source, False
    if source is not None and stored.to_dict() != source.to_dict():
        keys = ", ".join(_differing_keys(stored.to_dict(), source.to_dict()))
        logger.warning(
            f"Resuming {store.root.name} under its stored settings "
            f"({store.config_path}); {source_label} differs in [{keys}] and is "
            f"not applied. Pass --experiment.overwrite to start over from "
            f"{source_label}."
        )
    return stored, True


# The two arms of a comparison. A rollout labelled with one of these while
# the pipeline is configured for the other would be reported under the wrong
# arm, and nothing else would notice.
POLICY_SETTINGS = ("baseline", "guided")


def check_policy_setting(args: Args) -> None:
    """Refuse a ``--experiment.policy-setting`` the flags contradict.

    ``guided`` needs the whole guidance chain switched on -- retrieval to
    find a demo, demo analysis to turn it into distance pairs, and the
    guidance gradient itself -- because with any one of them off the
    steering is silently inert and the rollouts would be a baseline
    recorded as guided. On the real arm it also needs a retrieval source
    that exists (the defaults are simulator datasets) and the real
    perception stack. ``baseline`` needs the gradient *off*, whatever else
    is running. Any other label is the operator's business.
    """
    setting = args.experiment.policy_setting
    if setting not in POLICY_SETTINGS or args.experiment.calibrate:
        return
    if setting == "baseline":
        if args.guidance.enable:
            raise SystemExit(
                "--experiment.policy-setting baseline, but guidance is enabled: "
                "the rollouts would be steered and reported as the baseline. "
                "Pass --no-guide (or --guidance.no-enable to keep perception "
                "running unguided)."
            )
        return

    off = [
        flag
        for flag, on in (
            ("--guide", args.guide),
            ("--guidance.enable", args.guidance.enable),
            ("--retrieval.enable", args.retrieval.enable),
            ("--demo-analysis.enable", args.demo_analysis.enable),
        )
        if not on
    ]
    if off:
        raise SystemExit(
            "--experiment.policy-setting guided, but the guidance chain is "
            f"switched off at {', '.join(off)}: the steering would be inert and "
            "the rollouts a baseline recorded as guided. Drop --no-guide and "
            "the .no-enable flags, or label the run baseline."
        )
    if isinstance(args.env, Ur5EnvArgs):
        dirs = [str(d) for d in (args.retrieval.dataset_dirs or [])]
        missing = [d for d in dirs if not pathlib.Path(d).is_dir()]
        if not dirs or missing:
            raise SystemExit(
                "--experiment.policy-setting guided on env:ur5 needs "
                "--retrieval.dataset-dirs pointing at a folder of encoded demo "
                "rollouts (the launchers take it from RETRIEVAL_DIR, e.g. "
                "data/reference_dataset_cable)"
                + (f"; these do not exist: {missing}" if missing else "")
                + ". Without it retrieval falls back to the LIBERO simulator "
                "datasets, which are not on this host."
            )
        if args.perception.mode != PerceptionMode.REAL_PERCEPTION:
            raise SystemExit(
                "--experiment.policy-setting guided on env:ur5 needs "
                "--perception.mode REAL_PERCEPTION: the arm publishes no "
                f"ground-truth segmentation, and {args.perception.mode.name} "
                "would leave guidance with nothing to measure distances on."
            )


def _report_only(args: Args) -> int:
    store = _resolve_store(args, None)
    if not store.exists:
        raise SystemExit(
            f"No experiment at {store.root}. Check --experiment.experiment-name "
            f"and --output.run."
        )
    summary = build_report(store)
    totals = summary["totals"]
    print(
        f"\n{summary['experiment']}: {totals['successes']}/{totals['collected']} "
        f"({totals['success_rate']:.0%}), "
        f"95% CI [{totals['ci95'][0]:.0%}, {totals['ci95'][1]:.0%}]"
    )
    print(f"Report: {store.report_dir / 'report.md'}")
    return 0


def _calibrate(args: Args, config: ExperimentConfig) -> int:
    """Project the configured grid onto one live frame and save the picture.

    Deliberately minimal and read-only. Defining a table grid is an iterative,
    physical process — measure a corner, look at the overlay, nudge a number —
    and the fastest loop is "edit the YAML, re-run this, look at the PNG". The
    arm is never commanded.

    No policy is loaded: this path only needs a camera frame and the
    camera's matrices, and a pi05 checkpoint costs about three minutes to
    load and compile — which would be paid on every turn of a loop whose
    whole point is being fast.
    """
    import imageio.v3 as iio

    from policy_abstraction.experiments.overlay import render_staging_view

    run_dir = pathlib.Path(f"data/{args.output.run}/calibration")
    run_dir.mkdir(parents=True, exist_ok=True)
    args.guidance.debug = True  # in-process dummy policy; nothing to load
    backends = start_backends(args, run_dir)
    try:
        backends.env.make_env(
            task_id=args.experiment.task_id,
            camera_heights=args.camera.height,
            camera_widths=args.camera.width,
            seed=args.seed,
        )
        # No session and no experiment directory here: calibration has
        # no plan to run, and must not create one — the operator is
        # still deciding what the grid *is*, and a plan frozen now
        # would be frozen against a grid that is about to change.
        camera = resolve_staging_camera(backends.env.env_spec, config.camera)
        params = fetch_camera_params(
            backends.env,
            camera,
            height=args.camera.height,
            width=args.camera.width,
        )
        intrinsic = params.get("intrinsic")
        extrinsic = params.get("extrinsic")
        image = frame_from_obs(backends.env.get_obs(), camera)
        if image is None:
            raise SystemExit(
                f"Camera {camera.name if camera else '?'} published no "
                "image, so there is nothing to project onto."
            )

        plan = config.build_plan()
        assignments = dict(plan[0].assignments) if plan else {}
        annotated = render_staging_view(
            image,
            config.factors,
            assignments,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            camera=camera,
            scale=config.view_scale,
            banner=f"calibration preview - {config.name}",
        )
        out = run_dir / f"{config.name}_grid_preview.png"
        iio.imwrite(out, annotated)
        print(f"\nWrote {out}")
        if config.task:
            print(f"\nPrompt for the first condition: {config.task_for(assignments)!r}")
        if intrinsic is None or extrinsic is None:
            print(
                "\nThe camera published no intrinsics/extrinsics, so the grid "
                "could not be projected. On the real cell that means "
                "--env.calibration-path is unset or names no extrinsic for this "
                "camera; see docs/ur5.md."
            )
        else:
            print(
                "Check that the grid lands on the work surface and that the "
                "cell numbering runs the way you want it to read. Adjust "
                f"`origin`, `u_axis`, `v_axis` and the cell sizes in "
                f"{args.experiment.experiment_config or args.experiment.reference_experiment} "
                "and re-run."
            )
        # Every factor's own coordinates, in the robot base frame, so they can
        # be checked against a tape measure rather than against a rendering.
        for factor in config.factors:
            lines = factor.geometry_lines()
            if lines:
                print("")
                print("\n".join(lines))
        return 0
    finally:
        backends.shutdown()


def _ask(prompt: str, current) -> str:
    """One prompt with its current value shown; blank keeps it."""
    reply = input(f"  {prompt} [{current}]: ").strip()
    return reply


def _headless_settings(
    session: ExperimentSession, reviser: ExperimentReviser
) -> ExperimentConfig:
    """The settings editor, for a machine with no display.

    Covers the same fields the GUI form does for a single-grid experiment —
    which is every experiment run so far — and applies them through the same
    :class:`ExperimentReviser`, so the rules about what may be overwritten are
    identical to the window's. Anything more exotic is a YAML edit.
    """
    config = reviser.config
    data = config.to_dict()
    print("\n--- settings (blank keeps the current value) ---")
    name = _ask("experiment name", reviser.name) or reviser.name
    try:
        for key, label, cast in (
            ("name", "report title", str),
            ("task", "task", str),
            ("repeats", "repetitions per cell", int),
            ("order", "order (systematic/random)", str),
            ("max_steps", "max steps", int),
            ("timeout_s", "timeout (s)", float),
        ):
            reply = _ask(label, data.get(key))
            if reply:
                data[key] = cast(reply)
        for entry in data.get("factors", []):
            grid = entry.get("grid")
            if not grid:
                continue
            print(f"  grid {entry['name']!r}:")
            # A config always writes its cells out in full, even when they are
            # every cell there is. Showing that list back would pin the old
            # shape the moment the operator enlarges the grid, so a full list
            # is shown -- and kept -- as "every cell" instead.
            covers_all = _covers_all(entry.get("cells"), grid)
            for key, label in (("rows", "rows"), ("cols", "columns")):
                reply = _ask(f"  {label}", grid[key])
                if reply:
                    grid[key] = int(reply)
            shown = "" if covers_all else ", ".join(entry.get("cells") or [])
            reply = _ask("  cells (blank = every cell)", shown)
            if reply:
                entry["cells"] = [c.strip() for c in reply.split(",") if c.strip()]
            elif covers_all:
                entry["cells"] = None
            # A subset written for the old grid may no longer fit the new one.
            entry["cells"] = _clip_cells(entry.get("cells"), grid)
        edited = ExperimentConfig.from_dict(data)
    except (EOFError, KeyboardInterrupt):
        print("\n  cancelled; settings unchanged")
        return config
    except (ValueError, KeyError) as exc:
        print(f"  not applied: {exc}")
        return config

    try:
        applied = reviser.apply(edited, name)
    except ExperimentConflict as conflict:
        rev = conflict.revision
        print(f"\n  {rev.describe()}")
        print("    [n] start a new experiment    [o] DELETE those results")
        choice = input("  > ").strip().lower()
        if choice == "n":
            suggested = reviser.suggest_name(rev.name)
            new_name = _ask("new experiment name", suggested) or suggested
            return _apply_headless(session, reviser, edited, new_name, destroy=False)
        if choice == "o":
            print(f"  This permanently deletes {rev.root} ({rev.records} records).")
            if input("  Delete? [yes/no] > ").strip().lower() != "yes":
                return config
            if input("  Type the experiment name to confirm > ").strip() != rev.name:
                print("  did not match; nothing was deleted")
                return config
            return _apply_headless(session, reviser, edited, rev.name, destroy=True)
        print("  cancelled; settings unchanged")
        return config
    session.reconfigure(applied.config, applied.store)
    print(f"  {applied.describe()}")
    return applied.config


def _covers_all(cells, grid: dict) -> bool:
    """Whether *cells* is the whole grid rather than a subset."""
    if not cells:
        return True
    from policy_abstraction.experiments.grid import cell_label

    first_col = int(grid.get("first_col", 1))
    return set(cells) == {
        cell_label(r, c, first_col=first_col)
        for r in range(grid["rows"])
        for c in range(grid["cols"])
    }


def _clip_cells(cells, grid: dict):
    """Drop cell labels that fall outside a resized grid."""
    if not cells:
        return None
    from policy_abstraction.experiments.grid import cell_label, parse_cell_label

    first_col = int(grid.get("first_col", 1))
    kept = []
    for label in cells:
        row, col = parse_cell_label(label, first_col=first_col)
        if 0 <= row < grid["rows"] and 0 <= col < grid["cols"]:
            kept.append(cell_label(row, col, first_col=first_col))
    return kept or None


def _apply_headless(
    session: ExperimentSession,
    reviser: ExperimentReviser,
    config: ExperimentConfig,
    name: str,
    *,
    destroy: bool,
) -> ExperimentConfig:
    try:
        applied = reviser.apply(config, name, destroy=destroy)
    except ExperimentConflict as exc:
        print(f"  not applied: {exc}")
        return reviser.config
    session.reconfigure(applied.config, applied.store)
    print(f"  {applied.describe()}")
    return applied.config


def _ask_failure_modes() -> tuple:
    """List the operator's failure modes and read back the ones chosen."""
    from policy_abstraction.experiments.outcomes import OPERATOR_FAILURE_MODES

    for i, mode in enumerate(OPERATOR_FAILURE_MODES):
        print(f"    {i + 1}. {mode.label}")
    print("    (several: e.g. '3 2' or '3,2')")
    try:
        answer = input("  mode(s) > ")
    except EOFError:
        answer = ""
    return _parse_failure_modes(answer)


def _ask_acted_on(
    session: ExperimentSession, assignments: Optional[dict[str, str]] = None
) -> tuple[str, ...]:
    """Which named object(s) the policy went for, for a scene holding several.

    The same question the window asks, and skipped on the same terms: an
    experiment staging one object has nothing to confuse it with. Enter keeps
    the object the prompt named, which is the answer for every failure except
    the one this experiment exists to measure. Several numbers — ``"1 3"`` or
    ``"1,3"`` — record that it went for each in that order. *assignments*
    names the rollout being asked about when it is not the one on deck.
    """
    from policy_abstraction.experiments.outcomes import NO_OBJECT

    objects = session.config.scene_objects()
    if not objects:
        return ()
    factor = session.config.target_factor()
    if assignments is None and session.current is not None:
        assignments = session.current.assignments
    named = ""
    if factor is not None and assignments is not None:
        named = assignments.get(factor.name, "")
    print("  which object(s) did it go for? (several: '1 3', in order)")
    print("    0. no object -- never committed to one")
    for i, (object_id, label) in enumerate(objects):
        marker = "  (named)" if object_id == named else ""
        print(f"    {i + 1}. {label}{marker}")
    reply = input("  went for [named] > ").strip()
    return _parse_acted_on(reply, [o for o, _ in objects], named, NO_OBJECT)


def _parse_acted_on(
    reply: str, object_ids: list[str], named: str, no_object: str
) -> tuple[str, ...]:
    """``"1 3"`` -> the ids at those (1-based) positions, in that order.

    Empty or unparseable keeps the named object; ``0`` is "no object", and
    wins outright — it contradicts every other number. Out-of-range numbers
    are ignored rather than failing the whole answer.
    """
    from policy_abstraction.experiments.outcomes import normalize_actual_targets

    tokens = reply.replace(",", " ").split()
    if not tokens:
        return normalize_actual_targets(named)
    chosen: list[str] = []
    for token in tokens:
        try:
            index = int(token)
        except ValueError:
            continue
        if index == 0:
            return (no_object,)
        if 1 <= index <= len(object_ids):
            chosen.append(object_ids[index - 1])
    return normalize_actual_targets(chosen) or normalize_actual_targets(named)


def _parse_failure_modes(answer: str) -> tuple:
    """``"3 2"`` / ``"3,2"`` -> the modes at those (1-based) positions.

    Anything that does not name a mode is ignored; an answer that names none
    records ``OTHER``, so a slip of the hand still yields a verdict.
    """
    import re

    from policy_abstraction.experiments.outcomes import (
        OPERATOR_FAILURE_MODES,
        FailureMode,
        normalize_failure_modes,
    )

    chosen = []
    for token in re.split(r"[\s,;+]+", answer.strip()):
        if token.isdigit() and 1 <= int(token) <= len(OPERATOR_FAILURE_MODES):
            chosen.append(OPERATOR_FAILURE_MODES[int(token) - 1])
    return normalize_failure_modes(chosen) or (FailureMode.OTHER,)


def _headless_loop(
    session: ExperimentSession,
    config: ExperimentConfig,
    reviser: Optional[ExperimentReviser] = None,
) -> int:
    """Drive rollouts from stdin, for a machine with no display.

    Same verdicts and the same letters as the GUI, so muscle memory transfers.
    """
    from policy_abstraction.experiments.outcomes import (
        FailureMode,
        Outcome,
        StopReason,
    )
    from policy_abstraction.experiments.session import SessionState

    print(
        "\nKeys:  ENTER start   s success   f failure   c CATASTROPHIC   "
        "k skip   e settings   r report   q quit\n"
    )
    while True:
        status = session.status
        planned = session.current
        if planned is None:
            print(f"\nAll {status.total} rollouts collected. {status}")
            if reviser is None:
                return 0
            print("Press 'e' to edit the settings and collect more, or 'q' to quit.")
            try:
                key = input("> ").strip().lower()
            except EOFError:
                return 0
            if key == "e":
                config = _headless_settings(session, reviser)
                continue
            if key == "r":
                build_report(session.store, config)
                continue
            return 0
        print(f"\n--- rollout {planned.index + 1}/{len(session.plan)}   {status}")
        for line in session.instructions():
            print(f"    {line}")
        try:
            key = input("[ENTER to start] > ").strip().lower()
        except EOFError:
            return 0
        if key in ("q", "quit"):
            return 0
        if key == "k":
            session.skip_current()
            continue
        if key == "e":
            if reviser is None:
                print("  settings editing is unavailable in this session")
            else:
                config = _headless_settings(session, reviser)
            continue
        if key == "r":
            build_report(session.store, config)
            continue
        if key:
            continue

        session.start_rollout()
        print("  running - s success / f failure / c CATASTROPHIC")
        while session.state in (SessionState.RUNNING, SessionState.STOPPING):
            try:
                verdict = input("  verdict > ").strip().lower()
            except EOFError:
                verdict = "c"
            if verdict == "s":
                session.request_stop(Outcome.SUCCESS, FailureMode.NONE)
            elif verdict == "c":
                session.request_stop(Outcome.FAILURE, FailureMode.CATASTROPHIC)
            elif verdict == "f":
                session.request_stop(
                    Outcome.FAILURE,
                    _ask_failure_modes(),
                    StopReason.OPERATOR,
                    actual_targets=_ask_acted_on(session),
                )
            else:
                continue
            if session._thread is not None:
                session._thread.join(timeout=120)
        record = session.last_record
        if record is not None:
            print(
                f"  -> {record.outcome.value} "
                f"({record.steps} steps, {record.duration_s:.0f}s)"
            )
        timed_out = session.pop_timeout_to_classify()
        if timed_out is not None:
            # Same question the window asks: the verdict is already a
            # TIMEOUT failure, this says what kind. Enter keeps it as is.
            print("  the rollout ran out of time - what was it doing?")
            session.classify_failure(
                _ask_failure_modes(),
                actual_targets=_ask_acted_on(
                    session, assignments=timed_out.assignments
                ),
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _preflight_gui() -> None:
    """Fail now if the GUI cannot open, rather than after the policy loads.

    Everything expensive sits between here and the window: a pi05
    checkpoint is ~3 minutes of load and compile, the env server claims
    the cameras, and on hardware the operator has already staged the
    scene. Discovering a missing Qt platform plugin *then* costs the
    whole wait, and Qt does not raise — it calls ``qFatal`` and the
    process aborts, so there is nothing to catch in-process either.

    Hence a subprocess: it absorbs the abort, and its stderr carries
    Qt's own diagnostic, which names the missing piece.
    """
    probe = "from PyQt6.QtWidgets import QApplication;QApplication([]);print('ok')"
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"Could not probe the GUI ({exc}); trying anyway.")
        return
    if result.returncode == 0:
        return

    detail = (result.stderr or result.stdout).strip()
    hint = ""
    if "xcb-cursor" in detail or "xcb" in detail:
        # Qt 6.5+ needs libxcb-cursor0, which is not pulled in by the
        # PyQt6 wheel and is absent from most headless images.
        hint = (
            "\nOn Debian/Ubuntu this is usually one package:\n"
            "    sudo apt-get install -y libxcb-cursor0\n"
            "No root on this box? Vendor it into the venv instead:\n"
            "    scripts/setup/vendor_qt_xcb_cursor.sh\n"
            "Over SSH, also check X11 forwarding is up: `ssh -X`, and\n"
            "`echo $DISPLAY` non-empty on this end."
        )
    elif not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        hint = (
            "\nNo DISPLAY or WAYLAND_DISPLAY is set, so there is no\n"
            "screen to open a window on. Over SSH use `ssh -X` (or -Y)."
        )

    raise SystemExit(
        "The operator GUI cannot start, so nothing was loaded and the "
        "robot was not touched.\n\n"
        f"{detail}\n{hint}\n\n"
        "Or run with --experiment.headless to drive the same rollouts "
        "from stdin, with the same hotkey letters."
    )


def _announce_successor(store: ExperimentStore) -> None:
    """Say so when this experiment was continued elsewhere under a new name.

    Renaming from the settings editor leaves the old directory exactly as it
    was, which is the point — but it also means the launcher's default name
    still opens the old one. Silently resuming the superseded half of a
    comparison is a whole afternoon wasted, so it is called out.
    """
    successor = store.load_metadata().get("superseded_by")
    if not successor:
        return
    logger.warning(
        f"{store.root.name} was continued as {successor!r} after its settings "
        "were edited. Resuming this one. To continue the other, relaunch with "
        f"--name {successor} (or --experiment.experiment-name {successor})."
    )


def _follow_revision(applied: AppliedRevision) -> None:
    """Move the log with the experiment when the operator edits it.

    The sink opened at start-up points into the directory the session began in.
    After a revision that may be a different directory — and an experiment
    whose ``evaluation.log`` stops at the moment it was reconfigured is exactly
    the log you want when working out what happened.
    """
    logger.add(
        str(applied.store.log_path), rotation="10 MB", level="DEBUG", enqueue=True
    )
    logger.info(f"Experiment settings revised: {applied.describe()}")


def main(args: Args) -> int:
    exp = args.experiment

    if exp.report_only:
        return _report_only(args)

    usage = (
        "Pass --experiment.experiment-config <yaml> to create an experiment, "
        "--experiment.reference-experiment <name> to create one from an "
        "experiment already collected, or --experiment.experiment-name <name> "
        "of one that already exists. See configs/experiments/ for examples, "
        "and docs/real_experiments.md for what each field means."
    )
    if exp.experiment_config and exp.reference_experiment:
        raise SystemExit(
            "Pass either --experiment.experiment-config or "
            "--experiment.reference-experiment, not both: a reference "
            "experiment brings its own settings, and the YAML would not be "
            "applied to them."
        )
    if exp.objects and exp.reference_experiment:
        raise SystemExit(
            "--experiment.objects re-samples the layouts, which is the opposite "
            "of what --experiment.reference-experiment is for. Drop one of them."
        )
    has_source = bool(exp.experiment_config or exp.reference_experiment)
    if not has_source and not (exp.experiment_name and not exp.calibrate):
        raise SystemExit(usage)
    reference: Optional[ExperimentStore] = None
    source: Optional[ExperimentConfig] = None
    if exp.reference_experiment:
        source, reference = _load_reference_config(exp, args.output.run)
    elif exp.experiment_config:
        source = _load_config(exp)

    if exp.calibrate:
        assert source is not None
        return _calibrate(args, source)

    # Before the GUI, the store and the ~3 minute checkpoint load: a label the
    # flags contradict is a usage error, not a run.
    check_policy_setting(args)

    # Before the store, the servers and the ~3 minute checkpoint load.
    if not exp.headless:
        _preflight_gui()

    store = _resolve_store(args, source)
    if reference is not None and store.root.resolve() == reference.root.resolve():
        raise SystemExit(
            f"{store.root} is the reference experiment itself. Give the new "
            "experiment its own --experiment.experiment-name; to resume the "
            "reference, pass its name without --experiment.reference-experiment."
        )
    config, from_experiment = source, False
    if not exp.overwrite:
        # --overwrite means "start over from what the YAML says", so the
        # experiment's own settings are deliberately not consulted there.
        config, from_experiment = _effective_config(
            store,
            source,
            source_label=(
                f"the reference {reference.root.name}"
                if reference is not None
                else "the YAML"
            ),
        )
    if config is None:
        raise SystemExit(f"No experiment at {store.root} to resume. {usage}")
    if from_experiment:
        logger.info(f"Resuming under the settings stored in {store.config_path}")
        if reference is not None:
            logger.warning(
                f"--experiment.reference-experiment is not applied to an "
                f"experiment that already exists: {store.root.name} keeps its "
                "own plan. Pass --experiment.overwrite to recreate it from "
                f"{reference.root.name}."
            )
        if exp.objects:
            logger.warning(
                "--experiment.objects is ignored when resuming: the object list "
                "is part of the stored plan. Pass --experiment.overwrite to "
                "start over with the new objects."
            )

    metadata: dict = {
        "policy_setting": exp.policy_setting,
        "task_id": exp.task_id,
        "eval_args": dataclasses.asdict(args),
    }
    if source is not None and reference is not None:
        # Provenance for the other arm: which experiment's plan this one
        # was made to match.
        metadata["reference_experiment"] = str(reference.root.resolve())
        metadata["source_config"] = str(reference.config_path.resolve())
        metadata[SOURCE_SNAPSHOT] = source.to_dict()
    elif source is not None and exp.experiment_config:
        metadata["source_config"] = str(pathlib.Path(exp.experiment_config).resolve())
        # The YAML as parsed, not as text, so a reformatted-but-identical
        # file reads as the same launch.
        metadata[SOURCE_SNAPSHOT] = source.to_dict()
    try:
        store = ExperimentStore.open(
            store.root, config, overwrite=exp.overwrite, metadata=metadata
        )
    except ExperimentExistsError as exc:
        raise SystemExit(
            f"\n{exc}\n"
            f"The plan in {store.plan_path} was frozen under settings that\n"
            f"    {store.config_path}\n"
            "no longer reproduces. Restore that file from the plan's era, or "
            "pass\n--experiment.overwrite to start again from the YAML.\n"
        ) from None
    _announce_successor(store)

    # The reference's staged scenes, ghosted under the staging view. Read
    # back from the metadata on a resume, so the ghost does not depend on
    # the flag being repeated on every relaunch.
    if reference is None:
        remembered = store.load_metadata().get("reference_experiment")
        if remembered and ExperimentStore(remembered).exists:
            reference = ExperimentStore(remembered)
    reference_frames: Optional[ReferenceFrames] = None
    if reference is not None:
        reference_frames = ReferenceFrames(reference)
        logger.info(
            f"Ghosting {reference.root.name}'s staged scenes under the live view "
            "(G cycles off / faint / strong / reference only)"
        )

    logger.add(str(store.log_path), rotation="10 MB", level="DEBUG", enqueue=True)
    seed_everything(args.seed)

    status = store.status()
    if status.completed:
        logger.info(f"Resuming: {status}")
    write_provenance(store.root, collect_provenance(args))

    # Route the evaluator's episode/video output into the experiment directory
    # so a rollout's .vla sits beside the record that describes it.
    args.output.episode_data_path = str(store.episodes_dir)
    args.output.video_out_path = str(store.videos_dir)

    def _build_session() -> ExperimentSession:
        """Servers up, session built, arm parked and a first frame in hand.

        Everything slow lives in here — a pi05 checkpoint is about three
        minutes of load and compile — so that the GUI can run it on a worker
        thread with a window already on screen, instead of the operator
        watching a still terminal and guessing whether it has hung. The
        headless path calls it directly, where there is nothing to show anyway.
        """
        logger.info("Starting policy and env servers (loaded once for the session) …")
        backends = start_backends(args, store.root, install_signal_handlers=False)
        session = ExperimentSession(
            backends,
            config,
            store,
            task_id=exp.task_id,
            policy_setting=exp.policy_setting,
            reference=reference_frames,
        )
        try:
            # Talks to the robot: it homes. Deliberately in here rather than
            # before the window, so a slow homing move is something the
            # operator can watch rather than another silent wait.
            session.prepare_env()
        except Exception:
            session.shutdown()
            raise
        logger.info("Servers up. Opening the operator window …")
        return session

    # Lets the operator edit the grid, the repeats and the limits from the
    # window without paying the checkpoint load again. It owns which store the
    # session is writing to from here on, so the report and the closing summary
    # read it rather than the store opened above.
    reviser = ExperimentReviser(
        run=args.output.run,
        name=exp.experiment_name or config.name,
        config=config,
        store=store,
        source_config=exp.experiment_config,
        # Carried onto whatever directory an edit lands in, so the next launch
        # can still tell a GUI edit from an edit of the YAML.
        metadata=metadata,
        on_applied=_follow_revision,
    )

    def _report() -> dict:
        return build_report(reviser.store, reviser.config)

    if exp.headless:
        session = _build_session()
        try:
            code = _headless_loop(session, reviser.config, reviser)
        finally:
            session.shutdown()
    else:
        from policy_abstraction.experiments.gui import run_gui

        # The factory, not a session: the window opens now and the servers
        # come up behind it. run_gui owns the session it builds, including
        # shutting it down on every exit path.
        code = run_gui(
            _build_session,
            reviser.config,
            on_report=_report,
            refresh_hz=exp.refresh_hz,
            reviser=reviser,
        )

    store = reviser.store
    summary = _report()
    totals = summary["totals"]
    print(
        f"\n{'=' * 66}\n"
        f"{summary['experiment']} [{exp.policy_setting or 'unlabelled'}]\n"
        f"  {totals['successes']}/{totals['collected']} succeeded "
        f"({totals['success_rate']:.0%}), 95% CI "
        f"[{totals['ci95'][0]:.0%}, {totals['ci95'][1]:.0%}]\n"
        f"  {totals['collected']}/{totals['planned']} planned rollouts collected\n"
        f"  report: {store.report_dir / 'report.md'}\n"
        f"{'=' * 66}"
    )
    if totals["collected"] < totals["planned"]:
        if reviser.name == (exp.experiment_name or config.name):
            print("Re-run the same command to resume where this session stopped.\n")
        else:
            # The settings editor renamed it, so "the same command" would open
            # the experiment this one was started from instead.
            print(
                "The settings were edited during this session, so resume with\n"
                f"    --name {reviser.name}\n"
                f"(or --experiment.experiment-name {reviser.name}). The edited "
                "settings are\nstored with the results and are picked up "
                "automatically.\n"
            )
    print(
        json.dumps(summary["failure_modes"], indent=2)
        if summary["failure_modes"]
        else ""
    )
    return code


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
