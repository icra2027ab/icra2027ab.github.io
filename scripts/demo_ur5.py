"""Live guidance demo on the real UR5e cell.

One policy, loaded once. One rollout at a time, started and stopped from the
window. And a switch that turns guidance on and off *under the running
policy*: retrieval, perception and alignment keep running either way, so the
only thing the switch changes is whether the next chunk is steered. The window
shows the exterior camera with the tracked objects and the distance pairs
drawn on it, the wrist camera, a trend plot of the live distances against the
retrieved demo's, and a banner that says which mode the arm is in. A Rerun
viewer alongside shows the 3-D scene.

The env and policy backends are selected exactly as in ``scripts/evaluate.py``
(the same tyro subcommands, the same flags), because they are the same code.
The demo's own flags are under ``--demo.``.

The cell launcher wires the hardware in for you; this is what it runs::

    scripts/run_ur5_demo.sh                              # guided pipeline up, switch off
    RETRIEVAL_DIR=data/reference_dataset_cable scripts/run_ur5_demo.sh
    scripts/run_ur5_demo.sh --start-guided               # switch on from the first rollout
    scripts/run_ur5_demo.sh --simulate --fake-policy circular   # rehearsal, no arm, no checkpoint

Called directly, the guided configuration is the one that gives the switch
something to do::

    python scripts/demo_ur5.py --perception.mode REAL_PERCEPTION \\
        --retrieval.dataset-dirs data/reference_dataset_cable \\
        --viz.layout scene --viz.robot-description ur5e_mj_description \\
      env:ur5 --env.calibration-path configs/ur5_camera_calibration.json \\
      policy:lerobot --policy.model-path <checkpoint> --policy.rtc \\
        --policy.replan-steps 25 --policy.rtc-execution-horizon 25

Launched unguided (``--no-guide``) it still streams and runs rollouts; the
switch then has nothing to open and the window says so.

Hotkeys: Space start, S stop and home, G guidance, O overlays, T trend plot,
V viewer layout, Esc quit. See ``docs/demo.md``.
"""

from __future__ import annotations

import os

# Must precede any torch import; mirrors scripts/evaluate.py.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import dataclasses  # noqa: E402
import logging  # noqa: E402
import pathlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from typing import Literal, Optional  # noqa: E402

import tyro  # noqa: E402
from loguru import logger  # noqa: E402

from policy_abstraction.demo import DemoSession, OverlayMode  # noqa: E402
from policy_abstraction.evaluation.backends import start_backends  # noqa: E402
from policy_abstraction.evaluation.config import Args as EvalArgs  # noqa: E402
from policy_abstraction.evaluation.config import (  # noqa: E402
    LerobotPolicyArgs,
    Ur5EnvArgs,
)
from policy_abstraction.utils.provenance import (  # noqa: E402
    collect_provenance,
    write_provenance,
)
from policy_abstraction.utils.seeding import seed_everything  # noqa: E402

# Silence per-request HTTP/HF chatter from the in-process model loaders, as
# scripts/evaluate.py does. Runs at import time, before any HF call.
for _name in (
    "httpx",
    "httpcore",
    "urllib3",
    "huggingface_hub",
    "transformers",
    "filelock",
):
    logging.getLogger(_name).setLevel(logging.WARNING)


@dataclasses.dataclass
class DemoArgs:
    """The demo's own flags, on top of the shared evaluation ones."""

    guided: bool = False
    """Position of the guidance switch at the first rollout. The operator
    flips it from the window (G) whenever they like; this only says where it
    starts."""
    overlays: Literal["auto", "on", "off"] = "auto"
    """What the exterior view draws on the frame. ``auto`` draws the masks and
    pairs while guidance is on and the bare frame while it is off, so the
    picture itself says which mode the arm is in; ``on`` / ``off`` pin it.
    The O hotkey cycles through the three."""
    trends: bool = True
    """Show the trend plot from the start (T toggles it in the window)."""
    fullscreen: bool = True
    """Start maximised; F11 toggles true full screen."""
    max_steps: int = 0
    """Per-rollout cap. ``0`` uses the env's own horizon."""
    refresh_hz: float = 20.0
    """How often the window repaints from the live snapshot."""
    guided_warmup: bool = True
    """Also compile the guided path during start-up, so the first flip of the
    switch does not stall the arm. Costs one extra throw-away inference."""
    save_video: bool = False
    """Write a replay video after each rollout (slow; the episode itself is
    recorded regardless)."""
    name: str = "demo"
    """Output sub-folder under ``data/<run>/``."""
    task_id: int = 0
    """Which of the env server's tasks to run."""
    training_positions: Optional[str] = None
    """A ``*_grasp_positions.json`` from
    ``scripts/experiments/plot_training_on_grid.py``: where the training demos
    grasped the object, in the robot base frame. Given, the window can overlay
    their density on the live table as the training distribution (checkbox /
    H). The launcher passes the cell's file when it exists."""
    heatmap: bool = False
    """Start with the training heat map shown."""


@dataclasses.dataclass
class Args(EvalArgs):
    """Everything ``scripts/evaluate.py`` takes, plus the demo flags.

    Subclassing rather than nesting keeps every evaluation flag spelled
    exactly as it is for ``evaluate.py`` — ``--guidance.debug``,
    ``env:ur5``, ``policy:lerobot --policy.model-path ...`` — so a command can
    be moved between the two scripts by changing only the script name.
    """

    demo: DemoArgs = dataclasses.field(default_factory=DemoArgs)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def check_guidance_configuration(args: Args) -> None:
    """Say up front what the switch will be able to do.

    An unguided launch is allowed — the demo still streams and runs the
    policy — but the switch then has nothing to open, and that is worth a
    warning before three minutes of checkpoint load rather than a badge
    afterwards. A guided launch against a policy server told to refuse guided
    requests is not allowed: every flip would be an error at the server, and
    the operator would be showing an arm that ignores them.
    """
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
        logger.warning(
            "launched unguided — the guidance switch will have nothing to open; "
            "pass the guided flags (RETRIEVAL_DIR, --perception.mode "
            "REAL_PERCEPTION) to demo guidance "
            f"(switched off at {', '.join(off)})"
        )
        return
    if (
        isinstance(args.env, Ur5EnvArgs)
        and isinstance(args.policy, LerobotPolicyArgs)
        and args.policy.guidance == "off"
    ):
        raise SystemExit(
            "--policy.guidance off makes the policy server refuse guided "
            "requests, so every flip of the guidance switch would fail at the "
            "server. Drop it (the default, auto, steers whenever the run asks), "
            "or launch the demo unguided with --no-guide."
        )


def _preflight_gui() -> None:
    """Fail now if the GUI cannot open, rather than after the policy loads.

    Mirrors ``scripts/experiments/collect_experiment.py::_preflight_gui`` —
    copied rather than imported because ``scripts/`` is not a package.

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
        "The demo window cannot start, so nothing was loaded and the "
        "robot was not touched.\n\n"
        f"{detail}\n{hint}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _task_for(args: Args) -> Optional[str]:
    """The prompt handed to the policy, or ``None`` for the env's own."""
    if isinstance(args.env, Ur5EnvArgs):
        tasks = args.env.tasks
        if not tasks:
            return None
        return tasks[min(args.demo.task_id, len(tasks) - 1)]
    return None


def main(args: Args) -> int:
    # Before the GUI, the servers and the ~3 minute checkpoint load: a
    # configuration the switch cannot work with is a usage error, not a run.
    check_guidance_configuration(args)
    _preflight_gui()

    run_dir = pathlib.Path(f"data/{args.output.run}/{args.demo.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    # Route the evaluator's episode/video output into the demo directory so a
    # rollout's .vla sits beside the log that describes it.
    args.output.episode_data_path = str(run_dir / "episodes")
    args.output.video_out_path = str(run_dir / "videos")
    log_path = run_dir / "demo.log"
    logger.add(str(log_path), rotation="10 MB", level="DEBUG", enqueue=True)
    seed_everything(args.seed)
    write_provenance(run_dir, collect_provenance(args))

    task = _task_for(args)

    heatmap = None
    if args.demo.training_positions:
        from policy_abstraction.demo.heatmap import TrainingHeatmap

        try:
            heatmap = TrainingHeatmap.load(args.demo.training_positions)
            logger.info(
                f"Training heat map: {heatmap.count} grasps from "
                f"{args.demo.training_positions}"
            )
        except Exception as exc:
            logger.warning(
                f"Could not load the training positions from "
                f"{args.demo.training_positions} ({exc}); the heat map is off."
            )

    def _build_session() -> DemoSession:
        """Servers up, session built, arm parked, policy warmed.

        Everything slow lives in here — a pi05 checkpoint is about three
        minutes of load and compile, then the plain and guided warm-ups — so
        the GUI can run it on a worker thread with a window already on
        screen, instead of the operator watching a still terminal.
        """
        logger.info("Starting policy and env servers (loaded once for the session) …")
        backends = start_backends(args, run_dir, install_signal_handlers=False)
        session = DemoSession(
            backends,
            task_id=args.demo.task_id,
            task=task,
            max_steps=args.demo.max_steps,
            guided=args.demo.guided,
            overlay_mode=OverlayMode(args.demo.overlays),
            guided_warmup=args.demo.guided_warmup,
            save_video=args.demo.save_video,
            output_dir=run_dir / "episodes",
            training_heatmap=heatmap,
            show_heatmap=args.demo.heatmap,
        )
        try:
            # Talks to the robot: the env's own reset may home it. In here
            # rather than before the window, so a slow move is something the
            # operator can watch rather than another silent wait.
            session.prepare_env()
        except Exception:
            session.shutdown()
            raise
        logger.info("Servers up. Opening the demo window …")
        return session

    from policy_abstraction.demo.gui import run_demo_gui

    gui_kwargs: dict = dict(
        title=f"Guidance demo — {task or args.output.run}",
        refresh_hz=args.demo.refresh_hz,
        show_trends=args.demo.trends,
        fullscreen=args.demo.fullscreen,
    )

    # The factory, not a session: the window opens now and the servers come
    # up behind it. run_demo_gui owns the session it builds, including
    # shutting it down on every exit path.
    sessions: list[DemoSession] = []

    def _build_and_remember() -> DemoSession:
        session = _build_session()
        sessions.append(session)
        return session

    code = run_demo_gui(_build_and_remember, **gui_kwargs)

    rollouts = sum(s.rollouts for s in sessions)
    paths = [p for s in sessions for p in s.episode_paths]
    print(
        f"\n{'=' * 66}\n"
        f"Guidance demo [{task or args.output.run}]\n"
        f"  {rollouts} rollout{'s' if rollouts != 1 else ''} run\n"
        + "".join(f"  episode: {p}\n" for p in paths)
        + f"  log: {log_path}\n"
        f"{'=' * 66}"
    )
    for session in sessions:
        if session.last_error:
            print(f"last error: {session.last_error}")
    return int(code or 0)


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
