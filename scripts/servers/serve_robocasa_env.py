"""Entry-point script for the RoboCasa environment WebSocket server.

Run this in the robocasa simulator venv (``third_party/robocasa/.venv``)
*before* launching the evaluation script:

    # Terminal 1 — robocasa Python
    cd third_party/robocasa
    uv run python ../../scripts/servers/serve_robocasa_env.py \\
        --host 0.0.0.0 --port 8766 \\
        --task PnPCounterToCab

    # Terminal 2 — policy Python (root venv)
    python scripts/evaluate.py \\
        --env.host localhost --env.port 8766 ...

The WebSocket protocol matches :class:`LiberoEnvServer`, so the existing
:class:`policy_abstraction.envs.WebsocketEnvClient` connects unchanged.

Required transport deps in the robocasa venv (one-time):
    cd third_party/robocasa && uv add websockets msgpack-numpy tyro loguru
"""

import dataclasses
import logging
import os
import sys
from typing import Optional

import tyro

# Make src/ importable when launched from inside third_party/robocasa.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from policy_abstraction.envs.robocasa.server import RobocasaEnvServer  # noqa: E402


@dataclasses.dataclass
class Args:
    """Arguments for the RoboCasa environment server."""

    host: str = "0.0.0.0"
    """Interface to bind to (0.0.0.0 accepts all incoming connections)."""

    port: int = 8766
    """WebSocket port to listen on."""

    task: Optional[str] = None
    """Single RoboCasa env-class name (e.g. ``PnPCounterToCab``).

    Mutually exclusive with --tasks and --task-suite.
    """

    tasks: Optional[list[str]] = None
    """List of RoboCasa env-class names to expose as separate task_ids.

    Mutually exclusive with --task and --task-suite.
    """

    task_suite: Optional[str] = None
    """RoboCasa leaderboard task suite to evaluate.

    One of ``atomic_seen``, ``composite_seen``, ``composite_unseen`` — resolved
    server-side via ``robocasa.utils.dataset_registry.TASK_SET_REGISTRY``. When
    set, per-task ``max_steps`` is read from each task's ``horizon`` field
    (unless --max-steps is explicitly passed, which overrides as a uniform
    cap), and --split defaults to ``target`` to match the leaderboard.
    Mutually exclusive with --task and --tasks.
    """

    suite_name: str = "robocasa_atomic"
    """Logical suite label returned to the client via ``setup_suite``."""

    robots: str = "PandaOmron"
    """Robosuite robot to instantiate (``PandaOmron``, ``Panda``, …)."""

    split: Optional[str] = None
    """RoboCasa scene split: one of {None, "all", "pretrain", "target"}."""

    seed: int = 0
    """Base random seed; per-task init states use seed..seed+num_init_states-1."""

    num_init_states: int = 20
    """Number of pre-sampled initial states cached per task."""

    max_steps: Optional[int] = None
    """Per-episode step cap returned to the client via ``setup_suite``.

    When unset, the server applies its own default: with --task-suite, each
    task's leaderboard horizon; otherwise a uniform fallback. When set, this
    value uniformly caps every task (overrides leaderboard horizons).
    """

    primary_camera: str = "robot0_agentview_left"
    """Camera marked ``is_primary=True`` in the published EnvSpec."""

    no_right_view: bool = False
    """Drop ``robot0_agentview_right`` from the published EnvSpec."""


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    selectors_set = sum(x is not None for x in (args.task, args.tasks, args.task_suite))
    if selectors_set != 1:
        raise SystemExit(
            "Specify exactly one of --task, --tasks, or --task-suite "
            "<atomic_seen|composite_seen|composite_unseen>."
        )
    task_names: list[str] | None
    if args.task is not None:
        task_names = [args.task]
    elif args.tasks is not None:
        task_names = list(args.tasks)
    else:
        task_names = None  # task_suite path — server resolves the list itself

    server = RobocasaEnvServer(
        host=args.host,
        port=args.port,
        task_names=task_names,
        task_suite=args.task_suite,
        suite_name=args.suite_name,
        robots=args.robots,
        split=args.split,
        seed=args.seed,
        num_init_states=args.num_init_states,
        max_steps=args.max_steps,
        primary_camera=args.primary_camera,
        include_right_view=not args.no_right_view,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
