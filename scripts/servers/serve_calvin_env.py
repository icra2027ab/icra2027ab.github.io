"""Entry-point script for the CALVIN environment WebSocket server.

Run this in the calvin simulator venv (``third_party/calvin/.venv``) *before*
launching the evaluation script:

    # Terminal 1 — calvin Python
    cd third_party/calvin
    uv run python ../../scripts/servers/serve_calvin_env.py \\
        --host 0.0.0.0 --port 8767 \\
        --scenes calvin_scene_D

    # Terminal 2 — policy Python (root venv)
    python -c "from policy_abstraction.envs import WebsocketEnvClient; \\
        env = WebsocketEnvClient(port=8767); env.setup_suite()"

The wire protocol matches :class:`LiberoEnvServer` and
:class:`RobocasaEnvServer`, so the existing
:class:`policy_abstraction.envs.WebsocketEnvClient` connects without changes.

Required transport deps in the calvin venv (one-time):
    cd third_party/calvin && uv add websockets msgpack-numpy tyro loguru
"""

from __future__ import annotations

import dataclasses
import logging
import os
import sys
from typing import Optional

import tyro

# Make src/ importable when launched from inside third_party/calvin.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from policy_abstraction.envs.calvin.config import DEFAULT_MAX_STEPS  # noqa: E402
from policy_abstraction.envs.calvin.server import CalvinEnvServer  # noqa: E402


@dataclasses.dataclass
class Args:
    """Arguments for the CALVIN environment server."""

    host: str = "0.0.0.0"
    """Interface to bind to (0.0.0.0 accepts all incoming connections)."""

    port: int = 8767
    """WebSocket port to listen on."""

    scenes: Optional[list[str]] = None
    """List of CALVIN scene config names (e.g. ``calvin_scene_D``).

    Each scene becomes one ``task_id``.
    """

    suite_name: str = "calvin"
    """Logical suite label returned to the client via ``setup_suite``."""

    cameras_cfg: str = "static_and_gripper"
    """Hydra cameras group (matches conf/cameras/*.yaml)."""

    robot_cfg: str = "panda_longer_finger"
    """Hydra robot group (matches conf/robot/*.yaml)."""

    seed: int = 0
    """Base random seed."""

    num_init_states: int = 10
    """Number of pre-sampled initial states cached per task."""

    max_steps: int = DEFAULT_MAX_STEPS
    """Per-episode step cap returned via ``setup_suite``."""

    primary_camera: str = "static"
    """Camera marked ``is_primary=True`` in the published EnvSpec."""

    no_gripper: bool = False
    """Drop the ``gripper`` camera from the published EnvSpec."""

    show_gui: bool = False
    """Open the PyBullet GUI (debugging only — incompatible with EGL)."""

    no_egl: bool = False
    """Disable EGL off-screen rendering (use with --show-gui)."""

    no_segmentation: bool = False
    """Skip rendering instance-segmentation masks per camera (saves one
    extra ``getCameraImage`` call per camera per step)."""


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.scenes:
        raise SystemExit("Pass --scenes <name> [<name> ...] (at least one).")

    server = CalvinEnvServer(
        host=args.host,
        port=args.port,
        scenes=args.scenes,
        suite_name=args.suite_name,
        cameras_cfg=args.cameras_cfg,
        robot_cfg=args.robot_cfg,
        seed=args.seed,
        num_init_states=args.num_init_states,
        max_steps=args.max_steps,
        primary_camera=args.primary_camera,
        include_gripper=not args.no_gripper,
        show_gui=args.show_gui,
        use_egl=not args.no_egl,
        enable_segmentation=not args.no_segmentation,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
