"""Launcher helper for the CALVIN env server.

Mirrors :func:`policy_abstraction.envs.libero.utils.start_libero_env_server`
and :func:`policy_abstraction.envs.robocasa.utils.start_robocasa_env_server`,
but execs the calvin venv's Python so pybullet/hydra/calvin_env resolve from
``third_party/calvin/.venv``.
"""

from __future__ import annotations

import os
import subprocess
from typing import Iterable

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.utils.servers import spawn_server


def start_calvin_env_server(
    env_host: str | None,
    env_port: int | None,
    *,
    scenes: Iterable[str] | None = None,
    suite_name: str = "calvin",
    cameras_cfg: str = "static_and_gripper",
    robot_cfg: str = "panda_longer_finger",
    seed: int = 0,
    num_init_states: int = 10,
    max_steps: int | None = None,
    primary_camera: str = "static",
    include_gripper: bool = True,
    show_gui: bool = False,
    use_egl: bool = True,
) -> "subprocess.Popen":
    """Start the CALVIN WebSocket env server in a subprocess.

    Returns the subprocess so the caller can manage its lifecycle.
    """
    from loguru import logger

    calvin_root = PROJECT_ROOT / "third_party" / "calvin"
    venv_python = calvin_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(
            f"calvin venv python not found at {venv_python}. "
            "Set up with: cd third_party/calvin && uv sync"
        )

    if scenes is None:
        raise ValueError("Pass scenes=[…] (a list of CALVIN scene config names).")
    scenes_list = list(scenes)
    if not scenes_list:
        raise ValueError("scenes must be non-empty.")

    env_vars = os.environ.copy()
    env_vars["MPLBACKEND"] = "Agg"
    # Default to off-screen EGL unless the caller explicitly disables it.
    if use_egl and "PYOPENGL_PLATFORM" not in env_vars:
        env_vars["PYOPENGL_PLATFORM"] = "egl"

    cli_args: list[str] = [
        str(venv_python),
        str(PROJECT_ROOT / "scripts" / "servers" / "serve_calvin_env.py"),
        "--suite-name",
        suite_name,
        "--cameras-cfg",
        cameras_cfg,
        "--robot-cfg",
        robot_cfg,
        "--seed",
        str(seed),
        "--num-init-states",
        str(num_init_states),
        "--primary-camera",
        primary_camera,
    ]
    if env_host is not None:
        cli_args.extend(["--host", env_host])
    if env_port is not None:
        cli_args.extend(["--port", str(env_port)])
    if max_steps is not None:
        cli_args.extend(["--max-steps", str(max_steps)])
    if not include_gripper:
        cli_args.append("--no-gripper")
    if show_gui:
        cli_args.append("--show-gui")
    if not use_egl:
        cli_args.append("--no-egl")
    cli_args.append("--scenes")
    cli_args.extend(scenes_list)

    server_process = spawn_server(
        cli_args,
        env=env_vars,
        cwd=str(calvin_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    logger.info(f"Started CalvinEnvServer with PID {server_process.pid}")
    return server_process


__all__ = ["start_calvin_env_server"]
