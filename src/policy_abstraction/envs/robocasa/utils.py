"""Launcher helper for the RoboCasa env server.

Mirrors :func:`policy_abstraction.envs.libero.utils.start_libero_env_server`,
but execs the robocasa venv's Python so robosuite/MuJoCo resolve from
``third_party/robocasa/.venv``.
"""

from __future__ import annotations

import os
import subprocess
from typing import Iterable

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.utils.servers import spawn_server


def start_robocasa_env_server(
    env_host: str | None,
    env_port: int | None,
    *,
    task: str | None = None,
    tasks: Iterable[str] | None = None,
    task_suite: str | None = None,
    suite_name: str = "robocasa_atomic",
    robots: str = "PandaOmron",
    split: str | None = None,
    seed: int = 0,
    num_init_states: int = 20,
    max_steps: int | None = None,
    primary_camera: str = "robot0_agentview_left",
    include_right_view: bool = True,
) -> "subprocess.Popen":
    """Start the RoboCasa WebSocket env server in a subprocess.

    The server runs in ``third_party/robocasa/.venv`` (per the standalone
    package layout), while the caller (root venv) just connects to the port.

    Returns the subprocess so the caller can manage its lifecycle.
    """
    from loguru import logger

    robocasa_root = PROJECT_ROOT / "third_party" / "robocasa"
    venv_python = robocasa_root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(
            f"robocasa venv python not found at {venv_python}. "
            "Set up with: cd third_party/robocasa && uv sync"
        )

    selectors_set = sum(x is not None for x in (task, tasks, task_suite))
    if selectors_set != 1:
        raise ValueError(
            "Pass exactly one of task=…, tasks=[…], or "
            "task_suite=<atomic_seen|composite_seen|composite_unseen>; "
            f"got task={task!r}, tasks={tasks!r}, task_suite={task_suite!r}"
        )

    env_vars = os.environ.copy()
    env_vars["MUJOCO_GL"] = env_vars.get("MUJOCO_GL", "egl")
    env_vars["MPLBACKEND"] = "Agg"

    cli_args: list[str] = [
        str(venv_python),
        str(PROJECT_ROOT / "scripts" / "servers" / "serve_robocasa_env.py"),
        "--suite-name",
        suite_name,
        "--robots",
        robots,
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
    if split is not None:
        cli_args.extend(["--split", split])
    if max_steps is not None:
        cli_args.extend(["--max-steps", str(max_steps)])
    if not include_right_view:
        cli_args.append("--no-right-view")
    if task_suite is not None:
        cli_args.extend(["--task-suite", task_suite])
    elif task is not None:
        cli_args.extend(["--task", task])
    else:
        cli_args.append("--tasks")
        cli_args.extend(list(tasks or []))

    server_process = spawn_server(
        cli_args,
        env=env_vars,
        cwd=str(robocasa_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    logger.info(f"Started RobocasaEnvServer with PID {server_process.pid}")
    return server_process


__all__ = ["start_robocasa_env_server"]
