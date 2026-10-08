"""Environment utilities for LIBERO evaluations."""

from __future__ import annotations

import os
import pathlib
import subprocess

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.utils.servers import spawn_server


def start_libero_env_server(
    env_host: str | None,
    env_port: int | None,
    task_suite_name: str,
    evaluation_config_path: str,
    use_swap: bool = False,
    use_object: bool = False,
    use_language: bool = False,
    use_task: bool = False,
    use_environment: bool = False,
    seed: int = 0,
    log_dir: str | os.PathLike | None = None,
) -> "subprocess.Popen":
    """Start the LIBERO environment WebSocket server in another python process.

    All LIBERO-PRO suite / perturbation configuration is passed to the server
    subprocess so that all suite-related logic (config loading, perturbation
    setup, etc.) happens server-side.

    Args:
        log_dir: Directory to write ``libero_env_server.{stdout,stderr}.log``
            into. When ``None``, the child inherits the parent's stdio.
            NEVER pass ``subprocess.PIPE`` here without a reader: the OS
            pipe buffer (~64 KB on Linux) fills under sustained logging
            and blocks the server mid-write.

    Returns the subprocess so the caller can manage its lifecycle.
    """
    from loguru import logger

    libero_pro_root = PROJECT_ROOT / "third_party" / "LIBERO-PRO"

    env_vars = os.environ.copy()
    # LIBERO_CONFIG_PATH is set inside server.py to this package's configs/
    # dir. Don't set it here — that path lacks a config.yaml and triggers
    # LIBERO's interactive "specify a custom dataset folder?" prompt.
    env_vars["PYTHONPATH"] = str(libero_pro_root)
    env_vars["MPLBACKEND"] = "Agg"
    # Defaults below respect anything the user already exported (e.g. via .env).
    # On headless boxes set MUJOCO_GL=egl in .env; on desktops with X11 keep
    # glx. Hardcoding either one would break the other setup.
    env_vars.setdefault("MUJOCO_GL", "glx")

    cli_args = [
        str(libero_pro_root / ".venv" / "bin" / "python"),
        "scripts/servers/serve_libero_env.py",
        "--task-suite-name",
        task_suite_name,
        "--evaluation-config-path",
        evaluation_config_path,
        "--seed",
        str(seed),
    ]
    if env_host is not None:
        cli_args.extend(["--host", env_host])
    if env_port is not None:
        cli_args.extend(["--port", str(env_port)])
    for flag_name, flag_val in [
        ("use-swap", use_swap),
        ("use-object", use_object),
        ("use-language", use_language),
        ("use-task", use_task),
        ("use-environment", use_environment),
    ]:
        cli_args.append(f"--{flag_name}" if flag_val else f"--no-{flag_name}")

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "libero_env_server.stdout.log"
        stderr_path = log_dir / "libero_env_server.stderr.log"
        stdout_f = open(stdout_path, "w", buffering=1)
        stderr_f = open(stderr_path, "w", buffering=1)
        logger.info(f"LIBERO env server logs → {stdout_path}, {stderr_path}")
    else:
        stdout_f = None
        stderr_f = None

    server_process = spawn_server(
        cli_args,
        env=env_vars,
        stdout=stdout_f,
        stderr=stderr_f,
    )

    logger.info(f"Started LiberoEnvServer with PID {server_process.pid}")
    return server_process


def get_libero_env(task, resolution, seed):
    """
    Initialize and return the LIBERO environment, along with the task description.

    Args:
        task: LIBERO task object
        resolution: Camera resolution for rendering
        seed: Random seed for environment

    Returns:
        Tuple of (env, task_description)
    """
    # Lazy imports so that importing this module does not force-load robosuite
    # (which can fail in environments that don't have LIBERO installed).
    from libero.libero import get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
        "camera_segmentations": "instance",  # Enable segmentation
    }

    env = SegmentationRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed affects object positions even with fixed initial state
    return env, task_description


def get_max_steps_for_suite(task_suite_name):
    """
    Get the maximum number of steps for a given task suite.

    Args:
        task_suite_name: Name of the task suite

    Returns:
        Maximum number of steps

    Raises:
        ValueError: If task suite name is unknown
    """
    from policy_abstraction.envs.libero.config import TASK_MAX_STEPS

    # Check for exact match first
    if task_suite_name in TASK_MAX_STEPS:
        return TASK_MAX_STEPS[task_suite_name]

    # Check for prefix match (handles _temp, _test, etc. suffixes)
    for suite_name, max_steps in TASK_MAX_STEPS.items():
        if task_suite_name.startswith(suite_name):
            return max_steps

    raise ValueError(f"Unknown task suite: {task_suite_name}")
