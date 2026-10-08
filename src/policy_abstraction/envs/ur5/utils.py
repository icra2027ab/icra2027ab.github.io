"""Launcher helper for the UR5e env server.

Mirrors :func:`policy_abstraction.envs.libero.utils.start_libero_env_server`,
:func:`policy_abstraction.envs.robocasa.utils.start_robocasa_env_server` and
:func:`policy_abstraction.envs.calvin.utils.start_calvin_env_server`, but execs
the ``physical-ai-sdk`` venv's Python so ``lerobot``, ``rtde_control`` and
``lerobot_robot_ur5e`` resolve.

Unlike the simulator adapters, the server is often better run **by hand** in
its own terminal (so the operator sees the arm's logs and can use
``--success-mode stdin``). Pass ``--env.no-launch-server`` to ``evaluate.py``
to attach to an already-running server instead of spawning one.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import Iterable

from policy_abstraction.envs.ur5.config import DEFAULT_ROBOT_IP
from policy_abstraction.utils.ports import assert_port_free
from policy_abstraction.utils.servers import spawn_server

DEFAULT_SDK_PATH = "~/dev/physical-ai-sdk"


def resolve_sdk_python(sdk_path: str | os.PathLike = DEFAULT_SDK_PATH) -> pathlib.Path:
    """Locate the physical-ai-sdk venv interpreter, raising a useful error."""
    root = pathlib.Path(os.path.expanduser(str(sdk_path)))
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        raise FileNotFoundError(
            f"physical-ai-sdk venv python not found at {venv_python}. "
            f"Set up with: cd {root} && uv sync, then "
            f"uv pip install --python {root}/.venv websockets msgpack-numpy loguru"
        )
    return venv_python


def start_ur5_env_server(
    env_host: str | None,
    env_port: int | None,
    *,
    tasks: Iterable[str] | None = None,
    suite_name: str = "ur5",
    robot_ip: str = DEFAULT_ROBOT_IP,
    cameras: Iterable[str] | None = None,
    wrist_camera: str | None = None,
    primary_camera: str | None = None,
    control_hz: float | None = None,
    max_steps: int | None = None,
    num_init_states: int = 10,
    max_joint_delta: float | None = None,
    flip_gripper: bool = True,
    home_on_reset: bool = True,
    home_on_disconnect: bool = True,
    success_mode: str = "never",
    dry_run: bool = False,
    simulate_hardware: bool = False,
    tcp_offset: Iterable[float] | None = None,
    calibration_path: str | None = None,
    calibration_base_frame: str | None = None,
    sdk_path: str | os.PathLike = DEFAULT_SDK_PATH,
    log_dir: str | os.PathLike | None = None,
) -> "subprocess.Popen":
    """Start the UR5e WebSocket env server in a subprocess.

    Returns the subprocess so the caller can manage its lifecycle.

    Note: ``--stage-prompt`` is intentionally not exposed here. A spawned
    server inherits a non-TTY stdin, so an operator prompt would be skipped
    anyway; run the server by hand when you want one.
    """
    from loguru import logger

    from policy_abstraction import PROJECT_ROOT

    venv_python = resolve_sdk_python(sdk_path)

    task_list = list(tasks or [])
    if not task_list:
        raise ValueError(
            "Pass tasks=[…] — the natural-language instruction(s) the policy is "
            "prompted with, one per task_id."
        )

    env_vars = os.environ.copy()
    env_vars["MPLBACKEND"] = "Agg"
    # The server imports `policy_abstraction.envs.ur5` from this repo while
    # running under the SDK's interpreter.
    src_root = str(pathlib.Path(PROJECT_ROOT) / "src")
    env_vars["PYTHONPATH"] = (
        src_root + os.pathsep + env_vars["PYTHONPATH"]
        if env_vars.get("PYTHONPATH")
        else src_root
    )

    cli_args: list[str] = [
        str(venv_python),
        str(PROJECT_ROOT / "scripts" / "servers" / "serve_ur5_env.py"),
        "--suite-name",
        suite_name,
        "--robot-ip",
        robot_ip,
        "--num-init-states",
        str(num_init_states),
        "--success-mode",
        success_mode,
    ]
    if env_host is not None:
        cli_args.extend(["--host", env_host])
    if env_port is not None:
        cli_args.extend(["--port", str(env_port)])
    if wrist_camera is not None:
        cli_args.extend(["--wrist-camera", wrist_camera])
    if primary_camera is not None:
        cli_args.extend(["--primary-camera", primary_camera])
    if control_hz is not None:
        cli_args.extend(["--control-hz", str(control_hz)])
    if max_steps is not None:
        cli_args.extend(["--max-steps", str(max_steps)])
    if max_joint_delta is not None:
        cli_args.extend(["--max-joint-delta", str(max_joint_delta)])
    if calibration_path is not None:
        cli_args.extend(["--calibration-path", calibration_path])
    if calibration_base_frame is not None:
        cli_args.extend(["--calibration-base-frame", calibration_base_frame])
    if not flip_gripper:
        cli_args.append("--no-flip-gripper")
    if not home_on_reset:
        cli_args.append("--no-home-on-reset")
    if not home_on_disconnect:
        cli_args.append("--no-home-on-disconnect")
    if dry_run:
        cli_args.append("--dry-run")
    if simulate_hardware:
        cli_args.append("--simulate-hardware")
    if tcp_offset is not None:
        cli_args.append("--tcp-offset")
        cli_args.extend(str(v) for v in tcp_offset)
    if cameras:
        cli_args.append("--cameras")
        cli_args.extend(cameras)
    # Instructions last: they contain spaces, and tyro's list parsing consumes
    # everything up to the next flag.
    cli_args.append("--tasks")
    cli_args.extend(task_list)

    # Refuse to spawn onto an occupied port. Without this the child dies on
    # "address already in use" -- into a log file -- and the client connects
    # to whoever already held the port, running the whole evaluation against
    # that process's cameras, calibration and checkout instead of these args.
    if env_port is not None:
        assert_port_free(int(env_port), "UR5 env server", "--env.no-launch-server")

    # Never use subprocess.PIPE without a reader here: the OS pipe buffer fills
    # after a few minutes of 50 Hz logging and blocks the server mid-write,
    # which strands the arm in servo mode.
    if log_dir is not None:
        log_path = pathlib.Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        stdout_f = open(log_path / "ur5_env_server.stdout.log", "w", buffering=1)
        stderr_f = open(log_path / "ur5_env_server.stderr.log", "w", buffering=1)
        logger.info(f"UR5 env server logs → {log_path}/ur5_env_server.*.log")
    else:
        stdout_f = None
        stderr_f = None

    server_process = spawn_server(
        cli_args,
        env=env_vars,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_f,
        stderr=stderr_f,
    )

    logger.info(f"Started Ur5EnvServer with PID {server_process.pid}")
    return server_process


__all__ = [
    "DEFAULT_ROBOT_IP",
    "DEFAULT_SDK_PATH",
    "resolve_sdk_python",
    "start_ur5_env_server",
]
