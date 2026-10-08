"""Cross-backend smoke test for the WebSocket env server/client interface.

Verifies that LIBERO, RoboCasa, CALVIN and the real UR5e adapter all expose the
same wire protocol to :class:`policy_abstraction.envs.WebsocketEnvClient`. For
each backend whose venv is set up (``third_party/<name>/.venv``, or the
physical-ai-sdk checkout for ``ur5``):

1. Boot the server in a subprocess (the appropriate sim venv).
2. Connect via :class:`WebsocketEnvClient`.
3. Call ``setup_suite()`` and assert the response carries an ``env_spec`` that
   round-trips through :meth:`EnvSpec.from_dict`.
4. Shut the server down cleanly.

Backends whose venv is missing are skipped (not failed) so the script is
useful in partial setups. Exit code is 0 when every attempted backend passes,
1 otherwise.

Usage
-----
    # All available backends (the default; ur5 runs with --simulate-hardware,
    # so it needs no robot)
    python scripts/debug/smoke_test_env_servers.py

    # A subset
    python scripts/debug/smoke_test_env_servers.py --backends libero calvin

    # Custom ports / longer boot timeout
    python scripts/debug/smoke_test_env_servers.py --boot-timeout 60
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time
from typing import Optional

import tyro
from loguru import logger

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.envs.spec import EnvSpec
from policy_abstraction.envs.websocket_client import WebsocketEnvClient

# ---------------------------------------------------------------------------
# Per-backend descriptors
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class BackendSpec:
    name: str
    venv: pathlib.Path
    server_script: pathlib.Path
    extra_args: list[str]
    server_label: str
    expected_metadata_env: str
    cwd: pathlib.Path
    extra_env: dict[str, str]


def _backend_specs() -> dict[str, BackendSpec]:
    third_party = PROJECT_ROOT / "third_party"
    servers_dir = PROJECT_ROOT / "scripts" / "servers"

    return {
        "libero": BackendSpec(
            name="libero",
            venv=third_party / "LIBERO-PRO" / ".venv",
            server_script=servers_dir / "serve_libero_env.py",
            extra_args=[
                "--task-suite-name",
                "libero_spatial",
                "--evaluation-config-path",
                str(
                    PROJECT_ROOT
                    / "src/policy_abstraction/envs/libero/configs/evaluation_config.yaml"
                ),
                "--seed",
                "0",
            ],
            server_label="LiberoEnvServer",
            expected_metadata_env="libero",
            cwd=PROJECT_ROOT,
            extra_env={
                # LIBERO_CONFIG_PATH is set inside server.py.
                "LIBERO_PRO_PATH": str(third_party / "LIBERO-PRO"),
                "PYTHONPATH": str(third_party / "LIBERO-PRO"),
                "DISPLAY": ":0",
                "MUJOCO_GL": "glx",
                "MPLBACKEND": "Agg",
            },
        ),
        "robocasa": BackendSpec(
            name="robocasa",
            venv=third_party / "robocasa" / ".venv",
            server_script=servers_dir / "serve_robocasa_env.py",
            extra_args=["--task", "PnPCounterToCab", "--num-init-states", "1"],
            server_label="RobocasaEnvServer",
            expected_metadata_env="robocasa",
            cwd=third_party / "robocasa",
            extra_env={"MUJOCO_GL": "egl", "MPLBACKEND": "Agg"},
        ),
        "calvin": BackendSpec(
            name="calvin",
            venv=third_party / "calvin" / ".venv",
            server_script=servers_dir / "serve_calvin_env.py",
            extra_args=[
                "--scenes",
                "calvin_scene_D",
                "--num-init-states",
                "1",
                "--no-egl",
            ],
            server_label="CalvinEnvServer",
            expected_metadata_env="calvin",
            cwd=third_party / "calvin",
            extra_env={"MPLBACKEND": "Agg"},
        ),
        "ur5": BackendSpec(
            name="ur5",
            # The only backend whose venv lives outside third_party: the UR5e
            # driver ships with the physical-ai-sdk checkout.
            venv=pathlib.Path(os.path.expanduser("~/dev/physical-ai-sdk")) / ".venv",
            server_script=servers_dir / "serve_ur5_env.py",
            # --simulate-hardware keeps this a pure protocol check: no RTDE
            # session, no camera nodes, safe to run with the arm powered off.
            extra_args=[
                "--simulate-hardware",
                "--num-init-states",
                "1",
                "--tasks",
                "Put all cables in the bag.",
            ],
            server_label="Ur5EnvServer",
            expected_metadata_env="ur5",
            cwd=PROJECT_ROOT,
            extra_env={
                # The server imports policy_abstraction from this repo while
                # running under the SDK's interpreter.
                "PYTHONPATH": str(PROJECT_ROOT / "src"),
                "MPLBACKEND": "Agg",
            },
        ),
    }


# ---------------------------------------------------------------------------
# Subprocess + port helpers
# ---------------------------------------------------------------------------


def _venv_python(spec: BackendSpec) -> pathlib.Path:
    return spec.venv / "bin" / "python"


def _wait_for_port(host: str, port: int, deadline: float) -> bool:
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def _start_server(spec: BackendSpec, host: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env.update(spec.extra_env)

    cli = [
        str(_venv_python(spec)),
        str(spec.server_script),
        "--host",
        host,
        "--port",
        str(port),
        *spec.extra_args,
    ]
    logger.info(f"[{spec.name}] launching: {' '.join(cli)}")
    return subprocess.Popen(
        cli,
        env=env,
        cwd=str(spec.cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _kill_proc(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


# ---------------------------------------------------------------------------
# Per-backend probe
# ---------------------------------------------------------------------------


def _probe_one(spec: BackendSpec, host: str, port: int, boot_timeout: float) -> dict:
    """Boot one backend, run the protocol checks, return a summary dict."""
    if not _venv_python(spec).exists():
        return {
            "backend": spec.name,
            "status": "skipped",
            "reason": f"missing venv at {spec.venv}",
        }

    proc = _start_server(spec, host, port)
    try:
        deadline = time.time() + boot_timeout
        if not _wait_for_port(host, port, deadline):
            stdout = ""
            if proc.stdout is not None:
                try:
                    stdout = proc.stdout.read()  # type: ignore[union-attr]
                except Exception:
                    pass
            return {
                "backend": spec.name,
                "status": "failed",
                "reason": (
                    f"server did not bind within {boot_timeout}s; "
                    f"stdout/stderr tail:\n{stdout[-500:]}"
                ),
            }

        client = WebsocketEnvClient(host=host, port=port, retry_interval=1.0)
        try:
            metadata = client._server_metadata
            assert metadata.get("server") == spec.server_label, (
                f"unexpected handshake: {metadata}"
            )

            suite = client.setup_suite()
            assert isinstance(suite, dict), f"setup_suite returned {type(suite)}"
            for required in (
                "num_tasks",
                "task_languages",
                "task_suite_name",
                "max_steps",
                "env_spec",
            ):
                assert required in suite, f"missing {required!r} in suite_info"

            spec_dict = suite["env_spec"]
            env_spec = EnvSpec.from_dict(spec_dict)
            assert env_spec.metadata.get("env") == spec.expected_metadata_env, (
                f"env metadata mismatch: {env_spec.metadata}"
            )
            assert client.env_spec is not None, "client did not cache env_spec"
            assert client.env_spec.primary_camera.name == env_spec.primary_camera.name

            return {
                "backend": spec.name,
                "status": "ok",
                "num_tasks": int(suite["num_tasks"]),
                "primary_camera": env_spec.primary_camera.name,
                "action_dim": env_spec.robot.action_layout.action_dim,
            }
        finally:
            client.disconnect()
    except Exception as exc:
        return {
            "backend": spec.name,
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        _kill_proc(proc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Args:
    backends: Optional[list[str]] = None
    """Backends to test (default: libero, robocasa, calvin, ur5)."""

    host: str = "127.0.0.1"
    """Bind interface for the test servers."""

    libero_port: int = 18765
    robocasa_port: int = 18766
    calvin_port: int = 18767
    ur5_port: int = 18770

    boot_timeout: float = 90.0
    """Seconds to wait for each server to start listening."""


def main(args: Args) -> int:
    requested = args.backends or ["libero", "robocasa", "calvin", "ur5"]
    specs = _backend_specs()
    unknown = set(requested) - set(specs)
    if unknown:
        raise SystemExit(f"unknown backends: {sorted(unknown)}")

    port_map = {
        "libero": args.libero_port,
        "robocasa": args.robocasa_port,
        "calvin": args.calvin_port,
        "ur5": args.ur5_port,
    }

    results: list[dict] = []
    for name in requested:
        result = _probe_one(specs[name], args.host, port_map[name], args.boot_timeout)
        results.append(result)
        logger.info(f"[{name}] {result}")

    print()
    print("===== Backend smoke summary =====")
    for r in results:
        print(f"  {r['backend']:<10} {r['status']}")
        for k, v in r.items():
            if k in ("backend", "status"):
                continue
            print(f"      {k}: {v}")

    failed = [r for r in results if r["status"] == "failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
