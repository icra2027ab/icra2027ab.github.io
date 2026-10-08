"""Environment utilities for LIBERO evaluations."""

import os
import pathlib
import subprocess
from typing import Tuple

from policy_abstraction.utils.servers import spawn_server

# Default location for the JAX persistent compilation cache shared by every
# policy server we spawn.  Without this, pi05 recompiles from scratch on every
# server restart (~25-60s on cold cache), which exceeds the websockets default
# ``ping_timeout`` (20s) and surfaces as a ``1011`` keepalive failure on the
# client mid-first-inference.  Anything that lengthens compile (different state
# shape, new guidance closure pytree, GPU contention) tips it over the cliff.
DEFAULT_JAX_COMPILATION_CACHE_DIR = str(
    pathlib.Path.home() / ".cache" / "openpi_jax_compilation_cache"
)


def start_policy_server(
    port,
    gpus="0",
    dynamics_model_path: str | None = None,
    skip_weights: bool = False,
    log_dir: str | os.PathLike | None = None,
    policy_env: str = "LIBERO",
    policy_config: str = "pi05_libero_grounded",
    policy_dir: str = "gs://openpi-assets/checkpoints/pi05_libero",
) -> "subprocess.Popen":
    """Start the policy server in another python process.

    Args:
        port: Port to serve the policy on.
        gpus: CUDA_VISIBLE_DEVICES string.
        dynamics_model_path: Optional path to a Python file defining a
            JAX-differentiable ``integrate_actions`` function used for
            guidance.  When ``None`` the server uses its built-in Euler
            integration.
        skip_weights: Skip loading model weights (for debug_guidance mode).
        log_dir: Directory to write ``policy_server.{stdout,stderr}.log``
            into. When ``None``, the child inherits the parent's stdio.
            NEVER pass ``subprocess.PIPE`` here without a reader: the OS
            pipe buffer (~64 KB on Linux) fills after a minute of inference
            logs and blocks the server mid-write, which surfaces as a
            websocket ``1011`` keepalive timeout on the client.
        policy_env: openpi ``--env`` flag (e.g. ``"LIBERO"``).
        policy_config: openpi ``--policy.config`` flag.
        policy_dir: openpi ``--policy.dir`` flag.

    Returns the subprocess so the caller can manage its lifecycle.
    """
    from loguru import logger

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpus
    env_vars["MPLBACKEND"] = "Agg"
    env_vars.setdefault("JAX_COMPILATION_CACHE_DIR", DEFAULT_JAX_COMPILATION_CACHE_DIR)
    pathlib.Path(env_vars["JAX_COMPILATION_CACHE_DIR"]).mkdir(
        parents=True, exist_ok=True
    )

    cmd = [
        ".venv/bin/python",
        "scripts/serve_policy.py",
        "--port",
        str(port),
        "--env",
        policy_env,
    ]
    if dynamics_model_path is not None:
        cmd.extend(["--dynamics-model-path", os.path.abspath(dynamics_model_path)])
    if skip_weights:
        cmd.append("--skip-weights")
    # Subcommand args must come after top-level args (tyro ordering requirement)
    cmd.extend(
        [
            "policy:checkpoint",
            "--policy.config",
            policy_config,
            "--policy.dir",
            policy_dir,
        ]
    )

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "policy_server.stdout.log"
        stderr_path = log_dir / "policy_server.stderr.log"
        stdout_f = open(stdout_path, "w", buffering=1)
        stderr_f = open(stderr_path, "w", buffering=1)
        logger.info(f"Policy server logs → {stdout_path}, {stderr_path}")
    else:
        stdout_f = None
        stderr_f = None

    server_process = spawn_server(
        cmd,
        env=env_vars,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd="third_party/openpi",
    )

    logger.info(f"Started PolicyServer with PID {server_process.pid}")
    return server_process


class PatientWebsocketClient:
    """``WebsocketClientPolicy`` with a much longer keepalive timeout.

    The default ``ping_timeout`` (20 s) trips during the first inference call
    when JAX has to compile the pi05 graph against the current obs / guidance
    pytree.  That close arrives on the client as ``1011 internal error`` even
    though the server is still mid-compile.  Bumping ``ping_timeout`` to two
    minutes covers the cold compile; the JAX compilation cache (set up in
    ``start_policy_server``) makes subsequent runs fast.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int | None = None,
        api_key: str | None = None,
        ping_timeout: float = 120.0,
        open_timeout: float = 120.0,
    ) -> None:
        # Imported lazily so this module stays importable on machines that
        # don't have the openpi client (e.g. perception-only ablations).
        from openpi_client import msgpack_numpy

        self._packer = msgpack_numpy.Packer()
        self._unpack = msgpack_numpy.unpackb
        self._api_key = api_key
        self._ping_timeout = ping_timeout
        self._open_timeout = open_timeout
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self) -> Tuple[object, dict]:
        import logging
        import time

        import websockets.sync.client

        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = (
                    {"Authorization": f"Api-Key {self._api_key}"}
                    if self._api_key
                    else None
                )
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    open_timeout=self._open_timeout,
                    ping_timeout=self._ping_timeout,
                )
                metadata = self._unpack(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def infer(self, obs: dict) -> dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return self._unpack(response)

    def reset(self) -> None:
        pass
