"""NVIDIA Cosmos-Policy (Cosmos-Predict2) backend.

Mirrors the layout of [`policies.groot`](./groot.py): a `start_cosmos_server`
helper that spawns `scripts/servers/serve_cosmos.py` inside cosmos-policy's
own venv (Python 3.10 + torch 2.7.0+cu128 + flash-attn), plus a
`CosmosPolicyClient` adapter that translates PA's per-step element dict into
cosmos's observation schema and stacks the response back into PA's
``(num_chunks, horizon, action_dim)`` convention.

The wire protocol is ZMQ REQ/REP + msgpack with the same custom ndarray codec
as the GR00T backend, so cosmos has the same retry semantics and uses the same
``options`` channel to carry ``__gp_*__`` guidance keys to the grounded
denoising loop (see ``cosmos_grounded_model.py``).
"""

from __future__ import annotations

import io
import os
import pathlib
import subprocess
import time
from typing import Any

import msgpack
import numpy as np
import zmq

from policy_abstraction.policies.base import BasePolicyClient
from policy_abstraction.policies.groot import (
    GripperRemap,
    resolve_gripper_remap,
)
from policy_abstraction.utils.servers import spawn_server
from policy_steering.params import extract_guidance_params

# Cosmos action_dim is 7 for both LIBERO and RoboCasa (pos[3] + axisangle[3]
# + gripper[1]). Chunk size differs (LIBERO=16, RoboCasa=32); the actual chunk
# size is set server-side via --policy.chunk-size and validated client-side
# against the configured suite expectation.
_COSMOS_ACTION_DIM = 7
_COSMOS_CHUNK_BY_SUITE = {"libero": 16, "robocasa": 32}


# ---------------------------------------------------------------------------
# msgpack ndarray codec — must match scripts/servers/serve_cosmos.py
# ---------------------------------------------------------------------------


def _encode_custom(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode_custom(obj: Any) -> Any:
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


# ---------------------------------------------------------------------------
# ZMQ REQ/REP client — same shape as `_GrootZmqClient`.
# ---------------------------------------------------------------------------


class _CosmosZmqClient:
    """Thin REQ/REP client for the cosmos server's `{endpoint, data}` protocol."""

    def __init__(self, host: str, port: int, timeout_ms: int = 120_000) -> None:
        self._host = host
        self._port = port
        self._timeout_ms = timeout_ms
        self._current_timeout_ms = timeout_ms
        self._context = zmq.Context.instance()
        self._socket: zmq.Socket | None = None
        self._init_socket(timeout_ms)

    def _init_socket(self, timeout_ms: int | None = None) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
        if timeout_ms is None:
            timeout_ms = self._current_timeout_ms
        self._current_timeout_ms = timeout_ms
        sock = self._context.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        sock.connect(f"tcp://{self._host}:{self._port}")
        self._socket = sock

    def call(self, endpoint: str, data: dict | None = None) -> Any:
        request: dict = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            assert self._socket is not None
            self._socket.send(msgpack.packb(request, default=_encode_custom))
            message = self._socket.recv()
        except zmq.error.Again:
            # REQ socket needs reinit after a timed-out recv (the FSM is stuck
            # waiting for a reply that will never arrive).
            self._init_socket()
            raise
        response = msgpack.unpackb(message, object_hook=_decode_custom)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Cosmos server error: {response['error']}")
        return response

    def ping(self, timeout_ms: int = 2000) -> bool:
        prior = self._current_timeout_ms
        self._init_socket(timeout_ms)
        try:
            self.call("ping")
            return True
        except (zmq.error.ZMQError, RuntimeError):
            return False
        finally:
            self._init_socket(prior)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None


# ---------------------------------------------------------------------------
# Server subprocess
# ---------------------------------------------------------------------------


def start_cosmos_server(
    port: int,
    model_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    *,
    gpus: str = "0",
    host: str = "127.0.0.1",
    repo_path: str | os.PathLike = "third_party/cosmos-policy",
    venv_python: str = ".venv/bin/python",
    suite: str = "libero",
    config: str | None = None,
    chunk_size: int = 16,
    num_open_loop_steps: int = 16,
    num_denoising_steps_action: int = 5,
    num_third_person_images: int = 1,
    seed: int = 195,
    deterministic: bool = True,
    t5_embeddings_path: str | None = None,
    dataset_stats_path: str | None = None,
    log_dir: str | os.PathLike | None = None,
) -> subprocess.Popen:
    """Spawn `scripts/servers/serve_cosmos.py` in cosmos-policy's venv.

    Cosmos pins torch==2.7.0+cu128 / flash-attn, which would conflict with PA's
    root venv, so we run it isolated. Communication happens over ZMQ on
    ``port`` (loopback by default).

    Args:
        port: ZMQ TCP port the server should bind to.
        model_path: HF repo id (e.g. ``nvidia/Cosmos-Policy-LIBERO-Predict2-2B``)
            or local checkpoint dir. Auto-downloaded by cosmos on first use.
        gpus: ``CUDA_VISIBLE_DEVICES`` for the child.
        host: Bind address. Default loopback so the port is local-only.
        repo_path: Path to the cosmos-policy checkout. Relative paths resolve
            against ``PROJECT_ROOT``.
        venv_python: Cosmos venv interpreter, relative to ``repo_path``. Built
            with ``uv sync --extra cu128 --group libero --python 3.10`` from
            inside the cosmos-policy directory. The same libero-group venv
            also serves RoboCasa inference — `cosmos_utils.py` doesn't import
            `robosuite`/`mujoco`/`robocasa` on the inference path.
        suite: One of ``"libero"`` or ``"robocasa"``. Drives observation/latent
            layout server-side; must match the ``suite`` passed to
            ``CosmosPolicyClient``.
        config: Cosmos experiment config name. When ``None``, the server uses
            its LIBERO default. For RoboCasa pass
            ``"cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference"``.
        chunk_size, num_open_loop_steps, num_denoising_steps_action,
        num_third_person_images: Cosmos inference hyperparameters. Defaults
            match LIBERO.md; RoboCasa requires chunk_size=32 and
            num_third_person_images=2.
        seed, deterministic: Mirrors LIBERO.md's deterministic mode.
        t5_embeddings_path, dataset_stats_path: HF paths or local paths to the
            T5 cache pickle and dataset stats JSON. When ``None``, the server
            uses its defaults (which point at the LIBERO checkpoint's bundled
            files on HF). Override these if you fine-tuned and produced new
            stats.
        log_dir: Where to write ``cosmos_server.{stdout,stderr}.log``. If
            ``None`` the child inherits parent's stdio. Mirror of
            ``start_groot_server``: NEVER use ``subprocess.PIPE`` here without
            a reader thread — pipe backpressure stalls the server.
    """
    from loguru import logger

    repo_path = pathlib.Path(repo_path)
    if not repo_path.is_absolute():
        from policy_abstraction import PROJECT_ROOT

        repo_path = pathlib.Path(PROJECT_ROOT) / repo_path
    if not repo_path.exists():
        raise FileNotFoundError(
            f"cosmos-policy repo not found at {repo_path}. Pass --policy.repo-path "
            f"or clone cosmos-policy into third_party/cosmos-policy."
        )
    py = repo_path / venv_python
    if not py.exists():
        raise FileNotFoundError(
            f"cosmos-policy venv interpreter not found at {py}. Run "
            f"`uv sync --extra cu128 --group libero --python 3.10` inside "
            f"{repo_path} (then `uv pip install msgpack pyzmq tyro` for the "
            f"ZMQ server)."
        )

    # Resolve model_path: HF repo ids must NOT be absolutized (they look like
    # paths but aren't). Local paths get absolutized so they survive the
    # `cwd=repo_path` switch below.
    looks_like_hf = (
        "/" in str(model_path)
        and not str(model_path).startswith("/")
        and not str(model_path).startswith("./")
        and not pathlib.Path(model_path).exists()
    )
    resolved_model = (
        str(model_path) if looks_like_hf else os.path.abspath(str(model_path))
    )

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpus
    # Make PA + policy_steering importable inside the cosmos venv so the
    # grounded model/runtime under policy_abstraction.policies.cosmos_grounded_*
    # can be imported by serve_cosmos.py.
    from policy_abstraction import PROJECT_ROOT

    extra_pp = os.pathsep.join(
        [
            str(pathlib.Path(PROJECT_ROOT) / "src"),
            str(pathlib.Path(PROJECT_ROOT) / "packages" / "policy_steering" / "src"),
        ]
    )
    if env_vars.get("PYTHONPATH"):
        env_vars["PYTHONPATH"] = extra_pp + os.pathsep + env_vars["PYTHONPATH"]
    else:
        env_vars["PYTHONPATH"] = extra_pp

    serve_script = (
        pathlib.Path(PROJECT_ROOT) / "scripts" / "servers" / "serve_cosmos.py"
    )
    cmd = [
        str(py),
        str(serve_script),
        "--model-path",
        resolved_model,
        "--suite",
        suite,
        "--chunk-size",
        str(chunk_size),
        "--num-open-loop-steps",
        str(num_open_loop_steps),
        "--num-denoising-steps-action",
        str(num_denoising_steps_action),
        "--num-third-person-images",
        str(num_third_person_images),
        "--seed",
        str(seed),
        "--host",
        host,
        "--port",
        str(port),
    ]
    if config is not None:
        cmd.extend(["--config", config])
    if deterministic:
        cmd.append("--deterministic")
    else:
        cmd.append("--no-deterministic")
    if t5_embeddings_path is not None:
        cmd.extend(["--t5-text-embeddings-path", t5_embeddings_path])
    if dataset_stats_path is not None:
        cmd.extend(["--dataset-stats-path", dataset_stats_path])

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "cosmos_server.stdout.log"
        stderr_path = log_dir / "cosmos_server.stderr.log"
        stdout_f: Any = open(stdout_path, "w", buffering=1)
        stderr_f: Any = open(stderr_path, "w", buffering=1)
        logger.info(f"Cosmos server logs → {stdout_path}, {stderr_path}")
    else:
        stdout_f = None
        stderr_f = None

    server_process = spawn_server(
        cmd,
        env=env_vars,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd=str(repo_path),
    )

    logger.info(
        f"Started Cosmos PolicyServer on {host}:{port} (PID {server_process.pid})"
    )
    return server_process


# ---------------------------------------------------------------------------
# State conversions (PA libero element ↔ cosmos LIBERO observation)
# ---------------------------------------------------------------------------


def _axisangle_to_quat_xyzw(aa: np.ndarray) -> np.ndarray:
    """Inverse of robosuite's ``quat2axisangle`` (LIBERO uses xyzw convention).

    cosmos LIBERO proprio packs raw ``robot0_eef_quat`` (xyzw). PA's policy
    state layout squashes it to axis-angle via ``StateField("quat_to_axisangle",
    "robot0_eef_quat")``, so we reconstruct an equivalent quaternion here.
    Quaternion sign is arbitrary; the trained policy is invariant to the
    double-cover.
    """
    aa = np.asarray(aa, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = aa / angle
    half = angle * 0.5
    s = np.sin(half)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, np.cos(half)], dtype=np.float64
    )


def _pa_state_to_cosmos_proprio(state: np.ndarray) -> np.ndarray:
    """Convert PA's 8D libero policy state → cosmos's 9D libero proprio.

    PA layout (from envs/libero/config.py policy_input_keys.state_layout):
      ``[eef_pos(3), eef_axis_angle(3), robot0_gripper_qpos(2)]``
    Cosmos layout (from libero_utils.py:prepare_observation):
      ``[robot0_gripper_qpos(2), robot0_eef_pos(3), robot0_eef_quat_xyzw(4)]``
    """
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (8,):
        raise ValueError(
            f"Expected 8D PA libero state (pos+axisangle+gripper_qpos), "
            f"got shape {state.shape}."
        )
    eef_pos = state[0:3]
    eef_aa = state[3:6]
    gripper_qpos = state[6:8]
    eef_quat = _axisangle_to_quat_xyzw(eef_aa)
    return np.concatenate([gripper_qpos, eef_pos, eef_quat], axis=0)


def _undo_pa_hflip(img: np.ndarray) -> np.ndarray:
    """PA applies ``flip_vh`` (vflip+hflip = 180° rotation) to libero cameras
    in ``CameraSpec.apply_image_transform``. Cosmos was trained on raw mujoco
    images with **only** ``np.flipud`` (vflip) applied. Stripping the extra
    hflip here lets us pass ``flip_images=False`` to the cosmos server and
    keep the training distribution.
    """
    return np.ascontiguousarray(img[:, ::-1])


# ---------------------------------------------------------------------------
# BasePolicyClient adapter
# ---------------------------------------------------------------------------


class CosmosPolicyClient(BasePolicyClient):
    """`BasePolicyClient` adapter for a cosmos-policy server.

    Translates PA's per-step element dict (built by
    ``EpisodeRunner._build_policy_input``) into cosmos's LIBERO observation
    schema, sends it over ZMQ, and reshapes the action response back to the
    ``(num_chunks=1, horizon, action_dim)`` shape PA's EpisodeRunner expects.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5556,
        timeout_ms: int = 600_000,
        ready_timeout_s: float = 1200.0,
        server_process: subprocess.Popen | None = None,
        gripper_remap: str | GripperRemap | None = "passthrough",
        suite: str = "libero",
    ) -> None:
        """
        Args:
            server_process: Optional handle to the cosmos server subprocess.
                When provided, the wait-for-ready loop polls it each cycle and
                bails immediately if the child has exited.
            gripper_remap: Mirrors the GR00T pattern: cosmos LIBERO emits the
                same robosuite-style ``{-1, +1}`` gripper action that the
                LIBERO env consumes, so the default is ``"passthrough"``. Use
                a named entry from ``GRIPPER_REMAPS`` or a callable to adapt
                to a different env convention.
            timeout_ms: ZMQ per-call recv timeout. The first inference for a
                prompt that isn't in the bundled ``libero_t5_embeddings.pkl``
                cache triggers a lazy load of ``google-t5/t5-11b`` (~44 GB)
                inside ``get_t5_embedding_from_cache``, which can take several
                minutes on a cold filesystem cache. After that one cache-miss
                pays the cost, subsequent calls are ~1-2 s.
            ready_timeout_s: First-call timeout is long because the initial
                HF checkpoint download (~5 GB) + model load can take 10+ min
                on a cold machine.
            suite: One of ``"libero"`` or ``"robocasa"``. Selects the
                observation schema used by ``_build_cosmos_obs`` — RoboCasa
                requires a third (``secondary``) image; LIBERO does not. Must
                match the ``--suite`` arg passed to the server.
        """
        if suite not in _COSMOS_CHUNK_BY_SUITE:
            raise ValueError(
                f"Unsupported cosmos suite {suite!r}; "
                f"expected one of {sorted(_COSMOS_CHUNK_BY_SUITE)}."
            )
        self._suite = suite
        self._expected_chunk = _COSMOS_CHUNK_BY_SUITE[suite]
        self._client = _CosmosZmqClient(host, port, timeout_ms=timeout_ms)
        self._server_process = server_process
        self._gripper_remap = resolve_gripper_remap(gripper_remap)
        self._wait_for_server(ready_timeout_s)

    def _wait_for_server(self, ready_timeout_s: float) -> None:
        from loguru import logger

        deadline = time.monotonic() + ready_timeout_s
        last_err: Exception | None = None
        attempt = 0
        start = time.monotonic()
        while time.monotonic() < deadline:
            if self._server_process is not None:
                rc = self._server_process.poll()
                if rc is not None:
                    raise RuntimeError(
                        f"Cosmos server subprocess (PID {self._server_process.pid}) "
                        f"exited with code {rc} before becoming ready. Check "
                        f"the server log (e.g. cosmos_server.stderr.log)."
                    )
            try:
                if self._client.ping(timeout_ms=2000):
                    logger.info(
                        f"Cosmos server is ready (took {time.monotonic() - start:.1f}s)."
                    )
                    return
            except Exception as e:  # noqa: BLE001
                last_err = e
            attempt += 1
            if attempt % 10 == 1:
                logger.info(
                    f"Waiting for Cosmos server at "
                    f"{self._client._host}:{self._client._port} "
                    f"({time.monotonic() - start:.0f}s elapsed)..."
                )
            time.sleep(3.0)
        raise RuntimeError(
            f"Cosmos server at {self._client._host}:{self._client._port} did not "
            f"become ready within {ready_timeout_s}s (last error: {last_err})."
        )

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------

    def _build_cosmos_obs(self, element: dict) -> dict:
        """Translate PA element → cosmos observation dict (suite-aware).

        LIBERO observation: ``{primary_image, wrist_image, proprio, task_description}``.
        RoboCasa observation: adds ``secondary_image`` (the right agentview).
        Proprio layout is identical across suites (9D = ``[gripper_qpos(2),
        eef_pos(3), eef_quat_xyzw(4)]``), so ``_pa_state_to_cosmos_proprio``
        is reused unchanged.
        """
        try:
            img = element["observation/image"]
            wrist_img = element["observation/wrist_image"]
            state = element["observation/state"]
            prompt = element["prompt"]
        except KeyError as e:
            raise KeyError(
                f"Cosmos client expected key {e} in policy element. Check your "
                f"env spec's policy_input_keys.images."
            ) from None

        if img.dtype != np.uint8:
            raise ValueError(
                f"Cosmos expects uint8 images; got primary={img.dtype}. Check "
                f"camera preprocessing."
            )
        if wrist_img.dtype != np.uint8:
            raise ValueError(
                f"Cosmos expects uint8 images; got wrist={wrist_img.dtype}."
            )

        obs: dict = {
            "primary_image": _undo_pa_hflip(img),
            "wrist_image": _undo_pa_hflip(wrist_img),
            "proprio": _pa_state_to_cosmos_proprio(state),
            "task_description": str(prompt),
        }
        if self._suite == "robocasa":
            try:
                secondary_img = element["observation/secondary_image"]
            except KeyError:
                raise KeyError(
                    "Cosmos RoboCasa requires 'observation/secondary_image' in "
                    "the policy element. Confirm the RoboCasa env spec routes "
                    "robot0_agentview_right (see envs/robocasa/config.py)."
                ) from None
            if secondary_img.dtype != np.uint8:
                raise ValueError(
                    f"Cosmos expects uint8 images; got secondary={secondary_img.dtype}."
                )
            obs["secondary_image"] = _undo_pa_hflip(secondary_img)
        return obs

    def _shape_action_chunk(self, actions: np.ndarray) -> np.ndarray:
        """Reshape cosmos's (T, action_dim) → PA's (num_chunks=1, T, action_dim)
        and apply the env-specific gripper remap to the last axis index.
        """
        if actions.ndim != 2:
            raise ValueError(
                f"Expected cosmos actions shape (T, action_dim); got {actions.shape}."
            )
        actions = actions.astype(np.float32, copy=False)
        out = actions[None, :, :]  # (1, T, action_dim)
        out = out.copy()  # detach from msgpack-owned buffer before mutating
        out[..., -1] = self._gripper_remap(out[..., -1])
        return out

    # ------------------------------------------------------------------
    # BasePolicyClient interface
    # ------------------------------------------------------------------

    def infer(self, obs_dict: dict) -> dict:
        # Strip ``__gp_*__`` guidance keys before building the cosmos obs;
        # they're forwarded separately under ``options`` and consumed by the
        # grounded model on the server side. When no guidance keys are
        # present, ``options`` is ``None`` and the server takes its
        # unguided fast path.
        guidance_params, cleaned = extract_guidance_params(obs_dict)
        cosmos_obs = self._build_cosmos_obs(cleaned)
        t0 = time.monotonic()
        response = self._client.call(
            "get_action",
            {"observation": cosmos_obs, "options": guidance_params},
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if not isinstance(response, dict) or "actions" not in response:
            raise RuntimeError(
                f"Unexpected cosmos response: {type(response).__name__} "
                f"(keys={list(response) if isinstance(response, dict) else 'N/A'})."
            )
        actions = self._shape_action_chunk(np.asarray(response["actions"]))

        if (
            actions.shape[1] != self._expected_chunk
            or actions.shape[2] != _COSMOS_ACTION_DIM
        ):
            from loguru import logger

            logger.debug(
                f"Cosmos returned action shape {actions.shape} "
                f"(expected (1, {self._expected_chunk}, {_COSMOS_ACTION_DIM}) "
                f"for suite={self._suite}); proceeding."
            )

        return {
            "actions": actions,
            "policy_timing": response.get(
                "policy_timing", {"infer_ms": float(elapsed_ms)}
            ),
        }

    def reset(self) -> None:
        try:
            self._client.call("reset", {"options": None})
        except Exception:  # noqa: BLE001 — reset is best-effort
            pass

    def close(self) -> None:
        self._client.close()
