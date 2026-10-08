"""LeRobot (HuggingFace) policy backend — pi0 / pi0.5 / pi0-fast checkpoints.

Mirrors the layout of [`policies.cosmos`](./cosmos.py) and
[`policies.groot`](./groot.py): a ``start_lerobot_server`` helper that spawns
[`scripts/servers/serve_lerobot.py`](../../../scripts/servers/serve_lerobot.py)
inside a venv that has ``lerobot`` installed, plus a ``LerobotPolicyClient``
adapter that translates PA's per-step element dict into the checkpoint's own
feature keys and reshapes the response into PA's
``(num_chunks, horizon, action_dim)`` convention.

The wire protocol is ZMQ REQ/REP + msgpack with the same npy-based ndarray
codec the cosmos backend uses, and the same endpoints (``ping``,
``get_action``, ``reset``) plus an ``info`` endpoint used for startup logging.

Why a separate venv: ``lerobot`` pins ``transformers>=5.3,<5.4`` and its own
torch build, which collide with the root ``policy_abstraction`` venv. The only
``PYTHONPATH`` bridging needed is ``packages/policy_steering/src`` (numpy +
torch only) for the guidance kernel — unlike the GR00T/cosmos servers, nothing
from ``policy_abstraction`` itself is imported server-side.

Observation mapping
-------------------
PA's element carries fixed logical keys (``observation/image``,
``observation/wrist_image``, ``observation/state``, ``prompt``) while a LeRobot
checkpoint names its inputs after the dataset it was trained on
(``observation.images.exterior_image_1_left``, ``observation.state``, ...).
The client sends the logical view names as-is; the *server* resolves them
against the checkpoint's ``input_features`` (auto-derived, or pinned with
``--camera-map``). Keeping the mapping server-side means the client never has
to know which checkpoint is loaded.

No image transform is applied here. PA already hands cameras to the policy in
the same orientation openpi's LIBERO eval uses (``CameraSpec.flip_vh``), which
is the orientation the LIBERO LeRobot datasets were recorded in; real-robot
checkpoints (e.g. the UR5e pi0.5 runs) are fed their cameras unrotated. Cosmos
needs ``_undo_pa_hflip`` because it alone was trained on raw mujoco frames.

Guidance
--------
Supported. PA's ``__gp_*__`` params ride in the request's ``options`` field
(same channel as the GR00T and cosmos backends) and steer the flow-matching
integration server-side, mirroring openpi's ``Pi0Grounded.sample_actions``.
See ``scripts/servers/serve_lerobot.py`` for how the correction is injected
without copying lerobot's denoising loop.

Guidance acts on the checkpoint's *normalized* action space — as it does for
GR00T and cosmos — so ``--guidance.factor`` is in normalized units whenever the
checkpoint's ACTION normalization isn't ``IDENTITY``. The server logs which it
is at startup and reports it via the ``info`` endpoint.

Only pi0 / pi0.5 / smolvla expose the denoising hook guidance needs; the server
refuses a guided request for any other policy type rather than answering it
unguided.
"""

from __future__ import annotations

import io
import os
import pathlib
import subprocess
import time
from typing import Any, Sequence

import msgpack
import numpy as np
import zmq

from policy_abstraction.policies.base import BasePolicyClient
from policy_abstraction.policies.groot import GripperRemap, resolve_gripper_remap
from policy_abstraction.utils.ports import assert_port_free
from policy_abstraction.utils.servers import spawn_server
from policy_steering.params import extract_guidance_params

# PA element keys are ``observation/<view>``; this prefix is stripped to get
# the logical view name the server maps onto a checkpoint feature key.
_OBS_PREFIX = "observation/"

# Canonical view ordering used when the server has to auto-map views onto the
# checkpoint's camera features. Views outside this list are appended sorted,
# so the mapping stays deterministic across runs.
CANONICAL_VIEW_ORDER: tuple[str, ...] = ("image", "wrist_image", "secondary_image")


# ---------------------------------------------------------------------------
# msgpack ndarray codec — must match scripts/servers/serve_lerobot.py
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
# ZMQ REQ/REP client — same shape as `_CosmosZmqClient` / `_GrootZmqClient`.
# ---------------------------------------------------------------------------


class _LerobotZmqClient:
    """Thin REQ/REP client for the server's ``{endpoint, data}`` protocol."""

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
            raise RuntimeError(f"LeRobot server error: {response['error']}")
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


def start_lerobot_server(
    port: int,
    model_path: str,
    *,
    gpus: str = "0",
    host: str = "127.0.0.1",
    repo_path: str | os.PathLike = "third_party/physical-ai-sdk",
    venv_python: str = ".venv/bin/python",
    policy_type: str | None = None,
    device: str = "cuda",
    camera_map: str | None = None,
    state_adapter: str = "identity",
    num_inference_steps: int | None = None,
    compile_model: str = "auto",
    guidance: str = "auto",
    rtc: bool = False,
    rtc_execution_horizon: int = 10,
    rtc_max_guidance_weight: float = 10.0,
    rtc_prefix_attention_schedule: str = "linear",
    robot_type: str = "",
    guidance_action_space: str = "cartesian-delta",
    tcp_offset: Sequence[float] | None = None,
    gripper_geometry: str | None = None,
    log_dir: str | os.PathLike | None = None,
) -> subprocess.Popen:
    """Spawn ``scripts/servers/serve_lerobot.py`` in a lerobot-enabled venv.

    Args:
        port: ZMQ TCP port the server should bind to.
        model_path: Local checkpoint dir (the ``pretrained_model/`` folder that
            holds ``config.json`` + ``model.safetensors``) or a HF repo id.
        gpus: ``CUDA_VISIBLE_DEVICES`` for the child.
        host: Bind address. Default loopback so the port is local-only.
        repo_path: Directory holding the lerobot venv. Relative paths resolve
            against ``PROJECT_ROOT``. Any project whose venv has ``lerobot``
            installed works — symlink it into ``third_party/`` (e.g.
            ``ln -s ~/dev/physical-ai-sdk third_party/physical-ai-sdk``) or
            pass ``--policy.repo-path`` explicitly.
        venv_python: Interpreter path relative to ``repo_path``.
        policy_type: ``pi05``/``pi0``/``pi0_fast``/... Defaults to the ``type``
            recorded in the checkpoint's ``config.json``.
        device: Torch device for the server-side policy.
        camera_map: ``"<view>=<feature_key>,..."`` pinning PA view names onto
            checkpoint camera features. ``None`` auto-derives (see the server's
            ``resolve_camera_map``).
        state_adapter: ``"identity"`` (strict dim check, the default) or
            ``"pad_or_truncate"``.
        num_inference_steps: Override the checkpoint's denoising step count.
        compile_model: ``"auto"`` (honour the checkpoint's ``compile_model``),
            ``"on"`` or ``"off"``. Training configs often ship
            ``compile_model=true, compile_mode=max-autotune``, which spends
            minutes autotuning inside the *first* inference. Worth it for a
            full eval; pass ``"off"`` for short debug runs.
        guidance: ``"auto"`` steers the denoising loop whenever a request
            carries ``__gp_*__`` params; ``"off"`` refuses guided requests.
        rtc: Enable Real-Time Chunking, which blends each new chunk into the
            unexecuted tail of the previous one. ``evaluate.py`` turns it on
            by default (``--policy.no-rtc`` to disable). PA steps the env
            synchronously, so chunk-boundary continuity is the only thing
            RTC buys here (there is no inference latency to hide).
        rtc_execution_horizon, rtc_max_guidance_weight,
        rtc_prefix_attention_schedule: Forwarded to ``lerobot``'s ``RTCConfig``.
            Ignored when ``rtc`` is False.
        guidance_action_space: How the server's guidance loss reads a chunk.
            ``"cartesian-delta"`` Euler-integrates the first three channels
            as tool velocities (the simulators); ``"ur5e-joints"`` unnormalizes
            the chunk with the checkpoint's action stats and runs every joint
            target through the UR5e forward kinematics (the real arm).
        tcp_offset: ``[x, y, z, rx, ry, rz]`` TCP offset the UR controller
            applies, so ``ur5e-joints`` guidance measures the same point RTDE
            reports. Only sent when given.
        gripper_geometry: ``policy_steering`` robot config of the gripper being
            guided (``robotiq_2f140``). Only sent when given.
        log_dir: Where to write ``lerobot_server.{stdout,stderr}.log``. If
            ``None`` the child inherits the parent's stdio. Mirror of
            ``start_cosmos_server``: NEVER use ``subprocess.PIPE`` here without
            a reader thread — pipe backpressure stalls the server.
    """
    from loguru import logger

    from policy_abstraction import PROJECT_ROOT

    repo_path = pathlib.Path(repo_path)
    if not repo_path.is_absolute():
        repo_path = pathlib.Path(PROJECT_ROOT) / repo_path
    if not repo_path.exists():
        raise FileNotFoundError(
            f"lerobot venv host directory not found at {repo_path}. Pass "
            f"--policy.repo-path <dir containing .venv with lerobot installed>."
        )
    py = repo_path / venv_python
    if not py.exists():
        raise FileNotFoundError(
            f"lerobot venv interpreter not found at {py}. Create it with "
            f"`uv sync` inside {repo_path}, then add the ZMQ server deps: "
            f"`uv pip install pyzmq tyro msgpack`."
        )

    # HF repo ids look like relative paths but must not be absolutized; local
    # paths must be, because the child runs with cwd=repo_path.
    looks_like_hf = (
        "/" in str(model_path)
        and not str(model_path).startswith("/")
        and not str(model_path).startswith("./")
        and not pathlib.Path(model_path).exists()
    )
    resolved_model = (
        str(model_path) if looks_like_hf else os.path.abspath(str(model_path))
    )
    if not looks_like_hf and not pathlib.Path(resolved_model).exists():
        raise FileNotFoundError(
            f"LeRobot checkpoint not found at {resolved_model}. Point "
            f"--policy.model-path at the directory holding config.json + "
            f"model.safetensors (usually .../<run>/pretrained_model)."
        )

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpus
    # The guidance kernel lives in ``policy_steering`` (numpy + torch only), so
    # the lerobot venv needs it importable. Only that package is bridged —
    # unlike the GR00T/cosmos servers, nothing from ``policy_abstraction``
    # itself is imported server-side. Prepend so a user-set PYTHONPATH is kept.
    steering_src = str(
        pathlib.Path(PROJECT_ROOT) / "packages" / "policy_steering" / "src"
    )
    env_vars["PYTHONPATH"] = (
        steering_src + os.pathsep + env_vars["PYTHONPATH"]
        if env_vars.get("PYTHONPATH")
        else steering_src
    )
    # torch.compile's autotune/kernel cache defaults to /tmp/torchinductor_*,
    # which systemd-tmpfiles empties at boot -- and a cold max-autotune of the
    # pi0.5 RTC + guidance graphs is ~6 min on an RTX 5090, paid inside the
    # first inference of the day. Keep it somewhere that survives a reboot.
    # An explicit TORCHINDUCTOR_CACHE_DIR in the environment wins.
    env_vars.setdefault(
        "TORCHINDUCTOR_CACHE_DIR",
        os.path.join(
            os.path.expanduser("~"), ".cache", "policy_abstraction", "torchinductor"
        ),
    )
    # A relative PA_GUIDANCE_CAPTURE dir (guidance landscape dump) would land
    # under the lerobot repo, since the child runs with cwd=repo_path.
    capture_dir = env_vars.get("PA_GUIDANCE_CAPTURE")
    if capture_dir and not os.path.isabs(capture_dir):
        env_vars["PA_GUIDANCE_CAPTURE"] = os.path.abspath(capture_dir)

    serve_script = (
        pathlib.Path(PROJECT_ROOT) / "scripts" / "servers" / "serve_lerobot.py"
    )
    cmd = [
        str(py),
        str(serve_script),
        "--model-path",
        resolved_model,
        "--device",
        device,
        "--state-adapter",
        state_adapter,
        "--compile-model",
        compile_model,
        "--guidance",
        guidance,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if policy_type is not None:
        cmd.extend(["--policy-type", policy_type])
    if camera_map:
        cmd.extend(["--camera-map", camera_map])
    if num_inference_steps is not None:
        cmd.extend(["--num-inference-steps", str(num_inference_steps)])
    if rtc:
        cmd.extend(
            [
                "--rtc",
                "--rtc-execution-horizon",
                str(rtc_execution_horizon),
                "--rtc-max-guidance-weight",
                str(rtc_max_guidance_weight),
                "--rtc-prefix-attention-schedule",
                rtc_prefix_attention_schedule,
            ]
        )
    else:
        cmd.append("--no-rtc")
    if robot_type:
        # Handed to the inference frame builder for the tokenizer step;
        # "ur5_follower" for the real arm, empty for the simulators.
        cmd.extend(["--robot-type", robot_type])
    cmd.extend(["--action-space", guidance_action_space])
    if tcp_offset is not None:
        offset = [float(v) for v in tcp_offset]
        if len(offset) != 6:
            raise ValueError(
                f"tcp_offset must be [x, y, z, rx, ry, rz]; got {len(offset)} values."
            )
        cmd.extend(["--tcp-offset", *[str(v) for v in offset]])
    if gripper_geometry:
        cmd.extend(["--gripper-geometry", gripper_geometry])

    # Refuse to spawn onto an occupied port: the child would fail to bind and
    # this run would silently query whatever policy already held the port.
    assert_port_free(int(port), "LeRobot policy server", "--policy.no-launch-server")

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "lerobot_server.stdout.log"
        stderr_path = log_dir / "lerobot_server.stderr.log"
        stdout_f: Any = open(stdout_path, "w", buffering=1)
        stderr_f: Any = open(stderr_path, "w", buffering=1)
        logger.info(f"LeRobot server logs → {stdout_path}, {stderr_path}")
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
        f"Started LeRobot PolicyServer on {host}:{port} (PID {server_process.pid})"
    )
    return server_process


# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------


def extract_views(element: dict) -> dict[str, np.ndarray]:
    """Pull the camera images out of a PA policy element.

    Any entry holding an ``HxWx3`` uint8 array counts as a camera; the state
    vector (1-D), the prompt (a str) and the ``__dunder__`` bookkeeping keys
    are skipped. An ``observation/`` prefix is stripped when present, so both
    spec conventions work: LIBERO/RoboCasa name their policy keys
    ``observation/image``, while the UR5e spec uses the raw LeRobot camera
    names (``wrist_image_left``) so the server can pass them to
    ``build_dataset_frame`` unchanged. Returned in
    :data:`CANONICAL_VIEW_ORDER` first, then any extra views sorted by name,
    so the server's auto-mapping is order-stable.
    """
    found: dict[str, np.ndarray] = {}
    for key, val in element.items():
        if key.startswith("__") or not isinstance(val, np.ndarray):
            continue
        if val.ndim != 3 or val.shape[-1] != 3:
            continue
        view = key[len(_OBS_PREFIX) :] if key.startswith(_OBS_PREFIX) else key
        if val.dtype != np.uint8:
            raise ValueError(
                f"LeRobot expects uint8 images; got {val.dtype} for "
                f"{key!r}. Check camera preprocessing."
            )
        found[view] = val

    if not found:
        raise KeyError(
            "LeRobot client found no camera images in the policy element "
            f"(keys={sorted(element)}). Check your env spec's "
            "policy_input_keys.images."
        )

    ordered = [v for v in CANONICAL_VIEW_ORDER if v in found]
    ordered += sorted(v for v in found if v not in CANONICAL_VIEW_ORDER)
    return {v: found[v] for v in ordered}


# ---------------------------------------------------------------------------
# BasePolicyClient adapter
# ---------------------------------------------------------------------------


class LerobotPolicyClient(BasePolicyClient):
    """``BasePolicyClient`` adapter for a ``serve_lerobot.py`` server.

    Translates PA's per-step element dict (built by
    ``EpisodeRunner._build_policy_input``) into the server's view/state/task
    payload and reshapes the action response to the ``(num_chunks=1, horizon,
    action_dim)`` shape PA's ``EpisodeRunner`` expects.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5557,
        timeout_ms: int = 900_000,
        ready_timeout_s: float = 1200.0,
        server_process: subprocess.Popen | None = None,
        gripper_remap: str | GripperRemap | None = "passthrough",
        replan_steps: int = 5,
    ) -> None:
        """
        Args:
            server_process: Optional handle to the server subprocess. When
                provided, the wait-for-ready loop polls it each cycle and bails
                immediately if the child has exited (missing checkpoint, port
                collision, ...) instead of waiting out ``ready_timeout_s``.
            gripper_remap: How to translate the checkpoint's gripper output
                into the target env's convention (see
                ``policy_abstraction.policies.groot.GRIPPER_REMAPS``). Default
                ``"passthrough"``: a LeRobot checkpoint emits actions in its
                *training dataset's* convention, so a LIBERO-trained one
                already matches robosuite's ``{-1, +1}``.
            replan_steps: How many actions PA consumes from each chunk before
                re-querying. Sent with every request so the server can compute
                the RTC leftover; irrelevant when the server runs without RTC.
            ready_timeout_s: Generous because a cold first run downloads the
                PaliGemma tokenizer/backbone and loads a ~3B checkpoint.
            timeout_ms: ZMQ per-call recv timeout. Deliberately long: the
                *first* inference absorbs ``torch.compile``'s autotuning when
                the checkpoint config enables it (measured ~250 s for
                ``lerobot/pi0_libero_finetuned`` at ``max-autotune``, vs
                ~70 ms once warm). A timeout here leaves the REQ socket
                needing a reinit, so the ceiling is set well above the
                worst observed compile rather than close to it.
        """
        if replan_steps < 1:
            raise ValueError(f"replan_steps must be >= 1; got {replan_steps}.")
        self._client = _LerobotZmqClient(host, port, timeout_ms=timeout_ms)
        self._server_process = server_process
        self._gripper_remap = resolve_gripper_remap(gripper_remap)
        self._replan_steps = int(replan_steps)
        self._nonfinite_guidance_steps = 0
        self._server_info: dict = {}
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
                        f"LeRobot server subprocess (PID {self._server_process.pid}) "
                        f"exited with code {rc} before becoming ready. Check "
                        f"the server log (e.g. lerobot_server.stderr.log)."
                    )
            try:
                if self._client.ping(timeout_ms=2000):
                    logger.info(
                        f"LeRobot server is ready "
                        f"(took {time.monotonic() - start:.1f}s)."
                    )
                    self._fetch_server_info()
                    return
            except Exception as e:  # noqa: BLE001 — best-effort retry loop
                last_err = e
            attempt += 1
            if attempt % 10 == 1:
                logger.info(
                    f"Waiting for LeRobot server at "
                    f"{self._client._host}:{self._client._port} "
                    f"({time.monotonic() - start:.0f}s elapsed)..."
                )
            time.sleep(3.0)
        raise RuntimeError(
            f"LeRobot server at {self._client._host}:{self._client._port} did not "
            f"become ready within {ready_timeout_s}s (last error: {last_err})."
        )

    def _fetch_server_info(self) -> None:
        """Log the resolved checkpoint layout so obs-mapping mistakes surface
        in the run log instead of as a silent distribution shift mid-rollout.

        Called twice: once at connect (checkpoint dims) and once after the
        first inference, because the server can only resolve the camera map
        when it sees which views the env actually sends.
        """
        from loguru import logger

        try:
            info = self._client.call("info")
        except Exception as e:  # noqa: BLE001 — informational only
            logger.debug(f"LeRobot server 'info' endpoint unavailable: {e}")
            return
        if not isinstance(info, dict):
            return
        self._server_info = info
        logger.info(
            f"[lerobot] policy_type={info.get('policy_type')} "
            f"chunk_size={info.get('chunk_size')} "
            f"action_dim={info.get('action_dim')} "
            f"state_dim={info.get('state_dim')} "
            f"rtc={info.get('rtc_enabled')}"
        )
        camera_map = info.get("camera_map")
        if camera_map:
            logger.info(f"[lerobot] camera map: {camera_map}")
        else:
            logger.info(
                f"[lerobot] camera features (mapped on first inference): "
                f"{info.get('camera_features')}"
            )

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------

    def _build_payload(self, element: dict) -> dict:
        try:
            state = element["observation/state"]
            prompt = element["prompt"]
        except KeyError as e:
            raise KeyError(
                f"LeRobot client expected key {e} in policy element. Check "
                f"your env spec's policy_input_keys."
            ) from None

        return {
            "images": extract_views(element),
            "state": np.asarray(state, dtype=np.float32),
            "task": str(prompt),
            # Number of actions PA will execute from this chunk before the next
            # query — the server's RTC leftover is the rest of the chunk.
            "consumed_steps": self._replan_steps,
        }

    def _shape_action_chunk(self, actions: np.ndarray) -> np.ndarray:
        """``(T, action_dim)`` → ``(1, T, action_dim)`` + gripper remap."""
        if actions.ndim != 2:
            raise ValueError(
                f"Expected LeRobot actions shape (T, action_dim); got {actions.shape}."
            )
        out = actions.astype(np.float32, copy=True)[None, :, :]
        out[..., -1] = self._gripper_remap(out[..., -1])
        return out

    # ------------------------------------------------------------------
    # BasePolicyClient interface
    # ------------------------------------------------------------------

    def infer(self, obs_dict: dict) -> dict:
        # Split the ``__gp_*__`` guidance keys out of the element and send them
        # under ``options``, exactly as the GR00T and cosmos clients do. When
        # no sentinel key is present ``options`` is ``None`` and the server
        # takes its un-guided path.
        guidance_params, cleaned = extract_guidance_params(obs_dict)
        payload = self._build_payload(cleaned)
        t0 = time.monotonic()
        response = self._client.call(
            "get_action", {"observation": payload, "options": guidance_params}
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if not isinstance(response, dict) or "actions" not in response:
            raise RuntimeError(
                f"Unexpected LeRobot response: {type(response).__name__} "
                f"(keys={list(response) if isinstance(response, dict) else 'N/A'})."
            )
        actions = self._shape_action_chunk(np.asarray(response["actions"]))

        # The server drops a guidance correction that came out non-finite and
        # serves that denoising step unguided (see GuidanceKernel.apply). Say
        # so here, because a run that reports itself as guided while some of
        # its steps were not is otherwise indistinguishable from a clean one.
        nonfinite_steps = int(response.get("guidance_nonfinite_steps", 0) or 0)
        if nonfinite_steps:
            from loguru import logger

            self._nonfinite_guidance_steps += nonfinite_steps
            logger.warning(
                f"[lerobot] guidance was non-finite on {nonfinite_steps} denoising "
                f"step(s) of this inference and was skipped for them "
                f"({self._nonfinite_guidance_steps} so far this client). Check "
                "the server's stderr log for the stage that produced it."
            )

        # The server resolves the view→camera-feature pairing on its first
        # request; pull it into the run log now that it exists.
        if not self._server_info.get("camera_map"):
            self._fetch_server_info()

        return {
            "actions": actions,
            "policy_timing": response.get(
                "policy_timing", {"infer_ms": float(elapsed_ms)}
            ),
        }

    # Signals to ``EpisodeRunner`` that ``reset()`` must be called between
    # episodes. Opt-in rather than duck-typed: every policy client in the repo
    # happens to define ``reset``, but only this one carries state that would
    # corrupt the next episode if the call were skipped.
    resets_per_episode = True

    def reset(self) -> None:
        """Drop per-episode server state (the RTC previous-chunk buffer).

        Unlike the GR00T/cosmos clients — whose servers treat ``reset`` as a
        no-op, making a swallowed failure harmless — this call is load-bearing
        whenever RTC is on: it is the only thing stopping the previous
        episode's chunk tail from being inpainted into this episode's opening
        chunk. A dropped reset would silently contaminate the rollout, so it
        propagates rather than passing quietly.
        """
        try:
            self._client.call("reset")
        except Exception as e:
            if self._server_info.get("rtc_enabled"):
                raise RuntimeError(
                    f"LeRobot server reset failed ({type(e).__name__}: {e}). "
                    f"RTC is enabled, so continuing would blend the previous "
                    f"episode's action tail into this one."
                ) from e
            from loguru import logger

            logger.warning(f"LeRobot server reset failed, continuing: {e}")

    def close(self) -> None:
        self._client.close()
