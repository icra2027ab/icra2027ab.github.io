"""NVIDIA Isaac GR00T N1.7 policy backend.

Provides a drop-in `BasePolicyClient` implementation that talks to a
`gr00t/eval/run_gr00t_server.py` instance over ZMQ, and a helper to spawn that
server as a subprocess from GR00T's own venv (Python 3.10 + torch + CUDA),
mirroring the pattern in [`openpi.start_policy_server`](./openpi.py).

The wire protocol matches `gr00t.policy.server_client.PolicyServer` (REQ/REP
ZMQ + msgpack with a custom ndarray codec). We vendor a minimal client here
instead of importing GR00T directly so PA's venv doesn't need any GR00T
internals.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import time
from typing import Any, Callable

import msgpack_numpy as mnp
import numpy as np
import zmq

from policy_abstraction.policies.base import BasePolicyClient
from policy_abstraction.utils.servers import spawn_server
from policy_steering.params import extract_guidance_params

# Server-side action chunk horizon for LIBERO_PANDA. Keep in sync with
# GR00T's modality config; only used for logging/sanity checks.
_GROOT_ACTION_HORIZON = 8

# Ordered action component keys for LIBERO_PANDA. The GR00T server returns
# `action.<key>` arrays; we stack them along the last axis to form (T, 7).
_LIBERO_ACTION_ORDER: tuple[str, ...] = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",
)


# ---------------------------------------------------------------------------
# Gripper conventions
# ---------------------------------------------------------------------------
#
# GR00T emits ``action.gripper`` in RLDS canonical convention: float in
# ``[0, 1]`` with ``0=close, 1=open``. Different environments expect different
# input conventions, so the right remap is a property of the *target env*,
# not of GR00T. Each function takes and returns an ndarray of arbitrary shape
# (the last axis is the gripper component); use the registry below or pass
# your own callable to ``GrootPolicyClient(gripper_remap=...)``.

GripperRemap = Callable[[np.ndarray], np.ndarray]


def passthrough_gripper(g: np.ndarray) -> np.ndarray:
    """Identity remap. Use when the env consumes RLDS-style ``[0, 1]``."""
    return g.astype(np.float32, copy=False)


def libero_robosuite_gripper(g: np.ndarray) -> np.ndarray:
    """RLDS ``[0, 1]`` (0=close, 1=open) → robosuite ``{-1, +1}`` (-1=open, +1=close).

    Mirrors the two-step transform GR00T's own LIBERO sim env wrapper applies
    in its ``step`` method (``normalize_gripper_action(binarize=True)`` then
    ``invert_gripper_action``); see
    ``third_party/Isaac-GR00T/gr00t/eval/sim/LIBERO/libero_env.py``.

    Combined: ``out = sign(1 - 2*x)``.
    """
    return np.sign(1.0 - 2.0 * g).astype(np.float32)


GRIPPER_REMAPS: dict[str, GripperRemap] = {
    "passthrough": passthrough_gripper,
    "libero_robosuite": libero_robosuite_gripper,
}


def resolve_gripper_remap(remap: str | GripperRemap | None) -> GripperRemap:
    """Resolve a string name or callable into a `GripperRemap` callable."""
    if remap is None:
        return passthrough_gripper
    if callable(remap):
        return remap
    try:
        return GRIPPER_REMAPS[remap]
    except KeyError:
        raise ValueError(
            f"Unknown gripper remap {remap!r}. Known: {sorted(GRIPPER_REMAPS)}. "
            f"Or pass a callable np.ndarray -> np.ndarray."
        ) from None


# ---------------------------------------------------------------------------
# Vendored ZMQ client (minimal port of gr00t.policy.server_client.PolicyClient)
# ---------------------------------------------------------------------------


# NB: the server (gr00t.policy.server_client.MsgSerializer) encodes/decodes
# ndarrays with msgpack_numpy (mnp.encode/decode → {b'nd': ..., b'data': ...}).
# This client MUST use the same codec; a hand-rolled np.save format would be
# left undecoded by the server and arrive as a raw dict ("Video key 'image'
# must be a numpy array. Got <class 'dict'>").
def _encode_custom(obj: Any) -> Any:
    return mnp.encode(obj)


def _decode_custom(obj: Any) -> Any:
    return mnp.decode(obj)


class _GrootZmqClient:
    """Thin REQ/REP ZMQ client for GR00T's PolicyServer.

    Mirrors the request shape `{"endpoint": <name>, "data": <dict>}` and
    msgpack ndarray codec defined in `gr00t.policy.server_client.MsgSerializer`.
    """

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
            self._socket.send(mnp.packb(request, default=_encode_custom))
            message = self._socket.recv()
        except zmq.error.Again:
            # REQ socket is now in an invalid state (waiting for a reply that
            # will never arrive); recreate it before re-raising so the caller
            # can retry. Same recovery as gr00t's PolicyClient.
            self._init_socket()
            raise
        response = mnp.unpackb(message, object_hook=_decode_custom, raw=False)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping(self, timeout_ms: int = 2000) -> bool:
        """Probe server readiness with a short, dedicated timeout.

        Uses ``timeout_ms`` (default 2s) instead of the full inference timeout
        so retry loops can quickly tell "server still booting" from "server
        responding"; restores the long inference timeout afterward.
        """
        prior = self._current_timeout_ms
        self._init_socket(timeout_ms)
        try:
            self.call("ping")
            return True
        except (zmq.error.ZMQError, RuntimeError):
            return False
        finally:
            # Always restore the inference timeout — and reset the socket if
            # the ping failed mid-flight (REQ/REP state machine).
            self._init_socket(prior)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None


# ---------------------------------------------------------------------------
# Server subprocess
# ---------------------------------------------------------------------------


def start_groot_server(
    port: int,
    model_path: str,
    *,
    gpus: str = "0",
    embodiment_tag: str = "LIBERO_PANDA",
    host: str = "127.0.0.1",
    use_sim_policy_wrapper: bool = True,
    repo_path: str | os.PathLike = "third_party/Isaac-GR00T",
    venv_python: str = ".venv/bin/python",
    log_dir: str | os.PathLike | None = None,
) -> subprocess.Popen:
    """Spawn `gr00t/eval/run_gr00t_server.py` from GR00T's own venv.

    The GR00T repo lives outside PA's installable tree (gitignored), so we
    invoke it by absolute path with `cwd=<repo_path>` and the GR00T venv's
    Python (3.10 + torch + CUDA). PA's calling process keeps its own
    interpreter; communication happens over ZMQ on `port`.

    Args:
        port: ZMQ TCP port the server should bind to.
        model_path: Path to a GR00T checkpoint dir (e.g. the `libero_10/`
            subdir of `nvidia/GR00T-N1.7-LIBERO`). Resolved relative to the
            GR00T repo root, so absolute paths are usually safest.
        gpus: ``CUDA_VISIBLE_DEVICES`` for the child.
        embodiment_tag: GR00T embodiment id (default LIBERO_PANDA).
        host: Bind address. Default is loopback so the port is local-only.
        use_sim_policy_wrapper: Forwarded to the server. Required for the
            flat `video.*` / `state.*` / `action.*` keys this client speaks.
        repo_path: Path to the Isaac-GR00T checkout. Relative paths are
            resolved against ``PROJECT_ROOT`` if ``$PWD`` isn't the project
            root yet.
        venv_python: GR00T venv interpreter, relative to ``repo_path``.
        log_dir: Where to write `groot_server.{stdout,stderr}.log`. If
            ``None``, the child inherits the parent's stdio. Mirrors
            ``start_policy_server``: never use ``subprocess.PIPE`` here
            without a reader thread; pipe backpressure stalls the server.
    """
    from loguru import logger

    repo_path = pathlib.Path(repo_path)
    if not repo_path.is_absolute():
        # Resolve against PROJECT_ROOT so callers can pass the repo-relative
        # default regardless of cwd.
        from policy_abstraction import PROJECT_ROOT

        repo_path = pathlib.Path(PROJECT_ROOT) / repo_path
    if not repo_path.exists():
        raise FileNotFoundError(
            f"GR00T repo not found at {repo_path}. Pass --policy.groot-repo-path "
            f"or clone Isaac-GR00T into third_party/Isaac-GR00T."
        )
    py = repo_path / venv_python
    if not py.exists():
        raise FileNotFoundError(
            f"GR00T venv interpreter not found at {py}. Run "
            f"`uv sync --all-extras` inside {repo_path} to create it."
        )

    # The server is launched with cwd=repo_path, so relative paths would
    # resolve under the GR00T repo (where they don't exist) and HF would then
    # interpret them as repo IDs. Absolutize against the caller's cwd, matching
    # `openpi.start_policy_server`'s handling of `dynamics_model_path`.
    abs_model_path = os.path.abspath(str(model_path))
    if not pathlib.Path(abs_model_path).exists():
        # Hint early; HF's downstream error ("Repo id must be in the form...")
        # is misleading when the path simply doesn't resolve.
        logger.warning(
            f"GR00T model path does not exist locally: {abs_model_path}. "
            f"If this is a HuggingFace repo id, prefix it with the namespace "
            f"(e.g. 'nvidia/GR00T-N1.7-LIBERO/libero_10') instead of a local path."
        )

    env_vars = os.environ.copy()
    env_vars["CUDA_VISIBLE_DEVICES"] = gpus
    # The server runs with cwd=<GR00T repo>, so a relative PA_GUIDANCE_CAPTURE
    # dir (diagnostic landscape dump) would land under third_party. Absolutize
    # it here (eval-process cwd = project root) so captures go where expected.
    _cap = env_vars.get("PA_GUIDANCE_CAPTURE")
    if _cap and not os.path.isabs(_cap):
        env_vars["PA_GUIDANCE_CAPTURE"] = os.path.abspath(_cap)
    # Make ``policy_abstraction`` and ``policy_steering`` importable from the
    # GR00T venv subprocess so it can load the grounded action head + policy.
    # Prepend rather than overwrite to keep any user-provided PYTHONPATH.
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

    serve_script = pathlib.Path(PROJECT_ROOT) / "scripts" / "servers" / "serve_groot.py"
    cmd = [
        str(py),
        str(serve_script),
        "--model-path",
        abs_model_path,
        "--embodiment-tag",
        embodiment_tag,
        "--host",
        host,
        "--port",
        str(port),
    ]
    if use_sim_policy_wrapper:
        cmd.append("--use-sim-policy-wrapper")

    if log_dir is not None:
        log_dir = pathlib.Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "groot_server.stdout.log"
        stderr_path = log_dir / "groot_server.stderr.log"
        stdout_f: Any = open(stdout_path, "w", buffering=1)
        stderr_f: Any = open(stderr_path, "w", buffering=1)
        logger.info(f"GR00T server logs → {stdout_path}, {stderr_path}")
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
        f"Started GR00T PolicyServer on {host}:{port} (PID {server_process.pid})"
    )
    return server_process


# ---------------------------------------------------------------------------
# Client wrapper conforming to BasePolicyClient
# ---------------------------------------------------------------------------


class GrootPolicyClient(BasePolicyClient):
    """`BasePolicyClient` adapter for a GR00T server (sim-wrapper format).

    Translates PA's per-step element dict (built by
    ``EpisodeRunner._build_policy_input``) into the flat GR00T sim-wrapper
    keys (``video.*``, ``state.*``, ``annotation.human.action.task_description``)
    and stacks the per-component ``action.*`` response back into the
    ``(num_chunks, horizon, action_dim)`` shape the rest of PA expects.

    The `Gr00tSimPolicyWrapper` enforces the LIBERO state layout used by
    PA's `policy_input_keys.state_layout` (8D: pos(3) + axis-angle(3) +
    gripper qpos(2)), so no coordinate conversion is needed.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = 120_000,
        ready_timeout_s: float = 600.0,
        server_process: subprocess.Popen | None = None,
        gripper_remap: str | GripperRemap | None = None,
    ) -> None:
        """
        Args:
            server_process: Optional handle to the GR00T server subprocess.
                When provided, ``_wait_for_server`` polls it each cycle and
                bails immediately if the child has exited (e.g. port collision,
                missing checkpoint) instead of waiting out the full
                ``ready_timeout_s``.
            gripper_remap: How to translate GR00T's RLDS-style gripper output
                (``[0, 1]``, 0=close, 1=open) into the target env's expected
                convention. Pass a string from ``GRIPPER_REMAPS`` (e.g.
                ``"libero_robosuite"``), a callable, or ``None`` for
                passthrough. The right value depends on the *env*, not the
                policy — robosuite-based LIBERO needs ``"libero_robosuite"``,
                envs that already accept RLDS canonical can use ``None``,
                and other platforms should plug their own callable.
        """
        self._client = _GrootZmqClient(host, port, timeout_ms=timeout_ms)
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
                        f"GR00T server subprocess (PID {self._server_process.pid}) "
                        f"exited with code {rc} before becoming ready. Check "
                        f"the server log (e.g. groot_server.stderr.log)."
                    )
            try:
                if self._client.ping(timeout_ms=2000):
                    logger.info(
                        f"GR00T server is ready (took {time.monotonic() - start:.1f}s)."
                    )
                    return
            except Exception as e:  # noqa: BLE001 — best-effort retry loop
                last_err = e
            attempt += 1
            # Log every ~10s while loading; the GR00T model takes ~30-60s
            # to warm up on cold cache.
            if attempt % 5 == 1:
                logger.info(
                    f"Waiting for GR00T server at "
                    f"{self._client._host}:{self._client._port} "
                    f"({time.monotonic() - start:.0f}s elapsed)..."
                )
            time.sleep(2.0)
        raise RuntimeError(
            f"GR00T server at {self._client._host}:{self._client._port} did not "
            f"become ready within {ready_timeout_s}s (last error: {last_err})."
        )

    # ------------------------------------------------------------------
    # Translation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_image(img: np.ndarray) -> np.ndarray:
        """Add (B, T) leading dims and ensure uint8 contiguous HWC."""
        if img.dtype != np.uint8:
            raise ValueError(
                f"GR00T expects uint8 images; got {img.dtype}. Check "
                f"camera preprocessing."
            )
        if img.ndim != 3:
            raise ValueError(f"GR00T expects HxWxC images; got shape {img.shape}.")
        return np.ascontiguousarray(img)[None, None, ...]

    @staticmethod
    def _split_state(state: np.ndarray) -> dict[str, np.ndarray]:
        """Split PA's 8D LIBERO state vector into GR00T's per-component keys.

        Layout (matches `libero_env_spec().policy_input_keys.state_layout`):
          [0:3]  eef position           → state.x, state.y, state.z
          [3:6]  eef axis-angle         → state.roll, state.pitch, state.yaw
                                          (modality.json names them rpy but
                                          rotation_type is "axis_angle")
          [6:8]  gripper qpos (2-DOF)   → state.gripper
        """
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,):
            raise ValueError(
                f"GR00T LIBERO_PANDA expects an 8D state vector; "
                f"got shape {state.shape}."
            )
        out: dict[str, np.ndarray] = {}
        # Each scalar key gets shape (B=1, T=1, D=1).
        for i, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
            out[f"state.{key}"] = state[i : i + 1].reshape(1, 1, 1)
        # Gripper has D=2.
        out["state.gripper"] = state[6:8].reshape(1, 1, 2)
        return out

    def _build_groot_obs(self, element: dict) -> dict:
        """Translate PA element → GR00T flat sim-wrapper observation."""
        try:
            img = element["observation/image"]
            wrist_img = element["observation/wrist_image"]
            state = element["observation/state"]
            prompt = element["prompt"]
        except KeyError as e:
            raise KeyError(
                f"GR00T client expected key {e} in policy element. PA's "
                f"libero spec produces these by default; check your env spec."
            ) from None

        obs: dict = {
            "video.image": self._pack_image(img),
            "video.wrist_image": self._pack_image(wrist_img),
            **self._split_state(state),
            # Sim wrapper consumes a tuple/list of length B; B=1 here.
            "annotation.human.action.task_description": (str(prompt),),
        }
        return obs

    def _stack_action(self, action_dict: dict) -> np.ndarray:
        """Stack the 7 GR00T action components into shape (1, T, 7).

        Applies ``self._gripper_remap`` to the gripper component (last axis
        index 6) on the way out. GR00T emits gripper in RLDS canonical form
        (``[0, 1]``, 0=close, 1=open); the env-specific remap is configured
        per-instance — see ``GRIPPER_REMAPS`` and the constructor docstring.
        """
        try:
            chunks = [action_dict[f"action.{k}"] for k in _LIBERO_ACTION_ORDER]
        except KeyError as e:
            raise KeyError(
                f"GR00T action response missing {e}. Expected keys: "
                f"{['action.' + k for k in _LIBERO_ACTION_ORDER]}; "
                f"got {sorted(action_dict)}."
            ) from None

        arrs = [np.asarray(c, dtype=np.float32) for c in chunks]
        # Each arr is shape (B, T, 1). Squeeze the last dim of each component
        # before stacking to be tolerant of (B, T) vs (B, T, 1) shapes
        # returned by different server versions.
        normalized = []
        for arr in arrs:
            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.ndim != 2:
                raise ValueError(
                    f"Unexpected GR00T action component shape {arr.shape}; "
                    f"want (B, T) or (B, T, 1)."
                )
            normalized.append(arr)
        stacked = np.stack(normalized, axis=-1)  # (B, T, 7)

        stacked[..., 6] = self._gripper_remap(stacked[..., 6])

        return stacked

    # ------------------------------------------------------------------
    # BasePolicyClient interface
    # ------------------------------------------------------------------

    def infer(self, obs_dict: dict) -> dict:
        # Split out ``__gp_*__`` guidance keys before building the GR00T obs;
        # they ride in ``options`` so the (grounded) action head can consume
        # them. When no sentinel key is present, ``options`` stays ``None``
        # and the server's grounded subclass falls through to the upstream
        # un-guarded path.
        guidance_params, cleaned_obs = extract_guidance_params(obs_dict)
        groot_obs = self._build_groot_obs(cleaned_obs)
        t0 = time.monotonic()
        response = self._client.call(
            "get_action",
            {"observation": groot_obs, "options": guidance_params},
        )
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # Server returns a list/tuple [action_dict, info_dict]; tolerate both.
        if isinstance(response, (list, tuple)) and len(response) == 2:
            action_dict, info = response
        elif isinstance(response, dict):
            action_dict, info = response, {}
        else:
            raise RuntimeError(
                f"Unexpected GR00T response type: {type(response).__name__}"
            )

        actions = self._stack_action(action_dict)  # (1, T, 7)

        if actions.shape[1] != _GROOT_ACTION_HORIZON:
            from loguru import logger

            logger.debug(
                f"GR00T returned action horizon {actions.shape[1]} "
                f"(expected {_GROOT_ACTION_HORIZON}); proceeding."
            )

        return {
            "actions": actions,
            "policy_timing": {"infer_ms": float(elapsed_ms)},
            "groot_info": info if isinstance(info, dict) else {},
        }

    def reset(self) -> None:
        try:
            self._client.call("reset", {"options": None})
        except Exception:  # noqa: BLE001 — reset is best-effort
            pass

    def close(self) -> None:
        self._client.close()
