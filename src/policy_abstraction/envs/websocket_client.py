"""Generic WebSocket environment client.

:class:`WebsocketEnvClient` is a thin, simulator-agnostic RPC proxy.  Every
method call is forwarded to a compatible server (e.g.
:class:`~policy_abstraction.envs.libero.server.LiberoEnvServer`)
verbatim — the cmd, positional args and keyword args are serialised with
msgpack-numpy and sent over the wire; the return value is deserialised and
returned to the caller unchanged.

The client knows about *one* special lifecycle command beyond plain method
forwarding:

* ``make_env(**kwargs)`` — forwarded as ``{"cmd": "make_env", …}``; if the
  server response contains ``"language_instruction"`` it is cached so that
  ``env.language_instruction`` works as a plain attribute access.

Everything else (``reset``, ``set_init_state``, ``step``, ``seed``,
``close``, …) is forwarded generically via :meth:`__getattr__`.

The connection carries one request at a time, and a reply is only
identifiable by its position in the stream, so two threads calling into the
same client would interleave their sends and each collect the other's
answer — silently, and from then on permanently. :meth:`_rpc` therefore
holds :attr:`rpc_lock` across the whole send/receive pair. A caller that
must not block (a GUI timer, say) can take that same lock non-blockingly
and skip its turn. As a backstop the server echoes the command it answered
and the client checks it, so a desync that slips through is a loud error
rather than a stale observation acted on by a moving robot.

Example
-------
::

    from policy_abstraction.envs.websocket_client import WebsocketEnvClient

    env = WebsocketEnvClient(host="localhost", port=8765)

    env.make_env(
        bddl_file="/path/to/task.bddl",
        camera_heights=256,
        camera_widths=256,
        seed=7,
    )

    obs = env.reset()
    obs = env.set_init_state(initial_state)

    for _ in range(max_steps):
        obs, reward, done, info = env.step(action)
        if done:
            break

    env.disconnect()
"""

import threading
import time
from typing import Any, Optional

import msgpack_numpy

try:
    import websockets.sync.client
except ImportError as e:
    raise ImportError(
        "websockets is required. Install with: pip install websockets"
    ) from e

from loguru import logger

from policy_abstraction.envs.spec import EnvSpec


class WebsocketEnvClient:
    """Generic WebSocket proxy for a remote environment.

    Every attribute access that is not part of the client's own interface
    returns a :class:`_RemoteMethod` callable that, when called, sends the
    call to the server and returns the deserialised result.

    Parameters
    ----------
    host:
        Hostname / IP of the server, or a full ``ws://…`` URI.
    port:
        Port of the server (ignored when *host* is a full URI).
    retry_interval:
        Seconds to wait between connection attempts.
    server_process:
        Optional handle to the server subprocess. Without it the connect loop
        retries forever, so a server that dies at startup -- a camera already
        claimed, a port in use -- presents as an endless "Server not ready"
        rather than as the error it is.
    log_dir:
        Run directory holding ``*_env_server.stderr.log``. When given, a
        server that dies at startup reports its own traceback inline instead
        of sending the operator off to find the file.
    """

    # Attributes that must NOT be proxied to the server.
    _LOCAL_ATTRS = frozenset(
        {
            "_uri",
            "_packer",
            "_ws",
            "_server_metadata",
            "_retry_interval",
            "_server_process",
            "_log_dir",
            "rpc_lock",
            "language_instruction",
            "segmentation_name_mapping",
            "env_spec",
            "disconnect",
            "setup_suite",
        }
    )

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8765,
        retry_interval: float = 5.0,
        server_process: Any = None,
        log_dir: Any = None,
    ) -> None:
        self._uri: str = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._retry_interval = retry_interval
        self._server_process = server_process
        self._log_dir = log_dir
        # Re-entrant so a caller can hold it around a call that itself
        # makes one (see :attr:`rpc_lock`).
        self.rpc_lock = threading.RLock()
        self._ws, self._server_metadata = self._connect()
        # Cached after make_env returns; accessible as env.language_instruction.
        self.language_instruction: str = ""
        # Cached after make_env returns; {str(mask_id): instance_name}.
        self.segmentation_name_mapping: dict[str, str] = {}
        # Cached after setup_suite returns; describes the env's camera/robot
        # layout so downstream modules don't hardcode names or obs keys.
        self.env_spec: Optional[EnvSpec] = None

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def _connect(self):
        logger.info(f"Connecting to env server at {self._uri} …")
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    # Keepalive disabled — server-side RPCs (e.g. RoboCasa
                    # init-state sampling) can block for tens of seconds and
                    # would otherwise trip the default 20 s ping timeout.
                    ping_interval=None,
                    ping_timeout=None,
                    close_timeout=None,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logger.info(f"Connected. Server metadata: {metadata}")
                return conn, metadata
            except (ConnectionRefusedError, OSError):
                if self._server_process is not None:
                    rc = self._server_process.poll()
                    if rc is not None:
                        raise RuntimeError(
                            f"The env server (PID {self._server_process.pid}) "
                            f"exited with code {rc} before accepting a "
                            f"connection.{self._server_error()}"
                        ) from None
                logger.info(f"Server not ready, retrying in {self._retry_interval}s …")
                time.sleep(self._retry_interval)

    def _server_error(self) -> str:
        """The dead server's last words, for the connect-failure message."""
        import pathlib

        if self._log_dir is None:
            return " Check its stderr log in the run directory."
        logs = sorted(pathlib.Path(self._log_dir).glob("*_env_server.stderr.log"))
        if not logs:
            return f" No *_env_server.stderr.log found in {self._log_dir}."
        log = logs[-1]
        try:
            tail = log.read_text(errors="replace").strip().splitlines()[-15:]
        except OSError:
            return f" Check {log}."
        if not tail:
            return f" {log} is empty."
        return "\n\n--- " + str(log) + " ---\n" + "\n".join(tail)

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        try:
            self._ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Low-level RPC
    # ------------------------------------------------------------------

    def _rpc(self, cmd: str, args: tuple = (), kwargs: Optional[dict] = None) -> Any:
        """Send *cmd* with *args* / *kwargs* and return the deserialised result.

        The send and the receive are one atomic operation under
        :attr:`rpc_lock`: the server answers a connection's requests strictly
        in order, so two threads sending between each other's send and recv
        would each be handed the other's reply, for the rest of the session.

        Raises:
            RuntimeError: if the server signals an error, or answers a
                different command than the one asked (see below).
        """
        msg = {"cmd": cmd, "args": list(args), "kwargs": kwargs or {}}
        with self.rpc_lock:
            self._ws.send(self._packer.pack(msg))
            raw = self._ws.recv()
        response = msgpack_numpy.unpackb(raw)

        # Servers from this tree echo the command they answered. Checking it
        # turns a stream that has slipped out of step into an error here,
        # rather than a plausible-looking result -- a stale observation fed
        # to a policy that is driving a real arm. Older servers send no
        # ``cmd`` and are simply not checked.
        answered = response.get("cmd")
        if answered is not None and answered != cmd:
            raise RuntimeError(
                f"Env server answered {answered!r} to a {cmd!r} request. The "
                "request/response stream is out of step, which happens when "
                "two threads share one client without holding its rpc_lock. "
                "This connection cannot be trusted; restart the evaluation."
            )

        error = response.get("error")
        if error:
            logger.error(f"Server error for cmd={cmd!r}:\n{error}")
            raise RuntimeError(f"Remote env server error for cmd={cmd!r}:\n{error}")
        return response.get("result")

    # ------------------------------------------------------------------
    # make_env — explicit so we can cache language_instruction
    # ------------------------------------------------------------------

    def make_env(self, *args, **kwargs) -> Any:
        """Create (or re-create) the remote environment.

        All positional and keyword arguments are forwarded verbatim to the
        server.  If the server response dict contains ``"language_instruction"``
        it is cached as ``self.language_instruction``.
        """
        result = self._rpc("make_env", args, kwargs)
        if isinstance(result, dict):
            if "language_instruction" in result:
                self.language_instruction = result["language_instruction"]
            if "segmentation_name_mapping" in result:
                self.segmentation_name_mapping = result["segmentation_name_mapping"]
        return result

    # ------------------------------------------------------------------
    # setup_suite — explicit so we can cache env_spec
    # ------------------------------------------------------------------

    def setup_suite(self, *args, **kwargs) -> Any:
        """Initialise the suite and cache the returned :class:`EnvSpec`.

        The server response dict is returned verbatim. If it contains an
        ``"env_spec"`` key, it is parsed and cached on ``self.env_spec``.
        """
        result = self._rpc("setup_suite", args, kwargs)
        if isinstance(result, dict) and "env_spec" in result and result["env_spec"]:
            self.env_spec = EnvSpec.from_dict(result["env_spec"])
        return result

    # ------------------------------------------------------------------
    # Generic proxy for all other env methods
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        # Guard against infinite recursion for private / local attrs.
        if name.startswith("_") or name in self._LOCAL_ATTRS:
            raise AttributeError(name)

        def _remote_method(*args, **kwargs):
            return self._rpc(name, args, kwargs)

        _remote_method.__name__ = name
        _remote_method.__qualname__ = f"WebsocketEnvClient.{name}"
        return _remote_method
