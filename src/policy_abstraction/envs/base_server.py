"""Shared WebSocket env server scaffolding.

Holds the connection plumbing that's identical across simulator backends —
``serve_forever``, the asyncio handler, the msgpack-numpy framing, the health
check endpoint, and the result-packing helper. Concrete servers (LIBERO,
RoboCasa, …) subclass :class:`BaseEnvServer` and implement ``_dispatch``.

Wire protocol (matches what :class:`policy_abstraction.envs.WebsocketEnvClient`
expects):

  Client → Server:  msgpack({"cmd": "<command>", "args": [...], "kwargs": {...}})
  Server → Client:  msgpack({"cmd": "<command>",
                             "result": <value>, "error": null | "<traceback>"})

Requests on one connection are handled strictly in order, so a reply is
identified only by its position in the stream. The echoed ``cmd`` lets the
client notice a stream that has slipped out of step (two threads sharing a
client, say) instead of acting on another call's answer. Clients predating
the echo ignore the extra key.

The first message a client receives after connecting is a handshake metadata
dict — subclasses override :attr:`server_metadata` to identify themselves.
"""

from __future__ import annotations

import asyncio
import http
import logging
import traceback as tb
from typing import Any, Tuple

import msgpack_numpy
import numpy as np
import websockets.exceptions
import websockets.server as _server

logger = logging.getLogger(__name__)


class BaseEnvServer:
    """Generic WebSocket env server. Subclasses override :meth:`_dispatch`."""

    server_metadata: dict = {"server": "BaseEnvServer", "version": "1.0"}

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self._host = host
        self._port = port

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        """Block and serve connections until interrupted."""
        asyncio.run(self._run())

    # ------------------------------------------------------------------
    # Internal async implementation
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        async with _server.serve(  # type: ignore[attr-defined]
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(
                f"{type(self).__name__} listening on ws://{self._host}:{self._port}"
            )
            await server.serve_forever()

    @staticmethod
    async def _health_check(path, request_headers):
        if path == "/healthz":
            return http.HTTPStatus.OK, {}, b"OK\n"
        return None

    async def _handler(self, websocket) -> None:
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self.server_metadata))

        state: dict = self._initial_state()
        try:
            async for raw in websocket:
                msg = msgpack_numpy.unpackb(raw)
                cmd: str = msg.get("cmd", "")
                args: list = msg.get("args", [])
                kwargs: dict = msg.get("kwargs", {})

                try:
                    result, state = await self._dispatch(cmd, args, kwargs, state)
                    await websocket.send(
                        packer.pack({"cmd": cmd, "result": result, "error": None})
                    )
                except Exception:
                    err = tb.format_exc()
                    logger.error(f"Error handling cmd={cmd!r}:\n{err}")
                    await websocket.send(
                        packer.pack({"cmd": cmd, "result": None, "error": err})
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")
        finally:
            self._cleanup_state(state)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _initial_state(self) -> dict:
        """Per-connection state. Default: holds an ``"env"`` slot."""
        return {"env": None}

    def _cleanup_state(self, state: dict) -> None:
        """Best-effort env close on connection drop."""
        env = state.get("env")
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    async def _dispatch(
        self, cmd: str, args: list, kwargs: dict, state: dict
    ) -> Tuple[Any, dict]:
        """Handle one command. Return ``(result, state)``.

        Subclasses must implement.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Result packing
# ---------------------------------------------------------------------------


def pack_result(result):
    """Recursively ensure numpy arrays survive msgpack serialisation.

    * ``None`` → ``None``
    * ``dict``  → pack each value
    * ``list`` / ``tuple`` → pack each element (step returns a 4-tuple)
    * scalar (int / float / bool / str) → as-is
    * ``np.ndarray`` → as-is (msgpack_numpy handles it)
    """
    if result is None:
        return None
    if isinstance(result, dict):
        return {k: pack_result(v) for k, v in result.items()}
    if isinstance(result, (list, tuple)):
        return [pack_result(v) for v in result]
    if isinstance(result, np.ndarray):
        return result
    return result


__all__ = ["BaseEnvServer", "pack_result"]
