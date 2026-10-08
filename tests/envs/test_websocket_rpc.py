"""The env RPC stream stays in step, even with two threads on one client.

A connection carries one request at a time and the server answers in
order, so a reply is identified purely by its position in the stream. If
two threads send between each other's send and receive, each collects the
other's answer — and every call after that is off by one, for the rest of
the session. The failure does not look like a failure: the caller gets a
well-formed value from a different command.

That is what the experiment GUI hit. Its preview timer polls ``get_obs``
on the Qt thread while the rollout thread drives the episode, and the
rollout was handed the preview's observation dict where it expected a
list of initial states.

Two mechanisms are pinned here: the client holds its lock across the
whole exchange, and the server echoes the command it answered so a
desync that slips through is loud rather than silent.
"""

from __future__ import annotations

import collections
import threading

import msgpack_numpy
import pytest

from policy_abstraction.envs.websocket_client import WebsocketEnvClient


class FakeSocket:
    """Answers each request in order with the command's own name.

    Optional hooks fire inside ``send`` / ``recv`` so a test can inspect
    the lock exactly while the exchange is mid-flight.
    """

    def __init__(self, on_send=None, on_recv=None):
        self.replies: collections.deque = collections.deque()
        self.sent: list[str] = []
        self._lock = threading.Lock()
        self._on_send = on_send
        self._on_recv = on_recv

    def send(self, raw: bytes) -> None:
        cmd = msgpack_numpy.unpackb(raw)["cmd"]
        with self._lock:
            self.sent.append(cmd)
            self.replies.append({"cmd": cmd, "result": cmd, "error": None})
        if self._on_send is not None:
            self._on_send()

    def recv(self) -> bytes:
        if self._on_recv is not None:
            self._on_recv()
        with self._lock:
            reply = self.replies.popleft()
        return msgpack_numpy.packb(reply)

    def close(self) -> None:
        pass


def _client(sock: FakeSocket) -> WebsocketEnvClient:
    """A client wired to *sock*, without connecting to anything."""
    client = object.__new__(WebsocketEnvClient)
    client._uri = "ws://fake"
    client._packer = msgpack_numpy.Packer()
    client._ws = sock
    client._server_metadata = {}
    client._retry_interval = 0.0
    client._server_process = None
    client._log_dir = None
    client.rpc_lock = threading.RLock()
    client.language_instruction = ""
    client.segmentation_name_mapping = {}
    client.env_spec = None
    return client


class TestTheLockCoversTheWholeExchange:
    def test_the_lock_is_held_across_send_and_recv(self):
        """Not just around the send: a lock released before the reply is
        read leaves exactly the window this bug needs."""
        held: list[tuple[str, bool]] = []
        client: list[WebsocketEnvClient] = []

        sock = FakeSocket(
            on_send=lambda: held.append(("send", client[0].rpc_lock._is_owned())),
            on_recv=lambda: held.append(("recv", client[0].rpc_lock._is_owned())),
        )
        client.append(_client(sock))

        assert client[0].get_obs() == "get_obs"
        assert held == [("send", True), ("recv", True)]

    def test_the_lock_is_released_afterwards(self):
        """A caller that skips its turn when the lock is taken (the GUI
        preview does) would otherwise skip every turn."""
        client = _client(FakeSocket())
        client.get_obs()
        assert client.rpc_lock.acquire(blocking=False)
        client.rpc_lock.release()

    def test_the_lock_is_released_when_the_server_errors(self):
        sock = FakeSocket()
        client = _client(sock)
        # Queued ahead of the fake's own echo, so this is what step() reads.
        sock.replies.append({"cmd": "step", "result": None, "error": "boom"})

        with pytest.raises(RuntimeError, match="boom"):
            client.step()
        assert client.rpc_lock.acquire(blocking=False)
        client.rpc_lock.release()

    def test_concurrent_callers_each_get_their_own_reply(self):
        """The end-to-end property, from the two threads that share a
        client in the experiment GUI."""
        client = _client(FakeSocket())
        mismatches: list[tuple[str, object]] = []

        def hammer(cmd: str) -> None:
            for _ in range(50):
                got = getattr(client, cmd)()
                if got != cmd:
                    mismatches.append((cmd, got))

        threads = [
            threading.Thread(target=hammer, args=(name,))
            for name in ("get_obs", "step", "home")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not mismatches


class TestTheEchoCatchesADesync:
    def test_a_reply_to_another_command_is_refused(self):
        """The backstop. Without it this is the UR5e failure: an
        observation dict returned where init states were asked for, and
        the caller indexing into it."""
        sock = FakeSocket()
        client = _client(sock)
        sock.replies.append(
            {"cmd": "get_obs", "result": {"joint_pos": [0]}, "error": None}
        )

        with pytest.raises(RuntimeError, match="answered 'get_obs'"):
            client.get_task_init_states(task_id=0)

    def test_the_message_names_the_cause_and_the_cure(self):
        sock = FakeSocket()
        client = _client(sock)
        sock.replies.append({"cmd": "home", "result": None, "error": None})

        with pytest.raises(RuntimeError) as excinfo:
            client.step()
        message = str(excinfo.value)
        assert "out of step" in message
        assert "rpc_lock" in message
        assert "restart" in message

    def test_a_server_that_sends_no_cmd_is_not_checked(self):
        """Servers predating the echo still work."""
        sock = FakeSocket()
        client = _client(sock)
        sock.replies.append({"result": 7, "error": None})
        assert client.anything() == 7


class TestServerEchoesTheCommand:
    """The other half of the check, on every server in the tree."""

    class _Ws:
        """Enough of a websocket for one handler pass."""

        remote_address = ("test", 0)

        def __init__(self, requests):
            self._requests = [msgpack_numpy.packb(r) for r in requests]
            self.sent: list[dict] = []

        async def send(self, raw):
            self.sent.append(msgpack_numpy.unpackb(raw))

        def __aiter__(self):
            async def _gen():
                for raw in self._requests:
                    yield raw

            return _gen()

    @staticmethod
    def _serve(requests, dispatch):
        import asyncio

        from policy_abstraction.envs.base_server import BaseEnvServer

        class _Server(BaseEnvServer):
            async def _dispatch(self, cmd, args, kwargs, state):
                return dispatch(cmd), state

        ws = TestServerEchoesTheCommand._Ws(requests)
        asyncio.run(_Server()._handler(ws))
        # First send is the handshake metadata.
        return ws.sent[1:]

    def test_a_result_carries_the_command_it_answers(self):
        replies = self._serve(
            [{"cmd": "get_obs"}, {"cmd": "step"}],
            lambda cmd: f"{cmd}-result",
        )
        assert [r["cmd"] for r in replies] == ["get_obs", "step"]
        assert [r["result"] for r in replies] == ["get_obs-result", "step-result"]

    def test_an_error_carries_it_too(self):
        """Otherwise a failed call would be the one reply a desynced
        client could not identify."""

        def _boom(cmd):
            raise ValueError("no such command")

        (reply,) = self._serve([{"cmd": "nonsense"}], _boom)
        assert reply["cmd"] == "nonsense"
        assert "no such command" in reply["error"]
