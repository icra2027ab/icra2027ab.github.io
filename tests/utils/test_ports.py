"""Port-holder detection for the server launchers.

The failure this guards against is silent: a launcher spawns a server onto
an occupied port, the child dies unread, and the client attaches to the
stranger already there — running the evaluation against that process's
configuration.
"""

from __future__ import annotations

import os
import socket

import pytest

from policy_abstraction.utils.ports import (
    assert_port_free,
    is_port_open,
    port_listener,
)


@pytest.fixture
def listening_port():
    """A real LISTEN socket owned by this process."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


@pytest.fixture
def free_port():
    """A port number nothing is listening on."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class TestPortListener:
    def test_finds_the_holding_process(self, listening_port):
        holder = port_listener(listening_port)
        assert holder is not None
        assert holder.split()[0] == str(os.getpid())

    def test_reports_the_holders_command_line(self, listening_port):
        """The cmdline is the whole point — it names the wrong checkout."""
        holder = port_listener(listening_port)
        assert holder is not None and len(holder.split()) > 1

    def test_free_port_has_no_listener(self, free_port):
        assert port_listener(free_port) is None

    def test_a_connected_socket_is_not_a_listener(self, listening_port):
        """Only LISTEN counts; an outbound connection must not register."""
        client = socket.create_connection(("127.0.0.1", listening_port))
        try:
            holder = port_listener(client.getsockname()[1])
            assert holder is None
        finally:
            client.close()


class TestAssertPortFree:
    def test_free_port_passes(self, free_port):
        assert_port_free(free_port, "test server", "--no-launch-server")

    def test_occupied_port_raises(self, listening_port):
        with pytest.raises(RuntimeError):
            assert_port_free(listening_port, "test server", "--no-launch-server")

    def test_error_names_holder_server_and_escape_hatch(self, listening_port):
        with pytest.raises(RuntimeError) as excinfo:
            assert_port_free(listening_port, "UR5 env server", "--env.no-launch-server")
        msg = str(excinfo.value)
        assert str(listening_port) in msg
        assert str(os.getpid()) in msg
        assert "UR5 env server" in msg
        assert "--env.no-launch-server" in msg


class TestIsPortOpen:
    def test_true_while_listening(self, listening_port):
        assert is_port_open("127.0.0.1", listening_port)

    def test_false_when_nothing_listens(self, free_port):
        assert not is_port_open("127.0.0.1", free_port)
