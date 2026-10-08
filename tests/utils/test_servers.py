"""Server children must not outlive the process that spawned them.

An orphaned server is not merely idle: it holds its port, so the next run's
child fails to bind and the client silently attaches to the stranger. On
hardware it also holds the cameras and an RTDE session.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

import pytest

from policy_abstraction.utils.servers import spawn_server, terminate_servers

SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A zombie is not alive for our purposes.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _wait_gone(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def reaped():
    """Track spawned pids and make sure none survives the test."""
    pids: list[int] = []
    yield pids
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except OSError:
            pass


class TestSpawnServer:
    def test_child_gets_its_own_session(self, reaped):
        """So a terminal Ctrl-C cannot interrupt a server mid-motion."""
        proc = spawn_server(SLEEPER)
        reaped.append(proc.pid)
        assert os.getpgid(proc.pid) != os.getpgid(os.getpid())

    @pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux")
    def test_child_dies_when_its_parent_is_sigkilled(self, reaped):
        """The ghost case: no atexit, no handler, no orderly shutdown.

        A grandchild is spawned through a parent we then SIGKILL, which is
        the one death that cannot be cleaned up in userspace.
        """
        parent = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import sys, time;"
                f"sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/src'!r});"
                "from policy_abstraction.utils.servers import spawn_server;"
                f"c = spawn_server({SLEEPER!r});"
                "print(c.pid, flush=True);"
                "time.sleep(120)",
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            assert parent.stdout is not None
            grandchild = int(parent.stdout.readline().strip())
            reaped.append(grandchild)
            assert _alive(grandchild)

            parent.kill()
            parent.wait(timeout=10)
            assert _wait_gone(grandchild), (
                "grandchild survived its parent's SIGKILL — it would hold the "
                "port and the cameras until noticed by hand"
            )
        finally:
            if parent.poll() is None:
                parent.kill()

    @pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_PDEATHSIG is Linux")
    def test_child_survives_the_thread_that_spawned_it(self, reaped):
        """PR_SET_PDEATHSIG's "parent" is a *thread*, not the process.

        The operator GUI builds its whole session — servers included — on a
        short-lived worker so the window is up while the checkpoint loads. When
        the fork happened on that worker, the kernel signalled both servers the
        moment it returned, and the first rollout died on a closed WebSocket
        seconds later.
        """
        box: dict[str, subprocess.Popen] = {}
        worker = threading.Thread(target=lambda: box.update(proc=spawn_server(SLEEPER)))
        worker.start()
        worker.join(timeout=30)
        assert not worker.is_alive()

        proc = box["proc"]
        reaped.append(proc.pid)
        # Generous: the kernel delivers the parent-death signal immediately, so
        # anything this side of it is the bug.
        time.sleep(1.0)
        assert _alive(proc.pid), (
            "server was killed when the thread that spawned it exited; the "
            "fork must be issued from the process-lifetime spawner thread"
        )
        assert proc.poll() is None

    def test_spawn_failure_still_raises(self):
        """A bad executable must not be swallowed by the preexec hook."""
        with pytest.raises((FileNotFoundError, OSError)):
            spawn_server(["/nonexistent/definitely-not-a-binary"])


class TestTerminateServers:
    def test_terminates_a_live_server(self, reaped):
        proc = spawn_server(SLEEPER)
        reaped.append(proc.pid)
        terminate_servers([proc], grace=10.0)
        assert proc.poll() is not None

    def test_ignores_already_dead_servers(self, reaped):
        proc = spawn_server(SLEEPER)
        reaped.append(proc.pid)
        proc.kill()
        proc.wait(timeout=10)
        terminate_servers([proc], grace=1.0)  # must not raise

    def test_is_idempotent(self, reaped):
        proc = spawn_server(SLEEPER)
        reaped.append(proc.pid)
        terminate_servers([proc], grace=10.0)
        terminate_servers([proc], grace=1.0)
        assert proc.poll() is not None

    def test_sigkills_a_server_that_ignores_sigterm(self, reaped):
        """The grace period is generous, but it does expire."""
        proc = spawn_server(
            [
                sys.executable,
                "-c",
                "import signal, time;"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                "time.sleep(120)",
            ]
        )
        reaped.append(proc.pid)
        terminate_servers([proc], grace=1.0)
        assert _wait_gone(proc.pid)

    def test_grace_is_shared_not_per_server(self, reaped):
        """N stubborn servers must cost one grace period, not N."""
        procs = []
        for _ in range(3):
            p = spawn_server(
                [
                    sys.executable,
                    "-c",
                    "import signal, time;"
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
                    "time.sleep(120)",
                ]
            )
            procs.append(p)
            reaped.append(p.pid)
        start = time.monotonic()
        terminate_servers(procs, grace=2.0)
        assert time.monotonic() - start < 5.0
        for p in procs:
            assert _wait_gone(p.pid)

    def test_empty_list_is_a_no_op(self):
        terminate_servers([], grace=1.0)
