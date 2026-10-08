"""Spawning long-lived server children that cannot outlive their parent.

Every backend here starts its servers with ``start_new_session=True``, which
puts the child in its own session so a Ctrl-C in the terminal cannot reach it
mid-motion — the parent is supposed to orchestrate an orderly SIGTERM
instead. That is right for the normal path and wrong for every other one:
if the parent dies without running its cleanup (SIGKILL, a hard crash, the
terminal closing, an exception raised between two spawns) nothing signals
the child at all, and it survives indefinitely.

An orphaned server is not merely idle. It holds the port, so the next run's
child fails to bind and the client silently attaches to the *stranger* —
running the whole evaluation against a stale checkout's configuration. On
the UR5 it also holds the cameras and an RTDE session the controller will
not hand out twice.

:func:`spawn_server` closes that gap with ``PR_SET_PDEATHSIG``: the kernel
signals the child when its parent dies, whatever killed the parent. The
child then runs its own shutdown — parking the arm, releasing RTDE and the
cameras — instead of becoming a ghost.

The leash has one sharp edge, and it is the reason for :func:`_run_on_spawner`
below: PR_SET_PDEATHSIG's "parent" is the *thread* that forked the child, not
the process. The kernel fires it when that thread exits, even though the
process is still very much alive. Spawning a server from any short-lived
worker — the operator GUI builds its whole session on one, so the window can
be up while a checkpoint loads — therefore killed the server seconds after it
came up, and the run then failed on a closed WebSocket. Every fork is issued
from one dedicated thread that lives as long as the process, so "parent
thread exits" and "process dies" are the same event again.
"""

from __future__ import annotations

import ctypes
import os
import queue
import signal
import subprocess
import sys
import threading
from typing import Any, Callable, Sequence

from loguru import logger

__all__ = ["spawn_server", "terminate_servers"]

# From <linux/prctl.h>. Not exposed by the stdlib.
_PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    """Ask the kernel to SIGTERM this child when its parent dies.

    Runs in the forked child between ``fork`` and ``exec``. The flag survives
    ``exec`` (it is cleared only for setuid/setgid binaries) and survives the
    ``setsid`` that ``start_new_session`` already performed, so the child
    keeps its own session *and* its leash.
    """
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        # Best effort: a child without a leash is the old behaviour, not a
        # reason to fail the run.
        return
    # The parent can die in the window between fork and prctl, in which case
    # the signal we just armed will never arrive. Re-parenting to init is how
    # that shows up.
    if os.getppid() == 1:
        os._exit(1)


# The one thread every server is forked from. Created on first use, daemon, and
# never joined: its whole job is to outlive every caller, so that the
# parent-death signal armed in `_die_with_parent` means "the process died" and
# nothing else. Deliberately a bare daemon thread rather than a
# ThreadPoolExecutor, whose workers are joined by an atexit hook -- that join
# is a thread exit *during* interpreter shutdown, which is precisely the event
# we are trying not to conflate with the process dying. A single worker also
# serialises the forks, which is what we want: `preexec_fn` runs between fork
# and exec, where very little is safe to do.
_spawner_lock = threading.Lock()
_spawner_queue: "queue.SimpleQueue[tuple[Callable[[], Any], _Result]]" = (
    queue.SimpleQueue()
)
_spawner_started = False


class _Result:
    """A one-shot slot for the spawner thread's return value or exception."""

    def __init__(self) -> None:
        self.done = threading.Event()
        self.value: Any = None
        self.error: BaseException | None = None

    def get(self) -> Any:
        self.done.wait()
        if self.error is not None:
            raise self.error
        return self.value


def _spawner_loop() -> None:
    while True:
        work, result = _spawner_queue.get()
        try:
            result.value = work()
        except BaseException as exc:  # noqa: BLE001 — handed to the caller
            result.error = exc
        finally:
            result.done.set()


def _run_on_spawner(work: "Callable[[], Any]") -> Any:
    """Run *work* on the process-lifetime spawner thread and return its result."""
    global _spawner_started
    with _spawner_lock:
        if not _spawner_started:
            threading.Thread(
                target=_spawner_loop, name="server-spawner", daemon=True
            ).start()
            _spawner_started = True
    result = _Result()
    _spawner_queue.put((work, result))
    return result.get()


def spawn_server(
    cli_args: Sequence[str],
    *,
    env: dict | None = None,
    cwd: str | os.PathLike | None = None,
    stdout: Any = None,
    stderr: Any = None,
    text: bool = True,
) -> subprocess.Popen:
    """``Popen`` a server child that dies with this process.

    Same contract as ``subprocess.Popen`` for the arguments it takes; adds
    its own session (so a terminal Ctrl-C does not interrupt the server
    mid-motion) and a parent-death signal (so it cannot be orphaned).

    The fork itself is handed to :func:`_run_on_spawner` rather than run on the
    calling thread — see the module docstring for why that distinction is
    load-bearing. The returned ``Popen`` is an ordinary one and is used from
    whichever thread asked for it.
    """
    args = list(cli_args)
    return _run_on_spawner(
        lambda: _popen_server(
            args,
            env=env,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            text=text,
        )
    )


def _popen_server(
    cli_args: list[str],
    *,
    env: dict | None,
    cwd: str | os.PathLike | None,
    stdout: Any,
    stderr: Any,
    text: bool,
) -> subprocess.Popen:
    """The actual fork. Runs only on the spawner thread."""
    return subprocess.Popen(
        cli_args,
        env=env,
        cwd=str(cwd) if cwd is not None else None,
        stdout=stdout,
        stderr=stderr,
        text=text,
        start_new_session=True,
        preexec_fn=_die_with_parent,  # noqa: PLW1509 - see module docstring
    )


def terminate_servers(processes: Sequence[Any], grace: float = 15.0) -> None:
    """SIGTERM each live server, then SIGKILL whatever is left after *grace*.

    The grace period is generous on purpose: a hardware server's SIGTERM
    handler parks the arm with a blocking ``moveJ`` before it releases the
    RTDE session, and killing it partway through leaves the session
    unreleased — which the controller then refuses to re-open.

    Signals the whole process group, since a server may have children of its
    own. Safe to call more than once; already-dead processes are skipped.
    """
    signalled: list[Any] = []
    for proc in processes:
        if proc.poll() is not None:
            continue
        logger.info(f"Terminating server PID {proc.pid}")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            signalled.append(proc)
        except (ProcessLookupError, PermissionError):
            pass

    # Wait on all of them together rather than serially, so N servers cost one
    # grace period instead of N.
    deadline = _monotonic() + grace
    for proc in signalled:
        try:
            proc.wait(timeout=max(0.0, deadline - _monotonic()))
        except Exception:
            logger.warning(
                f"Server PID {proc.pid} did not exit within {grace:.0f}s of "
                "SIGTERM; force-killing. If this is the arm's server, check "
                "that it parked and released RTDE."
            )
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def _monotonic() -> float:
    import time

    return time.monotonic()
