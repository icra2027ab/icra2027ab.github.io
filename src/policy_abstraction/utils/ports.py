"""Who is listening on a TCP port.

A launcher that spawns a server and then connects by port has a silent
failure mode: if the port is already held, the child dies on "address
already in use" (into a log file nobody reads) and the client connects to
the *stranger* instead. Everything then appears to work while running
against another process's configuration — a different checkout, a different
calibration, a different robot.

The cure is to look before spawning and name the holder, the same way
:func:`~policy_abstraction.envs.ur5.realsense.video_node_holders` names
whoever has the camera.
"""

from __future__ import annotations

import glob
import os
import socket

__all__ = ["port_listener", "assert_port_free"]


def _inode_to_pid_cmdline(inode: int) -> tuple[int, str] | None:
    """Find the process holding socket *inode*, as ``(pid, cmdline)``."""
    target = f"socket:[{inode}]"
    for fd_path in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            if os.readlink(fd_path) != target:
                continue
        except OSError:
            continue  # process exited, or not ours to inspect
        pid = int(fd_path.split("/")[2])
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            cmdline = "<unknown>"
        return pid, cmdline.strip()
    return None


def _listening_inode(port: int) -> int | None:
    """Socket inode of the LISTEN socket on *port*, if any.

    Reads ``/proc/net/tcp{,6}`` directly rather than shelling out to ``ss``,
    which is not installed everywhere this runs.
    """
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as fh:
                next(fh, None)  # header
                for line in fh:
                    parts = line.split()
                    if len(parts) < 10 or parts[3] != "0A":  # 0A == TCP_LISTEN
                        continue
                    if int(parts[1].rsplit(":", 1)[1], 16) != port:
                        continue
                    return int(parts[9])
        except (OSError, ValueError, IndexError):
            continue
    return None


def port_listener(port: int) -> str | None:
    """``"<pid> <cmdline>"`` for whoever is listening on *port*, else ``None``.

    Falls back to a bare ``"<unknown process>"`` when something holds the
    port but the owning process is not ours to inspect — the caller still
    needs to know the port is taken.
    """
    inode = _listening_inode(port)
    if inode is None:
        return None
    found = _inode_to_pid_cmdline(inode)
    if found is None:
        return "<unknown process>"
    pid, cmdline = found
    return f"{pid} {cmdline}"


def assert_port_free(port: int, what: str, attach_flag: str) -> None:
    """Raise if *port* is already served, naming the holder and the way out.

    Parameters
    ----------
    what:
        Human name for the server being started, e.g. ``"UR5 env server"``.
    attach_flag:
        The flag that deliberately attaches to a running server instead of
        spawning one, quoted in the error so the operator has both choices.
    """
    holder = port_listener(port)
    if holder is None:
        return
    raise RuntimeError(
        f"Cannot start the {what}: port {port} is already served by\n"
        f"  {holder}\n"
        f"The spawned server would fail to bind and this run would silently "
        f"attach to that process instead, using *its* configuration rather "
        f"than the flags passed here. Stop it first, or pass {attach_flag} to "
        f"attach to it on purpose."
    )


def is_port_open(host: str, port: int, timeout: float = 0.25) -> bool:
    """Whether something accepts connections at ``host:port`` right now."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
