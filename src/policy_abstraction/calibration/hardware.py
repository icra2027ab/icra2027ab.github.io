"""The two devices calibration reads from, behind small protocols.

Both are *read-only*: the arm is only ever observed over RTDE's receive
interface, never commanded, and the camera is one RealSense opened by serial
through the same adapter the env server uses, so the intrinsics reported here
are the ones the server will publish.
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np


class ArmReader(Protocol):
    """Joint state and TCP pose of the arm, as the controller reports them."""

    def joints(self) -> np.ndarray: ...

    def joint_velocities(self) -> np.ndarray: ...

    def tcp_pose(self) -> np.ndarray:
        """``[x, y, z, rx, ry, rz]`` in the controller's ``base`` frame."""
        ...

    def close(self) -> None: ...


class CameraSource(Protocol):
    """A colour (+ optional registered depth) stream with known intrinsics."""

    width: int
    height: int

    def read(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """``(rgb uint8 HxWx3, depth_m float32 HxW or None)`` for one instant."""
        ...

    @property
    def intrinsics(self) -> np.ndarray:
        """3x3 of the colour stream at the streaming resolution."""
        ...

    def close(self) -> None: ...


class RtdeArm:
    """``rtde_receive`` only. Nothing here can move the robot."""

    def __init__(self, robot_ip: str) -> None:
        import rtde_receive  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]

        self.robot_ip = robot_ip
        self._rec = rtde_receive.RTDEReceiveInterface(robot_ip)
        if not self._rec.isConnected():
            raise ConnectionError(f"RTDE receive did not connect to {robot_ip}")

    def joints(self) -> np.ndarray:
        return np.asarray(self._rec.getActualQ(), dtype=np.float64)

    def joint_velocities(self) -> np.ndarray:
        return np.asarray(self._rec.getActualQd(), dtype=np.float64)

    def tcp_pose(self) -> np.ndarray:
        return np.asarray(self._rec.getActualTCPPose(), dtype=np.float64)

    def close(self) -> None:
        try:
            self._rec.disconnect()
        except Exception:  # pragma: no cover - depends on the binding
            pass


class RealSenseSource:
    """One RealSense, colour + depth registered to colour, by serial."""

    def __init__(
        self, serial: str, width: int, height: int, fps: int, *, depth: bool = True
    ) -> None:
        from policy_abstraction.envs.ur5.realsense import RealSenseCamera

        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self._cam = RealSenseCamera(
            self.serial, width=width, height=height, fps=fps, depth=depth
        )
        self._cam.connect()
        # The intrinsics are read off the first frame.
        self._cam.async_read()

    def read(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        rgb = self._cam.async_read()
        depth = self._cam.read_depth_m() if self._cam.depth else None
        return rgb, depth

    @property
    def intrinsics(self) -> np.ndarray:
        K = self._cam.intrinsics
        if K is None:
            self._cam.async_read()
            K = self._cam.intrinsics
        assert K is not None
        return np.asarray(K, dtype=np.float64)

    def close(self) -> None:
        self._cam.disconnect()


__all__ = ["ArmReader", "CameraSource", "RealSenseSource", "RtdeArm"]
