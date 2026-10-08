"""In-process stand-in for the UR5e driver, for running with the arm powered off.

:class:`FakeUr5Driver` implements the small slice of the
``lerobot_robot_ur5e.UR5E`` surface that
:class:`policy_abstraction.envs.ur5.server.Ur5EnvServer` touches:
``get_observation``, ``send_action``, ``home``, ``disconnect``,
``is_connected``, ``rtde_ctrl`` and ``rtde_rec``.

It exists so the whole pipeline — env server, WebSocket protocol, episode
runner, policy client — can be exercised end-to-end before anyone powers on the
robot, and so :mod:`tests` can cover the adapter without hardware. Commanded
joint targets are applied instantly (a perfect servo), the TCP pose comes from
the closed-form UR5e forward kinematics in
:mod:`policy_abstraction.dynamics.ur5_kinematics` composed with the configured
``tcp_offset`` — exactly what RTDE's ``getActualTCPPose()`` reports on the real
controller, so the runner's FK-predicted chunks and the published end-effector
agree — and camera frames are synthetic gradients so image plumbing bugs are
still visible.

Enable it with ``serve_ur5_env.py --simulate-hardware``. It is *not* a
simulator: nothing about the scene, contact or object motion is modelled.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from policy_abstraction.dynamics.ur5_kinematics import ur5e_forward_kinematics
from policy_abstraction.envs.ur5.config import UR5E_HOME_JOINT_POSITIONS


class _FakeRtdeReceive:
    """Minimal ``rtde_receive.RTDEReceiveInterface`` stand-in."""

    def __init__(self, driver: "FakeUr5Driver") -> None:
        self._driver = driver

    def getActualQ(self) -> list[float]:  # noqa: N802 — mirrors the RTDE API
        return [float(v) for v in self._driver.joints]

    def getActualQd(self) -> list[float]:  # noqa: N802
        return [float(v) for v in self._driver.joint_velocities]

    def getActualTCPPose(self) -> list[float]:  # noqa: N802
        pos, rotvec = ur5e_forward_kinematics(
            self._driver.joints, self._driver.tcp_offset
        )
        return [*(float(v) for v in pos), *(float(v) for v in rotvec)]

    def isConnected(self) -> bool:  # noqa: N802
        return True


class _FakeRtdeControl:
    """Minimal ``rtde_control.RTDEControlInterface`` stand-in."""

    def __init__(self, driver: "FakeUr5Driver") -> None:
        self._driver = driver

    def servoStop(self) -> None:  # noqa: N802
        self._driver.joint_velocities[:] = 0.0

    def getTCPOffset(self) -> list[float]:  # noqa: N802
        """The offset the fake "controller" is configured with — what the
        server's pre-flight compares ``--tcp-offset`` against."""
        off = self._driver.tcp_offset
        return [0.0] * 6 if off is None else [float(v) for v in off]

    def servoJ(self, q, *_args, **_kwargs) -> None:  # noqa: N802
        self._driver.set_joints(q)

    def moveJ(self, q, *_args, **_kwargs) -> bool:  # noqa: N802
        self._driver.set_joints(q)
        return True

    def isConnected(self) -> bool:  # noqa: N802
        return True

    def disconnect(self) -> None:
        pass


class FakeUr5Driver:
    """Hardware-free stand-in for ``lerobot_robot_ur5e.UR5E``."""

    def __init__(
        self,
        camera_specs: list[dict],
        control_period: float,
        seed: int = 0,
        tcp_offset=None,
    ) -> None:
        self.joints = np.array(UR5E_HOME_JOINT_POSITIONS, dtype=np.float64)
        # ``[x, y, z, rx, ry, rz]`` as the UR pendant shows it; ``None`` reports
        # the bare flange pose, as a controller with no TCP configured would.
        self.tcp_offset = None if tcp_offset is None else [float(v) for v in tcp_offset]
        self.joint_velocities = np.zeros(6, dtype=np.float64)
        # Driver convention: 0 = open, 1 = closed.
        self.gripper = 0.0
        self._control_period = float(control_period)
        self._camera_specs = camera_specs
        self._rng = np.random.default_rng(seed)
        self._frame_idx = 0
        self.rtde_rec: Any = _FakeRtdeReceive(self)
        self.rtde_ctrl: Any = _FakeRtdeControl(self)

    # -- lifecycle ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def home(self) -> None:
        self.set_joints(UR5E_HOME_JOINT_POSITIONS)
        self.gripper = 0.0

    # -- state --------------------------------------------------------------

    def set_joints(self, q) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(6)
        self.joint_velocities = (q - self.joints) / max(self._control_period, 1e-6)
        self.joints = q

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.set_joints([float(action[f"joint_{i}"]) for i in range(6)])
        self.gripper = float(np.clip(action["gripper"], 0.0, 1.0))
        return action

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {f"joint_{i}": float(self.joints[i]) for i in range(6)}
        obs["gripper"] = float(self.gripper)
        for spec in self._camera_specs:
            obs[spec["name"]] = self._fake_frame(spec)
        self._frame_idx += 1
        return obs

    # -- synthetic frames ---------------------------------------------------

    def _fake_frame(self, spec: dict) -> np.ndarray:
        """A moving colour gradient, so orientation/keying bugs stay visible."""
        h, w = int(spec["height"]), int(spec["width"])
        ys = np.linspace(0, 255, h, dtype=np.float32)[:, None]
        xs = np.linspace(0, 255, w, dtype=np.float32)[None, :]
        phase = (self._frame_idx * 4) % 256
        frame = np.empty((h, w, 3), dtype=np.uint8)
        frame[..., 0] = np.clip(xs + phase, 0, 255).astype(np.uint8)
        frame[..., 1] = np.clip(ys, 0, 255).astype(np.uint8)
        frame[..., 2] = np.uint8(abs(hash(spec["name"])) % 256)
        return frame


__all__ = ["FakeUr5Driver"]
