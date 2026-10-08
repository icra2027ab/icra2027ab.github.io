"""Moving the arm through the capture poses itself, inside a box you set.

Freedrive works, but it asks the operator to invent forty varied poses while
watching a screen. Here the poses are planned: positions stratified over a
**safe box** given in the robot base frame, orientations jittered around the
pose the arm was left in (exterior camera) or aimed at the reference mark
(wrist camera). The arm is driven with slow linear moves and waits at each
pose so a still frame is taken.

Safety is layered, because this is the one place the tool commands the robot:

* every target is inside the box by construction, and is rejected unless the
  controller's own ``isPoseWithinSafetyLimits`` accepts it;
* the arm must already be inside the box when motion starts;
* the live TCP is checked against the box (plus a small margin) on every poll
  while motion is active, and a violation stops the arm immediately;
* moves are asynchronous so a stop request is honoured mid-move, and the
  speed and acceleration are capped;
* the GUI's red *Stop motion* button and ``Esc`` call :meth:`Mover.stop`.

None of this replaces the e-stop in the operator's hand.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Protocol

import numpy as np

from policy_abstraction.calibration.geometry import (
    CalibrationMode,
    matrix_to_pose,
    pose_to_matrix,
)
from policy_abstraction.dynamics.ur5_kinematics import rotvec_to_matrix

MAX_SPEED = 0.25
"""m/s. The UR5e's reduced-mode Cartesian limit; the default is well below."""
MAX_ACCEL = 0.8
"""m/s^2."""


@dataclasses.dataclass
class SafeBox:
    """Axis-aligned box in the robot base frame, metres."""

    x: tuple[float, float]
    y: tuple[float, float]
    z: tuple[float, float]

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            lo, hi = getattr(self, name)
            if not hi > lo:
                raise ValueError(f"box {name} range must be increasing, got {lo}..{hi}")
            setattr(self, name, (float(lo), float(hi)))

    @classmethod
    def around(
        cls,
        position,
        *,
        half_xy: float = 0.15,
        below: float = 0.0,
        above: float = 0.20,
    ) -> "SafeBox":
        """A box centred on *position* in x/y and rising from it in z.

        ``below`` defaults to zero on purpose: the operator parks the arm at a
        height they know is clear of the table, and the default box must not
        plan anything under that height."""
        p = np.asarray(position, dtype=np.float64).reshape(3)
        return cls(
            (p[0] - half_xy, p[0] + half_xy),
            (p[1] - half_xy, p[1] + half_xy),
            (p[2] - below, p[2] + above),
        )

    @property
    def lo(self) -> np.ndarray:
        return np.array([self.x[0], self.y[0], self.z[0]])

    @property
    def hi(self) -> np.ndarray:
        return np.array([self.x[1], self.y[1], self.z[1]])

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) / 2

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo

    def contains(self, p, margin: float = 0.0) -> bool:
        q = np.asarray(p, dtype=np.float64).reshape(-1)[:3]
        return bool(np.all(q >= self.lo - margin) and np.all(q <= self.hi + margin))

    def to_dict(self) -> dict:
        return {"x": list(self.x), "y": list(self.y), "z": list(self.z)}

    @classmethod
    def from_dict(cls, d: dict) -> "SafeBox":
        return cls(tuple(d["x"]), tuple(d["y"]), tuple(d["z"]))  # type: ignore[arg-type]


@dataclasses.dataclass
class MotionConfig:
    box: SafeBox
    n_targets: int = 40
    speed: float = 0.08
    """m/s, linear."""
    accel: float = 0.3
    """m/s^2."""
    tilt_deg: float = 20.0
    """Random tilt of the tool axis about the nominal orientation (exterior),
    or of the aim around the mark (wrist)."""
    roll_deg: float = 30.0
    """Random rotation about the tool axis."""
    dwell_s: float = 1.0
    """Time to hold at each pose once the arm is still."""
    margin_m: float = 0.02
    """How far outside the box the live TCP may stray before motion is stopped
    (controller overshoot, TCP-offset differences)."""
    move_timeout_s: float = 30.0
    min_move_s: float = 0.3
    """A move is never considered finished sooner than this after it was
    issued; the controller reports an asynchronous move as idle for a cycle
    or two before it starts."""
    seed: int = 0

    def __post_init__(self) -> None:
        self.speed = float(min(max(self.speed, 0.005), MAX_SPEED))
        self.accel = float(min(max(self.accel, 0.01), MAX_ACCEL))
        self.n_targets = int(self.n_targets)


def _rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _stratified_positions(
    box: SafeBox, n: int, rng: np.random.Generator, margin: float
) -> np.ndarray:
    """*n* points spread over the box: one per cell of a coarse grid, shuffled,
    then jittered inside the cell — so forty poses cover the volume instead of
    clumping where a uniform draw happens to land."""
    lo, hi = box.lo + margin, box.hi - margin
    if np.any(hi <= lo):
        raise ValueError("safe box is smaller than twice the margin")
    per_axis = max(1, int(math.ceil(n ** (1 / 3))))
    cells = np.array(
        [
            (i, j, k)
            for i in range(per_axis)
            for j in range(per_axis)
            for k in range(per_axis)
        ]
    )
    rng.shuffle(cells)
    reps = int(math.ceil(n / len(cells)))
    cells = np.concatenate([cells] * reps)[:n]
    u = (cells + rng.uniform(0, 1, size=cells.shape)) / per_axis
    return lo + u * (hi - lo)


def _look_at_tool(
    position: np.ndarray, target: np.ndarray, nominal_R: np.ndarray
) -> np.ndarray:
    """Tool rotation whose z axis points from *position* at *target*, keeping
    the tool's x axis as close to the nominal one as the geometry allows."""
    z = target - position
    norm = np.linalg.norm(z)
    if norm < 1e-6:
        return nominal_R.copy()
    z = z / norm
    x = nominal_R[:, 0] - np.dot(nominal_R[:, 0], z) * z
    if np.linalg.norm(x) < 1e-6:
        x = nominal_R[:, 1] - np.dot(nominal_R[:, 1], z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def _order_by_travel(targets: list[np.ndarray], start: np.ndarray) -> list[np.ndarray]:
    """Greedy nearest-neighbour tour, so the arm does not zigzag across the box."""
    remaining = list(targets)
    ordered: list[np.ndarray] = []
    cur = np.asarray(start[:3])
    while remaining:
        i = int(np.argmin([np.linalg.norm(t[:3] - cur) for t in remaining]))
        ordered.append(remaining.pop(i))
        cur = ordered[-1][:3]
    return ordered


def plan_targets(
    config: MotionConfig,
    nominal_pose,
    mode: CalibrationMode,
    *,
    reference_point_base=None,
    margin: float = 0.01,
) -> list[np.ndarray]:
    """Capture poses (``[x, y, z, rx, ry, rz]``) inside the box.

    *nominal_pose* is where the arm is now. For an exterior camera its
    orientation is kept, jittered by ``tilt_deg`` about the tool's x and y
    axes and ``roll_deg`` about z, so the fingertips keep facing roughly the
    same way (and stay visible). For a wrist camera the tool axis is aimed at
    the reference mark from each position, with the aim point jittered by
    ``tilt_deg`` and a random roll, so the mark stays in the image.
    """
    rng = np.random.default_rng(config.seed)
    T0 = pose_to_matrix(nominal_pose)
    R0 = T0[:3, :3]
    positions = _stratified_positions(config.box, config.n_targets, rng, margin)
    tilt = math.radians(config.tilt_deg)
    roll = math.radians(config.roll_deg)
    out: list[np.ndarray] = []
    for p in positions:
        if mode is CalibrationMode.EXTERIOR:
            R = R0 @ _rot_x(rng.uniform(-tilt, tilt)) @ _rot_y(rng.uniform(-tilt, tilt))
            R = R @ _rot_z(rng.uniform(-roll, roll))
        else:
            if reference_point_base is None:
                raise ValueError("wrist motion planning needs the reference mark")
            ref = np.asarray(reference_point_base, dtype=np.float64)
            dist = float(np.linalg.norm(ref - p))
            # Aim a little off the mark so the views differ in angle, not just
            # in position; the offset scales with distance to keep it ~tilt.
            aim = ref + rng.normal(0.0, math.tan(tilt) * dist / 2, size=3)
            R = _look_at_tool(p, aim, R0) @ _rot_z(rng.uniform(-roll, roll))
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        out.append(matrix_to_pose(T))
    return _order_by_travel(out, np.asarray(nominal_pose, dtype=np.float64))


class Mover(Protocol):
    """The slice of the controller that motion needs."""

    def move_to(self, pose, speed: float, accel: float) -> bool:
        """Start a linear move; return False if the controller refused it."""
        ...

    def is_moving(self) -> bool: ...

    def stop(self) -> None:
        """Decelerate to a stop now."""
        ...

    def pose_ok(self, pose) -> bool:
        """Whether the controller would accept *pose* (reachable, inside its
        own safety limits)."""
        ...

    def close(self) -> None: ...


class RtdeMover:
    """``rtde_control`` with asynchronous ``moveL``.

    Connecting takes over the controller (the robot must be in *Remote
    Control* on the pendant, not in freedrive), so the session only creates
    this when autonomous capture is actually started.
    """

    def __init__(self, robot_ip: str) -> None:
        import rtde_control  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]

        self.robot_ip = robot_ip
        self._c = rtde_control.RTDEControlInterface(robot_ip)
        if not self._c.isConnected():
            raise ConnectionError(
                f"RTDE control did not connect to {robot_ip}. Is the robot in Remote "
                "Control mode and out of freedrive?"
            )

    def tcp_offset(self) -> list[float]:
        return [float(v) for v in self._c.getTCPOffset()]

    def move_to(self, pose, speed: float, accel: float) -> bool:
        speed = float(min(speed, MAX_SPEED))
        accel = float(min(accel, MAX_ACCEL))
        return bool(self._c.moveL([float(v) for v in pose], speed, accel, True))

    def is_moving(self) -> bool:
        # Negative once no asynchronous operation is running.
        return int(self._c.getAsyncOperationProgress()) >= 0

    def stop(self) -> None:
        try:
            self._c.stopL(0.5)
        except Exception:  # pragma: no cover - the binding raises on a dead link
            pass

    def pose_ok(self, pose) -> bool:
        try:
            return bool(self._c.isPoseWithinSafetyLimits([float(v) for v in pose]))
        except Exception:
            return False

    def close(self) -> None:
        try:
            self.stop()
            self._c.disconnect()
        except Exception:  # pragma: no cover
            pass


def pose_error(pose_a, pose_b) -> tuple[float, float]:
    """(metres, radians) between two poses."""
    Ta, Tb = pose_to_matrix(pose_a), pose_to_matrix(pose_b)
    dR = Ta[:3, :3].T @ Tb[:3, :3]
    ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
    return float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3])), ang


__all__ = [
    "MAX_ACCEL",
    "MAX_SPEED",
    "MotionConfig",
    "Mover",
    "RtdeMover",
    "SafeBox",
    "plan_targets",
    "pose_error",
    "rotvec_to_matrix",
]
