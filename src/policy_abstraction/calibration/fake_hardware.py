"""Stand-ins for the arm, the camera and the tracker, for runs without the cell.

The fake camera renders a saturated magenta marker at the true projection of
the tracked point, computed from a *hidden* ground-truth extrinsic that only
this module knows. The fake tracker finds that marker by colour, frame by
frame, so the click -> track -> solve path is exercised honestly: the solver
never sees the truth, and a test can compare what it recovers against it.

``FakeArm`` drives :class:`~policy_abstraction.envs.ur5.fake_driver.FakeUr5Driver`
along a scripted freedrive — move to a random target, pause, repeat — so the
session's settle detection has something to detect.
"""

from __future__ import annotations

import dataclasses
from typing import Callable, Optional

import cv2
import numpy as np

from policy_abstraction.calibration.dataset import KalibDataset
from policy_abstraction.calibration.geometry import (
    CalibrationMode,
    keypoint_in_parent,
    pose_to_matrix,
    project,
)
from policy_abstraction.envs.ur5.config import UR5E_HOME_JOINT_POSITIONS
from policy_abstraction.envs.ur5.fake_driver import FakeUr5Driver

MARKER_RGB = (255, 0, 200)
MARK_RGB = (0, 220, 255)
"""Colour of fixed table marks the fake exterior camera renders."""


class FakeArm:
    """A fake arm that either freedrives itself or obeys commanded moves.

    Scripted mode (default) glides to random nearby joint targets and holds,
    like an operator in freedrive. Once :meth:`move_to` is called it becomes
    a :class:`~policy_abstraction.calibration.motion.Mover`: a small numerical
    IK turns the Cartesian target into joints and ``tick`` glides there.
    """

    def __init__(
        self,
        tcp_offset,
        *,
        seed: int = 0,
        step_rad: float = 0.02,
        hold_ticks: int = 12,
        spread_rad: float = 0.45,
    ) -> None:
        self._driver = FakeUr5Driver([], control_period=0.05, tcp_offset=tcp_offset)
        self._tcp_offset = tcp_offset
        self._rng = np.random.default_rng(seed)
        self._home = np.asarray(UR5E_HOME_JOINT_POSITIONS, dtype=np.float64)
        self._target = self._home.copy()
        self._step = float(step_rad)
        self._hold_ticks = int(hold_ticks)
        self._hold = 0
        self._spread = float(spread_rad)
        self.scripted = True
        self.commands = 0
        self.ticks = 0
        self._cart_target: Optional[np.ndarray] = None
        self.step_m = 0.01
        """Cartesian step per tick while executing a commanded move."""
        self.step_rad_cart = 0.05

    def tick(self) -> None:
        """Advance the motion by one poll."""
        self.ticks += 1
        if self._cart_target is not None:
            self._tick_cartesian()
            return
        q = self._driver.joints
        delta = self._target - q
        dist = float(np.abs(delta).max())
        if dist < 1e-9:
            self._driver.joint_velocities[:] = 0.0
            if not self.scripted:
                return
            if self._hold < self._hold_ticks:
                self._hold += 1
                return
            self._hold = 0
            self._target = self._home + self._rng.uniform(
                -self._spread, self._spread, size=6
            )
            return
        move = np.clip(delta, -self._step, self._step)
        self._driver.set_joints(q + move)

    def _tick_cartesian(self) -> None:
        """One step of a straight-line move, like the controller's ``moveL``."""
        from policy_abstraction.calibration.geometry import pose_to_matrix
        from policy_abstraction.dynamics.ur5_kinematics import (
            matrix_to_rotvec,
            rotvec_to_matrix,
        )

        assert self._cart_target is not None
        cur = self.tcp_pose()
        T_cur, T_tgt = pose_to_matrix(cur), pose_to_matrix(self._cart_target)
        dp = T_tgt[:3, 3] - T_cur[:3, 3]
        dist = float(np.linalg.norm(dp))
        drv = matrix_to_rotvec(T_tgt[:3, :3] @ T_cur[:3, :3].T)
        ang = float(np.linalg.norm(drv))
        if dist < 2e-4 and ang < 2e-3:
            q = self.solve_ik(self._cart_target, q0=self._driver.joints, iters=50)
            if q is not None:
                self._driver.set_joints(q)
            self._driver.joint_velocities[:] = 0.0
            self._target = self._driver.joints.copy()
            self._cart_target = None
            return
        frac = min(
            1.0,
            self.step_m / dist if dist > 1e-9 else 1.0,
            self.step_rad_cart / ang if ang > 1e-9 else 1.0,
        )
        T = np.eye(4)
        T[:3, :3] = rotvec_to_matrix(frac * drv) @ T_cur[:3, :3]
        T[:3, 3] = T_cur[:3, 3] + frac * dp
        from policy_abstraction.calibration.geometry import matrix_to_pose

        q = self.solve_ik(matrix_to_pose(T), q0=self._driver.joints, iters=50)
        if q is None:
            self.stop()
            return
        self._driver.set_joints(q)
        self._target = q.copy()

    # -- Mover ------------------------------------------------------------------

    def solve_ik(self, pose, q0=None, iters: int = 200) -> Optional[np.ndarray]:
        """Damped least-squares IK on the UR5e forward kinematics, or None."""
        from policy_abstraction.calibration.geometry import pose_to_matrix
        from policy_abstraction.dynamics.ur5_kinematics import (
            matrix_to_rotvec,
            ur5e_forward_kinematics,
        )

        target = pose_to_matrix(pose)
        q = np.asarray(
            self._driver.joints if q0 is None else q0, dtype=np.float64
        ).copy()

        def err(qq):
            pos, rv = ur5e_forward_kinematics(qq, self._tcp_offset)
            R = pose_to_matrix(np.concatenate([pos, rv]))[:3, :3]
            dR = target[:3, :3] @ R.T
            return np.concatenate([target[:3, 3] - pos, matrix_to_rotvec(dR)])

        for _ in range(iters):
            e = err(q)
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e[3:]) < 2e-3:
                return q
            J = np.zeros((6, 6))
            h = 1e-6
            for j in range(6):
                dq = np.zeros(6)
                dq[j] = h
                J[:, j] = (err(q + dq) - e) / (-h)
            # J = d(pose)/dq, and e = target - pose, so J dq = e (damped).
            dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(6), J.T @ e)
            q = q + np.clip(dq, -0.3, 0.3)
        return None

    def pose_ok(self, pose) -> bool:
        return self.solve_ik(pose) is not None

    def move_to(self, pose, speed: float, accel: float) -> bool:
        if self.solve_ik(pose) is None:
            return False
        self.scripted = False
        self._cart_target = np.asarray(pose, dtype=np.float64).copy()
        self.commands += 1
        return True

    def is_moving(self) -> bool:
        return self._cart_target is not None

    def stop(self) -> None:
        self._cart_target = None
        self._target = self._driver.joints.copy()
        self._driver.joint_velocities[:] = 0.0

    def joints(self) -> np.ndarray:
        return np.asarray(self._driver.rtde_rec.getActualQ(), dtype=np.float64)

    def joint_velocities(self) -> np.ndarray:
        return np.asarray(self._driver.rtde_rec.getActualQd(), dtype=np.float64)

    def tcp_pose(self) -> np.ndarray:
        return np.asarray(self._driver.rtde_rec.getActualTCPPose(), dtype=np.float64)

    def close(self) -> None:
        pass


@dataclasses.dataclass
class FakeTruth:
    """The hidden answer the fake camera renders with."""

    mode: CalibrationMode
    T_parent_cam: np.ndarray
    """Camera pose in the parent frame (base for exterior, TCP for wrist)."""
    reference_point_base: Optional[np.ndarray] = None
    point_offset_tcp: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def T_cam_parent(self) -> np.ndarray:
        return np.linalg.inv(self.T_parent_cam)


def look_at(position, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera pose (OpenCV axes: z forward, y down) at *position* looking at *target*."""
    pos = np.asarray(position, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - pos
    z /= np.linalg.norm(z)
    up = np.asarray(up, dtype=np.float64)
    x = np.cross(z, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, y, z
    T[:3, 3] = pos
    return T


def default_truth(mode: CalibrationMode, home_tcp_pose) -> FakeTruth:
    """A plausible camera for each mounting, aimed at the arm's starting pose so
    the tracked point is in the very first frame (which is where it is clicked)."""
    T_base_tcp = pose_to_matrix(home_tcp_pose)
    tcp = T_base_tcp[:3, 3]
    if mode is CalibrationMode.EXTERIOR:
        # ~0.9 m away, above the table, looking down at the workspace.
        cam_pos = tcp + np.array([0.35, -0.75, 0.45])
        return FakeTruth(mode=mode, T_parent_cam=look_at(cam_pos, tcp))
    # Wrist: camera a few cm off the TCP looking along the tool's z axis; the
    # reference mark sits 35 cm ahead of the tool at the starting pose.
    ahead = T_base_tcp[:3, :3] @ np.array([0.0, 0.0, 0.35]) + tcp
    T_tcp_cam = look_at([0.0, -0.06, -0.12], [0.0, 0.0, 0.35], up=(0.0, -1.0, 0.0))
    return FakeTruth(mode=mode, T_parent_cam=T_tcp_cam, reference_point_base=ahead)


class FakeCamera:
    """Renders the tracked point as a magenta marker over a textured background."""

    def __init__(
        self,
        arm: FakeArm,
        truth: FakeTruth,
        *,
        width: int = 640,
        height: int = 480,
        seed: int = 0,
        pixel_noise: float = 0.0,
        on_read: Optional[Callable[[], None]] = None,
        table_marks_base: Optional[list[np.ndarray]] = None,
    ) -> None:
        self._arm = arm
        self.truth = truth
        self._on_read = on_read
        self.table_marks_base = [
            np.asarray(m, dtype=np.float64) for m in (table_marks_base or [])
        ]
        """Fixed points in the base frame drawn as cyan discs (exterior mode),
        so a table mark can be clicked in the fake exterior image."""
        """Called before every frame — the launcher advances the fake freedrive
        here, so the scripted arm moves at the GUI's polling rate."""
        self.width = int(width)
        self.height = int(height)
        self._rng = np.random.default_rng(seed)
        self._noise = float(pixel_noise)
        f = 0.95 * self.width
        self._K = np.array(
            [[f, 0.0, self.width / 2 - 3.5], [0.0, f, self.height / 2 + 2.0], [0, 0, 1]]
        )
        # A fixed random texture so frames are not featureless.
        tex = self._rng.integers(
            60, 120, size=(height // 8, width // 8), dtype=np.uint8
        )
        self._background = cv2.resize(
            tex, (width, height), interpolation=cv2.INTER_NEAREST
        )

    @property
    def intrinsics(self) -> np.ndarray:
        return self._K.copy()

    def _keypoint_parent(self) -> np.ndarray:
        return keypoint_in_parent(
            self.truth.mode,
            self._arm.tcp_pose(),
            point_offset_tcp=self.truth.point_offset_tcp,
            reference_point_base=self.truth.reference_point_base,
        )

    def true_pixel(self) -> tuple[np.ndarray, float]:
        uv, z = project(self._K, self.truth.T_cam_parent, self._keypoint_parent()[None])
        return uv[0], float(z[0])

    def read(self) -> tuple[np.ndarray, Optional[np.ndarray]]:
        if self._on_read is not None:
            self._on_read()
        uv, z = self.true_pixel()
        rgb = np.repeat(self._background[..., None], 3, axis=2).copy()
        depth = np.full((self.height, self.width), np.nan, dtype=np.float32)
        if np.all(np.isfinite(uv)) and z > 0:
            if self._noise > 0:
                uv = uv + self._rng.normal(0.0, self._noise, size=2)
            # Depth: a gentle plane through the point so the sample at the
            # marker is exact and its neighbourhood plausible.
            vv = np.arange(self.height, dtype=np.float32)[:, None]
            depth[:] = z + 0.0005 * (vv - float(uv[1]))
            centre = (int(round(float(uv[0]))), int(round(float(uv[1]))))
            cv2.circle(rgb, centre, 6, MARKER_RGB, thickness=-1, lineType=cv2.LINE_AA)
        if self.truth.mode is CalibrationMode.EXTERIOR and self.table_marks_base:
            marks = np.stack(self.table_marks_base)
            muv, mz = project(self._K, self.truth.T_cam_parent, marks)
            for (mu, mv), z_m in zip(muv, mz, strict=True):
                if not (np.isfinite(mu) and np.isfinite(mv)) or z_m <= 0:
                    continue
                c = (int(round(float(mu))), int(round(float(mv))))
                cv2.circle(depth, c, 7, float(z_m), thickness=-1)
                cv2.circle(rgb, c, 5, MARK_RGB, thickness=-1, lineType=cv2.LINE_AA)
        return rgb, depth

    def close(self) -> None:
        pass


def find_marker(rgb: np.ndarray, colour: str = "magenta") -> Optional[np.ndarray]:
    """Centroid of the magenta tracked marker (or a cyan table mark), or None."""
    r = rgb[..., 0].astype(np.int16)
    g = rgb[..., 1].astype(np.int16)
    b = rgb[..., 2].astype(np.int16)
    if colour == "cyan":
        mask = (r < 60) & (g > 160) & (b > 180)
    else:
        mask = (r > 180) & (b > 120) & (g < 80)
    if mask.sum() < 4:
        return None
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()])


class FakeTracker:
    """Follows the marker by colour: an honest stand-in for SpaTracker."""

    def __init__(self, pixel_noise: float = 0.0, seed: int = 0) -> None:
        self._noise = float(pixel_noise)
        self._rng = np.random.default_rng(seed)

    def track(self, dataset: KalibDataset, click_uv, log=None):
        from policy_abstraction.calibration.kalib_runner import TrackingResult

        records = dataset.records()
        n = len(records)
        tracks = np.full((n, 2), np.nan)
        vis = np.zeros(n, dtype=bool)
        for i, rec in enumerate(records):
            c = find_marker(dataset.load_rgb(rec.index))
            if c is not None:
                if self._noise > 0:
                    c = c + self._rng.normal(0.0, self._noise, size=2)
                tracks[i] = c
                vis[i] = True
            if log and i % 25 == 0:
                log(f"[fake tracker] frame {i}/{n}")
        first = tracks[0] if vis[0] else None
        if first is not None and np.linalg.norm(first - np.asarray(click_uv)) > 25:
            if log:
                log(
                    "[fake tracker] warning: the click is far from the marker in "
                    "frame 0; tracking the marker anyway"
                )
        return TrackingResult(
            tracks=tracks, visibility=vis, kalib_T_cam_parent=None, kalib_reproj_px=None
        )


def marker_pixel_from_truth(
    camera: FakeCamera, tcp_pose, truth: FakeTruth
) -> np.ndarray:
    """Where the marker is for a given TCP pose — for tests, not for the solver."""
    P = keypoint_in_parent(
        truth.mode,
        tcp_pose,
        point_offset_tcp=truth.point_offset_tcp,
        reference_point_base=truth.reference_point_base,
    )
    uv, _ = project(camera.intrinsics, truth.T_cam_parent, P[None])
    return uv[0]


__all__ = [
    "FakeArm",
    "FakeCamera",
    "FakeTracker",
    "FakeTruth",
    "MARKER_RGB",
    "MARK_RGB",
    "default_truth",
    "find_marker",
    "look_at",
    "marker_pixel_from_truth",
]
