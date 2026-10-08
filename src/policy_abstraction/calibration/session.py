"""The calibration procedure as a state machine, with no Qt in it.

The GUI polls :meth:`CalibrationSession.poll` on a timer and calls the
transition methods from its buttons; the tests drive the very same object
with fake hardware. Steps, in order::

    [REFERENCE ->] CAPTURE -> SELECT_POINT -> TRACK -> VERIFY -> DONE

``REFERENCE`` exists only for a wrist camera: the operator touches the mark
on the table with the closed fingertips and its base-frame position is read
off the controller. From then on the two mountings share the procedure; only
:func:`~policy_abstraction.calibration.geometry.keypoint_in_parent` differs.

Capture takes two kinds of frame. *Keyframes* are taken when the arm has been
still and has moved at least ``keyframe_min_move_m`` since the last one; they
are the PnP correspondences, because an RTDE reading and a camera frame are
only guaranteed to describe the same instant when nothing is moving. *Bridge*
frames are taken at a steady rate while the arm moves, so the tracker sees a
continuous video instead of a slideshow; they are excluded from the solve.

Capture can be *manual* (the operator freedrives the arm) or *autonomous*
(:meth:`CalibrationSession.start_autonomous_capture`): a
:class:`~policy_abstraction.calibration.motion.Mover` drives the arm through
planned poses inside a safe box, and :meth:`poll` advances that plan one
step per call — command a move, wait until the arm is still, dwell so the
keyframe is taken, next. The live TCP is checked against the box on every
poll and any excursion stops the arm.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import pathlib
import time
from typing import Callable, Optional

import numpy as np
from loguru import logger

from policy_abstraction.calibration.dataset import (
    CaptureFrame,
    CaptureMeta,
    FrameRecord,
    KalibDataset,
)
from policy_abstraction.calibration.geometry import (
    CalibrationMode,
    keypoint_in_parent,
    project,
    tracked_point_base,
)
from policy_abstraction.calibration.hardware import ArmReader, CameraSource
from policy_abstraction.calibration.kalib_runner import Tracker, TrackingResult
from policy_abstraction.calibration.motion import MotionConfig, Mover, plan_targets
from policy_abstraction.calibration.output import (
    BASE_FRAME,
    TCP_PARENT,
    camera_entry,
    write_calibration,
)
from policy_abstraction.calibration.solve import (
    DepthAgreement,
    ExtrinsicSolution,
    PoseDifference,
    Verdict,
    depth_agreement,
    judge,
    pose_difference,
    solve_extrinsic,
)

WRIST_MIN_CLEARANCE_M = 0.05
"""A wrist-camera capture box must start this far above the recorded mark."""


class Step(str, enum.Enum):
    REFERENCE = "reference"
    CAPTURE = "capture"
    SELECT_POINT = "select_point"
    TRACK = "track"
    VERIFY = "verify"
    DONE = "done"


@dataclasses.dataclass
class SessionConfig:
    camera_name: str
    camera_device: str
    mode: CalibrationMode
    dataset_root: pathlib.Path
    output_path: pathlib.Path
    robot_ip: str = ""
    tcp_offset: list[float] = dataclasses.field(default_factory=lambda: [0.0] * 6)
    point_offset_tcp: tuple[float, float, float] = (0.0, 0.0, 0.0)
    width: int = 640
    height: int = 480
    fps: int = 30
    target_keyframes: int = 40
    max_frames: int = 400
    min_keyframes: int = 12
    settle_rad_s: float = 0.01
    keyframe_min_move_m: float = 0.02
    keyframe_min_interval_s: float = 0.6
    bridge_interval_s: float = 0.25
    ransac_px: float = 4.0
    notes: str = ""

    def __post_init__(self) -> None:
        # Absolute from the start: the Kalib driver runs in another directory.
        self.dataset_root = pathlib.Path(self.dataset_root).resolve()
        self.output_path = pathlib.Path(self.output_path).resolve()
        self.mode = CalibrationMode(self.mode)
        try:
            self.output_path.resolve().relative_to(self.dataset_root.resolve())
        except ValueError:
            return
        raise ValueError(
            f"output_path {self.output_path} lies inside dataset_root "
            f"{self.dataset_root}; Kalib counts every top-level *.json there as a "
            "frame. Write the calibration somewhere else."
        )


@dataclasses.dataclass
class LiveSample:
    rgb: np.ndarray
    depth_m: Optional[np.ndarray]
    joints: np.ndarray
    joint_velocities: np.ndarray
    tcp_pose: np.ndarray
    settled: bool
    timestamp: float
    accepted: Optional[str] = None
    """``"keyframe"`` / ``"bridge"`` when this poll stored the frame."""


@dataclasses.dataclass
class CaptureStatus:
    n_frames: int = 0
    n_keyframes: int = 0
    target_keyframes: int = 0
    max_frames: int = 0
    travel_since_keyframe_m: float = 0.0
    keyframe_positions: list[np.ndarray] = dataclasses.field(default_factory=list)

    @property
    def target_reached(self) -> bool:
        return self.n_keyframes >= self.target_keyframes

    @property
    def full(self) -> bool:
        return self.n_frames >= self.max_frames

    def extent_m(self) -> np.ndarray:
        """Range of keyframe TCP positions along x, y, z — coverage at a glance."""
        if len(self.keyframe_positions) < 2:
            return np.zeros(3)
        P = np.stack(self.keyframe_positions)
        return P.max(axis=0) - P.min(axis=0)


@dataclasses.dataclass
class MotionRun:
    """Progress of an autonomous capture."""

    config: MotionConfig
    targets: list[np.ndarray]
    index: int = 0
    phase: str = "idle"
    """``idle`` (about to command), ``moving``, ``dwell``, ``done``, ``stopped``,
    ``fault``."""
    phase_t: float = 0.0
    skipped: int = 0
    message: str = ""

    @property
    def active(self) -> bool:
        return self.phase in ("idle", "moving", "dwell")

    @property
    def current_target(self) -> Optional[np.ndarray]:
        if self.index < len(self.targets):
            return self.targets[self.index]
        return None


@dataclasses.dataclass
class CalibrationResult:
    solution: ExtrinsicSolution
    tracking: TrackingResult
    records: list[FrameRecord]
    depth: Optional[DepthAgreement]
    kalib_difference: Optional[PoseDifference]
    verdict: Verdict
    intrinsic: np.ndarray

    def reprojected(self) -> np.ndarray:
        """``(N, 2)`` pixel where the solved pose puts the point in each frame."""
        P = np.stack([r.keypoint for r in self.records])
        uv, _ = project(self.intrinsic, self.solution.T_cam_parent, P)
        return uv


class CalibrationSession:
    def __init__(
        self,
        config: SessionConfig,
        *,
        arm: ArmReader,
        camera: CameraSource,
        tracker: Tracker,
        clock: Callable[[], float] = time.monotonic,
        mover: Optional[Mover] = None,
        mover_factory: Optional[Callable[[], Mover]] = None,
    ) -> None:
        self.config = config
        self.arm = arm
        self.camera = camera
        self.tracker = tracker
        self._clock = clock
        self.mover = mover
        self._mover_factory = mover_factory
        """Creates the mover on first use, because connecting RTDE control
        takes the robot over and should not happen until motion is wanted."""
        self.motion: Optional[MotionRun] = None
        self.step: Step = (
            Step.REFERENCE if config.mode is CalibrationMode.WRIST else Step.CAPTURE
        )
        self.reference_point_base: Optional[np.ndarray] = None
        self.reference_source: str = ""
        self.skipped = False
        self.dataset: Optional[KalibDataset] = None
        self.capturing = False
        self.status = CaptureStatus(
            target_keyframes=config.target_keyframes, max_frames=config.max_frames
        )
        self.click_uv: Optional[np.ndarray] = None
        self.result: Optional[CalibrationResult] = None
        self.saved_path: Optional[pathlib.Path] = None
        self.last_sample: Optional[LiveSample] = None
        self._last_key_pos: Optional[np.ndarray] = None
        self._last_key_t = -1e9
        self._last_frame_t = -1e9
        self._intrinsic: Optional[np.ndarray] = None

    # -- helpers ------------------------------------------------------------

    @property
    def mode(self) -> CalibrationMode:
        return self.config.mode

    @property
    def parent_frame(self) -> str:
        return self.mode.parent_frame

    @property
    def intrinsic(self) -> np.ndarray:
        if self._intrinsic is None:
            self._intrinsic = np.asarray(self.camera.intrinsics, dtype=np.float64)
        return self._intrinsic

    def steps(self) -> list[Step]:
        steps = [Step.CAPTURE, Step.SELECT_POINT, Step.TRACK, Step.VERIFY, Step.DONE]
        if self.mode is CalibrationMode.WRIST:
            steps.insert(0, Step.REFERENCE)
        return steps

    def _keypoint(self, tcp_pose) -> np.ndarray:
        return keypoint_in_parent(
            self.mode,
            tcp_pose,
            point_offset_tcp=self.config.point_offset_tcp,
            reference_point_base=self.reference_point_base,
        )

    # -- live polling ---------------------------------------------------------

    def poll(self) -> LiveSample:
        """Read camera + arm once; store the frame if capturing and it qualifies."""
        rgb, depth = self.camera.read()
        q = self.arm.joints()
        qd = self.arm.joint_velocities()
        tcp = self.arm.tcp_pose()
        now = self._clock()
        settled = bool(np.max(np.abs(qd)) < self.config.settle_rad_s)
        sample = LiveSample(rgb, depth, q, qd, tcp, settled, now)
        if self.capturing:
            sample.accepted = self._maybe_store(sample)
        if self.motion is not None and self.motion.active:
            self._drive_motion(sample)
        self.last_sample = sample
        return sample

    def _maybe_store(self, s: LiveSample) -> Optional[str]:
        assert self.dataset is not None
        if self.status.full:
            return None
        pos = s.tcp_pose[:3]
        travel = (
            float("inf")
            if self._last_key_pos is None
            else float(np.linalg.norm(pos - self._last_key_pos))
        )
        self.status.travel_since_keyframe_m = 0.0 if travel == float("inf") else travel
        moved = travel >= self.config.keyframe_min_move_m
        if (
            s.settled
            and moved
            and s.timestamp - self._last_key_t >= self.config.keyframe_min_interval_s
        ):
            self._store(s, settled=True)
            self._last_key_pos = pos.copy()
            self._last_key_t = s.timestamp
            self.status.n_keyframes += 1
            self.status.keyframe_positions.append(pos.copy())
            return "keyframe"
        if (
            not s.settled or moved
        ) and s.timestamp - self._last_frame_t >= self.config.bridge_interval_s:
            self._store(s, settled=False)
            return "bridge"
        return None

    def _store(self, s: LiveSample, *, settled: bool) -> None:
        assert self.dataset is not None
        self.dataset.append(
            CaptureFrame(
                rgb=s.rgb,
                depth_m=s.depth_m,
                joints=s.joints,
                joint_velocities=s.joint_velocities,
                tcp_pose=s.tcp_pose,
                keypoint=self._keypoint(s.tcp_pose),
                settled=settled,
                timestamp=s.timestamp,
            )
        )
        self._last_frame_t = s.timestamp
        self.status.n_frames += 1

    # -- transitions ----------------------------------------------------------

    def record_reference(self) -> np.ndarray:
        """Wrist mode: the mark's base-frame position is where the TCP is now."""
        if self.mode is not CalibrationMode.WRIST:
            raise RuntimeError("only a wrist calibration has a reference point")
        tcp = self.arm.tcp_pose()
        return self.set_reference_point(
            tracked_point_base(tcp, self.config.point_offset_tcp), source="touched"
        )

    def set_reference_point(self, point_base, *, source: str = "given") -> np.ndarray:
        """Wrist mode: take the mark's base-frame position from elsewhere — a
        click in the calibrated exterior camera's image, say."""
        if self.mode is not CalibrationMode.WRIST:
            raise RuntimeError("only a wrist calibration has a reference point")
        p = np.asarray(point_base, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            raise ValueError("reference point is not finite")
        self.reference_point_base = p
        self.reference_source = source
        logger.info(f"Reference mark ({source}) at {np.round(p, 4).tolist()} m (base)")
        if self.step is Step.REFERENCE:
            self.step = Step.CAPTURE
        return p

    def skip(self, reason: str = "kept the saved calibration") -> None:
        """Leave this camera as it is in the calibration file."""
        if self.capturing:
            raise RuntimeError("finish or abandon the capture first")
        self.skipped = True
        self.step = Step.DONE
        self.saved_path = self.config.output_path
        logger.info(f"{self.config.camera_name}: {reason}")

    def start_capture(self) -> KalibDataset:
        if self.step not in (Step.CAPTURE, Step.SELECT_POINT):
            raise RuntimeError(f"cannot start capture from step {self.step.value}")
        if self.mode is CalibrationMode.WRIST and self.reference_point_base is None:
            raise RuntimeError("record the reference mark before capturing")
        meta = CaptureMeta(
            camera_name=self.config.camera_name,
            camera_device=self.config.camera_device,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
            intrinsic=self.intrinsic.tolist(),
            mode=self.mode.value,
            tcp_offset=[float(v) for v in self.config.tcp_offset],
            point_offset_tcp=[float(v) for v in self.config.point_offset_tcp],
            reference_point_base=(
                None
                if self.reference_point_base is None
                else [float(v) for v in self.reference_point_base]
            ),
            robot_ip=self.config.robot_ip,
            notes=self.config.notes,
        )
        self.dataset = KalibDataset.create(self.config.dataset_root, meta)
        self.status = CaptureStatus(
            target_keyframes=self.config.target_keyframes,
            max_frames=self.config.max_frames,
        )
        self._last_key_pos = None
        self._last_key_t = -1e9
        self._last_frame_t = -1e9
        self.click_uv = None
        self.result = None
        self.capturing = True
        self.step = Step.CAPTURE
        logger.info(f"Capturing to {self.dataset.root}")
        return self.dataset

    def finish_capture(self) -> None:
        if not self.capturing or self.dataset is None:
            raise RuntimeError("not capturing")
        if self.motion is not None and self.motion.active:
            self.stop_motion("capture finished")
        if self.status.n_keyframes < self.config.min_keyframes:
            raise RuntimeError(
                f"only {self.status.n_keyframes} still poses captured; need at least "
                f"{self.config.min_keyframes}. Keep moving the arm and pausing."
            )
        self.capturing = False
        self.step = Step.SELECT_POINT
        logger.info(
            f"Capture done: {self.status.n_frames} frames, {self.status.n_keyframes} "
            f"still poses, TCP extent {np.round(self.status.extent_m(), 2).tolist()} m"
        )

    # -- autonomous motion ------------------------------------------------------

    def ensure_mover(self) -> Mover:
        if self.mover is None:
            if self._mover_factory is None:
                raise RuntimeError(
                    "no motion interface: run without --no-motion, or capture manually"
                )
            self.mover = self._mover_factory()
        return self.mover

    def plan_motion(self, motion: MotionConfig) -> list[np.ndarray]:
        """Plan and vet the capture poses without moving anything."""
        tcp = self.arm.tcp_pose()
        if not motion.box.contains(tcp[:3]):
            raise RuntimeError(
                f"the TCP is at {np.round(tcp[:3], 3).tolist()} m, outside the safe "
                "box. Move the arm inside it (or re-centre the box) first."
            )
        if self.mode is CalibrationMode.WRIST:
            if self.reference_point_base is None:
                raise RuntimeError("record the reference mark before planning motion")
            clearance = motion.box.z[0] - float(self.reference_point_base[2])
            if clearance < WRIST_MIN_CLEARANCE_M:
                raise RuntimeError(
                    f"the safe box starts {clearance * 100:.1f} cm above the mark; "
                    f"it must start at least {WRIST_MIN_CLEARANCE_M * 100:.0f} cm above "
                    "it so the poses stay clear of the table. Raise the box's lower z."
                )
        targets = plan_targets(
            motion, tcp, self.mode, reference_point_base=self.reference_point_base
        )
        mover = self.ensure_mover()
        ok = [t for t in targets if mover.pose_ok(t)]
        dropped = len(targets) - len(ok)
        if dropped:
            logger.warning(
                f"{dropped} of {len(targets)} planned poses were rejected by the "
                "controller (unreachable or outside its safety limits)."
            )
        if len(ok) < self.config.min_keyframes:
            raise RuntimeError(
                f"only {len(ok)} reachable poses in the box; shrink the tilt, move the "
                "box, or add poses."
            )
        return ok

    def start_autonomous_capture(self, motion: MotionConfig) -> MotionRun:
        """Plan, then start capturing and let :meth:`poll` drive the arm."""
        targets = self.plan_motion(motion)
        self.config.target_keyframes = len(targets)
        self.start_capture()
        self.motion = MotionRun(config=motion, targets=targets, phase_t=self._clock())
        logger.info(
            f"Autonomous capture: {len(targets)} poses in box x{motion.box.x} "
            f"y{motion.box.y} z{motion.box.z} at {motion.speed} m/s"
        )
        return self.motion

    def stop_motion(self, reason: str = "stopped by operator") -> None:
        if self.mover is not None:
            self.mover.stop()
        if self.motion is not None and self.motion.active:
            self.motion.phase = "stopped"
            self.motion.message = reason
            logger.warning(f"Motion stopped: {reason}")

    def _drive_motion(self, s: LiveSample) -> None:
        m = self.motion
        assert m is not None and self.mover is not None
        now = s.timestamp
        if not m.config.box.contains(s.tcp_pose[:3], margin=m.config.margin_m):
            self.mover.stop()
            m.phase = "fault"
            m.message = (
                f"TCP left the safe box at {np.round(s.tcp_pose[:3], 3).tolist()} m; "
                "motion stopped"
            )
            logger.error(m.message)
            return
        if m.phase == "idle":
            target = m.current_target
            if target is None or self.status.full:
                m.phase = "done"
                m.message = f"{m.index - m.skipped} poses visited, {m.skipped} skipped"
                logger.info(f"Autonomous capture finished: {m.message}")
                return
            if not self.mover.move_to(target, m.config.speed, m.config.accel):
                logger.warning(f"controller refused pose {m.index}; skipping")
                m.index += 1
                m.skipped += 1
                return
            m.phase = "moving"
            m.phase_t = now
        elif m.phase == "moving":
            # The controller needs a cycle or two before an asynchronous move
            # registers as running, so a just-issued move must not look done.
            if (
                now - m.phase_t >= m.config.min_move_s
                and not self.mover.is_moving()
                and s.settled
            ):
                m.phase = "dwell"
                m.phase_t = now
            elif now - m.phase_t > m.config.move_timeout_s:
                self.mover.stop()
                logger.warning(
                    f"pose {m.index} timed out after {m.config.move_timeout_s} s; skipping"
                )
                m.index += 1
                m.skipped += 1
                m.phase = "idle"
        elif m.phase == "dwell":
            if now - m.phase_t >= m.config.dwell_s:
                m.index += 1
                m.phase = "idle"

    def use_existing_dataset(self, root: str | pathlib.Path) -> KalibDataset:
        """Skip capture and re-click / re-solve a previous capture."""
        ds = KalibDataset(root)
        meta = ds.meta
        if CalibrationMode(meta.mode) is not self.mode:
            raise ValueError(
                f"{root} was captured in {meta.mode} mode, this session is {self.mode.value}"
            )
        if ds.n_frames == 0:
            raise ValueError(f"{root} holds no frames")
        self.dataset = ds
        self._intrinsic = np.asarray(meta.intrinsic, dtype=np.float64)
        if meta.reference_point_base is not None:
            self.reference_point_base = np.asarray(meta.reference_point_base)
        recs = ds.records()
        self.status = CaptureStatus(
            n_frames=len(recs),
            n_keyframes=sum(r.settled for r in recs),
            target_keyframes=self.config.target_keyframes,
            max_frames=self.config.max_frames,
            keyframe_positions=[
                r.tcp_pose[:3] for r in recs if r.settled and r.tcp_pose is not None
            ],
        )
        self.capturing = False
        self.click_uv = None
        self.result = None
        self.step = Step.SELECT_POINT
        return ds

    def first_frame(self) -> np.ndarray:
        if self.dataset is None:
            raise RuntimeError("no capture yet")
        return self.dataset.load_rgb(self.dataset.records()[0].index)

    def set_click(self, u: float, v: float) -> None:
        if self.step not in (Step.SELECT_POINT, Step.VERIFY, Step.DONE):
            raise RuntimeError(f"cannot pick the point during step {self.step.value}")
        if not (0 <= u < self.config.width and 0 <= v < self.config.height):
            raise ValueError(f"click ({u:.1f}, {v:.1f}) is outside the image")
        self.click_uv = np.array([float(u), float(v)])
        self.step = Step.SELECT_POINT

    def run_tracking(
        self, log: Optional[Callable[[str], None]] = None
    ) -> CalibrationResult:
        """Track, solve, verify. Blocking; run it off the GUI thread."""
        if self.dataset is None or self.click_uv is None:
            raise RuntimeError("capture and click before tracking")
        log = log or logger.info
        self.step = Step.TRACK
        try:
            tracking = self.tracker.track(self.dataset, self.click_uv, log)
            result = self.solve(tracking, log)
        except Exception:
            self.step = Step.SELECT_POINT
            raise
        self.result = result
        self.step = Step.VERIFY
        return result

    def solve(
        self, tracking: TrackingResult, log: Optional[Callable[[str], None]] = None
    ) -> CalibrationResult:
        assert self.dataset is not None
        log = log or logger.info
        records = self.dataset.records()
        n = len(records)
        if tracking.tracks.shape[0] != n:
            raise ValueError(f"{tracking.tracks.shape[0]} tracks for {n} frames")
        P = np.stack([r.keypoint for r in records])
        settled = np.array([r.settled for r in records])
        cand = settled & tracking.visibility
        log(
            f"solving PnP from {int(cand.sum())} still+visible frames of {n} "
            f"({int(settled.sum())} still, {int(tracking.visibility.sum())} visible)"
        )
        solution = solve_extrinsic(
            P,
            tracking.tracks,
            self.intrinsic,
            candidates=cand,
            ransac_px=self.config.ransac_px,
        )
        log(
            f"PnP: {solution.n_used}/{solution.n_candidates} inliers, mean "
            f"{solution.mean_px:.2f} px, median {solution.median_px:.2f} px, "
            f"max {solution.max_inlier_px:.2f} px"
        )
        depth: Optional[DepthAgreement] = None
        if any(r.depth_path is not None for r in records):
            maps = [
                self.dataset.load_depth_m(r.index) if solution.used[i] else None
                for i, r in enumerate(records)
            ]
            depth = depth_agreement(
                maps, tracking.tracks, solution.T_cam_parent, P, mask=solution.used
            )
            log(
                f"depth check: {depth.n_valid} readings, median residual "
                f"{depth.median_m * 100:+.1f} cm, median |residual| {depth.median_abs_m * 100:.1f} cm"
            )
        diff: Optional[PoseDifference] = None
        if tracking.kalib_T_cam_parent is not None:
            diff = pose_difference(solution.T_cam_parent, tracking.kalib_T_cam_parent)
            log(
                f"Kalib's own PnP differs by {diff.translation_m * 100:.1f} cm / "
                f"{diff.rotation_deg:.2f} deg (its reprojection error: "
                f"{tracking.kalib_reproj_px if tracking.kalib_reproj_px is not None else float('nan'):.2f} px)"
            )
        verdict = judge(solution, depth)
        log(f"verdict: {verdict.level} — " + "; ".join(verdict.reasons))
        cam_pos = solution.T_parent_cam[:3, 3]
        log(
            f"camera origin in {self.parent_frame} frame: {np.round(cam_pos, 4).tolist()} m"
        )
        return CalibrationResult(
            solution=solution,
            tracking=tracking,
            records=records,
            depth=depth,
            kalib_difference=diff,
            verdict=verdict,
            intrinsic=self.intrinsic.copy(),
        )

    def entry(self) -> dict:
        """The camera's calibration-file entry for the current result."""
        if self.result is None or self.dataset is None:
            raise RuntimeError("nothing solved yet")
        r = self.result
        meta = {
            "tool": "scripts/setup/calibrate_ur5_cameras.py",
            "calibrated": _dt.datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode.value,
            "camera_device": self.config.camera_device,
            "resolution": [self.config.width, self.config.height],
            "intrinsic_at_calibration": r.intrinsic.tolist(),
            "n_frames": len(r.records),
            "n_inliers": r.solution.n_used,
            "mean_reproj_px": r.solution.mean_px,
            "depth_median_abs_m": (None if r.depth is None else r.depth.median_abs_m),
            "tcp_offset": [float(v) for v in self.config.tcp_offset],
            "point_offset_tcp": [float(v) for v in self.config.point_offset_tcp],
            "reference_point_base": (
                None
                if self.reference_point_base is None
                else [float(v) for v in self.reference_point_base]
            ),
            "reference_source": self.reference_source or None,
            "dataset": str(self.dataset.root),
            "verdict": r.verdict.level,
        }
        if r.kalib_difference is not None:
            meta["kalib_pnp_difference"] = {
                "translation_m": r.kalib_difference.translation_m,
                "rotation_deg": r.kalib_difference.rotation_deg,
            }
        parent = TCP_PARENT if self.mode is CalibrationMode.WRIST else BASE_FRAME
        return camera_entry(r.solution.T_parent_cam, parent=parent, meta=meta)

    def save(self) -> pathlib.Path:
        entry = self.entry()
        write_calibration(self.config.output_path, self.config.camera_name, entry)
        self.saved_path = self.config.output_path
        self.step = Step.DONE
        return self.saved_path

    def close(self, shared_ids: Optional[set[int]] = None) -> None:
        """Release the hardware. Sessions in a suite share the arm and the
        mover; *shared_ids* records what has been closed already so each
        device is closed once."""
        seen = shared_ids if shared_ids is not None else set()
        if self.motion is not None and self.motion.active:
            self.stop_motion("session closed")
        for dev in (self.mover, self.camera, self.arm):
            if dev is None or id(dev) in seen:
                continue
            seen.add(id(dev))
            try:
                dev.close()
            except Exception as exc:  # pragma: no cover - hardware teardown
                logger.warning(f"closing {dev!r}: {exc}")


__all__ = [
    "CalibrationResult",
    "CalibrationSession",
    "CaptureStatus",
    "LiveSample",
    "MotionRun",
    "SessionConfig",
    "Step",
]
