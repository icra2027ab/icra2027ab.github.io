"""Calibrate one of the UR5e cell's cameras against the robot, end to end.

A GUI walks through capture, point selection, Kalib tracking, a RANSAC PnP
solve, a depth cross-check and saving — writing straight into the calibration
file the rollout launchers read, so a guided run works on the new set-up with
no further conversion.

    # exterior camera, camera pose in the robot base frame
    scripts/setup/calibrate_ur5_cameras.sh

    # wrist camera, camera pose in the TCP frame
    scripts/setup/calibrate_ur5_cameras.sh --camera wrist

    # no hardware, no GPU: fake arm + camera + tracker, to learn the flow
    scripts/setup/calibrate_ur5_cameras.sh --simulate

Run through the ``.sh`` wrapper: it picks the ``physical-ai-sdk`` interpreter
(the one with ``pyrealsense2``, ``ur_rtde`` and PyQt6) and tells the tool
where the Kalib checkout and its conda env are. The arm is only ever *read*
over RTDE; the operator moves it in freedrive. The intrinsics come from the
RealSense at the streaming resolution, which is exactly what the env server
publishes, so nothing about the camera has to be measured by hand.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import sys
from typing import TYPE_CHECKING, Callable, Literal, Optional

import numpy as np
import tyro

if TYPE_CHECKING:
    from policy_abstraction.calibration.geometry import CalibrationMode

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
for _p in (os.path.join(_REPO_ROOT, "src"), _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from policy_abstraction.envs.ur5.config import (  # noqa: E402
    DEFAULT_EXTERIOR_CAMERA,
    DEFAULT_ROBOT_IP,
    DEFAULT_TCP_OFFSET,
    DEFAULT_WRIST_CAMERA,
    UR5_CAMERA_HEIGHT,
    UR5_CAMERA_WIDTH,
)

# The cell's RealSenses, by librealsense serial (see docs/ur5.md). The same
# environment variables the rollout launchers honour override them.
DEFAULT_EXTERIOR_SERIAL = os.environ.get("EXTERIOR_SERIAL", "262722073432")
DEFAULT_WRIST_SERIAL = os.environ.get("WRIST_SERIAL", "115222071166")


@dataclasses.dataclass
class Args:
    camera: Literal["both", "exterior", "wrist"] = "both"
    """What to calibrate. ``both`` (default): the exterior camera, then the
    wrist camera, in one window — the wrist stage can take its table mark from
    a click in the freshly calibrated exterior image. ``exterior``: a fixed
    camera alone, solved in the robot base frame by tracking the TCP.
    ``wrist``: the tool camera alone, solved in the TCP frame by tracking a
    fixed mark on the table (then the mark must be touched)."""

    exterior_name: str = DEFAULT_EXTERIOR_CAMERA
    """Observation key / EnvSpec name of the exterior camera."""

    wrist_name: str = DEFAULT_WRIST_CAMERA
    """Observation key / EnvSpec name of the wrist camera."""

    exterior_device: str = f"rs:{DEFAULT_EXTERIOR_SERIAL}"
    """``rs:<serial>`` of the exterior RealSense (``EXTERIOR_SERIAL`` env var)."""

    wrist_device: str = f"rs:{DEFAULT_WRIST_SERIAL}"
    """``rs:<serial>`` of the wrist RealSense (``WRIST_SERIAL`` env var)."""

    resolution: str = f"{UR5_CAMERA_WIDTH}x{UR5_CAMERA_HEIGHT}"
    """Stream resolution, ``WxH``. Keep it equal to what the env server
    streams: the intrinsics are queried at this resolution."""

    fps: int = 30

    robot_ip: str = DEFAULT_ROBOT_IP
    """UR controller. Read-only over RTDE."""

    tcp_offset: tuple[float, float, float, float, float, float] = DEFAULT_TCP_OFFSET
    """The TCP offset configured on the pendant, recorded with the result so a
    later change is caught. The tracked point *is* the controller's TCP."""

    point_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Where the clicked physical point sits relative to the TCP, in the TCP
    frame (metres). Zero when the pendant's TCP is the closed fingertips, as on
    this cell."""

    dataset_dir: Optional[pathlib.Path] = None
    """Parent folder for the frames; each camera gets ``<dataset_dir>/<camera_name>``.
    Defaults to ``data/calibration``."""

    output: pathlib.Path = pathlib.Path("configs/ur5_camera_calibration.json")
    """Calibration file to write/merge into — the one the launchers default to."""

    target_poses: int = 40
    """Still poses to aim for during capture."""

    max_frames: int = 400
    """Hard cap on stored frames (still + moving); bounds tracker memory."""

    ransac_px: float = 4.0
    """PnP inlier threshold."""

    no_depth: bool = False
    """Open the camera colour-only. Loses the depth cross-check; only for a USB
    layout that cannot stream depth."""

    simulate: bool = False
    """No hardware and no GPU: a scripted fake arm, a rendered fake camera and
    a colour-blob tracker stand in for the cell and for Kalib."""

    no_motion: bool = False
    """Never command the arm: only manual (freedrive) capture is offered. By
    default the window also offers autonomous capture, which connects RTDE
    control the moment you start it (the robot must be in Remote Control)."""

    kalib_root: Optional[str] = None
    """Kalib checkout (default ``$KALIB_ROOT`` or ``~/dev/Kalib``)."""

    kalib_python: Optional[str] = None
    """Kalib's interpreter (default ``$KALIB_PYTHON`` or the ``kalib`` conda env)."""

    kalib_device: Optional[int] = None
    """CUDA device for the tracker (default ``$KALIB_DEVICE`` or 0)."""

    notes: str = ""
    """Free text stored with the capture."""


def _parse_resolution(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x")
        return int(w), int(h)
    except ValueError as exc:
        raise SystemExit(f"--resolution must look like 640x480, got {text!r}") from exc


def _stage_modes(args: Args) -> list["CalibrationMode"]:
    from policy_abstraction.calibration.geometry import CalibrationMode

    if args.camera == "both":
        return [CalibrationMode.EXTERIOR, CalibrationMode.WRIST]
    return [
        CalibrationMode.WRIST if args.camera == "wrist" else CalibrationMode.EXTERIOR
    ]


def main(args: Args) -> int:
    from loguru import logger

    from policy_abstraction.calibration.geometry import CalibrationMode
    from policy_abstraction.calibration.hardware import ArmReader, CameraSource
    from policy_abstraction.calibration.kalib_runner import Tracker
    from policy_abstraction.calibration.motion import Mover
    from policy_abstraction.calibration.session import (
        CalibrationSession,
        SessionConfig,
    )
    from policy_abstraction.calibration.suite import CalibrationSuite

    modes = _stage_modes(args)
    width, height = _parse_resolution(args.resolution)

    def name_of(mode: CalibrationMode) -> str:
        return args.wrist_name if mode is CalibrationMode.WRIST else args.exterior_name

    def device_of(mode: CalibrationMode) -> str:
        return (
            args.wrist_device if mode is CalibrationMode.WRIST else args.exterior_device
        )

    arm: ArmReader
    tracker: Tracker
    cameras: dict[CalibrationMode, CameraSource] = {}
    mover: Optional[Mover] = None
    mover_factory: Optional[Callable[[], Mover]] = None

    if args.simulate:
        from policy_abstraction.calibration.fake_hardware import (
            FakeArm,
            FakeCamera,
            FakeTracker,
            default_truth,
        )

        fake_arm = FakeArm(list(args.tcp_offset), spread_rad=0.22)
        arm = fake_arm
        home = fake_arm.tcp_pose()
        truths = {m: default_truth(m, home) for m in modes}
        # The fake exterior camera also shows the wrist stage's table mark, so
        # picking it in the exterior image can be tried without hardware.
        wrist_truth = truths.get(CalibrationMode.WRIST)
        marks = (
            [wrist_truth.reference_point_base]
            if wrist_truth is not None and wrist_truth.reference_point_base is not None
            else []
        )
        for m in modes:
            cameras[m] = FakeCamera(
                fake_arm,
                truths[m],
                width=width,
                height=height,
                pixel_noise=0.3,
                # One camera advances the fake freedrive per poll; the other
                # is only read on demand.
                on_read=fake_arm.tick if m is modes[0] else None,
                table_marks_base=marks if m is CalibrationMode.EXTERIOR else None,
            )
        tracker = FakeTracker(pixel_noise=0.3)
        if not args.no_motion:
            mover = fake_arm  # the fake arm also executes commanded moves
        logger.warning(
            "--simulate: fake arm, fake cameras and a colour tracker. Nothing is "
            "real, including the result."
        )
    else:
        from policy_abstraction.calibration.hardware import RealSenseSource, RtdeArm
        from policy_abstraction.calibration.kalib_runner import (
            KalibConfig,
            KalibTracker,
        )
        from policy_abstraction.envs.ur5.realsense import (
            is_realsense_device,
            realsense_serial,
        )

        for m in modes:
            if not is_realsense_device(device_of(m)):
                raise SystemExit(
                    f"{device_of(m)!r} is not rs:<serial>; the intrinsics and "
                    "registered depth come from the RealSense SDK."
                )
        kalib = KalibConfig.from_env(
            root=args.kalib_root, python=args.kalib_python, device_id=args.kalib_device
        )
        problems = kalib.problems()
        if problems:
            raise SystemExit(
                "Kalib is not runnable:\n  - "
                + "\n  - ".join(problems)
                + "\nSet KALIB_ROOT / KALIB_PYTHON, or pass --kalib-root / --kalib-python."
            )
        logger.info(
            f"Kalib: {kalib.root} with {kalib.python} on cuda:{kalib.device_id}"
        )
        logger.info(f"Connecting to the arm at {args.robot_ip} (RTDE receive only) ...")
        arm = RtdeArm(args.robot_ip)
        for m in modes:
            dev = device_of(m)
            logger.info(f"Opening RealSense {dev} at {width}x{height}@{args.fps} ...")
            try:
                cameras[m] = RealSenseSource(
                    realsense_serial(dev),
                    width,
                    height,
                    args.fps,
                    depth=not args.no_depth,
                )
            except RuntimeError as exc:
                if args.no_depth or len(modes) == 1:
                    raise
                logger.warning(
                    f"{dev} would not stream colour+depth alongside the other camera "
                    f"({exc}); opening it colour-only. Its depth cross-check is lost."
                )
                cameras[m] = RealSenseSource(
                    realsense_serial(dev), width, height, args.fps, depth=False
                )
        tracker = KalibTracker(kalib)
        if not args.no_motion:
            from policy_abstraction.calibration.motion import RtdeMover

            robot_ip = args.robot_ip
            shared: list[Mover] = []

            def _connect_mover() -> Mover:
                if shared:
                    return shared[0]
                logger.info(
                    f"Connecting RTDE control to {robot_ip} for autonomous motion"
                )
                m = RtdeMover(robot_ip)
                configured = np.asarray(args.tcp_offset, dtype=np.float64)
                actual = np.asarray(m.tcp_offset(), dtype=np.float64)
                if not np.allclose(configured, actual, atol=1e-4):
                    logger.warning(
                        f"The controller's TCP offset {np.round(actual, 5).tolist()} "
                        f"differs from --tcp-offset {np.round(configured, 5).tolist()}; "
                        "the tracked point is the controller's TCP, so the pendant "
                        "value is what gets calibrated."
                    )
                shared.append(m)
                return m

            mover_factory = _connect_mover

    stages: list[CalibrationSession] = []
    for m in modes:
        name = name_of(m)
        config = SessionConfig(
            camera_name=name,
            camera_device="fake" if args.simulate else device_of(m),
            mode=m,
            dataset_root=(args.dataset_dir or pathlib.Path("data") / "calibration")
            / name,
            output_path=args.output,
            robot_ip="" if args.simulate else args.robot_ip,
            tcp_offset=[float(v) for v in args.tcp_offset],
            point_offset_tcp=(
                float(args.point_offset[0]),
                float(args.point_offset[1]),
                float(args.point_offset[2]),
            ),
            width=width,
            height=height,
            fps=args.fps,
            target_keyframes=args.target_poses,
            max_frames=args.max_frames,
            ransac_px=args.ransac_px,
            notes=args.notes,
        )
        stages.append(
            CalibrationSession(
                config,
                arm=arm,
                camera=cameras[m],
                tracker=tracker,
                mover=mover,
                mover_factory=mover_factory,
            )
        )
    suite = CalibrationSuite(stages, calibration_path=args.output)

    from policy_abstraction.calibration.gui import run_gui

    return run_gui(suite)


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
