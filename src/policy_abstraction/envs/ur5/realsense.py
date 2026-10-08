"""RealSense colour+depth capture for the UR5e cell.

The default UR5 camera path opens each RealSense's *colour* node over V4L2
through LeRobot's ``OpenCVCamera``, which is all the policy needs. Perception's
``reconstruct`` stage additionally needs per-pixel metric depth registered to
that colour image, and that is what this module provides.

Why not read the depth node directly over V4L2? A RealSense does expose depth as
``Z16`` on its own ``/dev/videoN``, but those frames are in the *depth sensor's*
frame and at a device-configured scale. Using them against colour pixels without
registration mis-associates depth by the stereo baseline (~15 mm on a D435),
which is more than enough to put a reconstructed object on the wrong side of a
grasp. ``rs.align(rs.stream.color)`` performs that registration, and the SDK also
reports the depth scale and the factory colour intrinsics — none of which V4L2
surfaces.

Cameras are addressed as ``rs:<serial>`` in ``--cameras``; everything else keeps
using the V4L2 path.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

RS_PREFIX = "rs:"


def is_realsense_device(device: str) -> bool:
    """Whether a ``--cameras`` device string selects the RealSense SDK path."""
    return str(device).startswith(RS_PREFIX)


def realsense_serial(device: str) -> str:
    """``rs:262722073432`` -> ``262722073432``."""
    if not is_realsense_device(device):
        raise ValueError(f"{device!r} is not a RealSense device string")
    return device[len(RS_PREFIX) :]


class RealSenseCamera:
    """One RealSense streaming colour + depth aligned to colour.

    Exposes the same ``connect`` / ``async_read`` / ``disconnect`` surface the
    server uses for ``OpenCVCamera``, plus :meth:`read_depth_m` and the
    intrinsics/scale the SDK reports.
    """

    def __init__(
        self,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        *,
        depth: bool = True,
        warmup_frames: int = 30,
    ) -> None:
        self.serial = str(serial)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth = bool(depth)
        self._warmup_frames = int(warmup_frames)

        self._pipeline: Any = None
        self._align: Any = None
        self._depth_scale: float | None = None
        self._intrinsics: np.ndarray | None = None
        # Latest aligned pair, refreshed by `async_read`, so a caller that wants
        # both colour and depth for the same instant does not re-poll the device
        # and get two different frames.
        self._last_color: np.ndarray | None = None
        self._last_depth_m: np.ndarray | None = None

    # -- lifecycle ----------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self) -> None:
        """Start streaming, persisting through a stall on the first start.

        A D435 sometimes starts its pipeline and then delivers nothing for its
        first seconds — right after another pipeline on it was stopped (the
        start-up pre-flight does that seconds earlier), and on this cell's
        exterior D435I (firmware 5.15, older than recommended) even from idle.
        A stop and a fresh start a moment later streams normally, and a
        ``hardware_reset`` clears the cases a restart does not. Seen on
        2026-09-24 as ``Frame didn't arrive within 5000`` at connect with the
        next attempt succeeding. So: three attempts, the last one after a
        hardware reset, each logged; only then is the failure reported.
        """
        import time

        attempts = 3
        for attempt in range(1, attempts + 1):
            try:
                self._start()
                return
            except RuntimeError as exc:
                if "didn't arrive" not in str(exc) and "did not arrive" not in str(exc):
                    raise
                self.disconnect()
                if attempt == attempts:
                    raise
                if attempt == attempts - 1:
                    logger.warning(
                        f"RealSense {self.serial} delivered no frames after "
                        f"starting ({exc}); resetting the device before a last try."
                    )
                    if self.serial not in reset_realsense([self.serial]):
                        raise RuntimeError(
                            f"RealSense {self.serial} did not come back after a "
                            "hardware reset"
                        ) from exc
                else:
                    logger.warning(
                        f"RealSense {self.serial} delivered no frames after "
                        f"starting ({exc}); restarting the pipeline."
                    )
                    time.sleep(2.5)

    def _start(self) -> None:
        import pyrealsense2 as rs

        cfg = rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        if self.depth:
            cfg.enable_stream(
                rs.stream.depth, self.width, self.height, rs.format.z16, self.fps
            )

        pipeline = rs.pipeline()
        profile = pipeline.start(cfg)
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color) if self.depth else None
        self._depth_scale = (
            float(profile.get_device().first_depth_sensor().get_depth_scale())
            if self.depth
            else None
        )

        # Auto-exposure needs a few frames; without this the first reads are
        # black, which silently poisons both the policy and perception.
        for _ in range(self._warmup_frames):
            self._grab()

        logger.info(
            f"RealSense {self.serial} streaming {self.width}x{self.height}@{self.fps} "
            + (
                f"colour+depth aligned (depth_scale={self._depth_scale})"
                if self.depth
                else "colour only"
            )
        )

    def disconnect(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._align = None

    # -- capture ------------------------------------------------------------

    def _grab(self) -> None:
        if self._pipeline is None:
            raise RuntimeError(f"RealSense {self.serial} is not connected")
        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if self.depth else None
        if not color_frame or (self.depth and not depth_frame):
            raise RuntimeError(
                f"RealSense {self.serial} returned an incomplete frameset"
            )

        self._last_color = np.ascontiguousarray(np.asanyarray(color_frame.get_data()))
        self._last_depth_m = (
            np.asanyarray(depth_frame.get_data()).astype(np.float32)
            * float(self._depth_scale or 0.0)
            if depth_frame is not None
            else None
        )
        if self._intrinsics is None:
            intr = color_frame.profile.as_video_stream_profile().intrinsics
            self._intrinsics = np.array(
                [[intr.fx, 0.0, intr.ppx], [0.0, intr.fy, intr.ppy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

    def async_read(self) -> np.ndarray:
        """Grab a fresh frameset and return the colour image (RGB uint8)."""
        self._grab()
        assert self._last_color is not None
        return self._last_color

    def read_depth_m(self) -> np.ndarray:
        """Depth for the most recent :meth:`async_read`, in metres.

        Does *not* re-poll: the depth must correspond to the colour frame the
        policy saw, otherwise masks and depth disagree by a frame.
        """
        if not self.depth:
            raise RuntimeError(
                f"RealSense {self.serial} was opened colour-only, so it has no "
                "depth to read. Depth is enabled for the primary camera and any "
                "camera with a calibration extrinsic; two D435s streaming depth "
                "over one USB controller exceeds its bandwidth."
            )
        if self._last_depth_m is None:
            self._grab()
        assert self._last_depth_m is not None
        return self._last_depth_m

    # -- calibration --------------------------------------------------------

    @property
    def depth_scale(self) -> float | None:
        return self._depth_scale

    @property
    def intrinsics(self) -> np.ndarray | None:
        """Factory colour intrinsics as a 3x3 matrix, or None before first read.

        These describe the *aligned* colour stream, so they are the right
        matrix for back-projecting the depth this class returns. A hand
        calibration can still override them via ``--calibration-path``.
        """
        return self._intrinsics


def probe_realsense_cameras(requests: list[dict]) -> list[tuple[str, str]]:
    """Open every RealSense at once, grab a frame from each, release them all.

    Returns ``[(name, error), ...]`` for the cameras that failed; empty when
    all are usable.

    Two things make this worth doing at startup. Cameras are otherwise only
    claimed on the first ``make_env``, which on this stack is ~3 minutes after
    launch (the policy server has to load and compile first), so any camera
    problem surfaced at the end of that wait. And the probe has to open them
    *together*, with the streams the real connect will use: two D435s each
    streaming colour+depth over one USB controller exceeds its bandwidth, and
    that failure is invisible when you test them one at a time.

    Each request is ``{name, serial, width, height, fps, depth}``.

    The check is advisory -- another process can still claim a device between
    the probe and the real connect. It catches the common case, not a race.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:  # pragma: no cover - depends on the venv
        return [(r["name"], f"pyrealsense2 unavailable ({exc})") for r in requests]

    failures: list[tuple[str, str]] = []
    started: list[tuple[str, Any]] = []
    try:
        for req in requests:
            cfg = rs.config()
            cfg.enable_device(str(req["serial"]))
            cfg.enable_stream(
                rs.stream.color,
                int(req["width"]),
                int(req["height"]),
                rs.format.rgb8,
                int(req["fps"]),
            )
            if req.get("depth", True):
                cfg.enable_stream(
                    rs.stream.depth,
                    int(req["width"]),
                    int(req["height"]),
                    rs.format.z16,
                    int(req["fps"]),
                )
            pipeline = rs.pipeline()
            try:
                pipeline.start(cfg)
            except RuntimeError as exc:
                failures.append((req["name"], str(exc)))
                continue
            started.append((req["name"], pipeline))

        # Starting can succeed while frames never arrive -- that is exactly how
        # a USB bandwidth conflict presents -- so every camera must deliver.
        for name, pipeline in started:
            try:
                pipeline.wait_for_frames(5000)
            except RuntimeError as exc:
                failures.append((name, str(exc)))
    finally:
        for _, pipeline in started:
            try:
                pipeline.stop()
            except RuntimeError:  # pragma: no cover
                pass
    return failures


def reset_realsense(serials: list[str], timeout_s: float = 20.0) -> list[str]:
    """Power-cycle the named cameras and wait for them to re-enumerate.

    A D435 left wedged by an abruptly killed process starts its pipeline
    happily and then never delivers a frame -- indistinguishable from a USB
    bandwidth problem from the outside, and it survives closing every handle.
    ``hardware_reset`` is the documented way out; the device drops off the bus
    and comes back in a couple of seconds.

    Returns the serials that came back.
    """
    import time

    import pyrealsense2 as rs

    wanted = {str(x) for x in serials}
    for dev in rs.context().query_devices():
        if dev.get_info(rs.camera_info.serial_number) in wanted:
            logger.warning(
                f"Resetting RealSense {dev.get_info(rs.camera_info.serial_number)}"
            )
            dev.hardware_reset()

    deadline = time.monotonic() + timeout_s
    back: set[str] = set()
    while time.monotonic() < deadline and back != wanted:
        time.sleep(1.0)
        back = {
            d.get_info(rs.camera_info.serial_number)
            for d in rs.context().query_devices()
        } & wanted
    # Re-enumerating is not the same as being ready to stream.
    if back:
        time.sleep(2.0)
    return sorted(back)


def video_node_holders() -> list[str]:
    """``<pid> <cmdline>`` for every process holding a /dev/video* node open.

    librealsense reports "Device or resource busy" without saying who has it,
    and the answer is nearly always another rollout or a playground script the
    operator forgot about, so name it for them.
    """
    import glob

    holders: dict[int, str] = {}
    for fd_path in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            target = os.readlink(fd_path)
        except OSError:
            continue  # process exited, or not ours to inspect
        if not target.startswith("/dev/video"):
            continue
        pid = int(fd_path.split("/")[2])
        if pid in holders:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmdline = fh.read().replace(b"\x00", b" ").decode(errors="replace")
        except OSError:
            cmdline = "<unknown>"
        holders[pid] = cmdline.strip()
    return [f"{pid} {cmd}" for pid, cmd in sorted(holders.items())]


__all__ = [
    "RS_PREFIX",
    "RealSenseCamera",
    "is_realsense_device",
    "probe_realsense_cameras",
    "reset_realsense",
    "realsense_serial",
    "video_node_holders",
]
