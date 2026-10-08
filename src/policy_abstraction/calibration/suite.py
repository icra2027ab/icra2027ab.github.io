"""Both cameras in one sitting: the exterior stage, then the wrist stage.

The stages are ordinary :class:`~policy_abstraction.calibration.session.CalibrationSession`
objects sharing the arm, the tracker and the motion interface; the suite only
says which one is active and carries what the second needs from the first.

That hand-over is the point. The wrist camera is calibrated against a fixed
mark on the table whose base-frame position must be known. Touching it with
the closed fingertips works, but once the exterior camera is calibrated the
mark can simply be **clicked in the exterior image**: its depth at that pixel,
the camera's intrinsics and the freshly solved extrinsic give the point in the
base frame (:meth:`CalibrationSuite.mark_from_exterior_click`). When the
exterior stage is skipped, the extrinsic comes from the calibration file.
"""

from __future__ import annotations

import json
import pathlib
from typing import Optional

import numpy as np
from loguru import logger

from policy_abstraction.calibration.geometry import CalibrationMode
from policy_abstraction.calibration.hardware import CameraSource
from policy_abstraction.calibration.output import BASE_FRAME, base_link_to_base
from policy_abstraction.calibration.session import CalibrationSession, Step
from policy_abstraction.calibration.solve import depth_at


def deproject(
    uv, depth_m: float, K: np.ndarray, T_parent_cam: np.ndarray
) -> np.ndarray:
    """Pixel + depth -> 3-D point in the camera's parent frame."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    x = (float(uv[0]) - K[0, 2]) / K[0, 0] * depth_m
    y = (float(uv[1]) - K[1, 2]) / K[1, 1] * depth_m
    p_cam = np.array([x, y, float(depth_m)])
    T = np.asarray(T_parent_cam, dtype=np.float64).reshape(4, 4)
    return T[:3, :3] @ p_cam + T[:3, 3]


def load_saved_extrinsic(
    path: str | pathlib.Path, camera_name: str
) -> Optional[np.ndarray]:
    """The camera's base-frame pose from an env-layout calibration file, or None.

    A ``base_link`` file is rotated into ``base``; a TCP-parented entry has no
    fixed base-frame pose and yields None.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    entry = raw.get(camera_name)
    if not isinstance(entry, dict) or "extrinsic" not in entry:
        return None
    if entry.get("parent", BASE_FRAME) != BASE_FRAME:
        return None
    E = np.asarray(entry["extrinsic"], dtype=np.float64).reshape(4, 4)
    frame = raw.get("base_frame", BASE_FRAME)
    if frame == "base_link":
        E = base_link_to_base() @ E
    elif frame != BASE_FRAME:
        raise ValueError(f"{path} declares unknown base_frame {frame!r}")
    return E


class CalibrationSuite:
    """An ordered list of stages, one per camera, sharing the cell."""

    def __init__(
        self, stages: list[CalibrationSession], *, calibration_path: str | pathlib.Path
    ) -> None:
        if not stages:
            raise ValueError("a suite needs at least one stage")
        self.stages = list(stages)
        self.calibration_path = pathlib.Path(calibration_path)
        self.active_index = 0

    # -- navigation -----------------------------------------------------------

    @property
    def active(self) -> CalibrationSession:
        return self.stages[self.active_index]

    @property
    def has_next(self) -> bool:
        return self.active_index + 1 < len(self.stages)

    def advance(self) -> CalibrationSession:
        if not self.has_next:
            raise RuntimeError("this was the last stage")
        self.active_index += 1
        logger.info(
            f"Stage {self.active_index + 1}/{len(self.stages)}: {self.active.config.camera_name}"
        )
        return self.active

    def label(self, index: int) -> str:
        s = self.stages[index]
        kind = (
            "exterior camera" if s.mode is CalibrationMode.EXTERIOR else "wrist camera"
        )
        return (
            f"Stage {index + 1} of {len(self.stages)}: {kind} ({s.config.camera_name})"
        )

    # -- what the wrist stage borrows from the exterior one -----------------------

    def exterior_stage(self) -> Optional[CalibrationSession]:
        for s in self.stages:
            if s.mode is CalibrationMode.EXTERIOR:
                return s
        return None

    def exterior_extrinsic(self) -> Optional[np.ndarray]:
        """Camera-to-base pose of the exterior camera: this sitting's result if
        it was saved, else the calibration file's entry."""
        ext = self.exterior_stage()
        if ext is not None and ext.result is not None and ext.step is Step.DONE:
            return ext.result.solution.T_parent_cam.copy()
        name = ext.config.camera_name if ext is not None else None
        if name is None:
            return None
        return load_saved_extrinsic(self.calibration_path, name)

    def exterior_camera(self) -> Optional[CameraSource]:
        ext = self.exterior_stage()
        return None if ext is None else ext.camera

    def exterior_frame(self, flush: int = 4) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """A fresh colour+depth pair from the exterior camera.

        The pipeline keeps delivering while the wrist stage is polling the
        other camera, so a few frames are discarded first to skip what queued.
        """
        cam = self.exterior_camera()
        if cam is None:
            raise RuntimeError("no exterior camera in this suite")
        rgb, depth = cam.read()
        for _ in range(max(0, flush)):
            rgb, depth = cam.read()
        return rgb, depth

    def mark_from_exterior_click(
        self, u: float, v: float, rgb_depth=None
    ) -> tuple[np.ndarray, float]:
        """Base-frame position of a table mark clicked in the exterior image.

        Returns ``(point_base, depth_m)``. Uses the depth pair given (the frame
        the operator clicked on), or grabs a fresh one.
        """
        E = self.exterior_extrinsic()
        if E is None:
            raise RuntimeError(
                "the exterior camera has no calibration yet — solve and save it "
                "first, or touch the mark with the fingertips instead."
            )
        cam = self.exterior_camera()
        if cam is None:
            raise RuntimeError("no exterior camera in this suite")
        if rgb_depth is None:
            rgb_depth = self.exterior_frame()
        _rgb, depth = rgb_depth
        if depth is None:
            raise RuntimeError(
                "the exterior camera has no depth stream; touch the mark instead"
            )
        d = depth_at(depth, (u, v), radius=3)
        if not np.isfinite(d):
            raise RuntimeError(
                f"no valid depth at ({u:.0f}, {v:.0f}); pick a spot the depth sensor "
                "sees (matte, not at an edge), or touch the mark instead."
            )
        point = deproject((u, v), d, cam.intrinsics, E)
        logger.info(
            f"Mark clicked at ({u:.1f}, {v:.1f}) px, depth {d:.3f} m -> "
            f"{np.round(point, 4).tolist()} m in base"
        )
        return point, float(d)

    def close(self) -> None:
        seen: set[int] = set()
        for s in self.stages:
            s.close(shared_ids=seen)


__all__ = ["CalibrationSuite", "deproject", "load_saved_extrinsic"]
