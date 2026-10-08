"""A capture on disk, in the layout Kalib reads plus what we need on top.

Kalib's ``find_data_in_dir`` wants ``NNNNNN.jpg`` + ``NNNNNN.json`` pairs in
one flat directory, each JSON carrying the tracked point's 3-D position and the
camera intrinsics under ``objects[0]``. We write exactly that, and add:

* ``meta/capture.json`` — camera, resolution, intrinsics, mode, TCP offset,
  reference point: everything the solve and the saved calibration need to be
  reproducible from the folder alone. It sits in a subdirectory because Kalib
  globs ``*.json`` at the top level and counts them against the images;
* ``depth/NNNNNN.png`` — the RealSense depth registered to the colour frame,
  in millimetres, for the depth-agreement check after solving. It lives in a
  subdirectory because Kalib globs ``*.png`` at the top level and would take
  the depth maps for video frames;
* per-frame ``tcp_pose`` and ``settled`` keys, which Kalib ignores.

``settled`` is the important one. Frames taken while the arm was moving keep
the video continuous for the tracker, but the RTDE reading and the camera
frame are not synchronised, so only frames where the arm was still are used
as PnP correspondences.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import pathlib
import shutil
from typing import Any, Optional

import cv2
import numpy as np

from policy_abstraction.calibration.geometry import CalibrationMode

KALIB_OUTPUT_DIR = "Kalib"
DEPTH_DIR = "depth"
META_DIR = "meta"
META_FILE = "meta/capture.json"
KEYPOINT_NAME = "tcp"


@dataclasses.dataclass
class CaptureMeta:
    """What the capture was made with. Written once per dataset."""

    camera_name: str
    camera_device: str
    width: int
    height: int
    fps: int
    intrinsic: list[list[float]]
    mode: str
    tcp_offset: list[float]
    point_offset_tcp: list[float] = dataclasses.field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    reference_point_base: Optional[list[float]] = None
    robot_ip: str = ""
    created: str = dataclasses.field(
        default_factory=lambda: _dt.datetime.now().isoformat(timespec="seconds")
    )
    notes: str = ""

    @property
    def calibration_mode(self) -> CalibrationMode:
        return CalibrationMode(self.mode)

    @property
    def parent_frame(self) -> str:
        return self.calibration_mode.parent_frame

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaptureMeta":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclasses.dataclass
class CaptureFrame:
    """One captured sample: image, robot state, and the point's 3-D position."""

    rgb: np.ndarray
    """``(H, W, 3)`` uint8, RGB."""
    joints: np.ndarray
    joint_velocities: np.ndarray
    tcp_pose: np.ndarray
    """``[x, y, z, rx, ry, rz]`` in the base frame, as RTDE reports it."""
    keypoint: np.ndarray
    """The tracked point's position in the PnP parent frame (base for an
    exterior camera, TCP for a wrist camera)."""
    settled: bool
    timestamp: float
    depth_m: Optional[np.ndarray] = None
    """``(H, W)`` float32 metres registered to ``rgb``, or ``None``."""


@dataclasses.dataclass
class FrameRecord:
    """A frame's non-image data, as read back from its JSON."""

    index: int
    keypoint: np.ndarray
    joints: np.ndarray
    tcp_pose: Optional[np.ndarray]
    settled: bool
    timestamp: float
    image_path: pathlib.Path
    depth_path: Optional[pathlib.Path]


class KalibDataset:
    """One capture directory."""

    def __init__(self, root: str | pathlib.Path) -> None:
        self.root = pathlib.Path(root)

    # -- creation -----------------------------------------------------------

    @classmethod
    def create(cls, root: str | pathlib.Path, meta: CaptureMeta) -> "KalibDataset":
        """Start a fresh capture at *root*, removing any previous one.

        Only files this class writes are removed (frames, depth maps, the
        Kalib output folder, the meta file), so pointing it at the wrong
        directory costs a previous capture, not unrelated data.
        """
        ds = cls(root)
        ds.root.mkdir(parents=True, exist_ok=True)
        ds.clear()
        (ds.root / DEPTH_DIR).mkdir(exist_ok=True)
        (ds.root / META_DIR).mkdir(exist_ok=True)
        (ds.root / META_FILE).write_text(json.dumps(meta.to_dict(), indent=2) + "\n")
        return ds

    def clear(self) -> None:
        for p in self.root.glob("[0-9][0-9][0-9][0-9][0-9][0-9].jpg"):
            p.unlink()
        for p in self.root.glob("[0-9][0-9][0-9][0-9][0-9][0-9].json"):
            p.unlink()
        for p in self.root.glob("outputs*.mp4"):
            p.unlink()
        shutil.rmtree(self.root / DEPTH_DIR, ignore_errors=True)
        self.clear_kalib_outputs()
        shutil.rmtree(self.root / META_DIR, ignore_errors=True)

    def clear_kalib_outputs(self) -> None:
        """Kalib skips tracking when it finds its previous result; remove it."""
        shutil.rmtree(self.root / KALIB_OUTPUT_DIR, ignore_errors=True)

    # -- writing ------------------------------------------------------------

    def append(self, frame: CaptureFrame) -> int:
        idx = self.n_frames
        stem = f"{idx:06d}"
        meta = self.meta
        rgb = np.ascontiguousarray(frame.rgb)
        if rgb.shape[:2] != (meta.height, meta.width):
            raise ValueError(
                f"frame is {rgb.shape[1]}x{rgb.shape[0]} but the capture was "
                f"declared {meta.width}x{meta.height}"
            )
        ok = cv2.imwrite(
            str(self.root / f"{stem}.jpg"),
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not ok:
            raise OSError(f"could not write {self.root / (stem + '.jpg')}")
        if frame.depth_m is not None:
            depth_mm = np.clip(
                np.nan_to_num(np.asarray(frame.depth_m, dtype=np.float64)) * 1000.0,
                0,
                65535,
            ).astype(np.uint16)
            (self.root / DEPTH_DIR).mkdir(exist_ok=True)
            cv2.imwrite(str(self.root / DEPTH_DIR / f"{stem}.png"), depth_mm)
        payload = {
            "objects": [
                {
                    "class": "ur5e",
                    "visibility": 1,
                    "camera_intrinsics": meta.intrinsic,
                    "joint_positions": [float(v) for v in frame.joints],
                    "local_to_world_matrix": np.eye(4).tolist(),
                    "keypoints": [
                        {
                            "name": KEYPOINT_NAME,
                            "location": [float(v) for v in frame.keypoint],
                            "projected_location": [-999.0, -999.0],
                            "predicted_location": [-999.0, -999.0],
                        }
                    ],
                }
            ],
            "tcp_pose": [float(v) for v in frame.tcp_pose],
            "joint_velocities": [float(v) for v in frame.joint_velocities],
            "settled": bool(frame.settled),
            "timestamp": float(frame.timestamp),
        }
        (self.root / f"{stem}.json").write_text(json.dumps(payload, indent=2))
        return idx

    # -- reading ------------------------------------------------------------

    @property
    def meta(self) -> CaptureMeta:
        path = self.root / META_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing: {self.root} is not a capture made by this tool."
            )
        return CaptureMeta.from_dict(json.loads(path.read_text()))

    @property
    def n_frames(self) -> int:
        return len(self.image_paths())

    def image_paths(self) -> list[pathlib.Path]:
        return sorted(self.root.glob("[0-9][0-9][0-9][0-9][0-9][0-9].jpg"))

    @property
    def first_frame_path(self) -> pathlib.Path:
        paths = self.image_paths()
        if not paths:
            raise FileNotFoundError(f"no frames in {self.root}")
        return paths[0]

    def records(self) -> list[FrameRecord]:
        out: list[FrameRecord] = []
        for img in self.image_paths():
            idx = int(img.stem)
            js = self.root / f"{img.stem}.json"
            if not js.exists():
                raise FileNotFoundError(f"{img} has no {js.name}")
            d = json.loads(js.read_text())
            obj = d["objects"][0]
            depth = self.root / DEPTH_DIR / f"{img.stem}.png"
            out.append(
                FrameRecord(
                    index=idx,
                    keypoint=np.asarray(
                        obj["keypoints"][0]["location"], dtype=np.float64
                    ),
                    joints=np.asarray(obj["joint_positions"], dtype=np.float64),
                    tcp_pose=(
                        np.asarray(d["tcp_pose"], dtype=np.float64)
                        if "tcp_pose" in d
                        else None
                    ),
                    settled=bool(d.get("settled", True)),
                    timestamp=float(d.get("timestamp", 0.0)),
                    image_path=img,
                    depth_path=depth if depth.exists() else None,
                )
            )
        return out

    def load_rgb(self, index: int) -> np.ndarray:
        bgr = cv2.imread(str(self.root / f"{index:06d}.jpg"), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(self.root / f"{index:06d}.jpg")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def load_depth_m(self, index: int) -> Optional[np.ndarray]:
        path = self.root / DEPTH_DIR / f"{index:06d}.png"
        if not path.exists():
            return None
        mm = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mm is None:
            return None
        return mm.astype(np.float32) / 1000.0

    @property
    def kalib_dir(self) -> pathlib.Path:
        return self.root / KALIB_OUTPUT_DIR


__all__ = [
    "CaptureFrame",
    "CaptureMeta",
    "DEPTH_DIR",
    "FrameRecord",
    "KALIB_OUTPUT_DIR",
    "KEYPOINT_NAME",
    "KalibDataset",
    "META_DIR",
    "META_FILE",
]
