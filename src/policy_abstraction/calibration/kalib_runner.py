"""Running Kalib's tracker on a capture, from outside its conda env.

Kalib (SpaTracker, PyTorch, its own CUDA build) lives in the ``kalib`` conda
env and cannot be imported from the venv that owns the RealSense and RTDE
bindings, so tracking runs as a subprocess: ``scripts/setup/_kalib_track_driver.py``
is executed with Kalib's interpreter, with the operator's click passed on the
command line instead of through Kalib's own OpenCV click window. It writes one
JSON file with the tracked pixel per frame, and this module reads it back.

Kalib's own PnP result rides along as a cross-check; the pose that gets saved
comes from :mod:`policy_abstraction.calibration.solve`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import subprocess
import threading
import time
from typing import Callable, Optional, Protocol

import numpy as np

from policy_abstraction.calibration.dataset import KalibDataset

LogFn = Callable[[str], None]

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_KALIB_ROOT = pathlib.Path("~/dev/Kalib")
DEFAULT_KALIB_PYTHON = pathlib.Path("~/miniconda3/envs/kalib/bin/python")
DEFAULT_DRIVER = _REPO_ROOT / "scripts" / "setup" / "_kalib_track_driver.py"
RESULT_FILE = "tracking_result.json"
LOG_FILE = "kalib_log.txt"


@dataclasses.dataclass
class TrackingResult:
    tracks: np.ndarray
    """``(N, 2)`` pixel per frame; nan where the tracker lost the point."""
    visibility: np.ndarray
    """``(N,)`` bool from the tracker."""
    kalib_T_cam_parent: Optional[np.ndarray] = None
    """Kalib's own PnP answer (``X_cam = T @ X_parent``) when it ran."""
    kalib_reproj_px: Optional[float] = None
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tracks": [
                [float(p[0]), float(p[1])] if np.all(np.isfinite(p)) else [None, None]
                for p in np.asarray(self.tracks).reshape(-1, 2)
            ],
            "visibility": [bool(v) for v in self.visibility],
            "kalib_T_cam_parent": (
                None
                if self.kalib_T_cam_parent is None
                else np.asarray(self.kalib_T_cam_parent).tolist()
            ),
            "kalib_reproj_px": self.kalib_reproj_px,
            "elapsed_s": self.elapsed_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrackingResult":
        raw = d["tracks"]
        tracks = np.array(
            [
                [np.nan, np.nan] if (p is None or p[0] is None) else [p[0], p[1]]
                for p in raw
            ],
            dtype=np.float64,
        ).reshape(-1, 2)
        vis = np.asarray(d.get("visibility", [True] * len(tracks)), dtype=bool)
        T = d.get("kalib_T_cam_parent")
        return cls(
            tracks=tracks,
            visibility=vis,
            kalib_T_cam_parent=None if T is None else np.asarray(T, dtype=np.float64),
            kalib_reproj_px=d.get("kalib_reproj_px"),
            elapsed_s=float(d.get("elapsed_s", 0.0)),
        )


class Tracker(Protocol):
    def track(
        self, dataset: KalibDataset, click_uv, log: Optional[LogFn] = None
    ) -> TrackingResult: ...


@dataclasses.dataclass
class KalibConfig:
    root: pathlib.Path = DEFAULT_KALIB_ROOT
    python: pathlib.Path = DEFAULT_KALIB_PYTHON
    driver: pathlib.Path = DEFAULT_DRIVER
    device_id: int = 0
    tracker: str = "spatial_tracker"
    cuda_home: Optional[str] = None
    """Prepended to PATH/LD_LIBRARY_PATH for the subprocess. Kalib's SpaTracker
    needed CUDA 12.9 on this host (see ~/dev/notes.txt); auto-detected when
    unset."""
    timeout_s: float = 3600.0

    def __post_init__(self) -> None:
        self.root = pathlib.Path(os.path.expanduser(str(self.root)))
        self.python = pathlib.Path(os.path.expanduser(str(self.python)))
        self.driver = pathlib.Path(os.path.expanduser(str(self.driver)))
        if self.cuda_home is None:
            for cand in (
                "/usr/local/cuda-12.9",
                "/usr/local/cuda-12.8",
                "/usr/local/cuda",
            ):
                if pathlib.Path(cand).is_dir():
                    self.cuda_home = cand
                    break

    @classmethod
    def from_env(cls, **overrides) -> "KalibConfig":
        kw: dict = {}
        if os.environ.get("KALIB_ROOT"):
            kw["root"] = os.environ["KALIB_ROOT"]
        if os.environ.get("KALIB_PYTHON"):
            kw["python"] = os.environ["KALIB_PYTHON"]
        if os.environ.get("KALIB_DEVICE"):
            kw["device_id"] = int(os.environ["KALIB_DEVICE"])
        if os.environ.get("KALIB_CUDA_HOME"):
            kw["cuda_home"] = os.environ["KALIB_CUDA_HOME"]
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    def problems(self) -> list[str]:
        """Why tracking would fail, before it does."""
        out: list[str] = []
        if not self.root.is_dir():
            out.append(f"Kalib checkout not found at {self.root} (set KALIB_ROOT)")
        elif not (self.root / "easycalib_demo.py").exists():
            out.append(f"{self.root} has no easycalib_demo.py; is it the Kalib repo?")
        elif self.tracker == "spatial_tracker":
            ckpt = self.root / "third_party" / "spatial_tracker" / "checkpoints"
            if not any(ckpt.glob("*.pth")) if ckpt.is_dir() else True:
                out.append(f"no SpaTracker checkpoint under {ckpt}")
        if not self.python.exists():
            out.append(
                f"Kalib interpreter not found at {self.python} (set KALIB_PYTHON)"
            )
        if not self.driver.exists():
            out.append(f"driver script missing at {self.driver}")
        return out

    def environment(self) -> dict[str, str]:
        env = dict(os.environ)
        if self.cuda_home:
            env["CUDA_HOME"] = self.cuda_home
            env["PATH"] = f"{self.cuda_home}/bin:" + env.get("PATH", "")
            env["LD_LIBRARY_PATH"] = f"{self.cuda_home}/lib64:" + env.get(
                "LD_LIBRARY_PATH", ""
            )
        # Kalib's demo switches matplotlib to TkAgg at import; keep it headless.
        env.setdefault("MPLBACKEND", "Agg")
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def command(
        self, dataset: KalibDataset, click_uv, result_path: pathlib.Path
    ) -> list[str]:
        u, v = float(click_uv[0]), float(click_uv[1])
        return [
            str(self.python),
            str(self.driver),
            "--root-dir",
            str(dataset.root.resolve()),
            "--kalib-root",
            str(self.root),
            "--click",
            f"{u:.3f}",
            f"{v:.3f}",
            "--device",
            str(self.device_id),
            "--tracker",
            self.tracker,
            "--result",
            str(pathlib.Path(result_path).resolve()),
        ]


class KalibTracker:
    """Runs Kalib's tracker (and PnP) in its own env and reads the result back."""

    def __init__(self, config: Optional[KalibConfig] = None) -> None:
        self.config = config or KalibConfig.from_env()
        self._proc: Optional[subprocess.Popen] = None

    def track(
        self,
        dataset: KalibDataset,
        click_uv,
        log: Optional[LogFn] = None,
        cancel: Optional[threading.Event] = None,
    ) -> TrackingResult:
        problems = self.config.problems()
        if problems:
            raise RuntimeError("Kalib is not runnable:\n  - " + "\n  - ".join(problems))
        log = log or (lambda _s: None)
        dataset.clear_kalib_outputs()
        dataset.kalib_dir.mkdir(parents=True, exist_ok=True)
        # The driver runs with Kalib's checkout as its working directory, so
        # every path handed to it must be absolute.
        result_path = dataset.kalib_dir.resolve() / RESULT_FILE
        log_path = dataset.kalib_dir.resolve() / LOG_FILE
        cmd = self.config.command(dataset, click_uv, result_path)
        log("$ " + " ".join(cmd))
        t0 = time.monotonic()
        lines: list[str] = []
        with open(log_path, "w", encoding="utf-8") as fh:
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.config.root),
                env=self.config.environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                line = line.rstrip("\n")
                fh.write(line + "\n")
                lines.append(line)
                log(line)
                if cancel is not None and cancel.is_set():
                    self._proc.terminate()
                    raise RuntimeError("tracking cancelled")
                if time.monotonic() - t0 > self.config.timeout_s:
                    self._proc.kill()
                    raise RuntimeError(
                        f"tracking exceeded {self.config.timeout_s:.0f} s"
                    )
            rc = self._proc.wait()
        self._proc = None
        if rc != 0:
            tail = "\n".join(lines[-25:])
            raise RuntimeError(
                f"Kalib driver exited with status {rc}. Last output:\n{tail}\n"
                f"(full log: {log_path})"
            )
        if not result_path.exists():
            raise RuntimeError(f"Kalib driver produced no {result_path}")
        result = TrackingResult.from_dict(json.loads(result_path.read_text()))
        result.elapsed_s = time.monotonic() - t0
        n = dataset.n_frames
        if result.tracks.shape[0] != n:
            raise RuntimeError(
                f"tracker returned {result.tracks.shape[0]} frames for a capture of {n}"
            )
        return result

    def cancel(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


__all__ = [
    "DEFAULT_DRIVER",
    "DEFAULT_KALIB_PYTHON",
    "DEFAULT_KALIB_ROOT",
    "KalibConfig",
    "KalibTracker",
    "LOG_FILE",
    "RESULT_FILE",
    "Tracker",
    "TrackingResult",
]
