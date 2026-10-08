"""Where the training demos put the object, as a heat map on the live table.

The checkpoint was trained from demonstrations recorded in this cell, and
every grasp in them is a place the policy has seen the object. Laid over the
live camera as a density on the table plane, that is the training
distribution the audience can compare the staged scene against: a cable on
the warm part of the map is in distribution, one on the cold table is not.

The positions come from ``scripts/experiments/plot_training_on_grid.py``,
which reads each demo's proprioception and takes every moment the gripper
closed low as a grasp, writing them to ``<dataset>_grasp_positions.json`` in
the robot base frame. That file is what :meth:`TrainingHeatmap.load` reads.
The camera never moves during a demo, so the projected layer is rendered
once per camera and blended into every frame.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Optional

import cv2
import numpy as np

from policy_abstraction.utils.pcd import project_points_to_pixels


@dataclasses.dataclass
class TrainingHeatmap:
    """A kernel density of grasp positions, projectable onto a camera frame.

    Args:
        grasps: ``[N, 3]`` base-frame positions (metres).
        bandwidth: Gaussian kernel width on the table, metres.
        alpha_max: Opacity at the density's peak.
        floor: Densities below this fraction of the peak are fully
            transparent, so the map's tail does not tint the whole table.
        margin: Table margin around the grasps that the raster covers.
        resolution: Raster cell size on the table, metres.
    """

    grasps: np.ndarray
    bandwidth: float = 0.04
    alpha_max: float = 0.55
    floor: float = 0.05
    margin: float = 0.12
    resolution: float = 0.005
    gamma: float = 0.5
    """Exponent applied to the normalised density before colouring. Below
    one it lifts the tail, so the spread of the demos shows and not only
    their densest spot."""
    source: str = ""

    def __post_init__(self) -> None:
        self.grasps = np.asarray(self.grasps, dtype=np.float64).reshape(-1, 3)
        if len(self.grasps) == 0:
            raise ValueError("a training heat map needs at least one grasp")
        self._density: Optional[np.ndarray] = None
        self._cache_key: Optional[tuple] = None
        self._layer: Optional[tuple[np.ndarray, np.ndarray]] = None

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | pathlib.Path, **kwargs) -> "TrainingHeatmap":
        """Read a ``*_grasp_positions.json`` written by
        ``plot_training_on_grid.py`` (or any ``{"demos": {name: {"grasps":
        [[x, y, z], ...]}}}`` / ``{"grasps": [...]}`` file)."""
        path = pathlib.Path(path)
        data = json.loads(path.read_text())
        if "demos" in data:
            pts = [p for demo in data["demos"].values() for p in demo.get("grasps", [])]
        else:
            pts = list(data.get("grasps", []))
        return cls(np.asarray(pts, dtype=np.float64), source=str(path), **kwargs)

    @property
    def count(self) -> int:
        return int(len(self.grasps))

    @property
    def table_z(self) -> float:
        """The plane the map is drawn on: the grasps' median height."""
        return float(np.median(self.grasps[:, 2]))

    # ------------------------------------------------------------------

    def _table_raster(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Top-down density over the grasps' footprint plus a margin.

        Returns ``(density [len(ys), len(xs)], xs, ys)`` with the density's
        peak at 1. Computed once.
        """
        if self._density is not None:
            return self._density, self._xs, self._ys
        x0, y0 = self.grasps[:, :2].min(axis=0) - self.margin
        x1, y1 = self.grasps[:, :2].max(axis=0) + self.margin
        self._xs = np.arange(x0, x1 + self.resolution, self.resolution)
        self._ys = np.arange(y0, y1 + self.resolution, self.resolution)
        xx, yy = np.meshgrid(self._xs, self._ys)
        dens = np.zeros_like(xx)
        inv = 1.0 / (2.0 * self.bandwidth * self.bandwidth)
        for px, py in self.grasps[:, :2]:
            dens += np.exp(-((xx - px) ** 2 + (yy - py) ** 2) * inv)
        dens /= max(float(dens.max()), 1e-9)
        self._density = dens
        return dens, self._xs, self._ys

    def layer(
        self,
        height: int,
        width: int,
        K: np.ndarray,
        E: np.ndarray,
        *,
        preprocessed: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The map warped into a ``height x width`` camera image.

        Returns ``(rgb [H, W, 3] float32, alpha [H, W] float32)``, cached
        for the camera, so per-frame blending is a single weighted sum.
        Raises ``ValueError`` when the table footprint does not project.
        """
        K = np.asarray(K, dtype=np.float64)
        K3 = K[:3, :3] if K.shape == (4, 4) else K
        E = np.asarray(E, dtype=np.float64)
        key = (height, width, bool(preprocessed), K3.tobytes(), E.tobytes())
        if self._layer is not None and self._cache_key == key:
            return self._layer

        dens, xs, ys = self._table_raster()
        z = self.table_z
        corners = np.array(
            [
                [xs[0], ys[0], z],
                [xs[-1], ys[0], z],
                [xs[-1], ys[-1], z],
                [xs[0], ys[-1], z],
            ]
        )
        px, ok = project_points_to_pixels(
            corners, K3, E, height, width, preprocessed=preprocessed
        )
        if not ok.all():
            raise ValueError("the training footprint does not project into the camera")
        src = np.array(
            [[0, 0], [len(xs) - 1, 0], [len(xs) - 1, len(ys) - 1], [0, len(ys) - 1]],
            dtype=np.float32,
        )
        H = cv2.getPerspectiveTransform(src, px.astype(np.float32))
        warped = cv2.warpPerspective(
            dens.astype(np.float32), H, (width, height), flags=cv2.INTER_LINEAR
        )
        from matplotlib import colormaps

        shaped = np.clip(warped, 0, 1) ** max(float(self.gamma), 1e-3)
        rgb = (colormaps["inferno"](shaped)[..., :3] * 255).astype(np.float32)
        alpha = (
            np.clip((shaped - self.floor) / max(1.0 - self.floor, 1e-6), 0, 1)
            * self.alpha_max
        ).astype(np.float32)
        self._layer = (rgb, alpha)
        self._cache_key = key
        return self._layer

    def blend(
        self,
        image: np.ndarray,
        K: np.ndarray,
        E: np.ndarray,
        *,
        preprocessed: bool = False,
    ) -> None:
        """Blend the map into *image* (``[H, W, 3]`` uint8 RGB), in place."""
        h, w = image.shape[:2]
        rgb, alpha = self.layer(h, w, K, E, preprocessed=preprocessed)
        a = alpha[..., None]
        image[:] = np.clip(
            a * rgb + (1.0 - a) * image.astype(np.float32), 0, 255
        ).astype(np.uint8)


__all__ = ["TrainingHeatmap"]
