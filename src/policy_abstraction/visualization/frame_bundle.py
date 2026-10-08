"""FrameBundle: container for all rendered images at one evaluation timestep."""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass
class FrameBundle:
    """All annotated images produced by :class:`VisualizationHandler` for one timestep.

    Camera-keyed dicts (instead of fixed primary/wrist fields) so envs with
    1, 2, or N cameras all flow through the same shape. Consumers (Rerun
    visualizer, video saver, abstraction-side renderers) look up images by
    camera name.
    """

    images: dict[str, np.ndarray]
    """Raw upright frame per camera, keyed by ``CameraSpec.name``."""

    primary_name: str
    """Name of the primary camera (key into ``images``)."""

    chunk_overlays: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    """Per-camera image with the action-chunk EEF trajectory projected onto it.
    Only populated for cameras the chunk projector has intrinsics for."""

    combined_replay: Optional[np.ndarray] = None
    """Horizontal concatenation of available chunk overlays / raw frames.
    Used by the video saver."""

    ref_images: dict[str, np.ndarray] = dataclasses.field(default_factory=dict)
    """Reference demo frame per camera at ``aligned_t``."""

    ref_image_annotated: Optional[np.ndarray] = None
    """Primary reference frame with pair lines drawn."""

    live_image_annotated: Optional[np.ndarray] = None
    """Primary live frame with pair lines and distance annotations drawn."""

    alignment_plot: Optional[np.ndarray] = None
    """``[H, W, 3]`` uint8 RGB alignment multiplot (ref overview + live + zoom)."""

    constraint_panel: np.ndarray | None = None
    """``[H, W, 3]`` uint8 RGB constraint-satisfaction panel, grouped by sub-trajectory."""

    # ------------------------------------------------------------------
    # Convenience accessors (keep older callers ergonomic without forcing
    # them to re-derive primary / wrist names).
    # ------------------------------------------------------------------

    @property
    def primary_image(self) -> np.ndarray:
        return self.images[self.primary_name]

    @property
    def primary_chunk_overlay(self) -> Optional[np.ndarray]:
        return self.chunk_overlays.get(self.primary_name)

    @property
    def ref_primary_image(self) -> Optional[np.ndarray]:
        return self.ref_images.get(self.primary_name)
