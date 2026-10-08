from __future__ import annotations

import numpy as np
import pytest
from open_perception.types import CameraInfo, PerceptionInput


def test_normalized_scalar_rgb_expands_to_singleton_list():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    norm = PerceptionInput(queries="mug", rgb=rgb).normalized()
    assert norm.queries == ["mug"]
    assert len(norm.rgbs) == 1
    assert norm.rgbs[0] is rgb
    assert norm.depths == [None]
    assert norm.cameras == [None]


def test_normalized_multiview_rgb_as_stack():
    rgbs = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    norm = PerceptionInput(queries=["mug"], rgb=rgbs).normalized()
    assert len(norm.rgbs) == 3


def test_normalized_depth_rgb_mismatch_raises():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = [np.zeros((8, 8), dtype=np.float32), np.zeros((8, 8), dtype=np.float32)]
    with pytest.raises(ValueError, match="depth"):
        PerceptionInput(queries="mug", rgb=rgb, depth=depth).normalized()


def test_normalized_cameras_singleton_broadcasts():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    cam = CameraInfo(intrinsics=np.eye(3, dtype=np.float32))
    norm = PerceptionInput(queries="mug", rgb=rgb, cameras=cam).normalized()
    assert norm.cameras == [cam]


def test_normalized_defaults_options():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    norm = PerceptionInput(queries="mug", rgb=rgb).normalized()
    assert norm.options.drift_policy == "surface"
