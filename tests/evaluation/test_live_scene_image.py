"""The live scene image is read through the env spec, not a fixed key.

A simulator publishes ``agentview_image`` upside-down; a real camera
publishes whatever it was named, upright. Hard-coding either one is what
broke abstraction setup on the UR5.
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.envs.spec import CameraSpec
from policy_abstraction.evaluation.abstraction import (
    _extract_live_scene_image,
    _render_relevant_seg_image,
)


def _cam(
    name: str = "cam",
    image_key: str = "cam",
    image_transform: str = "none",
    segmentation_key: str | None = None,
) -> CameraSpec:
    return CameraSpec(
        name=name,
        role="world",
        image_key=image_key,
        segmentation_key=segmentation_key,
        image_transform=image_transform,
        is_primary=True,
    )


@pytest.fixture
def rgb() -> np.ndarray:
    return np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)


class TestExtractLiveSceneImage:
    def test_reads_the_specs_image_key(self, rgb):
        """A real camera's key is its own name, not 'agentview_image'."""
        cam = _cam(name="exterior_image_1_left", image_key="exterior_image_1_left")
        got = _extract_live_scene_image({"exterior_image_1_left": rgb}, cam)
        assert got is not None and np.array_equal(got, rgb)

    def test_flip_vh_spec_uprights_a_simulator_frame(self, rgb):
        cam = _cam(image_key="agentview_image", image_transform="flip_vh")
        got = _extract_live_scene_image({"agentview_image": rgb}, cam)
        assert got is not None and np.array_equal(got, rgb[::-1, ::-1])

    def test_none_transform_leaves_a_real_frame_alone(self, rgb):
        """The old hard-coded flip would have mirrored this."""
        got = _extract_live_scene_image({"cam": rgb}, _cam(image_transform="none"))
        assert got is not None and np.array_equal(got, rgb)

    def test_missing_key_returns_none(self, rgb):
        assert _extract_live_scene_image({"other": rgb}, _cam()) is None

    def test_result_is_contiguous(self, rgb):
        got = _extract_live_scene_image({"cam": rgb}, _cam(image_transform="flip_vh"))
        assert got is not None and got.flags["C_CONTIGUOUS"]


class TestRenderRelevantSegImage:
    def test_camera_without_segmentation_returns_none(self, rgb):
        """Every real camera is in this case; it must degrade, not raise."""
        cam = _cam(segmentation_key=None)
        assert _render_relevant_seg_image({"cam": rgb}, {1}, cam) is None

    def test_missing_seg_in_obs_returns_none(self, rgb):
        cam = _cam(segmentation_key="cam_seg")
        assert _render_relevant_seg_image({"cam": rgb}, {1}, cam) is None

    def test_renders_when_the_spec_names_a_seg_key(self, rgb):
        cam = _cam(segmentation_key="cam_seg")
        seg = np.zeros((2, 3), dtype=np.int32)
        seg[0, :2] = 1
        out = _render_relevant_seg_image({"cam": rgb, "cam_seg": seg}, {1}, cam)
        assert out is not None and out.shape == rgb.shape
