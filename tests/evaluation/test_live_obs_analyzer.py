"""Tests for LiveObsAnalyzer in policy_abstraction.evaluation.abstraction."""

from __future__ import annotations

import numpy as np

from policy_abstraction.evaluation.abstraction import LiveObsAnalyzer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_obs(
    H: int = 16,
    W: int = 16,
    seg_ids: list[int] | None = None,
    include_image: bool = True,
    include_seg: bool = True,
) -> dict:
    """Return a minimal synthetic obs dict."""
    obs: dict = {}
    if include_image:
        rng = np.random.default_rng(0)
        obs["agentview_image"] = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
    if include_seg:
        seg = np.zeros((H, W), dtype=np.int32)
        if seg_ids:
            # Tile seg IDs across rows
            for i, sid in enumerate(seg_ids):
                row_start = (i * H) // len(seg_ids)
                row_end = ((i + 1) * H) // len(seg_ids)
                seg[row_start:row_end, :] = sid
        obs["agentview_segmentation_instance"] = seg
    return obs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLiveObsAnalyzer:
    """Tests for LiveObsAnalyzer.analyze."""

    def test_missing_seg_key_returns_empty_result(self):
        """obs without segmentation key → empty objects and seg_ids."""
        obs = _make_obs(include_seg=False)
        analyzer = LiveObsAnalyzer()
        result = analyzer.analyze(obs, "pick up the mug")
        assert result.objects == {}
        assert result.all_seg_ids == set()
        assert result.seg_index_image is None

    def test_visible_ids_collected(self):
        """Seg IDs present in mask are all collected in all_seg_ids."""
        obs = _make_obs(seg_ids=[1, 2, 3])
        analyzer = LiveObsAnalyzer()
        result = analyzer.analyze(obs, "task")
        assert {1, 2, 3} == result.all_seg_ids

    def test_background_zero_excluded(self):
        """Background id=0 is never included in all_seg_ids or objects."""
        obs = _make_obs(seg_ids=[0])  # all-background mask
        analyzer = LiveObsAnalyzer()
        result = analyzer.analyze(obs, "task")
        assert 0 not in result.all_seg_ids
        assert 0 not in result.objects

    def test_name_map_applied(self):
        """Seg IDs in the name map are resolved to their names."""
        obs = _make_obs(seg_ids=[1, 2])
        name_map = {"1": "mug", "2": "plate"}
        analyzer = LiveObsAnalyzer(seg_name_map=name_map)
        result = analyzer.analyze(obs, "task")
        assert result.objects[1] == "mug"
        assert result.objects[2] == "plate"

    def test_fallback_name_is_str_seg_id(self):
        """Seg IDs absent from the name map fall back to str(seg_id)."""
        obs = _make_obs(seg_ids=[7])
        analyzer = LiveObsAnalyzer()  # no name map
        result = analyzer.analyze(obs, "task")
        assert result.objects[7] == "7"

    def test_scene_image_is_flipped(self):
        """scene_image should be the 180°-flipped version of agentview_image."""
        H, W = 4, 4
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[0, 0] = [255, 0, 0]  # top-left corner is red
        obs = _make_obs(H=H, W=W, include_seg=False)
        obs["agentview_image"] = img
        analyzer = LiveObsAnalyzer()
        result = analyzer.analyze(obs, "task")
        assert result.scene_image is not None
        # After 180° flip, the red corner should now be at bottom-right
        np.testing.assert_array_equal(result.scene_image[-1, -1], [255, 0, 0])
        np.testing.assert_array_equal(result.scene_image[0, 0], [0, 0, 0])

    def test_seg_index_image_shape_and_dtype(self):
        """seg_index_image (when generated) has shape (H, W, 3) and dtype uint8."""
        H, W = 8, 8
        obs = _make_obs(H=H, W=W, seg_ids=[1, 2])
        analyzer = LiveObsAnalyzer()
        result = analyzer.analyze(obs, "task")
        if result.seg_index_image is not None:
            assert result.seg_index_image.shape == (H, W, 3)
            assert result.seg_index_image.dtype == np.uint8

    def test_no_image_seg_index_is_none(self):
        """Without agentview_image, seg_index_image cannot be generated."""
        obs = _make_obs(include_image=False, seg_ids=[1])
        analyzer = LiveObsAnalyzer()
        result = analyzer.analyze(obs, "task")
        assert result.seg_index_image is None

    def test_scene_image_none_when_no_image(self):
        """Without agentview_image, scene_image is None."""
        obs = _make_obs(include_image=False, include_seg=False)
        analyzer = LiveObsAnalyzer()
        result = analyzer.analyze(obs, "task")
        assert result.scene_image is None

    def test_partial_name_map_applies_to_known_ids(self):
        """Name map is applied only to known IDs; unknown ones get fallback."""
        obs = _make_obs(seg_ids=[1, 2, 5])
        name_map = {"1": "mug"}  # only id=1 named
        analyzer = LiveObsAnalyzer(seg_name_map=name_map)
        result = analyzer.analyze(obs, "task")
        assert result.objects[1] == "mug"
        assert result.objects[2] == "2"  # fallback
        assert result.objects[5] == "5"  # fallback
