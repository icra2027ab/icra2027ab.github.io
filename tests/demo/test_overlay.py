"""The demo overlay: masks, pairs and readouts on a synthetic frame."""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.demo.overlay import (
    COLOR_GUIDED,
    draw_masks,
    object_colors,
    render_live_view,
)
from policy_abstraction.demo.session import LiveView
from policy_abstraction.visualization.drawing import pair_colors
from policy_abstraction.vlm.types import DistancePair

H, W = 120, 160
K = np.array([[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1.0]])
# Camera 1 m above the origin looking straight down: world (x, y, 0) lands
# near the image centre, offset by x and y.
E = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


class _Result:
    def __init__(self, pairs):
        self.pairs = pairs
        self.live_distances = {p.label: 0.15 for p in pairs}
        self.ref_distances = {p.label: 0.12 for p in pairs}
        self.distance_errors = {p.label: 0.03 for p in pairs}
        self.entity_centroids = {
            "ee": np.array([-0.2, 0.0, 0.0]),
            5: np.array([0.2, 0.0, 0.0]),
            6: np.array([0.0, 0.2, 0.0]),
        }
        self.current_subtraj_idx = 0


def _view(pairs, *, calibrated=True, seg=True) -> LiveView:
    image = np.full((H, W, 3), 40, dtype=np.uint8)
    seg_map = np.zeros((H, W), dtype=np.int32)
    if seg:
        seg_map[50:70, 100:130] = 5
        seg_map[10:30, 60:90] = 6
    return LiveView(
        image=image,
        step=4,
        timestamp=1.0,
        guided=True,
        intrinsic=K if calibrated else None,
        extrinsic=E if calibrated else None,
        abstraction=_Result(pairs),
        seg=seg_map if seg else None,
        names={5: "cup", 6: "bowl"},
        ee_pos=np.array([-0.2, 0.0, 0.0]),
        colors=pair_colors(pairs),
    )


@pytest.fixture
def pairs():
    return [
        DistancePair("ee", 5, "ee_to_cup", subtraj_idx=0),
        DistancePair("ee", 6, "ee_to_bowl", subtraj_idx=1),
    ]


def test_masks_tint_exactly_the_instances_named(pairs):
    view = _view(pairs)
    img = view.image.copy()
    draw_masks(img, view.seg, view.names, object_colors(view.names), label=False)
    inside = img[55:65, 105:125]
    outside = img[90:110, 10:30]
    assert not np.all(inside == 40), "the cup's pixels were tinted"
    assert np.all(outside == 40), "the table was left alone"
    # The two objects have different colours.
    assert not np.array_equal(img[60, 115], img[20, 75])


def test_render_draws_pairs_in_their_own_colours(pairs):
    view = _view(pairs)
    out = render_live_view(
        view, show_overlays=True, banner="GUIDED", banner_color=COLOR_GUIDED
    )
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    assert np.all(view.image == 40), "the source frame is never modified"
    colour = np.array(view.colors["ee_to_cup"])
    # The ee->cup line runs horizontally through the image centre; some
    # pixel on that row carries the pair's colour.
    hits = np.all(np.abs(out.astype(int) - colour) < 40, axis=-1)
    assert hits.any()
    # And the picture differs from the bare frame far more than a banner would.
    assert (out != 40).any(axis=-1).mean() > 0.05


def test_without_overlays_only_the_banner_is_drawn(pairs):
    view = _view(pairs)
    out = render_live_view(view, show_overlays=False, banner="UNGUIDED")
    changed = (out != 40).any(axis=-1)
    assert changed[: H // 2].sum() == 0, "nothing above the bottom strip"
    assert changed[H - 5 :].any()


def test_no_calibration_says_so_instead_of_guessing(pairs):
    view = _view(pairs, calibrated=False)
    out = render_live_view(view, show_overlays=True)
    # Masks still drawn (they need no projection), badge in the corner.
    assert (out[55:65, 105:125] != 40).any()
    assert (out[:30, : W // 2] != 40).any(), "a badge was drawn top-left"


def test_an_unanalysed_view_renders_the_frame(pairs):
    view = LiveView(image=np.full((H, W, 3), 40, dtype=np.uint8), timestamp=1.0)
    out = render_live_view(view, show_overlays=True, badges=["waiting"])
    assert out.shape == (H, W, 3)
    with pytest.raises(ValueError):
        render_live_view(LiveView(), show_overlays=False)


def test_pair_colours_match_the_shared_palette(pairs):
    colors = pair_colors(pairs)
    assert set(colors) == {"ee_to_cup", "ee_to_bowl"}
    assert colors["ee_to_cup"] != colors["ee_to_bowl"]
    assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in colors.values())
