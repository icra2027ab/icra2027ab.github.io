"""The training heat map: loading, projection, blending, caching."""

from __future__ import annotations

import json

import numpy as np
import pytest

from policy_abstraction.demo.heatmap import TrainingHeatmap
from policy_abstraction.demo.overlay import render_live_view
from policy_abstraction.demo.session import LiveView

H, W = 120, 160
K = np.array([[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1.0]])
# One metre above the origin looking down: base (x, y) lands near the centre.
E = np.array([[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]])


@pytest.fixture
def positions_file(tmp_path):
    grasps = [[0.05 * i - 0.1, 0.02 * i - 0.05, 0.04] for i in range(6)]
    path = tmp_path / "demo_grasp_positions.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "demo",
                "frame": "robot base",
                "demos": {"a": {"grasps": grasps[:3]}, "b": {"grasps": grasps[3:]}},
            }
        )
    )
    return path


def test_load_reads_every_demos_grasps(positions_file):
    hm = TrainingHeatmap.load(positions_file)
    assert hm.count == 6
    assert hm.table_z == pytest.approx(0.04)
    assert hm.source.endswith("demo_grasp_positions.json")


def test_a_plain_grasps_list_also_loads(tmp_path):
    path = tmp_path / "g.json"
    path.write_text(json.dumps({"grasps": [[0, 0, 0.03], [0.1, 0, 0.03]]}))
    assert TrainingHeatmap.load(path).count == 2


def test_no_grasps_is_refused():
    with pytest.raises(ValueError):
        TrainingHeatmap(np.zeros((0, 3)))


def test_blend_warms_the_table_where_the_demos_grasped(positions_file):
    hm = TrainingHeatmap.load(positions_file)
    img = np.full((H, W, 3), 40, dtype=np.uint8)
    hm.blend(img, K, E)
    changed = (img != 40).any(axis=-1)
    assert changed.any(), "the map was drawn"
    # Grasps sit near the image centre; the corners stay untouched.
    assert changed[H // 2 - 5 : H // 2 + 5, W // 2 - 10 : W // 2 + 10].mean() > 0.5
    assert not changed[:5, :5].any() and not changed[-5:, -5:].any()


def test_the_layer_is_cached_per_camera(positions_file):
    hm = TrainingHeatmap.load(positions_file)
    a = hm.layer(H, W, K, E)
    b = hm.layer(H, W, K, E)
    assert a[0] is b[0], "same camera, same layer object"
    c = hm.layer(H, W, K * 1.1, E)
    assert c[0] is not a[0], "a new camera re-projects"


def test_a_footprint_behind_the_camera_is_refused(positions_file):
    hm = TrainingHeatmap.load(positions_file)
    E_behind = E.copy()
    E_behind[2, 3] = -1.0  # camera below the table, looking down: all behind
    with pytest.raises(ValueError):
        hm.layer(H, W, K, E_behind)


def test_render_live_view_underlays_the_map_and_says_when_it_cannot(positions_file):
    hm = TrainingHeatmap.load(positions_file)
    view = LiveView(
        image=np.full((H, W, 3), 40, dtype=np.uint8),
        timestamp=1.0,
        intrinsic=K,
        extrinsic=E,
    )
    out = render_live_view(view, show_overlays=False, heatmap=hm)
    assert (out[H // 2, W // 2] != 40).any()
    assert np.all(view.image == 40)
    uncalibrated = LiveView(image=view.image, timestamp=1.0)
    out2 = render_live_view(uncalibrated, show_overlays=False, heatmap=hm)
    # Nothing but a badge top-left: the centre pixel is untouched.
    assert (out2[H // 2, W // 2] == 40).all()
    assert (out2[:30, : W // 2] != 40).any()
