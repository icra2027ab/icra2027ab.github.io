"""The trend panel: reference above, live below, guided stretches shaded."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("PyQt6", reason="PyQt6 is not installed")
pytest.importorskip("pyqtgraph", reason="pyqtgraph is not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from policy_abstraction.demo.gui.trends import TrendsPanel  # noqa: E402
from policy_abstraction.demo.session import ReferenceTrends, TrendSample  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture
def trends() -> ReferenceTrends:
    t = np.linspace(0, 1, 100, dtype=np.float32)
    ref = np.stack([0.3 - 0.25 * t, 0.1 + 0.2 * np.sin(3 * t)], axis=1)
    return ReferenceTrends(
        ref_traj=ref,
        ref_std=np.full_like(ref, 0.01),
        pair_labels=("ee_to_cup", "ee_to_bowl"),
        pair_subtraj=(0, 1),
        subtrajectories=((0, 50), (50, 100)),
        colors={"ee_to_cup": (255, 80, 80), "ee_to_bowl": (80, 80, 255)},
        task="pour",
    )


def _samples(n: int, guided_range=(20, 35)) -> list[TrendSample]:
    out = []
    for step in range(n):
        live = np.array([0.3 - 0.002 * step, 0.1 + 0.001 * step], dtype=np.float32)
        if step % 7 == 3:
            live[1] = np.nan
        out.append(
            TrendSample(
                step=step,
                live=live,
                aligned_t=min(step, 99),
                aligned_ts=np.array([min(step, 99), min(step + 2, 99)], dtype=np.int32),
                current_subtraj_idx=0 if step < 50 else 1,
                guidance_weights=np.ones(2, dtype=np.float32),
                relevances={"cup": 1.0 - step / 100},
                guided=guided_range[0] <= step <= guided_range[1],
                replan=step % 5 == 0,
                chunk_size=5,
                # Half strength for the first guided chunk, full after.
                guidance_factor=(
                    (0.5 if step < guided_range[0] + 5 else 1.0)
                    if guided_range[0] <= step <= guided_range[1]
                    else 0.0
                ),
                guidance_scale=1.0,
            )
        )
    return out


def test_reference_makes_one_curve_per_pair(qapp, trends):
    panel = TrendsPanel()
    panel.set_reference(trends)
    assert len(panel._ref_curves) == 2
    assert len(panel._ref_bands) == 2, "a std band per pair"
    assert len(panel._ref_regions) == 2, "a band per sub-trajectory"
    assert panel.pair_count == 2
    assert panel.legend is None, "no legend by default: the pairs panel is it"


def test_the_legend_is_optional(qapp, trends):
    panel = TrendsPanel(show_legend=True)
    panel.set_reference(trends)
    assert panel.legend is not None
    assert panel.legend.labels() == ["ee_to_cup", "ee_to_bowl"]
    assert "ee_to_cup" in panel.legend.text()
    assert "#ff5050" in panel.legend.rows[0].text(), "the row is in the pair colour"
    assert panel.legend.width() == 220
    assert panel.legend.mode.text() == panel.mode_text
    panel.clear()
    assert panel.legend.labels() == [] and panel.legend.text() == ""


def test_history_appends_and_shades_the_guided_stretch(qapp, trends):
    panel = TrendsPanel()
    panel.set_reference(trends)
    panel.update_history(_samples(50))
    assert panel.sample_count == 50
    fraction = panel.guidance_fraction
    assert len(fraction) == 50
    assert np.all(fraction[:20] == 0) and np.all(fraction[36:] == 0)
    assert np.allclose(fraction[20:25], 0.5) and np.allclose(fraction[25:36], 1.0)
    shade = panel._shade.image
    assert shade is not None and shade.shape == (50, 1)
    assert shade[30, 0] > shade[22, 0] > shade[10, 0] == 0
    x, y = panel._live_curves[0].getData()
    assert len(x) == 50 and np.isfinite(y).all()
    _, y1 = panel._live_curves[1].getData()
    assert np.isnan(y1).any(), "NaNs are kept, and break the line"
    # The cursor sits at the aligned time of the newest sample.
    assert panel.ref_cursor.value() == pytest.approx(49)

    panel.update_history(_samples(80))
    assert panel.sample_count == 80
    x, _ = panel._live_curves[0].getData()
    assert len(x) == 80
    assert len(panel.guidance_fraction) == 80


def test_a_shorter_history_is_a_new_rollout(qapp, trends):
    panel = TrendsPanel()
    panel.set_reference(trends)
    panel.update_history(_samples(30))
    panel.update_history(_samples(4))
    assert panel.sample_count == 4
    assert np.all(panel.guidance_fraction == 0)


def test_clear_leaves_nothing(qapp, trends):
    panel = TrendsPanel()
    panel.set_reference(trends)
    panel.update_history(_samples(10))
    panel.clear()
    assert panel._ref_curves == [] and panel._live_curves == []
    assert panel.sample_count == 0
    assert "chunks around the aligned time" in panel.mode_text
    panel.update_history([])  # still fine with nothing to draw


def test_samples_before_a_reference_still_plot(qapp):
    panel = TrendsPanel()
    panel.update_history(_samples(12))
    assert panel.pair_count == 2
    assert panel.sample_count == 12


def _xrange(plot) -> tuple[float, float]:
    lo, hi = plot.viewRange()[0]
    return float(lo), float(hi)


def test_the_default_view_is_a_snapshot_around_now(qapp, trends):
    panel = TrendsPanel()
    assert panel.windowed
    assert panel.window_chunks_ref == 3 and panel.window_chunks_live == 6
    panel.set_reference(trends)
    panel.update_history(_samples(80))
    # Live: six chunks behind the latest step, one ahead of it.
    lo, hi = _xrange(panel.live_plot)
    assert lo == pytest.approx(79 - 30, abs=0.5)
    assert hi == pytest.approx(79 + 5, abs=0.5)
    # Reference: three chunks either side of the aligned time.
    lo, hi = _xrange(panel.ref_plot)
    aligned = 79
    assert lo == pytest.approx(aligned - 15, abs=0.5)
    assert hi == pytest.approx(aligned + 15, abs=0.5)
    assert (lo + hi) / 2 == pytest.approx(aligned, abs=0.5)
    assert panel.mode_text == (
        "window: ±15 steps around the aligned time; last 30 live steps"
    )
    # The curves still hold the whole history; only the view moved.
    x, _ = panel._live_curves[0].getData()
    assert len(x) == 80 and x[0] == 0


def test_the_snapshot_is_clamped_to_the_trajectory(qapp, trends):
    panel = TrendsPanel()
    panel.set_reference(trends)
    panel.update_history(_samples(3))
    lo, hi = _xrange(panel.live_plot)
    assert lo == pytest.approx(0, abs=0.5), "never before step 0"
    assert hi == pytest.approx(2 + 5, abs=0.5)
    lo, hi = _xrange(panel.ref_plot)
    assert lo == pytest.approx(0, abs=0.5)
    assert hi - lo >= 10, "clamping never squeezes it under two chunks"
    panel.update_history(_samples(120))  # aligned_t saturates at 99
    lo, hi = _xrange(panel.ref_plot)
    assert hi == pytest.approx(99, abs=0.5), "never past the demo's end"
    assert lo == pytest.approx(99 - 15, abs=0.5)


def test_full_mode_widens_to_the_whole_extents(qapp, trends):
    panel = TrendsPanel()
    panel.set_reference(trends)
    panel.update_history(_samples(80))
    panel.set_windowed(False)
    assert not panel.windowed
    lo, hi = _xrange(panel.live_plot)
    assert lo <= 0 and hi >= 79
    lo, hi = _xrange(panel.ref_plot)
    assert lo <= 0 and hi >= 99
    assert panel.mode_text == "full trajectory"
    # More samples in full mode keep the whole rollout in view …
    panel.update_history(_samples(90))
    lo, hi = _xrange(panel.live_plot)
    assert lo <= 0 and hi >= 89
    # … and going back to the snapshot re-centres on the newest sample.
    panel.set_windowed(True)
    lo, hi = _xrange(panel.live_plot)
    assert lo == pytest.approx(89 - 30, abs=0.5)
    assert "window" in panel.mode_text


def test_relevance_row_is_hidden_by_default(qapp, trends):
    panel = TrendsPanel()
    assert panel.show_relevance is False
    panel.set_reference(trends)
    panel.update_history(_samples(20))
    assert panel.rel_plot.isVisible() is False
    assert panel._rel_curves == {}, "not even drawn while hidden"
    panel.show_relevance = True
    panel.update_history(_samples(25))
    assert "cup" in panel._rel_curves


def test_the_reference_plot_sits_left_of_the_live_plot(qapp, trends):
    panel = TrendsPanel()
    panel.set_reference(trends)
    grid = panel.graphics.ci.layout
    ref_rect = panel.ref_plot.geometry()
    live_rect = panel.live_plot.geometry()
    qapp.processEvents()
    assert grid.columnCount() >= 2
    assert ref_rect.top() == pytest.approx(live_rect.top(), abs=2), "same row"
    assert ref_rect.left() < live_rect.left(), "reference on the left"
