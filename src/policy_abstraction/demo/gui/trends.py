"""The trend plot: the retrieved demo's distances above, the live ones below.

This is the picture that explains the demo to an audience. The upper plot is
the reference — one curve per distance pair over the demo's own time, the
sub-trajectories shaded behind it, and a cursor at the point the alignment
says the robot has reached. The lower plot is the rollout so far, the same
pairs in the same colours over live control steps, with the stretches where
guidance was on shaded green. When the switch is off the live curves drift
away from the demo's; when it is on they are pulled back — that is the whole
demonstration, and it should be legible from across the room.

Colours come from :attr:`ReferenceTrends.colors`, the palette the 3-D scene
and the camera overlay use for the same pairs, so a line here is the same
line there. The panel draws no legend of its own by default: in the demo
window the DISTANCE PAIRS panel beside it *is* the legend — the same colours,
the labels, and the live and reference values — and it takes the view-mode
line from :attr:`TrendsPanel.mode_text`. ``TrendsPanel(show_legend=True)``
adds the old :class:`TrendsLegend` column to the right of the plots for a
panel used on its own.

The view is a *snapshot* by default (:attr:`TrendsPanel.windowed`): the
reference plot shows a few chunks either side of the aligned time and the
live plot the last few chunks up to now, so a trace's shape is legible at
the height the window gives the row. The whole history is always logged;
only the visible x-range moves, and y follows the visible data.
:meth:`TrendsPanel.set_windowed` switches to the full trajectory and back.

Built to be refreshed at the window's rate (15–20 Hz) for a rollout of a few
thousand steps: every curve is one :class:`pyqtgraph.PlotDataItem` created
once and fed with ``setData``, the history lives in growable numpy arrays
and only the samples that arrived since the last call are ingested. Nothing
here is thread-aware — the window calls it from the GUI thread with
snapshots it already holds.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, cast

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from policy_abstraction.demo.overlay import COLOR_GUIDED, object_color
from policy_abstraction.demo.session import ReferenceTrends, TrendSample
from policy_abstraction.visualization.drawing import subtraj_color

BACKGROUND = "#101014"
FOREGROUND = "#c8c8d2"
_AXIS = "#6a6a74"
_GREY = (150, 150, 158)
_CURSOR = pg.mkPen(color=(240, 240, 245), width=2)
SHADE_ALPHA_MAX = 0.45
"""Opacity of the live plot's green background at full guidance."""
_NO_PEN = pg.mkPen(None)
LEGEND_WIDTH = 220
"""The legend column's fixed width, to the right of the plots."""
_LEGEND_ROW_STYLE = "color: #d8d8e0; font-size: 12px;"
_LEGEND_MODE_STYLE = "color: #8f8fa0; font-size: 11px;"


class _Columns:
    """A ``[n, width]`` float array that grows by doubling.

    ``np.concatenate`` per tick is fine for a few hundred rows and quadratic
    over a long rollout; this keeps appends amortised O(1) and hands out a
    view rather than a copy.
    """

    def __init__(self, width: int, capacity: int = 256) -> None:
        self._data = np.full((max(capacity, 1), width), np.nan, dtype=np.float32)
        self.n = 0

    @property
    def width(self) -> int:
        return self._data.shape[1]

    def append(self, rows: np.ndarray) -> None:
        """*rows* is ``[k, width]``; a zero-width table still counts them."""
        rows = np.asarray(rows, dtype=np.float32)
        needed = self.n + rows.shape[0]
        if needed > len(self._data):
            grown = np.full(
                (max(needed, 2 * len(self._data)), self.width), np.nan, np.float32
            )
            grown[: self.n] = self._data[: self.n]
            self._data = grown
        self._data[self.n : needed] = rows
        self.n = needed

    def add_column(self) -> None:
        """One more series, NaN for every row so far."""
        extra = np.full((len(self._data), 1), np.nan, dtype=np.float32)
        self._data = np.hstack([self._data, extra])

    @property
    def array(self) -> np.ndarray:
        return self._data[: self.n]


def _rgb(color: Sequence[int]) -> tuple[int, int, int]:
    r, g, b = (int(c) for c in color[:3])
    return (r, g, b)


class TrendsLegend(QWidget):
    """The legend column: a coloured dot and the label per pair, one row
    each, and a last dim line saying which view mode the plots are in.

    A fixed-width column beside the plots rather than a line above them:
    labels are long (``gripper_to_red_cable_connector``) and a line of them
    either clipped or took a second row from the plots' height. Here each
    label wraps within the column and the plots keep the whole height.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(LEGEND_WIDTH)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 4, 4, 4)
        self._layout.setSpacing(6)
        self.rows: list[QLabel] = []
        """One rich-text label per pair, in the order of the reference."""
        self._labels: list[str] = []
        self.mode = QLabel("")
        self.mode.setWordWrap(True)
        self.mode.setStyleSheet(_LEGEND_MODE_STYLE)
        self.mode.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._layout.addStretch(1)
        self._layout.addWidget(self.mode)

    def set_pairs(
        self, labels: Sequence[str], colors: Sequence[tuple[int, int, int]]
    ) -> None:
        for row in self.rows:
            self._layout.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        self.rows = []
        self._labels = list(labels)
        for i, label in enumerate(labels):
            color = colors[i] if i < len(colors) else _GREY
            row = QLabel(
                f"<span style='color:{_hex(color)}'>&#9679;</span> "
                f"<span style='color:{_hex(color)}'>{label}</span>"
            )
            row.setTextFormat(Qt.TextFormat.RichText)
            row.setWordWrap(True)
            row.setStyleSheet(_LEGEND_ROW_STYLE)
            # Ignored: a long unbreakable label must wrap or clip inside the
            # column, never widen it.
            row.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            self._layout.insertWidget(i, row)
            self.rows.append(row)

    def set_mode(self, text: str) -> None:
        self.mode.setText(text)

    def labels(self) -> list[str]:
        """The pair labels shown, plain text, in row order."""
        return list(self._labels)

    def text(self) -> str:
        """Every row's rich text joined, for tests and logs."""
        return "\n".join(row.text() for row in self.rows)


class TrendsPanel(QWidget):
    """Reference distances above, live distances below, shared pair colours.

    Drive it with :meth:`set_reference` once the rollout has retrieved its
    demo, :meth:`update_history` with the *whole* history every tick (only
    the new tail is processed), and :meth:`clear` when a new rollout starts.
    """

    show_reference_snippets: bool = True
    """At every replan, draw the reference chunk the policy was steered toward,
    shifted to start at that live step, as a dim dashed curve."""
    show_relevance: bool = False
    """A third, short row with the per-object relevances in ``[0, 1]``. Off
    by default so the reference and the live plot share the row's height."""
    windowed: bool = True
    """Snapshot view: the x-ranges follow the aligned time and the latest
    step (see :attr:`window_chunks_ref`, :attr:`window_chunks_live`). ``False``
    shows the whole reference and the whole rollout so far."""
    window_chunks_ref: int = 3
    """Snapshot half-width of the reference plot, in action chunks either
    side of the aligned time."""
    window_chunks_live: int = 6
    """Snapshot depth of the live plot, in action chunks behind the latest
    step; one chunk ahead of it is kept free for the next one."""

    def __init__(
        self, parent: Optional[QWidget] = None, *, show_legend: bool = False
    ) -> None:
        super().__init__(parent)
        self._trends: Optional[ReferenceTrends] = None
        self._colors: list[tuple[int, int, int]] = []
        self._n_pairs = 0
        self._count = 0
        self._latest: Optional[TrendSample] = None
        self._mode_text = ""

        layout = QHBoxLayout(self)
        # Side margins keep the plots off the splitter handles either side.
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(4)

        # pyqtgraph forwards PlotItem calls to its ViewBox at runtime and its
        # stubs know neither that nor the grid layout, so these are untyped.
        self.graphics: Any = pg.GraphicsLayoutWidget()
        self.graphics.setBackground(BACKGROUND)
        grid = cast(Any, self.graphics.ci.layout)
        grid.setSpacing(6)
        layout.addWidget(self.graphics, stretch=1)

        self.legend: Optional[TrendsLegend] = None
        """The legend column, only with ``show_legend``; the window's pairs
        panel plays that part otherwise."""
        if show_legend:
            self.legend = TrendsLegend()
            layout.addWidget(self.legend, stretch=0)

        # Side by side: the demo on the left, the live rollout on the right,
        # so the eye reads left-to-right from what the demo did to what the
        # arm is doing. The relevance row (off by default) spans both.
        self.ref_plot = self._make_plot(0, 0, "reference time", "distance (m)")
        self.live_plot = self._make_plot(0, 1, "live step", "distance (m)")
        self.rel_plot = self._make_plot(1, 0, "live step", "relevance", colspan=2)
        # The y-axes follow the data that is *visible* in the x-window, so a
        # snapshot's traces fill the height rather than the whole rollout's
        # extremes flattening them.
        for plot in (self.ref_plot, self.live_plot):
            plot.enableAutoRange(axis="y")
            plot.setAutoVisible(y=True)
        self.rel_plot.setYRange(-0.05, 1.05, padding=0)
        self.rel_plot.setMaximumHeight(90)
        self.rel_plot.setXLink(self.live_plot)
        self._relevance_in_layout = True
        self._set_relevance_row(self.show_relevance)
        grid.setRowStretchFactor(0, 5)
        grid.setRowStretchFactor(1, 2)
        grid.setColumnStretchFactor(0, 1)
        grid.setColumnStretchFactor(1, 1)
        self._update_mode_line()

        # Static reference items, rebuilt by set_reference.
        self._ref_curves: list[pg.PlotDataItem] = []
        self._ref_bands: list[tuple[pg.PlotDataItem, pg.PlotDataItem, object]] = []
        self._ref_regions: list[pg.LinearRegionItem] = []
        self._chunk_curves: list[pg.PlotDataItem] = []
        self._chunk_dots: list[pg.ScatterPlotItem] = []
        self.ref_cursor = pg.InfiniteLine(angle=90, movable=False, pen=_CURSOR)
        self.ref_cursor.setZValue(20)
        self.ref_plot.addItem(self.ref_cursor)

        # Live items, sized per pair by set_reference / the first sample.
        self._live_curves: list[pg.PlotDataItem] = []
        self._snippet_curves: list[pg.PlotDataItem] = []
        # The green background of the live plot: one column per step, its
        # opacity the fraction of guidance applied to the chunk on the arm.
        # Excluded from auto-range, since it spans a huge y extent on purpose.
        self._shade = pg.ImageItem()
        self._shade.setZValue(-10)
        # A float image through a lookup table: one green whose alpha rises
        # with the value. The LUT path is the one pyqtgraph renders reliably;
        # a raw RGBA array was silently not painted on this build.
        lut = np.zeros((256, 4), dtype=np.uint8)
        lut[:, 0], lut[:, 1], lut[:, 2] = COLOR_GUIDED
        lut[:, 3] = np.linspace(0, SHADE_ALPHA_MAX * 255, 256).astype(np.uint8)
        self._shade.setLookupTable(lut)
        self.live_plot.getViewBox().addItem(self._shade, ignoreBounds=True)
        self.live_cursor = pg.InfiniteLine(angle=90, movable=False, pen=_CURSOR)
        self.live_cursor.setZValue(20)
        self.live_plot.addItem(self.live_cursor)

        self._rel_curves: dict[str, pg.PlotDataItem] = {}
        self._reset_history()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _set_relevance_row(self, shown: bool) -> None:
        """Add or remove the relevance row. A hidden GraphicsLayout item keeps
        its row's height, so it is taken out of the layout rather than hidden,
        and the two distance plots get the whole panel."""
        if shown == self._relevance_in_layout:
            return
        grid = self.graphics.ci.layout  # self.graphics is Any
        if shown:
            self.graphics.ci.addItem(self.rel_plot, row=1, col=0, colspan=2)
            self.rel_plot.show()
            grid.setRowStretchFactor(1, 2)
        else:
            self.graphics.ci.removeItem(self.rel_plot)
            self.rel_plot.hide()
            # An empty row with a stretch factor still takes its share of the
            # height; give it none so the two plots fill the panel.
            grid.setRowStretchFactor(1, 0)
            grid.setRowPreferredHeight(1, 0)
            grid.setRowMaximumHeight(1, 0)
        self._relevance_in_layout = shown

    def _make_plot(
        self, row: int, col: int, xlabel: str, ylabel: str, *, colspan: int = 1
    ) -> Any:
        plot: Any = self.graphics.addPlot(row=row, col=col, colspan=colspan)
        plot.showGrid(x=False, y=True, alpha=0.15)
        plot.setMenuEnabled(False)
        plot.hideButtons()
        plot.setMouseEnabled(x=False, y=False)
        for side in ("left", "bottom"):
            axis = plot.getAxis(side)
            axis.setPen(pg.mkPen(_AXIS))
            axis.setTextPen(pg.mkPen(FOREGROUND))
            axis.setStyle(tickFont=pg.QtGui.QFont("sans", 8))
            # Metres stay metres: no "(x0.001)" prefix on a 5–30 cm range.
            axis.enableAutoSIPrefix(False)
        plot.setLabel("bottom", xlabel, color=FOREGROUND, size="9pt")
        plot.setLabel("left", ylabel, color=FOREGROUND, size="9pt")
        return plot

    def _pen(self, i: int, width: int, *, dashed: bool = False, alpha: int = 255):
        color = self._colors[i] if i < len(self._colors) else _GREY
        style = Qt.PenStyle.DashLine if dashed else Qt.PenStyle.SolidLine
        return pg.mkPen(color=(*color, alpha), width=width, style=style)

    def _new_curve(self, plot: Any, pen, z: int = 10) -> pg.PlotDataItem:
        item = pg.PlotDataItem(
            pen=pen, connect="finite", antialias=True, skipFiniteCheck=False
        )
        item.setZValue(z)
        plot.addItem(item)
        return item

    # ------------------------------------------------------------------
    # Reference
    # ------------------------------------------------------------------

    def set_reference(self, trends: Optional[ReferenceTrends]) -> None:
        """Rebuild the reference plot for *trends*; ``None`` clears everything.

        Also resets the live history: the pairs, and so the series, are
        those of this reference.
        """
        self._clear_reference()
        self._trends = trends
        if trends is None:
            self._colors = []
            self._n_pairs = 0
            self._reset_history()
            if self.legend is not None:
                self.legend.set_pairs([], [])
            self._update_mode_line()
            return
        self._colors = [
            _rgb(trends.colors.get(lbl, _GREY)) for lbl in trends.pair_labels
        ]
        self._n_pairs = trends.n_pairs
        ref = np.asarray(trends.ref_traj, dtype=np.float32)
        if ref.ndim == 1:
            ref = ref[:, None]
        x = np.arange(ref.shape[0], dtype=np.float32)

        for idx, (start, end) in enumerate(trends.subtrajectories):
            region = pg.LinearRegionItem(
                values=(float(start), float(end)),
                movable=False,
                brush=pg.mkBrush(*subtraj_color(idx, bright=False), 30),
                pen=_NO_PEN,
            )
            region.setZValue(-10)
            self.ref_plot.addItem(region)
            self._ref_regions.append(region)

        std = trends.ref_std
        for i in range(self._n_pairs):
            if std is not None and np.asarray(std).shape == ref.shape:
                s = np.asarray(std, dtype=np.float32)[:, i]
                upper = self._new_curve(self.ref_plot, _NO_PEN, z=0)
                lower = self._new_curve(self.ref_plot, _NO_PEN, z=0)
                upper.setData(x, ref[:, i] + s)
                lower.setData(x, ref[:, i] - s)
                band = pg.FillBetweenItem(
                    upper, lower, brush=pg.mkBrush(*self._colors[i], 45)
                )
                band.setZValue(1)
                self.ref_plot.addItem(band)
                self._ref_bands.append((upper, lower, band))
            curve = self._new_curve(self.ref_plot, self._pen(i, 2))
            curve.setData(x, ref[:, i])
            self._ref_curves.append(curve)
            chunk = self._new_curve(self.ref_plot, self._pen(i, 4), z=15)
            self._chunk_curves.append(chunk)
            dot = pg.ScatterPlotItem(
                size=9, pen=_NO_PEN, brush=pg.mkBrush(*self._colors[i])
            )
            dot.setZValue(16)
            self.ref_plot.addItem(dot)
            self._chunk_dots.append(dot)

        self.ref_cursor.setValue(0)
        self.ref_cursor.setVisible(True)
        if self.legend is not None:
            self.legend.set_pairs(trends.pair_labels, self._colors)
        self._reset_history()
        self._apply_ranges()

    def _clear_reference(self) -> None:
        for item in self._ref_curves + self._chunk_curves:
            self.ref_plot.removeItem(item)
        for upper, lower, band in self._ref_bands:
            for item in (band, upper, lower):
                self.ref_plot.removeItem(item)
        for item in self._ref_regions + self._chunk_dots:
            self.ref_plot.removeItem(item)
        self._ref_curves, self._chunk_curves = [], []
        self._ref_bands, self._ref_regions, self._chunk_dots = [], [], []
        self.ref_cursor.setVisible(False)

    # ------------------------------------------------------------------
    # Live history
    # ------------------------------------------------------------------

    def _reset_history(self) -> None:
        for item in self._live_curves + self._snippet_curves:
            self.live_plot.removeItem(item)
        self._shade.clear()
        for curve in self._rel_curves.values():
            self.rel_plot.removeItem(curve)
        self._live_curves, self._snippet_curves = [], []
        self._rel_curves = {}
        self._count = 0
        self._latest = None
        self._steps = _Columns(1)
        self._live = _Columns(max(self._n_pairs, 1))
        self._relevance = _Columns(0)
        self._relevance_names: list[str] = []
        # Snippets are one PlotDataItem per pair, NaN-separated, so a long
        # rollout does not become hundreds of items.
        self._snippets: list[list[np.ndarray]] = [[] for _ in range(self._n_pairs)]
        self._snippet_dirty = False
        self._fraction = _Columns(1)
        self.live_cursor.setVisible(False)
        for i in range(self._n_pairs):
            self._live_curves.append(self._new_curve(self.live_plot, self._pen(i, 2)))
            self._snippet_curves.append(
                self._new_curve(
                    self.live_plot, self._pen(i, 1, dashed=True, alpha=140), z=5
                )
            )

    def _adopt_pairs(self, n: int) -> None:
        """Live samples before any reference: size the series from them."""
        self._n_pairs = n
        self._colors = [_GREY] * n
        self._reset_history()

    def clear(self) -> None:
        """Drop the reference and the history; leaves an empty pair of axes."""
        self.set_reference(None)

    def update_history(self, samples: list[TrendSample]) -> None:
        """Take the *full* history so far; only the new tail is ingested.

        A history shorter than the last one means a new rollout, and the
        panel starts over.
        """
        if len(samples) < self._count:
            self._reset_history()
        if not samples:
            self._redraw()
            return
        if self._n_pairs == 0 or self._n_pairs != len(samples[0].live):
            self._adopt_pairs(len(samples[0].live))
        new = samples[self._count :]
        if not new:
            return
        self._ingest(new)
        self._count = len(samples)
        self._latest = samples[-1]
        self._redraw()

    def _ingest(self, new: list[TrendSample]) -> None:
        steps = np.fromiter((s.step for s in new), dtype=np.float32, count=len(new))
        self._steps.append(steps[:, None])
        live = np.full((len(new), self._n_pairs), np.nan, dtype=np.float32)
        for k, s in enumerate(new):
            v = np.asarray(s.live, dtype=np.float32).reshape(-1)
            live[k, : min(len(v), self._n_pairs)] = v[: self._n_pairs]
        self._live.append(live)

        fraction = np.zeros((len(new), 1), dtype=np.float32)
        for k, s in enumerate(new):
            scale = max(float(getattr(s, "guidance_scale", 1.0) or 1.0), 1e-6)
            applied = float(getattr(s, "guidance_factor", 0.0) or 0.0)
            if applied <= 0.0 and s.guided:
                # An older sample without a factor: guided means full strength.
                applied = scale if not hasattr(s, "guidance_factor") else applied
            fraction[k, 0] = float(np.clip(applied / scale, 0.0, 1.0))
            if s.replan and self._trends is not None:
                self._add_snippet(s)
        self._fraction.append(fraction)

        names = self._relevance_names
        for s in new:
            for name in s.relevances:
                if name not in names:
                    names.append(name)
                    self._relevance.add_column()
        # Appended even with no names yet, so the rows stay in step with
        # the steps when the first relevance arrives mid-rollout.
        rel = np.full((len(new), len(names)), np.nan, dtype=np.float32)
        for k, s in enumerate(new):
            for name, value in s.relevances.items():
                if value is not None:
                    rel[k, names.index(name)] = float(value)
        self._relevance.append(rel)

    def _add_snippet(self, s: TrendSample) -> None:
        assert self._trends is not None
        ref = self._trends.ref_traj
        T = ref.shape[0]
        n = max(int(s.chunk_size), 1)
        for i in range(self._n_pairs):
            t0 = int(
                np.clip(
                    s.aligned_ts[i] if i < len(s.aligned_ts) else s.aligned_t, 0, T - 1
                )
            )
            y = np.asarray(ref[t0 : t0 + n, i], dtype=np.float32)
            x = s.step + np.arange(len(y), dtype=np.float32)
            # A NaN row ends this snippet, so connect="finite" lifts the pen.
            block = np.column_stack([np.append(x, np.nan), np.append(y, np.nan)])
            self._snippets[i].append(block)
        self._snippet_dirty = True

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        steps = self._steps.array[:, 0]
        live = self._live.array
        for i, curve in enumerate(self._live_curves):
            curve.setData(steps, live[:, i])

        self._draw_shade(steps)

        if self._snippet_dirty or not self.show_reference_snippets:
            for i, curve in enumerate(self._snippet_curves):
                blocks = self._snippets[i]
                if self.show_reference_snippets and blocks:
                    data = np.concatenate(blocks)
                    curve.setData(data[:, 0], data[:, 1])
                else:
                    curve.setData([], [])
            self._snippet_dirty = False

        self._set_relevance_row(self.show_relevance)
        if self.show_relevance and self._relevance_names:
            rel = self._relevance.array
            for j, name in enumerate(self._relevance_names):
                curve = self._rel_curves.get(name)
                if curve is None:
                    curve = self._new_curve(
                        self.rel_plot, pg.mkPen(color=object_color(j), width=1)
                    )
                    self._rel_curves[name] = curve
                curve.setData(steps, rel[:, j])

        latest = self._latest
        if latest is None:
            self.live_cursor.setVisible(False)
            for chunk, dot in zip(self._chunk_curves, self._chunk_dots, strict=False):
                chunk.setData([], [])
                dot.setData([], [])
            self._apply_ranges()
            self._update_mode_line()
            return
        self.live_cursor.setValue(float(latest.step))
        self.live_cursor.setVisible(True)
        self._highlight_chunk(latest)
        self._apply_ranges()
        self._update_mode_line()  # QLabel.setText is a no-op on the same text

    def _draw_shade(self, steps: np.ndarray) -> None:
        """Paint the guidance fraction as a column per step behind the live
        curves: transparent where the policy ran alone, full green where the
        whole configured factor was applied, and every shade between as the
        objects' relevance scaled it."""
        from PyQt6.QtCore import QRectF

        fraction = self._fraction.array[:, 0]
        n = len(fraction)
        if n == 0:
            self._shade.clear()
            return
        self._shade.setImage(
            np.clip(fraction, 0, 1).astype(np.float32)[:, None],
            autoLevels=False,
            levels=(0.0, 1.0),
        )
        x0 = float(steps[0]) - 0.5
        width = float(steps[-1]) - float(steps[0]) + 1.0
        # Far taller than any distance in metres, so the band always fills
        # the visible height whatever the y range auto-fits to.
        self._shade.setRect(QRectF(x0, -100.0, width, 200.0))

    def _highlight_chunk(self, s: TrendSample) -> None:
        if self._trends is None:
            return
        ref = self._trends.ref_traj
        T = ref.shape[0]
        self.ref_cursor.setValue(float(np.clip(s.aligned_t, 0, T - 1)))
        n = max(int(s.chunk_size), 1)
        for i, (chunk, dot) in enumerate(
            zip(self._chunk_curves, self._chunk_dots, strict=False)
        ):
            t0 = int(
                np.clip(
                    s.aligned_ts[i] if i < len(s.aligned_ts) else s.aligned_t, 0, T - 1
                )
            )
            y = np.asarray(ref[t0 : t0 + n, i], dtype=np.float32)
            chunk.setData(t0 + np.arange(len(y), dtype=np.float32), y)
            dot.setData([float(t0)], [float(ref[t0, i])])

    # ------------------------------------------------------------------
    # View window
    # ------------------------------------------------------------------

    def set_windowed(self, windowed: bool) -> None:
        """Snapshot around now (``True``) or the whole trajectory (``False``).

        The curves are untouched; only the visible x-ranges move, and
        :attr:`mode_text` says which it is.
        """
        self.windowed = bool(windowed)
        self._apply_ranges()
        self._update_mode_line()

    def _apply_ranges(self) -> None:
        latest = self._latest
        chunk = max(int(latest.chunk_size), 1) if latest is not None else 1

        if self._trends is not None:
            T = int(self._trends.ref_traj.shape[0])
            last = max(T - 1, 1)
            if self.windowed and latest is not None:
                t = int(np.clip(latest.aligned_t, 0, T - 1))
                half = self.window_chunks_ref * chunk
                lo, hi = max(0, t - half), min(last, t + half)
                # Clamping at either end must not squeeze the window below
                # two chunks — the highlighted chunk and one before it.
                width = 2 * chunk
                if hi - lo < width:
                    if lo == 0:
                        hi = min(last, lo + width)
                    else:
                        lo = max(0, hi - width)
                self.ref_plot.setXRange(lo, max(hi, lo + 1), padding=0)
            else:
                self.ref_plot.setXRange(0, last, padding=0.01)

        if latest is not None:
            step = float(latest.step)
            if self.windowed:
                lo = max(0.0, step - self.window_chunks_live * chunk)
                self.live_plot.setXRange(lo, step + chunk, padding=0)
            else:
                self.live_plot.setXRange(0, max(step, 1.0), padding=0.02)

    def _update_mode_line(self) -> None:
        self._mode_text = self._describe_mode()
        if self.legend is not None:
            self.legend.set_mode(self._mode_text)

    def _describe_mode(self) -> str:
        if not self.windowed:
            return "full trajectory"
        latest = self._latest
        if latest is None:
            return f"window: ±{self.window_chunks_ref} chunks around the aligned time"
        chunk = max(int(latest.chunk_size), 1)
        return (
            f"window: ±{self.window_chunks_ref * chunk} steps around the aligned "
            f"time; last {self.window_chunks_live * chunk} live steps"
        )

    # ------------------------------------------------------------------

    @property
    def mode_text(self) -> str:
        """One line naming the view — ``window: ±15 steps around the aligned
        time; last 30 live steps`` or ``full trajectory`` — for whoever plays
        the legend."""
        return self._mode_text

    @property
    def pair_count(self) -> int:
        return self._n_pairs

    @property
    def sample_count(self) -> int:
        return self._count

    @property
    def guidance_fraction(self) -> np.ndarray:
        """Per ingested step, the fraction of the configured guidance factor
        that was applied to the chunk on the arm — what the background shade
        shows."""
        return self._fraction.array[:, 0].copy()


def _hex(color: Sequence[int]) -> str:
    r, g, b = _rgb(color)
    return f"#{r:02x}{g:02x}{b:02x}"


__all__ = ["BACKGROUND", "FOREGROUND", "LEGEND_WIDTH", "TrendsLegend", "TrendsPanel"]
