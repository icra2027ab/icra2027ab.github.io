"""Geometry of the staging grid."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from policy_abstraction.experiments.grid import (
    TableGrid,
    cell_label,
    parse_cell_label,
)


class TestCellLabels:
    def test_round_trip(self):
        for row in range(0, 60):
            for col in range(0, 12):
                assert parse_cell_label(cell_label(row, col)) == (row, col)

    def test_reads_like_a_spreadsheet(self):
        assert cell_label(0, 0) == "A1"
        assert cell_label(1, 3) == "B4"
        assert cell_label(25, 0) == "Z1"

    def test_wraps_past_z_without_colliding(self):
        assert cell_label(26, 0) == "AA1"
        assert cell_label(27, 0) == "AB1"
        labels = {cell_label(r, 0) for r in range(60)}
        assert len(labels) == 60

    def test_parse_is_case_and_space_insensitive(self):
        assert parse_cell_label(" b3 ") == parse_cell_label("B3")

    @pytest.mark.parametrize("bad", ["", "3", "B", "B0", "??", "B-1"])
    def test_malformed_labels_raise(self, bad):
        with pytest.raises(ValueError):
            parse_cell_label(bad)

    def test_columns_can_be_numbered_from_zero(self):
        # A grid widened by one column in front of A1 after collection:
        # the new column is 0, every old label still names the same cell.
        assert cell_label(1, 0, first_col=0) == "B0"
        assert cell_label(1, 1, first_col=0) == "B1"
        assert parse_cell_label("B0", first_col=0) == (1, 0)
        assert parse_cell_label("B1", first_col=0) == (1, 1)
        for row in range(4):
            for col in range(6):
                label = cell_label(row, col, first_col=0)
                assert parse_cell_label(label, first_col=0) == (row, col)


class TestConstruction:
    def test_axes_are_normalised_and_squared_up(self):
        # A config author gives table edges by eye; they are rarely
        # exactly perpendicular or unit length.
        grid = TableGrid(origin=(0, 0, 0), u_axis=(3, 0, 0), v_axis=(0.4, 2, 0))
        assert np.allclose(np.linalg.norm(grid.u_axis), 1.0)
        assert np.allclose(np.linalg.norm(grid.v_axis), 1.0)
        assert abs(float(np.dot(grid.u_axis, grid.v_axis))) < 1e-9

    def test_parallel_axes_rejected(self):
        with pytest.raises(ValueError, match="parallel"):
            TableGrid(origin=(0, 0, 0), u_axis=(1, 0, 0), v_axis=(2, 0, 0))

    @pytest.mark.parametrize("rows,cols", [(0, 3), (3, 0), (-1, 2)])
    def test_empty_grid_rejected(self, rows, cols):
        with pytest.raises(ValueError):
            TableGrid(origin=(0, 0, 0), rows=rows, cols=cols)

    def test_non_positive_cell_size_rejected(self):
        with pytest.raises(ValueError):
            TableGrid(origin=(0, 0, 0), cell_size_u=0.0)

    def test_up_hint_in_plane_rejected(self):
        with pytest.raises(ValueError, match="up_hint"):
            TableGrid(
                origin=(0, 0, 0),
                u_axis=(1, 0, 0),
                v_axis=(0, 1, 0),
                up_hint=(1, 0, 0),
            )


class TestGeometry:
    @pytest.fixture
    def grid(self):
        return TableGrid(
            origin=(0.30, -0.20, 0.02),
            u_axis=(0, 1, 0),
            v_axis=(1, 0, 0),
            cell_size_u=0.10,
            cell_size_v=0.10,
            rows=3,
            cols=4,
            marker_height=0.015,
        )

    def test_cell_centres_are_half_a_cell_from_the_corner(self, grid):
        assert np.allclose(grid.cell_center((0, 0)), [0.35, -0.15, 0.02])

    def test_cells_are_a_cell_apart(self, grid):
        step_u = grid.cell_center((0, 1)) - grid.cell_center((0, 0))
        step_v = grid.cell_center((1, 0)) - grid.cell_center((0, 0))
        assert np.allclose(np.linalg.norm(step_u), grid.cell_size_u)
        assert np.allclose(np.linalg.norm(step_v), grid.cell_size_v)

    def test_marker_height_lifts_off_the_table_not_into_it(self, grid):
        flat = grid.cell_center((1, 1))
        lifted = grid.cell_center((1, 1), lifted=True)
        assert lifted[2] == pytest.approx(flat[2] + 0.015)

    def test_marker_lifts_off_whichever_axis_order_is_used(self):
        """u x v flips sign with axis order; up_hint must absorb that."""
        common = dict(
            origin=(0, 0, 0.5), cell_size_u=0.1, cell_size_v=0.1, marker_height=0.02
        )
        one = TableGrid(u_axis=(0, 1, 0), v_axis=(1, 0, 0), **common)
        other = TableGrid(u_axis=(1, 0, 0), v_axis=(0, 1, 0), **common)
        assert one.cell_center((0, 0), lifted=True)[2] == pytest.approx(0.52)
        assert other.cell_center((0, 0), lifted=True)[2] == pytest.approx(0.52)

    def test_cell_of_inverts_cell_center(self, grid):
        for cell in grid.cells():
            assert grid.cell_of(grid.cell_center(cell)) == cell

    def test_cell_of_ignores_height_above_the_plane(self, grid):
        centre = grid.cell_center((1, 2)) + 0.3 * grid.normal
        assert grid.cell_of(centre) == (1, 2)

    def test_cell_of_returns_none_outside(self, grid):
        assert grid.cell_of([9.0, 9.0, 0.02]) is None

    def test_corners_bound_the_centre(self, grid):
        corners = grid.cell_corners((2, 3))
        assert corners.shape == (4, 3)
        assert np.allclose(corners.mean(axis=0), grid.cell_center((2, 3)))

    def test_out_of_range_cell_raises(self, grid):
        with pytest.raises(KeyError):
            grid.cell_center((3, 0))

    def test_grid_lines_cover_every_edge(self, grid):
        lines = grid.grid_lines()
        assert len(lines) == (grid.rows + 1) + (grid.cols + 1)
        assert all(line.shape[1] == 3 for line in lines)

    def test_round_trip_through_dict(self, grid):
        assert TableGrid.from_dict(grid.to_dict()) == grid

    def test_first_col_defaults_to_one_for_an_older_config(self, grid):
        data = grid.to_dict()
        del data["first_col"]
        assert TableGrid.from_dict(data).first_col == 1

    def test_zero_numbered_grid_keeps_old_labels_in_place(self):
        """Widening the 4x4 cable grid by one column in front of column 1.

        The origin moves one cell back along u and the columns are numbered
        from 0, so ``B1`` is the same point on the table as before and the
        new column is ``B0``.
        """
        old = TableGrid(
            origin=(-0.15, 0.05, 0.0),
            u_axis=(0, -1, 0),
            v_axis=(-1, 0, 0),
            cell_size_u=0.075,
            cell_size_v=0.15,
            rows=4,
            cols=4,
        )
        widened = dataclasses.replace(
            old, origin=(-0.15, 0.125, 0.0), cols=5, first_col=0
        )
        assert widened.label((1, 0)) == "B0"
        assert widened.parse_label("B1") == (1, 1)
        assert np.allclose(
            widened.cell_center(widened.parse_label("B1")),
            old.cell_center(old.parse_label("B1")),
        )
        assert np.allclose(
            widened.cell_center(widened.parse_label("B0")),
            old.cell_center(old.parse_label("B1")) + 0.075 * np.array([0, 1, 0]),
        )
        assert [widened.label(c) for c in widened.cells()][:5] == [
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
        ]

    @pytest.mark.parametrize("first_col", [-1, 2])
    def test_other_column_numberings_rejected(self, first_col):
        with pytest.raises(ValueError, match="first_col"):
            TableGrid(origin=(0, 0, 0), first_col=first_col)
