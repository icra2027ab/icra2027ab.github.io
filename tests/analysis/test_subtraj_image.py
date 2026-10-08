"""The sub-trajectory overlay lands on the demo frame it is drawn on.

The retrieval panel shows the retrieved demo's first frame with its EE path
drawn over it. That frame follows the *dataset's* convention — a simulator
stores it 180°-rotated, a recording from the arm stores it upright — and the
projection used to assume the former for both.
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.analysis.demo_analyzer import _draw_subtraj_image
from policy_abstraction.utils.pcd import project_points_to_pixels

W, H = 640, 480
K = np.array([[600.0, 0.0, W / 2], [0.0, 600.0, H / 2], [0.0, 0.0, 1.0]])
E = np.array(
    [[1.0, 0, 0, 0], [0, -1.0, 0, 0], [0, 0, -1.0, 1.0], [0, 0, 0, 1.0]]
)  # 1 m above the table, looking down
EE = np.array([[0.05 + 0.02 * i, 0.0, 0.0] for i in range(12)])


def painted_columns(img: np.ndarray) -> np.ndarray:
    """Columns where the overlay put ink on an otherwise black frame."""
    return np.where(img.any(axis=(0, 2)))[0]


@pytest.mark.parametrize("preprocessed", [False, True])
def test_the_path_lands_where_that_convention_projects(preprocessed):
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    img = _draw_subtraj_image(
        blank, EE, [(0, len(EE))], K, E, preprocessed=preprocessed
    )
    expected, _ = project_points_to_pixels(EE, K, E, H, W, preprocessed=preprocessed)
    cols = painted_columns(img)
    assert cols.min() >= expected[:, 0].min() - 8
    assert cols.max() <= expected[:, 0].max() + 8


def test_an_upright_recording_is_not_mirrored():
    """The regression: a real-arm demo's path was drawn on the mirror side."""
    blank = np.zeros((H, W, 3), dtype=np.uint8)
    upright = painted_columns(
        _draw_subtraj_image(blank, EE, [(0, len(EE))], K, E, preprocessed=False)
    )
    mirrored = painted_columns(
        _draw_subtraj_image(blank, EE, [(0, len(EE))], K, E, preprocessed=True)
    )
    # The EE moves +x, which this camera sees as rightwards: the path must sit
    # right of centre when upright, and the two conventions must be mirrors.
    assert upright.mean() > W / 2 > mirrored.mean()
    assert abs((W - 1 - upright.mean()) - mirrored.mean()) < 8
