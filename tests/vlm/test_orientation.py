"""Smoke tests for :mod:`policy_abstraction.vlm._orientation`."""

from __future__ import annotations

import numpy as np

from policy_abstraction.vlm._orientation import to_vlm_orientation


def test_to_vlm_orientation_column_flips_rgb():
    # Build a 2x3 RGB image with distinct corner pixels so the mapping is
    # unambiguous. Format: (H=2, W=3, C=3).
    img = np.array(
        [
            [[10, 0, 0], [20, 0, 0], [30, 0, 0]],
            [[40, 0, 0], [50, 0, 0], [60, 0, 0]],
        ],
        dtype=np.uint8,
    )

    out = to_vlm_orientation(img)

    # Same shape and dtype; columns reversed; rows untouched.
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    np.testing.assert_array_equal(
        out,
        np.array(
            [
                [[30, 0, 0], [20, 0, 0], [10, 0, 0]],
                [[60, 0, 0], [50, 0, 0], [40, 0, 0]],
            ],
            dtype=np.uint8,
        ),
    )


def test_to_vlm_orientation_handles_segmentation_mask_2d():
    # Seg masks are 2-D int arrays — the helper must touch only the spatial
    # axes regardless of the trailing channel layout.
    seg = np.array(
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int32,
    )

    out = to_vlm_orientation(seg)

    assert out.shape == seg.shape
    assert out.dtype == seg.dtype
    np.testing.assert_array_equal(out, np.array([[3, 2, 1], [6, 5, 4]], dtype=np.int32))


def test_to_vlm_orientation_handles_singleton_channel():
    # Some perception outputs carry a trailing 1 channel (e.g., (H, W, 1)
    # seg masks). Ensure the helper preserves layout.
    seg = np.arange(6, dtype=np.uint8).reshape(2, 3, 1)

    out = to_vlm_orientation(seg)

    assert out.shape == seg.shape
    np.testing.assert_array_equal(out[..., 0], seg[..., 0][:, ::-1])


def test_to_vlm_orientation_returns_contiguous():
    # PIL.Image.fromarray + base64 encoding downstream requires a contiguous
    # buffer; a strided view would silently work in numpy but blow up in PIL.
    img = np.zeros((4, 5, 3), dtype=np.uint8)
    out = to_vlm_orientation(img)
    assert out.flags["C_CONTIGUOUS"]


def test_to_vlm_orientation_is_self_inverse():
    # Applying twice returns the original — useful sanity for callers that
    # accidentally double-apply.
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(5, 7, 3), dtype=np.uint8)
    np.testing.assert_array_equal(to_vlm_orientation(to_vlm_orientation(img)), img)
