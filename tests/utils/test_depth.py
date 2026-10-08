"""
Tests for policy_abstraction.utils.depth.

All tests use only synthetic data (no LIBERO / GPU / network required).
"""

from __future__ import annotations

import numpy as np

from policy_abstraction.utils.depth import depth_to_color, mujoco_depth_to_metric

NEAR = 0.01
FAR = 10.0


class TestMujocoDepthToMetric:
    def test_zero_depth_returns_near_plane(self):
        """d=0 → real depth = near plane."""
        result = mujoco_depth_to_metric(np.array([0.0]), NEAR, FAR)
        np.testing.assert_allclose(result, [NEAR], rtol=1e-10)

    def test_one_depth_returns_far_plane(self):
        """d=1 → real depth = far plane."""
        result = mujoco_depth_to_metric(np.array([1.0]), NEAR, FAR)
        np.testing.assert_allclose(result, [FAR], rtol=1e-6)

    def test_midpoint_value(self):
        """Verify formula at d=0.5: near / (1 - 0.5*(1 - near/far))."""
        d = 0.5
        expected = NEAR / (1.0 - d * (1.0 - NEAR / FAR))
        result = mujoco_depth_to_metric(np.array([d]), NEAR, FAR)
        np.testing.assert_allclose(result, [expected], rtol=1e-10)

    def test_array_input(self):
        """Works with 2-D depth maps."""
        d = np.linspace(0.0, 1.0, 12).reshape(3, 4)
        result = mujoco_depth_to_metric(d, NEAR, FAR)
        assert result.shape == (3, 4)
        # All values must be within [near, far]
        assert result.min() >= NEAR - 1e-9
        assert result.max() <= FAR + 1e-6

    def test_monotonically_increasing(self):
        """Metric depth should increase monotonically with normalized depth."""
        d = np.linspace(0.0, 1.0, 100)
        result = mujoco_depth_to_metric(d, NEAR, FAR)
        assert np.all(np.diff(result) > 0)


class TestDepthToColor:
    def _make_depth(self):
        return np.linspace(0.5, 3.0, 20).reshape(4, 5)

    def test_output_shape_and_dtype(self):
        """Output is [H, W, 3] uint8 BGR."""
        depth = self._make_depth()
        out = depth_to_color(depth)
        assert out.shape == (4, 5, 3)
        assert out.dtype == np.uint8

    def test_max_depth_parameter(self):
        """Supplying max_depth clips values above it."""
        depth = self._make_depth()
        out1 = depth_to_color(depth, max_depth=2.0)
        out2 = depth_to_color(depth, max_depth=3.0)
        assert out1.shape == out2.shape == depth.shape + (3,)

    def test_uniform_depth_produces_constant_image(self):
        """Uniform depth triggers divide-by-zero internally; output is still [H,W,3] uint8."""
        import warnings

        depth = np.full((8, 8), 1.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = depth_to_color(depth)
        assert out.shape == (8, 8, 3)
        assert out.dtype == np.uint8
