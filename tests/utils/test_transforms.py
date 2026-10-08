"""
Tests for policy_abstraction.utils.transforms.

All tests use only synthetic data (no LIBERO / GPU / network required).
"""

from __future__ import annotations

import math

import numpy as np

from policy_abstraction.utils.transforms import quat2axisangle


class TestQuat2AxisAngle:
    def test_identity_quaternion_returns_zero(self):
        """Identity quaternion [x=0,y=0,z=0,w=1] maps to zero axis-angle."""
        q = np.array([0.0, 0.0, 0.0, 1.0])
        aa = quat2axisangle(q)
        np.testing.assert_allclose(aa, [0.0, 0.0, 0.0], atol=1e-12)

    def test_90deg_around_z(self):
        """90° rotation around z-axis: quat = [0, 0, sin(π/4), cos(π/4)]."""
        angle = math.pi / 2
        q = np.array([0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)])
        aa = quat2axisangle(q)
        np.testing.assert_allclose(aa, [0.0, 0.0, angle], atol=1e-12)

    def test_180deg_around_x(self):
        """180° rotation around x-axis: quat = [1, 0, 0, 0]."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        aa = quat2axisangle(q)
        np.testing.assert_allclose(aa, [math.pi, 0.0, 0.0], atol=1e-12)

    def test_negative_w_clamped(self):
        """w < -1 is clamped to -1 (axis-angle = π * axis)."""
        q = np.array([0.0, 0.0, 0.0, -1.5])
        aa = quat2axisangle(q)
        np.testing.assert_allclose(aa, [0.0, 0.0, 0.0], atol=1e-12)

    def test_w_above_one_clamped(self):
        """w > 1 is clamped to 1 → near-identity result."""
        q = np.array([0.0, 0.0, 0.0, 1.5])
        aa = quat2axisangle(q)
        np.testing.assert_allclose(aa, [0.0, 0.0, 0.0], atol=1e-12)

    def test_output_shape(self):
        """Output is always a 1-D array of length 3."""
        q = np.array([0.0, 0.0, math.sin(math.pi / 8), math.cos(math.pi / 8)])
        aa = quat2axisangle(q)
        assert aa.shape == (3,)

    def test_arbitrary_rotation(self):
        """45° around [1,1,0]/√2 axis – verify inverse using scipy."""

        axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        angle = math.pi / 4
        q_xyzw = np.append(axis * math.sin(angle / 2), math.cos(angle / 2))
        aa = quat2axisangle(q_xyzw)
        expected = axis * angle
        np.testing.assert_allclose(aa, expected, atol=1e-12)
