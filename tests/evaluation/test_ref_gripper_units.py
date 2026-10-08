"""The reference gripper aperture is measured in the live side's units.

The live aperture pair is a gap in metres (``gripper_qpos_to_metric``). A
real-cell demo records a normalised openness, and comparing that raw value
against metres made the aperture pair read the live gripper as "at the start
of the demo" for the whole rollout, pinning the alignment cursor there and
sending the guidance the wrong way. The reference has to be scaled with the
demo's own stroke.
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.evaluation.abstraction import (
    _gripper_aperture_metric,
    _gripper_aperture_scalar,
)
from policy_abstraction.utils.gripper import gripper_qpos_to_metric


def test_single_dof_openness_is_scaled_like_the_live_side():
    openness = np.array([[0.11], [0.471], [0.706], [0.988]], dtype=np.float32)
    ref = _gripper_aperture_metric(openness, "robotiq_2f140")
    assert ref.shape == (4, 1)
    live = [float(gripper_qpos_to_metric(o, "robotiq_2f140")) for o in openness[:, 0]]
    assert ref[:, 0] == pytest.approx(live, abs=1e-6)
    # A 2F-140 at 70 % open is about a 10 cm gap, nowhere near 0.7 m.
    assert 0.08 < ref[2, 0] < 0.12


def test_panda_joint_difference_is_already_metres():
    qpos = np.array([[0.04, -0.04], [0.01, -0.01]], dtype=np.float32)
    assert _gripper_aperture_metric(qpos, None)[:, 0] == pytest.approx([0.08, 0.02])
    assert _gripper_aperture_scalar(qpos)[:, 0] == pytest.approx([0.08, 0.02])


def test_raw_scalar_keeps_the_recording_units():
    """The dynamics roll-out's initial state wants the raw frame."""
    openness = np.array([[0.706]], dtype=np.float32)
    assert _gripper_aperture_scalar(openness)[0, 0] == pytest.approx(0.706)


def test_none_passes_through():
    assert _gripper_aperture_metric(None, "robotiq_2f140") is None
