"""
Tests for policy_abstraction.core.guidance.

All tests use only synthetic data (no LIBERO / GPU / network required).
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.core.guidance import ActionSpaceGuidance, SO3Guidance


class TestSO3Guidance:
    def test_instantiation(self):
        """SO3Guidance can be instantiated."""
        guidance = SO3Guidance()
        assert guidance is not None

    def test_compute_loss_raises_not_implemented(self):
        """compute_loss raises NotImplementedError (stub pending implementation)."""
        guidance = SO3Guidance()
        with pytest.raises(NotImplementedError):
            guidance.compute_loss(
                robot_state=np.zeros(7),
                pcd=np.zeros((10, 7)),
                reference_features={},
            )


class TestActionSpaceGuidance:
    def _make_guidance(self):
        from policy_abstraction.dynamics.identity import IdentityDynamicsModel

        so3 = SO3Guidance()
        dyn = IdentityDynamicsModel()
        return ActionSpaceGuidance(so3_guidance=so3, dynamics_model=dyn)

    def test_instantiation_stores_components(self):
        """Constructor stores so3_guidance and dynamics_model attributes."""
        from policy_abstraction.dynamics.identity import IdentityDynamicsModel

        so3 = SO3Guidance()
        dyn = IdentityDynamicsModel()
        guidance = ActionSpaceGuidance(so3_guidance=so3, dynamics_model=dyn)
        assert guidance.so3_guidance is so3
        assert guidance.dynamics_model is dyn

    def test_grad_raises_not_implemented(self):
        """grad() raises NotImplementedError (stub pending implementation)."""
        guidance = self._make_guidance()
        with pytest.raises(NotImplementedError):
            guidance.grad(
                actions=np.zeros((5, 7)),
                robot_state=np.zeros(7),
                pcd=np.zeros((10, 7)),
                reference_features={},
            )
