"""`--perception.flip-u` resolution.

`flip_u` mirrors the u axis during back-projection to compensate for the
abstractor pre-flipping simulator frames. Applying it to an env that publishes
upright frames mirrors every reconstructed object about the optical axis —
measured at ~0.8 m of displacement on the UR5 cell — and nothing errors. Hence
tests on the resolution rule rather than trust in a default.
"""

import sys

import pytest

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.envs.calvin.config import calvin_env_spec
from policy_abstraction.envs.libero.config import libero_env_spec
from policy_abstraction.envs.robocasa.config import robocasa_env_spec
from policy_abstraction.envs.ur5.config import ur5_env_spec

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from evaluate import _resolve_flip_u  # noqa: E402


class TestExplicitOverrides:
    @pytest.mark.parametrize("spec", [libero_env_spec(), ur5_env_spec(), None])
    def test_true_and_false_ignore_the_spec(self, spec):
        assert _resolve_flip_u("true", spec) is True
        assert _resolve_flip_u("false", spec) is False

    def test_rejects_an_unknown_setting(self):
        with pytest.raises(ValueError, match="auto/true/false"):
            _resolve_flip_u("yes", ur5_env_spec())


class TestAuto:
    def test_simulated_envs_flip(self):
        """They publish 180-rotated frames the abstractor pre-flips."""
        for spec in (libero_env_spec(), robocasa_env_spec(), calvin_env_spec()):
            env = spec.metadata.get("env")
            transform = spec.primary_camera.image_transform
            # Only meaningful for envs that actually pre-flip.
            expected = transform != "none"
            assert _resolve_flip_u("auto", spec) is expected, env

    def test_libero_specifically_still_flips(self):
        """Guard the sim path against a regression: it must stay True."""
        assert libero_env_spec().primary_camera.image_transform == "flip_vh"
        assert _resolve_flip_u("auto", libero_env_spec()) is True

    def test_ur5_does_not_flip(self):
        """Upright frames from a real camera must not be mirrored."""
        assert ur5_env_spec().primary_camera.image_transform == "none"
        assert _resolve_flip_u("auto", ur5_env_spec()) is False

    def test_missing_spec_keeps_the_simulator_default(self):
        """No spec to consult: don't silently change existing geometry."""
        assert _resolve_flip_u("auto", None) is True
