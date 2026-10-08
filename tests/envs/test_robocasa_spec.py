"""Smoke test: ``robocasa_env_spec()`` builds a valid EnvSpec.

Validates the layout the RoboCasa server publishes to clients without
requiring robosuite/MuJoCo (the spec module is leaf-level by design).
"""

from policy_abstraction.envs.robocasa.config import robocasa_env_spec
from policy_abstraction.envs.spec import EnvSpec


def test_robocasa_env_spec_round_trips():
    spec = robocasa_env_spec()

    # EnvSpec.__post_init__ enforces unique camera names, exactly one primary,
    # and that policy_input_keys references known cameras.
    assert isinstance(spec, EnvSpec)
    assert spec.primary_camera.name == "robot0_agentview_left"
    assert {c.name for c in spec.cameras} == {
        "robot0_agentview_left",
        "robot0_agentview_right",
        "robot0_eye_in_hand",
    }
    assert spec.metadata == {"env": "robocasa"}

    # Round-trip through the wire format used by WebsocketEnvClient.
    roundtripped = EnvSpec.from_dict(spec.to_dict())
    assert roundtripped.primary_camera.name == spec.primary_camera.name
    assert roundtripped.robot.eef_pos_key == "robot0_eef_pos"
    assert roundtripped.robot.action_layout.action_dim == 7


def test_robocasa_env_spec_no_right_view():
    spec = robocasa_env_spec(include_right_view=False)
    assert {c.name for c in spec.cameras} == {
        "robot0_agentview_left",
        "robot0_eye_in_hand",
    }


def test_robocasa_env_spec_alternate_primary():
    spec = robocasa_env_spec(primary_camera="robot0_eye_in_hand")
    assert spec.primary_camera.name == "robot0_eye_in_hand"
    # Exactly one camera carries is_primary=True (validated by EnvSpec).
    assert sum(int(c.is_primary) for c in spec.cameras) == 1
