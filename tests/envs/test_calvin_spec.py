"""Smoke test: ``calvin_env_spec()`` builds a valid EnvSpec.

Validates the layout the CALVIN server publishes to clients without requiring
pybullet/hydra (the spec module is leaf-level by design).
"""

from policy_abstraction.envs.calvin.config import calvin_env_spec
from policy_abstraction.envs.spec import EnvSpec


def test_calvin_env_spec_round_trips():
    spec = calvin_env_spec()

    assert isinstance(spec, EnvSpec)
    assert spec.primary_camera.name == "static"
    assert {c.name for c in spec.cameras} == {"static", "gripper"}
    assert spec.metadata == {"env": "calvin"}

    roundtripped = EnvSpec.from_dict(spec.to_dict())
    assert roundtripped.primary_camera.name == spec.primary_camera.name
    assert roundtripped.robot.eef_pos_key == "tcp_pos"
    assert roundtripped.robot.eef_quat_key == "tcp_orn_quat"
    assert roundtripped.robot.action_layout.action_dim == 7


def test_calvin_env_spec_static_only():
    spec = calvin_env_spec(include_gripper=False)
    assert {c.name for c in spec.cameras} == {"static"}
    assert sum(int(c.is_primary) for c in spec.cameras) == 1


def test_calvin_env_spec_alternate_primary():
    spec = calvin_env_spec(primary_camera="gripper")
    assert spec.primary_camera.name == "gripper"
    assert sum(int(c.is_primary) for c in spec.cameras) == 1
