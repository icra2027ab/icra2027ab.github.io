"""Placing the robot model must preserve the frame its MJCF was authored in.

Menagerie's UR5e declares its root body with ``quat="0 0 0 -1"`` — a 180°
rotation about z — because the meshes are built in the ROS ``base_link`` frame
while the arm is driven in the controller's ``base`` frame. Replacing that quat
with the base orientation the env reports (the identity, for a UR5e whose base
*is* the world origin) renders the robot facing backwards.

The reference is :func:`ur5e_forward_kinematics`, which was checked against the
real arm to 1.0 mm.
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.dynamics.ur5_kinematics import ur5e_forward_kinematics

mujoco = pytest.importorskip("mujoco")
mj_loader = pytest.importorskip("robot_descriptions.loaders.mujoco")

from policy_abstraction.visualization.rerun import RerunVisualizer  # noqa: E402

# Non-singular poses; a yaw error is invisible at some configurations.
POSES = [
    [0.0, -1.5707963, 1.5707963, -1.5707963, -1.5707963, 0.0],
    [0.3, -1.2, 1.4, -1.7, -1.5, 0.2],
    [-0.6, -1.0, 1.1, -1.4, -1.6, 0.5],
]


def _viz_with(model, base_pos, base_ori, *, authored=None) -> RerunVisualizer:
    """A visualiser holding *model*, with the base applied as in a real run.

    *authored* overrides the snapshot of the model's own root pose, which is
    how the pre-fix behaviour (substitute rather than compose) is reproduced.
    ``_apply_base_to_model`` mutates the model, so callers must pass a model
    nothing else has touched.
    """
    viz = RerunVisualizer.__new__(RerunVisualizer)
    viz._mj_model = model
    viz._mj_data = mujoco.MjData(model)
    viz._model_base_pose0 = authored or (
        model.body_pos[1].copy(),
        model.body_quat[1].copy(),
    )
    viz._base_pos = np.asarray(base_pos, dtype=np.float64)
    viz._base_ori = np.asarray(base_ori, dtype=np.float64)
    viz._apply_base_to_model()
    return viz


def _flange(viz, q) -> np.ndarray:
    """World position of the tool flange for joint vector *q*.

    Read from the model's own ``attachment_site`` rather than reconstructed
    from the wrist body: menagerie's link frames don't follow the DH chain
    (the site sits at ``0 0.1 0``, along +y of wrist_3), so hand-adding a d6
    offset along +z lands somewhere else entirely.
    """
    data = viz._mj_data
    data.qpos[:6] = q
    mujoco.mj_kinematics(viz._mj_model, data)
    sid = mujoco.mj_name2id(viz._mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    assert sid >= 0, "menagerie ur5e should expose attachment_site"
    return data.site_xpos[sid].copy()


@pytest.fixture
def ur5e_model():
    """Fresh per test: ``_apply_base_to_model`` mutates the model in place, so a
    shared instance would let one test's placement leak into the next."""
    return mj_loader.load_robot_description("ur5e_mj_description")


class TestUr5eBaseFrame:
    def test_menagerie_ur5e_authors_a_180_degree_yaw(self, ur5e_model):
        """If this ever becomes the identity, the compose logic is moot and the
        test below stops proving anything — so assert the premise."""
        np.testing.assert_allclose(
            ur5e_model.body_quat[1], [0.0, 0.0, 0.0, -1.0], atol=1e-9
        )

    @pytest.mark.parametrize("q", POSES)
    def test_identity_base_preserves_the_authored_frame(self, ur5e_model, q):
        """A UR5e's base *is* the world origin, so the env reports identity.
        The rendered flange must still match the hardware-verified FK."""
        viz = _viz_with(ur5e_model, np.zeros(3), np.eye(3))
        expected, _ = ur5e_forward_kinematics(q, None)
        np.testing.assert_allclose(_flange(viz, q), expected, atol=2e-3)

    @pytest.mark.parametrize("q", POSES)
    def test_overwriting_the_quat_would_face_backwards(self, ur5e_model, q):
        """The regression itself: substituting identity for the authored quat
        mirrors x and y, which is what "facing backwards" looks like."""
        viz = _viz_with(
            ur5e_model,
            np.zeros(3),
            np.eye(3),
            authored=(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])),
        )
        expected, _ = ur5e_forward_kinematics(q, None)
        got = _flange(viz, q)
        np.testing.assert_allclose(got[:2], -expected[:2], atol=2e-3)
        assert got[2] == pytest.approx(expected[2], abs=2e-3)  # z unaffected

    def test_base_translation_still_applies(self, ur5e_model):
        """Composing must not break placing the robot somewhere else."""
        offset = np.array([0.25, -0.10, 0.40])
        q = POSES[1]
        at_origin = _flange(_viz_with(ur5e_model, np.zeros(3), np.eye(3)), q)
        fresh = mj_loader.load_robot_description("ur5e_mj_description")
        moved = _flange(_viz_with(fresh, offset, np.eye(3)), q)
        np.testing.assert_allclose(moved - at_origin, offset, atol=1e-6)

    def test_base_rotation_composes_on_top(self, ur5e_model):
        """A world yaw of 180° must cancel the authored one, not stack oddly."""
        q = POSES[1]
        yaw180 = np.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
        got = _flange(_viz_with(ur5e_model, np.zeros(3), yaw180), q)
        expected, _ = ur5e_forward_kinematics(q, None)
        np.testing.assert_allclose(got[:2], -expected[:2], atol=2e-3)


class TestPandaUnaffected:
    def test_panda_authored_quat_is_identity(self):
        """Composing is a no-op for the simulator envs, so LIBERO/RoboCasa
        rendering is unchanged by this fix."""
        model = mj_loader.load_robot_description("panda_mj_description")
        np.testing.assert_allclose(model.body_quat[1], [1.0, 0.0, 0.0, 0.0], atol=1e-9)
