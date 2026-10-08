"""The rendered gripper is bolted on the way the real one is.

A gripper can sit on the flange at any of several yaws. Rendered at the wrong
one its fingers open across the real ones, and every finger-anchored distance
pair in the 3-D view looks rotated 90°. Two things have to agree for the
overlay to mean anything: the mount yaw the viewer renders at, and the
``finger_axis`` the geometry YAML gives the fingertip FK. On this cell the
2F-140 opens along tool x -- menagerie's default mount -- and the last test
here pins the render and the FK to that same flange axis.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("robot_descriptions")

from policy_abstraction.envs.ur5.config import ur5_env_spec  # noqa: E402
from policy_abstraction.evaluation.config import VisualizerArgs  # noqa: E402
from policy_abstraction.utils.gripper import (  # noqa: E402
    compute_finger_tip_positions_single,
)
from policy_abstraction.visualization.robot_assembly import (  # noqa: E402
    compose_arm_with_gripper,
)
from policy_steering.robot_config import load_gripper_geometry  # noqa: E402

PADS = ("robotiq_2f140/left_pad", "robotiq_2f140/right_pad")


def finger_line(model, joints=None) -> tuple[np.ndarray, np.ndarray]:
    """(unit vector between the pads, flange z axis) at a bent pose."""
    data = mujoco.MjData(model)
    data.qpos[:6] = joints if joints is not None else [0.0, -1.2, 1.6, -2.0, -1.6, 0.3]
    mujoco.mj_kinematics(model, data)

    def body(name: str) -> np.ndarray:
        return data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]

    line = body(PADS[0]) - body(PADS[1])
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    flange_z = data.site_xmat[site].reshape(3, 3)[:, 2]
    return line / np.linalg.norm(line), flange_z


def angle_deg(u, v) -> float:
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(u, v)), 0.0, 1.0))))


@pytest.mark.parametrize("yaw", [90.0, -90.0])
def test_a_quarter_turn_swings_the_fingers_a_quarter_turn(yaw):
    default = compose_arm_with_gripper("ur5e_mj_description", "robotiq_2f140")
    turned = compose_arm_with_gripper(
        "ur5e_mj_description", "robotiq_2f140", mount_yaw_deg=yaw
    )
    line0, flange_z = finger_line(default.model)
    line1, _ = finger_line(turned.model)
    assert angle_deg(line0, line1) == pytest.approx(90.0, abs=0.5)


def test_the_turn_is_about_the_flange_axis_not_some_other_one():
    turned = compose_arm_with_gripper(
        "ur5e_mj_description", "robotiq_2f140", mount_yaw_deg=90.0
    )
    line, flange_z = finger_line(turned.model)
    # Fingers still open perpendicular to the tool axis: yaw does not tilt them.
    assert angle_deg(line, flange_z) == pytest.approx(90.0, abs=1.0)


def test_zero_yaw_is_menagerie_as_authored():
    a = compose_arm_with_gripper("ur5e_mj_description", "robotiq_2f140")
    b = compose_arm_with_gripper(
        "ur5e_mj_description", "robotiq_2f140", mount_yaw_deg=0.0
    )
    assert angle_deg(finger_line(a.model)[0], finger_line(b.model)[0]) < 1e-6


def test_the_rendered_pads_and_the_fk_fingertips_share_a_flange_axis():
    """The viewer's mount yaw and the FK's finger_axis describe the same bolt
    pattern. If one is changed without the other, the drawn gripper and the
    fingertips guidance measures from disagree by a quarter turn -- which is
    exactly the bug this guards against, and it is invisible to either side's
    own tests."""
    # env:ur5 selects the 2F-140 model and leaves the mount yaw at the
    # VisualizerArgs default (test_backends pins both), so that default is
    # what the real rollout renders at.
    viz = VisualizerArgs()
    assembled = compose_arm_with_gripper(
        "ur5e_mj_description",
        "robotiq_2f140",
        mount_yaw_deg=viz.gripper_mount_yaw_deg,
    )
    model = assembled.model
    data = mujoco.MjData(model)
    data.qpos[:6] = [0.0, -1.2, 1.6, -2.0, -1.6, 0.3]
    # Pose the linkage a good way open so the pads are clearly apart.
    linkage = assembled.linkage
    idx = int(np.searchsorted(linkage.apertures, 0.10))
    data.qpos[linkage.qpos_adr] = linkage.table[idx]
    mujoco.mj_kinematics(model, data)

    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    flange_pos = data.site_xpos[site]
    flange_rot = data.site_xmat[site].reshape(3, 3)

    def pad_in_flange(name: str) -> np.ndarray:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return flange_rot.T @ (data.xpos[bid] - flange_pos)

    rendered = pad_in_flange(PADS[1]) - pad_in_flange(PADS[0])  # left -> right
    rendered /= np.linalg.norm(rendered)

    geometry = load_gripper_geometry(ur5_env_spec().robot.gripper.geometry_id)
    tips = compute_finger_tip_positions_single(
        np.zeros(3), np.zeros(3), np.array([0.7]), geometry
    )
    fk = tips["gripper_r"] - tips["gripper_l"]
    fk /= np.linalg.norm(fk)

    # Same axis *and* same sign: the left pad is the FK's gripper_l.
    assert float(np.dot(rendered, fk)) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(fk, geometry.finger_open_direction)
