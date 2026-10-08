"""CALVIN configuration constants and EnvSpec factory.

Kept leaf-level (no pybullet/hydra imports) so it can be imported from the
calvin venv without pulling the full ``policy_abstraction`` ML stack —
mirrors what :mod:`policy_abstraction.envs.libero.config` and
:mod:`policy_abstraction.envs.robocasa.config` do.

The CALVIN env returns nested ``rgb_obs`` / ``depth_obs`` dicts and a
1-D ``robot_obs`` vector. The server flattens that into a single dict whose
keys match the :class:`EnvSpec` published here, so downstream consumers
(``EpisodeRunner`` etc.) work without code changes.
"""

from __future__ import annotations

from policy_abstraction.envs.spec import (
    ActionLayout,
    CameraSpec,
    EnvSpec,
    GripperSpec,
    PolicyIOKeys,
    RobotSpec,
    StateField,
    StateLayout,
)

# CALVIN evaluation rollouts cap at 360 steps (matches the public benchmark).
DEFAULT_MAX_STEPS = 360

# Native CALVIN render resolutions. Static cam: 200×200, gripper cam: 84×84.
# We expose a single resolution at the wire level; the server resizes as needed.
CALVIN_ENV_RESOLUTION = 200

# Same arm-only zero step LIBERO uses; CALVIN accepts a 7-D relative action
# directly, so no padding is needed.
CALVIN_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# PyBullet z-buffer near/far per camera (from the CALVIN hydra configs).
CALVIN_STATIC_DEPTH_NEAR: float = 0.01
CALVIN_STATIC_DEPTH_FAR: float = 10.0
CALVIN_GRIPPER_DEPTH_NEAR: float = 0.01
CALVIN_GRIPPER_DEPTH_FAR: float = 2.0


def calvin_env_spec(
    primary_camera: str = "static",
    include_gripper: bool = True,
    enable_segmentation: bool = True,
) -> EnvSpec:
    """Build the EnvSpec for a CALVIN PlayTable env.

    Observation key contract (the server is responsible for emitting these
    keys verbatim from the nested CALVIN obs dict):

    * ``tcp_pos`` — (3,) end-effector position
    * ``tcp_orn_quat`` — (4,) [x, y, z, w] end-effector quaternion
    * ``gripper_qpos`` — (1,) gripper opening width as a 1-D array
    * ``arm_joint_states`` — (7,) Panda arm joint positions
    * ``static_image`` / ``static_depth`` — primary camera
    * ``gripper_image`` / ``gripper_depth`` — wrist camera (when included)
    * ``scene_obs`` — variable-length scene state vector

    The published action layout is 7-D relative
    ``[dx, dy, dz, drx, dry, drz, gripper]`` — exactly what CALVIN's
    ``relative_to_absolute`` consumes when the env is in relative-action mode.
    """
    cameras: list[CameraSpec] = [
        CameraSpec(
            name="static",
            role="world",
            image_key="static_image",
            depth_key="static_depth",
            segmentation_key="static_segmentation" if enable_segmentation else None,
            depth_near=CALVIN_STATIC_DEPTH_NEAR,
            depth_far=CALVIN_STATIC_DEPTH_FAR,
            image_transform="none",
            is_primary=(primary_camera == "static"),
        ),
    ]
    if include_gripper:
        cameras.append(
            CameraSpec(
                name="gripper",
                role="wrist",
                image_key="gripper_image",
                depth_key="gripper_depth",
                segmentation_key="gripper_segmentation"
                if enable_segmentation
                else None,
                depth_near=CALVIN_GRIPPER_DEPTH_NEAR,
                depth_far=CALVIN_GRIPPER_DEPTH_FAR,
                image_transform="none",
                is_primary=(primary_camera == "gripper"),
            )
        )

    return EnvSpec(
        cameras=tuple(cameras),
        robot=RobotSpec(
            eef_pos_key="tcp_pos",
            eef_quat_key="tcp_orn_quat",
            gripper_qpos_key="gripper_qpos",
            joint_pos_key="arm_joint_states",
            state_layout=StateLayout(
                fields=(
                    StateField("pos", "tcp_pos"),
                    StateField("quat_to_axisangle", "tcp_orn_quat"),
                    StateField("gripper_aperture", "gripper_qpos"),
                )
            ),
            gripper=GripperSpec(
                # CALVIN uses a Franka Panda; the qpos here is already a
                # 1-D opening-width scalar (sum of two finger joints inside
                # the env), so the identity op picks it up as-is.
                geometry_id="franka_panda",
                finger_count=2,
                aperture_op="identity",
                speed_threshold=0.002,
            ),
            action_layout=ActionLayout(
                action_dim=7,
                eef_pos_slice=(0, 3),
                eef_rot_slice=(3, 6),
                gripper_dim=6,
                integrate_eef=True,
            ),
            robot_description="panda_mj_description",
            entity_names=("ee", "gripper_l", "gripper_r"),
            visualizer_negate_gripper_joint=False,
        ),
        policy_input_keys=PolicyIOKeys(
            state_key="observation/state",
            prompt_key="prompt",
            images=(
                ("static", "observation/image"),
                *((("gripper", "observation/wrist_image"),) if include_gripper else ()),
            ),
            state_layout=StateLayout(
                fields=(
                    StateField("pos", "tcp_pos"),
                    StateField("quat_to_axisangle", "tcp_orn_quat"),
                    StateField("vector", "gripper_qpos"),
                )
            ),
        ),
        metadata={"env": "calvin"},
    )
