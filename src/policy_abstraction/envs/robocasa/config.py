"""RoboCasa configuration constants and EnvSpec factory.

Kept leaf-level (no robosuite/MuJoCo imports) so it can be imported from the
robocasa venv without pulling the full ``policy_abstraction`` ML stack —
mirrors what :mod:`policy_abstraction.envs.libero.config` does for LIBERO.
"""

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

# Default rollout cap for RoboCasa atomic tasks. Composite tasks override this
# at server startup if needed.
DEFAULT_MAX_STEPS = 500

# Standard render resolution used in robocasa demos (matches collect_demos.py).
ROBOCASA_ENV_RESOLUTION = 256

# Same arm-only zero step LIBERO uses; RoboCasa server zero-pads to PandaOmron's
# 12-D action.
ROBOCASA_DUMMY_ACTION = [0.0] * 6 + [-1.0]

# Depth z-buffer near/far. RoboCasa kitchens have larger scene extents than
# LIBERO tabletop. These are first-cut defaults — tune empirically per scene.
ROBOCASA_DEPTH_NEAR: float = 0.01
ROBOCASA_DEPTH_FAR: float = 1000.0


def robocasa_env_spec(
    primary_camera: str = "robot0_agentview_left",
    include_right_view: bool = True,
) -> EnvSpec:
    """Build the EnvSpec describing a RoboCasa env with a PandaOmron arm.

    The arm-only action layout (7-D) matches the LIBERO/pi0 contract — the
    server is responsible for zero-padding to PandaOmron's native 12-D action
    (base_motion=0, control_mode=0) before calling ``env.step``.
    """
    cameras: list[CameraSpec] = [
        CameraSpec(
            name="robot0_agentview_left",
            role="world",
            image_key="robot0_agentview_left_image",
            depth_key="robot0_agentview_left_depth",
            segmentation_key="robot0_agentview_left_segmentation_instance",
            depth_near=ROBOCASA_DEPTH_NEAR,
            depth_far=ROBOCASA_DEPTH_FAR,
            image_transform="flip_vh",
            is_primary=(primary_camera == "robot0_agentview_left"),
        ),
    ]
    if include_right_view:
        cameras.append(
            CameraSpec(
                name="robot0_agentview_right",
                role="world",
                image_key="robot0_agentview_right_image",
                depth_key="robot0_agentview_right_depth",
                segmentation_key="robot0_agentview_right_segmentation_instance",
                depth_near=ROBOCASA_DEPTH_NEAR,
                depth_far=ROBOCASA_DEPTH_FAR,
                image_transform="flip_vh",
                is_primary=(primary_camera == "robot0_agentview_right"),
            )
        )
    cameras.append(
        CameraSpec(
            name="robot0_eye_in_hand",
            role="wrist",
            image_key="robot0_eye_in_hand_image",
            depth_key="robot0_eye_in_hand_depth",
            segmentation_key=None,
            depth_near=ROBOCASA_DEPTH_NEAR,
            depth_far=ROBOCASA_DEPTH_FAR,
            image_transform="flip_vh",
            is_primary=(primary_camera == "robot0_eye_in_hand"),
        )
    )

    policy_images: list[tuple[str, str]] = [
        ("robot0_agentview_left", "observation/image"),
    ]
    if include_right_view:
        policy_images.append(("robot0_agentview_right", "observation/secondary_image"))
    policy_images.append(("robot0_eye_in_hand", "observation/wrist_image"))

    return EnvSpec(
        cameras=tuple(cameras),
        robot=RobotSpec(
            eef_pos_key="robot0_eef_pos",
            eef_quat_key="robot0_eef_quat",
            gripper_qpos_key="robot0_gripper_qpos",
            joint_pos_key="robot0_joint_pos",
            state_layout=StateLayout(
                fields=(
                    StateField("pos", "robot0_eef_pos"),
                    StateField("quat_to_axisangle", "robot0_eef_quat"),
                    StateField("gripper_aperture", "robot0_gripper_qpos"),
                )
            ),
            gripper=GripperSpec(
                geometry_id="franka_panda",
                finger_count=2,
                aperture_op="diff",
                speed_threshold=0.002,
            ),
            # 7-D arm-only action published to clients; server pads to 12-D
            # (PandaOmron base_motion + control_mode) before env.step.
            action_layout=ActionLayout(
                action_dim=7,
                eef_pos_slice=(0, 3),
                eef_rot_slice=(3, 6),
                gripper_dim=6,
                integrate_eef=True,
            ),
            robot_description="panda_mj_description",
            entity_names=("ee", "gripper_l", "gripper_r"),
            visualizer_negate_gripper_joint=True,
        ),
        policy_input_keys=PolicyIOKeys(
            state_key="observation/state",
            prompt_key="prompt",
            images=tuple(policy_images),
            state_layout=StateLayout(
                fields=(
                    StateField("pos", "robot0_eef_pos"),
                    StateField("quat_to_axisangle", "robot0_eef_quat"),
                    StateField("vector", "robot0_gripper_qpos"),
                )
            ),
        ),
        metadata={"env": "robocasa"},
    )
