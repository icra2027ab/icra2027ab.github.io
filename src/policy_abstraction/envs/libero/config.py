"""Configuration constants for LIBERO evaluations."""

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

# LIBERO environment constants
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data

# Maximum steps per task suite (based on longest training demo)
TASK_MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}

__all__ = [
    "LIBERO_DUMMY_ACTION",
    "LIBERO_ENV_RESOLUTION",
    "TASK_MAX_STEPS",
    "LIBERO_DEPTH_NEAR",
    "LIBERO_DEPTH_FAR",
    "DEFAULT_RESIZE_SIZE",
    "libero_env_spec",
]

# Image preprocessing
DEFAULT_RESIZE_SIZE = 224

# ─────────────────────────────────────────────────────────────────────────────
# Depth conversion  (normalised MuJoCo zbuffer → metric metres)
# ─────────────────────────────────────────────────────────────────────────────
# Default near/far for LIBERO tabletop environments
# (znear_frac=0.001, zfar_frac=50.0, scene extent ≈ 10.6 m, measured empirically).
LIBERO_DEPTH_NEAR: float = 0.0106
LIBERO_DEPTH_FAR: float = 530.49


def libero_env_spec() -> EnvSpec:
    """Build the EnvSpec describing every LIBERO suite.

    All LIBERO suites share the same camera layout (agentview + wrist), the
    same Franka Panda robot, and the same observation-key naming, so a single
    spec covers the entire benchmark.
    """
    return EnvSpec(
        cameras=(
            CameraSpec(
                name="agentview",
                role="world",
                image_key="agentview_image",
                depth_key="agentview_depth",
                segmentation_key="agentview_segmentation_instance",
                depth_near=LIBERO_DEPTH_NEAR,
                depth_far=LIBERO_DEPTH_FAR,
                image_transform="flip_vh",
                is_primary=True,
            ),
            CameraSpec(
                name="robot0_eye_in_hand",
                role="wrist",
                image_key="robot0_eye_in_hand_image",
                depth_key="robot0_eye_in_hand_depth",
                segmentation_key="robot0_eye_in_hand_segmentation_instance",
                depth_near=LIBERO_DEPTH_NEAR,
                depth_far=LIBERO_DEPTH_FAR,
                image_transform="flip_vh",
            ),
        ),
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
            images=(
                ("agentview", "observation/image"),
                ("robot0_eye_in_hand", "observation/wrist_image"),
            ),
            # pi0/LIBERO checkpoints expect the raw 2-DOF gripper qpos
            # (state_dim=8), distinct from the internal 1-D aperture used
            # for dynamics / visualisation.
            state_layout=StateLayout(
                fields=(
                    StateField("pos", "robot0_eef_pos"),
                    StateField("quat_to_axisangle", "robot0_eef_quat"),
                    StateField("vector", "robot0_gripper_qpos"),
                )
            ),
        ),
        metadata={"env": "libero"},
    )
