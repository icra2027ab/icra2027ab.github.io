"""UR5e configuration constants and :class:`EnvSpec` factory.

Kept leaf-level (no ``rtde_*`` / ``lerobot`` imports) so it can be imported
both from the evaluation venv and from the ``physical-ai-sdk`` venv that runs
the server — mirroring what :mod:`policy_abstraction.envs.libero.config`,
:mod:`policy_abstraction.envs.robocasa.config` and
:mod:`policy_abstraction.envs.calvin.config` do.

Differences from the simulated adapters
---------------------------------------
* **Action space.** LIBERO / RoboCasa / CALVIN emit normalised Cartesian
  deltas for an OSC-style controller. The UR5e is driven with ``servoJ``, so
  the policy's action vector is *absolute joint targets in radians* plus an
  absolute gripper channel — ``ActionLayout.space == "joint_absolute"``.
* **No depth / segmentation.** The exterior and wrist cameras are plain UVC
  colour cameras, so ``depth_key`` and ``segmentation_key`` stay ``None`` and
  the GT-perception modes are unavailable. Use
  ``--perception.mode REAL_PERCEPTION`` when running with guidance.
* **Gripper convention.** ``lerobot_robot_ur5e`` reports the Robotiq position
  as ``0 = open, 1 = closed``. The server republishes it as an *aperture*
  (``1 = open``) under ``gripper_qpos`` so it matches the sign convention the
  rest of the repo assumes, and converts the policy's gripper channel back on
  the way out. See :mod:`policy_abstraction.envs.ur5.server`.

Observation key contract (emitted verbatim by the server):

* ``joint_pos`` — (6,) arm joint positions, radians
* ``joint_vel`` — (6,) arm joint velocities, rad/s
* ``gripper_qpos`` — (1,) gripper aperture in ``[0, 1]`` (1 = open)
* ``eef_pos`` — (3,) TCP position in the robot base frame, metres
* ``eef_quat`` — (4,) TCP orientation as ``[x, y, z, w]``
* ``eef_axis_angle`` — (3,) TCP orientation as the raw RTDE rotation vector
* ``<camera name>`` — (H, W, 3) uint8 frame, one key per configured camera
"""

from __future__ import annotations

import os

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

# ---------------------------------------------------------------------------
# Hardware / control constants
# ---------------------------------------------------------------------------

# The one place the arm's address is written down. Everything that needs it —
# the env server, its launcher, evaluate.py, the shell wrappers — reads this,
# so pointing at a different cell is one env var rather than a grep. It used to
# be hardcoded in four files at two different values, which is exactly the
# failure this prevents.
DEFAULT_ROBOT_IP: str = os.environ.get("UR5_ROBOT_IP", "192.168.1.23")


# The TCP offset configured on this cell's controller: the closed fingertip
# plane of the Robotiq 2F-140, 223 mm out along the tool z axis. Every FK
# prediction and the camera calibration tool measure to this point, so it
# must match the teach pendant; the env server warns at connect if it does
# not. Override with UR5_TCP_OFFSET="x y z rx ry rz" for another cell.
def _tcp_offset_from_env() -> tuple[float, float, float, float, float, float]:
    raw = os.environ.get("UR5_TCP_OFFSET", "0.00052 0.00109 0.22347 0 0 0").split()
    if len(raw) != 6:
        raise ValueError(
            f"UR5_TCP_OFFSET must hold six numbers (x y z rx ry rz), got {raw!r}"
        )
    x, y, z, rx, ry, rz = (float(v) for v in raw)
    return (x, y, z, rx, ry, rz)


DEFAULT_TCP_OFFSET: tuple[float, float, float, float, float, float] = (
    _tcp_offset_from_env()
)

# Actions must be executed at the rate they were recorded at, otherwise the arm
# lags behind the trajectory the policy predicted. 50 Hz matches the datasets
# the pi0/pi05 UR5e checkpoints were trained on.
DEFAULT_CONTROL_HZ: float = 50.0

# Per-episode step cap. 50 Hz × 60 s.
DEFAULT_MAX_STEPS: int = 3000

# Per-step joint-target safety clamp (radians). A well-behaved 50 Hz policy
# commands far less than this; the clamp only bounds how far a single malformed
# action can throw the arm. 0.05 rad/step ≈ 2.5 rad/s at 50 Hz, below the UR5e
# joint speed limit of π rad/s.
DEFAULT_MAX_JOINT_DELTA: float = 0.05

# UR5e joint limits (radians). The controller enforces ±2π; we clamp tighter to
# keep the arm inside a sane working envelope for tabletop manipulation.
UR5E_JOINT_LIMITS: tuple[tuple[float, float], ...] = (
    (-6.2831853, 6.2831853),
    (-6.2831853, 6.2831853),
    (-3.1415927, 3.1415927),
    (-6.2831853, 6.2831853),
    (-6.2831853, 6.2831853),
    (-6.2831853, 6.2831853),
)

# Joint pose the arm returns to at the start and end of every episode. Matches
# ``lerobot_robot_ur5e.UR5E.HOME_JOINT_POSITIONS``; kept here so the launcher
# and docs can reference it without importing the hardware driver.
UR5E_HOME_JOINT_POSITIONS: tuple[float, ...] = (
    0.0,
    -1.5707963,
    1.5707963,
    -1.5707963,
    -1.5707963,
    0.0,
)

# LeRobot camera keys used by the pi0 / pi05 UR5e checkpoints. These double as
# the EnvSpec camera *names* and the raw observation keys so a single
# identifier flows from the driver through the wire format into the policy.
DEFAULT_EXTERIOR_CAMERA = "exterior_image_1_left"
DEFAULT_WRIST_CAMERA = "wrist_image_left"

# Native capture resolution of the UVC cameras used during data collection.
UR5_CAMERA_HEIGHT = 480
UR5_CAMERA_WIDTH = 640

# Names of the 7 hardware action channels, in order. Matches
# ``lerobot_robot_ur5e.UR5E.action_features``.
UR5_MOTOR_NAMES: tuple[str, ...] = (
    "joint_0",
    "joint_1",
    "joint_2",
    "joint_3",
    "joint_4",
    "joint_5",
    "gripper",
)


def ur5_env_spec(
    cameras: tuple[str, ...] = (DEFAULT_EXTERIOR_CAMERA, DEFAULT_WRIST_CAMERA),
    wrist_camera: str = DEFAULT_WRIST_CAMERA,
    primary_camera: str = DEFAULT_EXTERIOR_CAMERA,
    tcp_offset: tuple[float, ...] | None = None,
    control_hz: float = DEFAULT_CONTROL_HZ,
    depth_cameras: tuple[str, ...] = (),
) -> EnvSpec:
    """Build the :class:`EnvSpec` for a UR5e + Robotiq 2F-140 cell.

    Args:
        cameras: Camera keys, in the order the server publishes them. Each
            becomes both an ``EnvSpec`` camera name and the observation key
            holding that camera's frame.
        wrist_camera: Which of *cameras* is mounted on the tool. Pass a name
            that isn't in *cameras* (e.g. ``""``) for a cell with no wrist cam.
        primary_camera: Which of *cameras* is tagged ``is_primary``.
        tcp_offset: The ``[x, y, z, rx, ry, rz]`` TCP offset configured on the
            UR controller, published in ``metadata`` so
            :class:`~policy_abstraction.dynamics.ur5_kinematics.Ur5KinematicsDynamicsModel`
            can predict poses that match ``getActualTCPPose()``.
        control_hz: Rate the server paces ``step`` at, published in
            ``metadata`` so consumers can derive the control ``dt``.
        depth_cameras: Which of *cameras* also publish metric depth (the ones
            streamed through the RealSense SDK). These get a ``<name>_depth``
            key and ``depth_is_metric=True``; the rest stay colour-only and are
            skipped by the perception pipeline.

    Raises:
        ValueError: if *cameras* is empty, *primary_camera* isn't in it, or
            *depth_cameras* names a camera that isn't in *cameras*.
    """
    if not cameras:
        raise ValueError("ur5_env_spec requires at least one camera")
    if primary_camera not in cameras:
        raise ValueError(
            f"primary_camera={primary_camera!r} not in cameras={list(cameras)}"
        )
    unknown_depth = [c for c in depth_cameras if c not in cameras]
    if unknown_depth:
        raise ValueError(
            f"depth_cameras={unknown_depth} not in cameras={list(cameras)}"
        )

    camera_specs = tuple(
        CameraSpec(
            name=name,
            role="wrist" if name == wrist_camera else "world",
            image_key=name,
            # Depth only for cameras streamed through the RealSense SDK, which
            # registers it to the colour frame. Colour-only cameras stay
            # depth_key=None and the perception pipeline skips them.
            depth_key=f"{name}_depth" if name in depth_cameras else None,
            depth_is_metric=name in depth_cameras,
            # No instance segmentation on any of these — GT-perception modes
            # are unavailable; use --perception.mode REAL_PERCEPTION.
            segmentation_key=None,
            image_transform="none",
            is_primary=(name == primary_camera),
        )
        for name in cameras
    )

    return EnvSpec(
        cameras=camera_specs,
        robot=RobotSpec(
            eef_pos_key="eef_pos",
            eef_quat_key="eef_quat",
            gripper_qpos_key="gripper_qpos",
            joint_pos_key="joint_pos",
            # Internal state used for dynamics + visualisation: Cartesian TCP
            # pose plus a scalar aperture, same 7-D layout every other env uses
            # so ChunkProjector / guidance need no special-casing.
            state_layout=StateLayout(
                fields=(
                    StateField("pos", "eef_pos"),
                    StateField("quat_to_axisangle", "eef_quat"),
                    StateField("gripper_aperture", "gripper_qpos"),
                )
            ),
            gripper=GripperSpec(
                # Selects configs/robotiq_2f140.yaml in policy_steering, which
                # holds the 140 mm stroke and the fingertip-plane TCP offsets.
                # The demo side keeps Panda geometry — see RolloutAbstractor.
                geometry_id="robotiq_2f140",
                finger_count=2,
                # The driver already reports a single scalar; the server has
                # flipped it to openness, so pass it straight through.
                aperture_op="identity",
                speed_threshold=0.01,
            ),
            action_layout=ActionLayout(
                action_dim=7,
                eef_pos_slice=None,
                eef_rot_slice=None,
                gripper_dim=6,
                integrate_eef=False,
                joint_pos_slice=(0, 6),
                space="joint_absolute",
            ),
            robot_description="ur5e_mj_description",
            entity_names=("ee", "gripper_l", "gripper_r"),
            visualizer_negate_gripper_joint=False,
        ),
        policy_input_keys=PolicyIOKeys(
            state_key="observation/state",
            prompt_key="prompt",
            # Every policy client reads views out of the element by the
            # ``observation/`` prefix, so the key has to carry it -- without
            # it the LeRobot client sees no cameras at all. The view name
            # after the prefix stays the raw camera key: the server pairs
            # views with the checkpoint's ``observation.images.*`` features
            # by name, and these already match a DROID/UR5e checkpoint's.
            images=tuple((name, f"observation/{name}") for name in cameras),
            # pi0/pi05 UR5e checkpoints consume the 7-D joint+gripper vector,
            # not the Cartesian state used internally.
            state_layout=StateLayout(
                fields=(
                    StateField("vector", "joint_pos"),
                    StateField("vector", "gripper_qpos"),
                )
            ),
        ),
        metadata={
            "env": "ur5",
            "robot_type": "ur5_follower",
            "control_hz": float(control_hz),
            "motor_names": list(UR5_MOTOR_NAMES),
            "tcp_offset": (list(tcp_offset) if tcp_offset is not None else [0.0] * 6),
        },
    )


__all__ = [
    "DEFAULT_CONTROL_HZ",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MAX_JOINT_DELTA",
    "DEFAULT_TCP_OFFSET",
    "DEFAULT_EXTERIOR_CAMERA",
    "DEFAULT_WRIST_CAMERA",
    "UR5E_JOINT_LIMITS",
    "UR5E_HOME_JOINT_POSITIONS",
    "UR5_CAMERA_HEIGHT",
    "UR5_CAMERA_WIDTH",
    "UR5_MOTOR_NAMES",
    "ur5_env_spec",
]
