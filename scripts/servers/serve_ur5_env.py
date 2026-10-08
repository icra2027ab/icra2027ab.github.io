"""Entry-point script for the UR5e environment WebSocket server.

Run this in the ``physical-ai-sdk`` venv (which carries ``lerobot``,
``rtde_control``/``rtde_receive`` and ``lerobot_robot_ur5e``) *before*
launching the evaluation script — or let ``scripts/evaluate.py`` spawn it for
you via ``policy_abstraction.envs.ur5.utils.start_ur5_env_server``.

    # Terminal 1 — physical-ai-sdk Python (the robot host)
    ~/dev/physical-ai-sdk/.venv/bin/python scripts/servers/serve_ur5_env.py \\
        --host 0.0.0.0 --port 8770 \\
        --robot-ip "$UR5_ROBOT_IP" \\
        --tasks "Put all cables in the bag." \\
        --cameras exterior_image_1_left=/dev/video4:640x480@30 \\
                  wrist_image_left=/dev/video12:640x480@30

    # Terminal 2 — policy Python (root venv)
    python -c "from policy_abstraction.envs import WebsocketEnvClient; \\
        env = WebsocketEnvClient(port=8770); print(env.setup_suite())"

The wire protocol matches :class:`LiberoEnvServer`, :class:`RobocasaEnvServer`
and :class:`CalvinEnvServer`, so the existing
:class:`policy_abstraction.envs.WebsocketEnvClient` connects without changes.

Required transport deps in the physical-ai-sdk venv (one-time):

    uv pip install --python ~/dev/physical-ai-sdk/.venv \\
        websockets msgpack-numpy loguru

Safety
------
``--dry-run`` connects to the arm and reads sensors but never commands motion.
``--simulate-hardware`` skips the arm entirely and serves synthetic
observations, so the whole pipeline can be exercised with the robot powered
off. Start with one of those before running a policy for real, and keep a hand
on the e-stop the first time you don't.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import signal
import sys
from typing import Optional

import tyro

# Make src/ importable when launched from an arbitrary cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from policy_abstraction.envs.ur5.config import (  # noqa: E402
    DEFAULT_CONTROL_HZ,
    DEFAULT_EXTERIOR_CAMERA,
    DEFAULT_MAX_JOINT_DELTA,
    DEFAULT_MAX_STEPS,
    DEFAULT_ROBOT_IP,
    DEFAULT_WRIST_CAMERA,
)
from policy_abstraction.envs.ur5.server import Ur5EnvServer  # noqa: E402


@dataclasses.dataclass
class Args:
    """Arguments for the UR5e environment server."""

    host: str = "0.0.0.0"
    """Interface to bind to (0.0.0.0 accepts all incoming connections)."""

    port: int = 8770
    """WebSocket port to listen on."""

    tasks: Optional[list[str]] = None
    """Natural-language instructions, one per ``task_id``.

    These are what the policy is prompted with, so they must match the
    checkpoint's training instruction (e.g. ``"Put all cables in the bag."``).
    """

    suite_name: str = "ur5"
    """Logical suite label returned to the client via ``setup_suite``."""

    robot_ip: str = DEFAULT_ROBOT_IP
    """IP address of the UR controller."""

    cameras: Optional[list[str]] = None
    """Camera specs as ``name=/dev/videoN[:WxH][@FPS]``.

    The names double as the observation keys and the policy's image keys, so
    they must match the checkpoint's camera names (LeRobot strips the
    ``observation.images.`` prefix). Defaults to
    ``exterior_image_1_left=/dev/video4`` and ``wrist_image_left=/dev/video12``.

    One camera per *device node* — reusing the same ``/dev/videoN`` twice fails
    on the second ``connect()``. Find working nodes with ``v4l2-ctl
    --list-devices`` and ``--list-formats-ext``.
    """

    wrist_camera: str = DEFAULT_WRIST_CAMERA
    """Which camera is tool-mounted (tagged ``role="wrist"`` in the EnvSpec)."""

    primary_camera: str = DEFAULT_EXTERIOR_CAMERA
    """Which camera is tagged ``is_primary`` in the published EnvSpec."""

    control_hz: float = DEFAULT_CONTROL_HZ
    """Rate ``step`` paces itself to. Must match the rate the policy's training
    data was recorded at, otherwise the arm trails the predicted trajectory."""

    max_steps: int = DEFAULT_MAX_STEPS
    """Per-episode step cap returned via ``setup_suite``."""

    num_init_states: int = 10
    """How many (placeholder) initial states to advertise per task. Must be at
    least ``--iter.num-trials-per-task`` on the evaluation side."""

    max_joint_delta: float = DEFAULT_MAX_JOINT_DELTA
    """Safety clamp: max radians a single commanded joint target may sit away
    from the measured joint position."""

    no_flip_gripper: bool = False
    """Don't invert the gripper channel. The Robotiq driver reports 0=open /
    1=closed while the pi0/pi05 UR5e checkpoints use openness, so inverting
    (the default) is almost always right."""

    no_home_on_reset: bool = False
    """Skip the homing move at the start of each episode."""

    no_home_on_disconnect: bool = False
    """Don't park the arm at its home pose when a client disconnects. Parking
    on disconnect is what guarantees the arm is home before the evaluation
    script's (short) shutdown window expires."""

    stage_prompt: bool = False
    """Wait for ENTER on stdin after homing so the operator can stage the
    scene. Ignored when stdin isn't a TTY (e.g. spawned by evaluate.py)."""

    success_mode: str = "never"
    """``never`` (episodes end at max_steps) or ``stdin`` (type ``s``/``f`` +
    ENTER during a rollout to end it as success/failure). ``stdin`` requires
    running this server by hand in its own terminal."""

    dry_run: bool = False
    """Connect and read sensors, but never command motion."""

    simulate_hardware: bool = False
    """Don't touch the robot at all; serve synthetic observations from
    ``FakeUr5Driver``. For pipeline smoke tests with the arm powered off."""

    tcp_offset: Optional[list[float]] = None
    """``x y z rx ry rz`` TCP offset configured on the UR controller. Published
    in the EnvSpec metadata so the UR5 FK dynamics model predicts poses that
    match ``getActualTCPPose()``. Read it off the teach pendant."""

    calibration_base_frame: Optional[str] = None
    """Which UR base frame the calibration's extrinsic is expressed in.

    ``base`` (default) is the controller's frame — what RTDE reports, and what
    this adapter publishes. ``base_link`` is the ROS/URDF convention, rotated
    180 degrees about z; calibration tools built on a URDF (Kalib included)
    commonly produce that one. Mixing them mirrors the scene in x and y while
    leaving the table perfectly horizontal, so it survives a plane-fit check and
    only shows up against the robot's own reported TCP."""

    calibration_path: Optional[str] = None
    """Camera calibration JSON. Two layouts are accepted:

    * this repo's — ``{camera: {extrinsic: 4x4, intrinsic?: 3x3}}``, extrinsic
      being the camera-to-base pose;
    * Kalib's ``T_cam_base.json`` as written — attached to ``--primary-camera``,
      with the transform direction taken from its ``convention`` field.

    Required for reconstruction and for action-chunk projection overlays; omit
    it and perception skips the camera and projection is simply not drawn.
    ``intrinsic`` is optional for ``rs:`` cameras (the SDK reports it)."""


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if not args.tasks:
        raise SystemExit(
            'Pass --tasks "<instruction>" [...] (at least one). This is the '
            "prompt handed to the policy, so it must match the checkpoint."
        )

    server = Ur5EnvServer(
        host=args.host,
        port=args.port,
        tasks=args.tasks,
        suite_name=args.suite_name,
        robot_ip=args.robot_ip,
        cameras=args.cameras,
        wrist_camera=args.wrist_camera,
        primary_camera=args.primary_camera,
        control_hz=args.control_hz,
        max_steps=args.max_steps,
        num_init_states=args.num_init_states,
        max_joint_delta=args.max_joint_delta,
        flip_gripper=not args.no_flip_gripper,
        home_on_reset=not args.no_home_on_reset,
        home_on_disconnect=not args.no_home_on_disconnect,
        stage_prompt=args.stage_prompt,
        success_mode=args.success_mode,
        dry_run=args.dry_run,
        simulate_hardware=args.simulate_hardware,
        tcp_offset=args.tcp_offset,
        calibration_path=args.calibration_path,
        calibration_base_frame=args.calibration_base_frame,
    )

    # Claiming a camera is the one startup failure that would otherwise wait
    # for the first episode -- i.e. until the policy server has loaded.
    server.preflight_cameras()

    # A killed server must not leave the arm mid-servo.
    def _shutdown(signum, _frame):
        logging.getLogger(__name__).info(f"Signal {signum} — stopping the arm.")
        server.shutdown()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    try:
        server.serve_forever()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main(tyro.cli(Args))
