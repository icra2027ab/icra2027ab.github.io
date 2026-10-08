# ruff: noqa: E402
"""UR5e WebSocket environment server.

Run from the ``physical-ai-sdk`` venv via ``scripts/servers/serve_ur5_env.py``.
Implements the same wire protocol as the LIBERO, RoboCasa and CALVIN servers,
so the existing :class:`policy_abstraction.envs.WebsocketEnvClient` connects
unchanged and ``EpisodeRunner`` cannot tell a real arm from a simulator.

Suite model
-----------
Each ``task_id`` maps to one natural-language instruction passed on the CLI.
``get_task_init_states`` returns *N* identical placeholder states — a real cell
has no savable simulator state, so "resetting" means moving the arm back to its
home joint configuration and letting the operator re-arrange the scene.
``set_init_state`` performs that homing move and then blocks until the operator
confirms the scene is staged (see ``--stage-prompt``).

Control model
-------------
``step`` streams **one** absolute joint target through ``servoJ`` and paces
itself to ``--control-hz`` so the arm tracks the rate the policy was trained at.
Safety is enforced here rather than in the caller:

* every target is clamped to :data:`UR5E_JOINT_LIMITS`;
* every target is clamped to at most ``--max-joint-delta`` radians away from the
  *measured* joint positions, so a single malformed action cannot throw the arm;
* non-finite actions are rejected outright;
* ``--dry-run`` reads sensors and reports the clamped target without commanding
  any motion — use it to exercise the whole pipeline before powering the arm.

The UR5e is physical hardware, so — unlike the simulator servers — a single
driver instance is shared for the server's lifetime and survives client
reconnects. Dropping a connection stops motion and parks the arm at its home
pose, but keeps the RTDE session open for the next client.
"""

from __future__ import annotations

import os
import sys

# Make policy_abstraction importable when running from the physical-ai-sdk venv.
# server.py is at src/policy_abstraction/envs/ur5/server.py; dirname×4 = src/.
_SRC_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import json
import logging
import math
import select
import time
from typing import Any, Optional

import numpy as np

from policy_abstraction.dynamics.ur5_kinematics import rotvec_to_matrix
from policy_abstraction.envs.base_server import BaseEnvServer, pack_result
from policy_abstraction.envs.ur5.config import (
    DEFAULT_CONTROL_HZ,
    DEFAULT_EXTERIOR_CAMERA,
    DEFAULT_MAX_JOINT_DELTA,
    DEFAULT_MAX_STEPS,
    DEFAULT_ROBOT_IP,
    DEFAULT_WRIST_CAMERA,
    UR5_CAMERA_HEIGHT,
    UR5_CAMERA_WIDTH,
    UR5E_HOME_JOINT_POSITIONS,
    UR5E_JOINT_LIMITS,
    ur5_env_spec,
)
from policy_abstraction.envs.ur5.realsense import is_realsense_device

logger = logging.getLogger(__name__)

# Success/failure reporting via stdin (``--success-mode stdin``).
_SUCCESS_KEYS = ("s", "y", "1")
_FAILURE_KEYS = ("f", "n", "0")


# ---------------------------------------------------------------------------
# Camera spec parsing
# ---------------------------------------------------------------------------


def _probe_v4l2(device: str) -> str | None:
    """Open and release a V4L2 node; return an error string, or None."""
    import cv2

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            return "could not be opened (busy, or not a capture device)"
    finally:
        cap.release()
    return None


def parse_camera_arg(spec: str) -> dict:
    """Parse ``name=/dev/videoN[:WxH][@FPS]`` into a camera descriptor dict.

    Examples
    --------
    ``exterior_image_1_left=/dev/video4``
    ``exterior_image_1_left=/dev/video4:640x480@30``

    Raises:
        ValueError: on a malformed spec.
    """
    if "=" not in spec:
        raise ValueError(
            f"camera spec {spec!r} must look like 'name=/dev/videoN[:WxH][@FPS]'"
        )
    name, rest = spec.split("=", 1)
    name = name.strip()
    rest = rest.strip()
    if not name or not rest:
        raise ValueError(f"camera spec {spec!r} has an empty name or device")

    fps = 30
    if "@" in rest:
        rest, fps_str = rest.rsplit("@", 1)
        fps = int(fps_str)

    width, height = UR5_CAMERA_WIDTH, UR5_CAMERA_HEIGHT
    # Split on the LAST ':' so device paths keep working; a resolution suffix
    # always contains an 'x' between two integers.
    if ":" in rest:
        head, tail = rest.rsplit(":", 1)
        if "x" in tail:
            w_str, h_str = tail.lower().split("x", 1)
            width, height = int(w_str), int(h_str)
            rest = head

    return {
        "name": name,
        "device": rest,
        "width": width,
        "height": height,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Small math helpers (numpy-only so the sim venvs never need scipy here)
# ---------------------------------------------------------------------------


def rotvec_to_quat(rotvec) -> np.ndarray:
    """Rotation vector (axis * angle, radians) → quaternion ``[x, y, z, w]``.

    RTDE's ``getActualTCPPose()`` returns the orientation in exactly this
    axis-angle form, and the rest of the repo speaks xyzw quaternions.
    """
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    axis = r / theta
    s = math.sin(theta / 2.0)
    return np.array(
        [axis[0] * s, axis[1] * s, axis[2] * s, math.cos(theta / 2.0)],
        dtype=np.float64,
    )


#: SDK module names carrying the UR5e driver, newest first. The package was
#: renamed and moved when physical-ai-sdk grew a robot registry
#: (``physical_ai_sdk/lerobot_robot_ur5e`` -> ``physical_ai_sdk/robots/
#: lerobot_robot_ur5``); both spellings export ``UR5E`` and ``UR5EConfig``, so
#: which one is installed is not something this adapter needs an opinion about.
_UR5_DRIVER_MODULES = ("lerobot_robot_ur5", "lerobot_robot_ur5e")


def _import_ur5_driver() -> tuple[type, type]:
    """Return ``(UR5E, UR5EConfig)`` from whichever SDK layout is installed.

    Raises with both names when neither imports: the bare
    ``ModuleNotFoundError`` names only the spelling this code happened to try,
    which sends you looking for a module that was merely renamed.
    """
    import importlib

    errors: list[str] = []
    for name in _UR5_DRIVER_MODULES:
        try:
            mod = importlib.import_module(name)
        except ImportError as exc:
            errors.append(f"{name}: {exc}")
            continue
        try:
            return mod.UR5E, mod.UR5EConfig
        except AttributeError as exc:  # imported, but not the module we mean
            errors.append(f"{name}: {exc}")
    raise ImportError(
        "Could not import the UR5e driver from the physical-ai-sdk venv. Tried "
        + ", ".join(_UR5_DRIVER_MODULES)
        + ". The SDK moved this package into a robot registry, so an older "
        "checkout provides `lerobot_robot_ur5e` and a newer one "
        "`lerobot_robot_ur5`; an editable install left over from before the "
        "move points at a path that no longer exists and fails exactly like a "
        "missing dependency. Re-sync the SDK venv (`uv sync` in "
        "--policy.repo-path) if neither name resolves.\n  " + "\n  ".join(errors)
    )


def stdin_is_interactive() -> bool:
    """Whether stdin is a TTY we can poll for operator input.

    A server spawned by ``evaluate.py`` inherits a non-interactive stdin, so
    the operator-driven features (``--stage-prompt``, ``--success-mode stdin``)
    check this once at startup instead of guarding every read.
    """
    return sys.stdin is not None and not sys.stdin.closed and sys.stdin.isatty()


def _poll_stdin_verdict() -> Optional[bool]:
    """Non-blocking read of an operator verdict from stdin.

    Returns ``True`` on success, ``False`` on failure, ``None`` when nothing
    was typed. Only call this when :func:`stdin_is_interactive`.
    """
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return None
    line = sys.stdin.readline().strip().lower()
    if not line:
        return None
    if line[0] in _SUCCESS_KEYS:
        return True
    if line[0] in _FAILURE_KEYS:
        return False
    return None


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class Ur5EnvServer(BaseEnvServer):
    """Serves a physical UR5e + Robotiq cell over a WebSocket connection."""

    server_metadata = {"server": "Ur5EnvServer", "version": "1.0"}

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8770,
        *,
        tasks: list[str] | None = None,
        suite_name: str = "ur5",
        robot_ip: str = DEFAULT_ROBOT_IP,
        cameras: list[str] | None = None,
        wrist_camera: str = DEFAULT_WRIST_CAMERA,
        primary_camera: str = DEFAULT_EXTERIOR_CAMERA,
        control_hz: float = DEFAULT_CONTROL_HZ,
        max_steps: int = DEFAULT_MAX_STEPS,
        num_init_states: int = 10,
        max_joint_delta: float = DEFAULT_MAX_JOINT_DELTA,
        flip_gripper: bool = True,
        home_on_reset: bool = True,
        home_on_disconnect: bool = True,
        stage_prompt: bool = False,
        success_mode: str = "never",
        dry_run: bool = False,
        simulate_hardware: bool = False,
        tcp_offset: list[float] | None = None,
        calibration_path: str | None = None,
        calibration_base_frame: str | None = None,
    ) -> None:
        super().__init__(host=host, port=port)
        if not tasks:
            raise ValueError("Ur5EnvServer requires a non-empty tasks list.")
        self._tasks = list(tasks)
        self._suite_name = suite_name
        self._robot_ip = robot_ip
        self._camera_specs = [
            parse_camera_arg(c)
            for c in (
                cameras
                if cameras
                else [
                    f"{DEFAULT_EXTERIOR_CAMERA}=/dev/video4",
                    f"{DEFAULT_WRIST_CAMERA}=/dev/video12",
                ]
            )
        ]
        self._camera_names = tuple(c["name"] for c in self._camera_specs)
        if primary_camera not in self._camera_names:
            raise ValueError(
                f"--primary-camera {primary_camera!r} is not among the configured "
                f"cameras {list(self._camera_names)}"
            )
        self._wrist_camera = wrist_camera
        self._primary_camera = primary_camera
        self._control_hz = float(control_hz)
        self._control_period = 1.0 / self._control_hz
        self._max_steps = int(max_steps)
        self._num_init_states = int(num_init_states)
        self._max_joint_delta = float(max_joint_delta)
        self._flip_gripper = bool(flip_gripper)
        self._home_on_reset = bool(home_on_reset)
        self._home_on_disconnect = bool(home_on_disconnect)
        self._stage_prompt = bool(stage_prompt)
        if success_mode not in ("never", "stdin"):
            raise ValueError(
                f"success_mode must be 'never' or 'stdin', got {success_mode!r}"
            )
        # Both operator-driven features need a TTY. Resolve that once here so
        # neither the per-step poll nor the reset path has to re-check.
        self._interactive = stdin_is_interactive()
        if not self._interactive and (success_mode == "stdin" or stage_prompt):
            logger.warning(
                "stdin is not a TTY, so --success-mode stdin / --stage-prompt "
                "have no effect. Start this server by hand in its own terminal "
                "(and pass --env.no-launch-server to evaluate.py) to use them."
            )
        self._success_mode = success_mode
        self._dry_run = bool(dry_run)
        self._simulate_hardware = bool(simulate_hardware)
        if tcp_offset is not None and len(tcp_offset) != 6:
            raise ValueError(
                f"tcp_offset must be [x, y, z, rx, ry, rz]; got {tcp_offset!r}"
            )
        self._tcp_offset = (
            [float(v) for v in tcp_offset] if tcp_offset is not None else [0.0] * 6
        )
        self._calibration = self._load_calibration(
            calibration_path, primary_camera, calibration_base_frame
        )
        self._check_calibration_tcp_offset()

        # Only SDK-driven cameras can publish depth registered to their colour
        # frame, so they are the only ones perception can reconstruct from --
        # and of those, only the ones it can actually place in the world. Two
        # D435s streaming colour+depth over a single USB controller exceed its
        # bandwidth (the second camera starts, then never delivers a frame), so
        # depth is spent where it is used rather than everywhere it is possible.
        self._depth_camera_names = tuple(
            c["name"]
            for c in self._camera_specs
            if is_realsense_device(c["device"]) and self._wants_depth(c["name"])
        )

        # The arm is physical: one driver for the whole server lifetime, shared
        # across client connections.
        self._robot: Any = None
        # name -> RealSenseCamera for the cameras streamed via the SDK.
        self._realsense: dict[str, Any] = {}
        # Requested output resolution, set by make_env.
        self._out_hw: tuple[int, int] | None = None
        # servoJ pacing state.
        self._next_tick: float | None = None
        self._last_step_end: float | None = None
        # Steps where this server overran its own deadline …
        self._late_steps = 0
        # … versus steps that were late because the caller stalled (inference).
        self._stall_steps = 0
        self._total_steps = 0
        self._consecutive_clamps = 0
        # Last commanded gripper openness, replayed when an action says "hold".
        self._last_gripper_openness = 1.0
        self._hold_warned = False

    # ------------------------------------------------------------------
    # Per-connection state
    # ------------------------------------------------------------------

    def _initial_state(self) -> dict:
        return {"env": None, "task_id": None, "step_idx": 0}

    def _cleanup_state(self, state: dict) -> None:
        """Park the arm when a client drops, but keep the RTDE session alive.

        The base class would call ``env.close()``; for hardware that would tear
        down the driver every time the evaluation script reconnects.

        Homing happens here rather than only at process shutdown because the
        evaluation script disconnects *before* it signals the server, and its
        cleanup only allows a few seconds for the child to exit — not enough
        for a blocking ``moveJ``. Parking on disconnect means the arm is
        already home by then.
        """
        self._halt()
        if self._home_on_disconnect and self._robot is not None:
            self._home()

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, cmd: str, args: list, kwargs: dict, state: dict) -> tuple:
        # ------------------------------------------------------------------ #
        # Suite-level commands                                                 #
        # ------------------------------------------------------------------ #

        if cmd == "setup_suite":
            return {
                "num_tasks": len(self._tasks),
                "task_languages": list(self._tasks),
                "task_suite_name": self._suite_name,
                "max_steps": self._max_steps,
                "evaluation_cfg": {
                    "robot_ip": self._robot_ip,
                    "control_hz": self._control_hz,
                    "dry_run": self._dry_run,
                    "flip_gripper": self._flip_gripper,
                    "max_joint_delta": self._max_joint_delta,
                    "success_mode": self._success_mode,
                },
                "env_spec": ur5_env_spec(
                    cameras=self._camera_names,
                    wrist_camera=self._wrist_camera,
                    primary_camera=self._primary_camera,
                    tcp_offset=tuple(self._tcp_offset),
                    control_hz=self._control_hz,
                    depth_cameras=self._depth_camera_names,
                ).to_dict(),
            }, state

        if cmd == "get_task_init_states":
            # A real cell has no savable state — every "initial state" is the
            # same home pose, and the scene is staged by the operator. We still
            # return one entry per trial so the episode loop indexes safely.
            return pack_result(
                [{"home": True, "index": i} for i in range(self._num_init_states)]
            ), state

        # ------------------------------------------------------------------ #
        # Environment lifecycle commands                                       #
        # ------------------------------------------------------------------ #

        if cmd == "make_env":
            task_id = int(kwargs.get("task_id", 0))
            if not (0 <= task_id < len(self._tasks)):
                raise IndexError(
                    f"task_id={task_id} out of range for {len(self._tasks)} task(s)"
                )
            self._out_hw = (
                int(kwargs.get("camera_heights", UR5_CAMERA_HEIGHT)),
                int(kwargs.get("camera_widths", UR5_CAMERA_WIDTH)),
            )
            self._ensure_robot()
            state["env"] = self._robot
            state["task_id"] = task_id
            state["step_idx"] = 0
            return {
                "language_instruction": self._tasks[task_id],
                # No instance segmentation on a plain UVC camera.
                "segmentation_name_mapping": {},
            }, state

        if cmd == "get_segmentation_name_mapping":
            return {}, state

        if cmd == "get_robot_base":
            # RTDE reports TCP poses in the robot base frame, so the base is
            # the world origin as far as this adapter is concerned.
            return pack_result(
                {
                    "base_pos": np.zeros(3, dtype=np.float64),
                    "base_ori": np.eye(3, dtype=np.float64),
                }
            ), state

        if cmd == "get_camera_params":
            requested = list(kwargs.get("camera_names", self._camera_names))
            requested = [n.decode() if isinstance(n, bytes) else n for n in requested]
            return pack_result(self._camera_params(requested)), state

        if cmd == "seed":
            # Nothing stochastic server-side; accepted for protocol parity.
            return None, state

        if cmd == "close":
            self._halt()
            state["env"] = None
            return None, state

        # ------------------------------------------------------------------ #
        # Commands that need a live robot                                      #
        # ------------------------------------------------------------------ #

        self._ensure_robot()
        state["env"] = self._robot

        if cmd in ("reset", "set_init_state"):
            obs = self._reset_episode()
            state["step_idx"] = 0
            return pack_result(obs), state

        if cmd == "home":
            self._home()
            return pack_result(self._read_obs()), state

        if cmd == "step":
            action = args[0] if args else kwargs.get("action")
            if action is None:
                raise ValueError("step requires an action arg.")
            obs, reward, done, info = self._step(action)
            state["step_idx"] = int(state.get("step_idx", 0)) + 1
            info["step_idx"] = state["step_idx"]
            return pack_result((obs, reward, done, info)), state

        if cmd == "get_obs":
            return pack_result(self._read_obs()), state

        raise AttributeError(
            f"Ur5EnvServer has no command {cmd!r}. Supported: setup_suite, "
            "get_task_init_states, make_env, reset, set_init_state, step, home, "
            "get_obs, get_robot_base, get_camera_params, "
            "get_segmentation_name_mapping, seed, close."
        )

    # ------------------------------------------------------------------
    # Hardware lifecycle
    # ------------------------------------------------------------------

    def _ensure_robot(self) -> None:
        """Create and connect the UR5e driver once, lazily."""
        if self._robot is not None:
            return

        if self._simulate_hardware:
            from policy_abstraction.envs.ur5.fake_driver import FakeUr5Driver

            # Real RealSense cameras are still opened when configured: the arm
            # and the cameras fail independently, and perception work needs
            # real frames + depth far more often than it needs a moving arm.
            self._connect_realsense()
            live = sorted(self._realsense)
            logger.warning(
                "--simulate-hardware: using FakeUr5Driver. No real arm is being "
                + (
                    f"commanded; cameras {live} are REAL, the rest are synthetic."
                    if live
                    else "commanded and the camera frames are synthetic."
                )
            )
            self._robot = FakeUr5Driver(
                self._camera_specs, self._control_period, tcp_offset=self._tcp_offset
            )
            return

        # Imported lazily so `--help`, spec tests and the config module all
        # work without the hardware stack installed.
        from pathlib import Path

        from lerobot.cameras.configs import Cv2Backends
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

        UR5E, UR5EConfig = _import_ur5_driver()

        # RealSense cameras are driven by their own SDK (for registered depth);
        # the driver only manages the V4L2 colour cameras.
        camera_config = {
            spec["name"]: OpenCVCameraConfig(
                index_or_path=Path(spec["device"]),
                backend=Cv2Backends.V4L2,
                width=spec["width"],
                height=spec["height"],
                fps=spec["fps"],
            )
            for spec in self._camera_specs
            if not is_realsense_device(spec["device"])
        }
        self._connect_realsense()
        logger.info(
            f"Connecting to UR5e at {self._robot_ip} with cameras "
            f"{ {s['name']: s['device'] for s in self._camera_specs} }"
        )
        robot = UR5E(
            UR5EConfig(
                id="ur5_follower",
                ip=self._robot_ip,
                cameras=camera_config,
                async_action_duration=self._control_period,
            )
        )
        robot.connect()
        # `UR5E.connect` swallows connection errors and just logs them, so the
        # only reliable readiness check is the driver's own flag.
        if not robot.is_connected:
            raise RuntimeError(
                f"Failed to connect to the UR5e at {self._robot_ip}. Check that "
                "the arm is powered on, the controller is in remote mode, the "
                "cameras are free (no other process holding /dev/video*), and "
                "that this host can reach the robot."
            )
        self._robot = robot
        logger.info("UR5e connected.")
        self._check_tcp_offset(robot)

    def _camera_params(self, requested: list[str]) -> dict:
        """Intrinsics + extrinsics per camera, for projection and reconstruction.

        Two sources, deliberately separate:

        * **Intrinsics** come from the RealSense SDK for SDK-driven cameras —
          they describe the *aligned colour* stream, which is exactly the frame
          the depth is registered to. A hand calibration in the JSON file
          overrides them (it may model distortion the factory profile zeroes).
        * **Extrinsics** can only come from the calibration file: the camera's
          pose in the robot base frame is something only a hand-eye
          calibration knows.

        A camera with an intrinsic but no extrinsic is omitted, because every
        consumer needs both — ``ChunkProjector`` skips it, and
        ``_iter_camera_views`` would drop it anyway.

        A camera whose entry says ``"parent": "tcp"`` (a wrist camera, calibrated
        by ``scripts/setup/calibrate_ur5_cameras.py --camera wrist``) stores its
        pose *in the TCP frame*; it is composed with the controller's current TCP
        pose here, on every call. ``ChunkProjector.fetch_params`` and the guided
        runner re-fetch at each replan for exactly this reason. Before the arm is
        connected there is no TCP pose, and the camera is left out.
        """
        params: dict[str, dict[str, np.ndarray]] = {}
        T_base_tcp: np.ndarray | None = None
        for name in requested:
            calib = self._calibration.get(name, {})
            K = calib.get("intrinsic")
            if K is None:
                cam = self._realsense.get(name)
                K = None if cam is None else cam.intrinsics
            E = calib.get("extrinsic")
            if K is None or E is None:
                continue
            if calib.get("parent") == "tcp":
                if T_base_tcp is None:
                    T_base_tcp = self._current_tcp_matrix()
                if T_base_tcp is None:
                    logger.debug(
                        f"camera {name!r} is parented to the TCP but the arm is not "
                        "connected yet; omitting it from get_camera_params"
                    )
                    continue
                E = T_base_tcp @ np.asarray(E, dtype=np.float64)
            params[name] = {"intrinsic": np.asarray(K), "extrinsic": np.asarray(E)}
        return params

    def _current_tcp_matrix(self) -> np.ndarray | None:
        """``T_base_tcp`` from the controller right now, or None without an arm."""
        robot = self._robot
        rec = getattr(robot, "rtde_rec", None) if robot is not None else None
        if rec is None:
            return None
        pose = np.asarray(rec.getActualTCPPose(), dtype=np.float64).reshape(6)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rotvec_to_matrix(pose[3:6])
        T[:3, 3] = pose[:3]
        return T

    def preflight_cameras(self) -> None:
        """Fail now if the cameras cannot all stream, not three minutes from now.

        Cameras are opened lazily, in ``_ensure_robot`` -- which runs on the
        first ``make_env``, and the evaluation only gets there once the policy
        server has finished loading and compiling. On this stack that is ~3
        minutes, so any camera problem used to present as a crash at the end of
        the wait. The probe opens them the way the real connect will: all at
        once, with the same streams, because a USB bandwidth conflict only
        appears when they run together.
        """
        from policy_abstraction.envs.ur5.realsense import (
            probe_realsense_cameras,
            realsense_serial,
            reset_realsense,
            video_node_holders,
        )

        if self._simulate_hardware:
            logger.info("--simulate-hardware: skipping the camera pre-flight.")
            return

        failures: list[str] = []
        requests = []
        for spec in self._camera_specs:
            device = spec["device"]
            if is_realsense_device(device):
                requests.append(
                    {
                        "name": spec["name"],
                        "serial": realsense_serial(device),
                        "width": spec["width"],
                        "height": spec["height"],
                        "fps": spec["fps"],
                        "depth": self._wants_depth(spec["name"]),
                    }
                )
            else:
                error = _probe_v4l2(device)
                if error:
                    failures.append(f"  {spec['name']} ({device}): {error}")

        rs_failures = probe_realsense_cameras(requests)

        # A camera left wedged by an abruptly killed run starts its pipeline and
        # then never delivers a frame. Power-cycling it is the documented
        # recovery, and it costs a few seconds -- worth one automatic attempt
        # before handing the operator a failure they would fix the same way.
        # Never reset a camera another process is legitimately using.
        recoverable = [
            name
            for name, error in rs_failures
            if "busy" not in error.lower() and "No device connected" not in error
        ]
        if recoverable:
            by_name = {r["name"]: r for r in requests}
            logger.warning(
                f"Cameras {recoverable} started but delivered no frames; "
                "power-cycling them once and retrying."
            )
            reset_realsense([by_name[n]["serial"] for n in recoverable])
            rs_failures = probe_realsense_cameras([by_name[n] for n in recoverable]) + [
                (n, e) for n, e in rs_failures if n not in recoverable
            ]

        for name, error in rs_failures:
            device = next(s["device"] for s in self._camera_specs if s["name"] == name)
            failures.append(f"  {name} ({device}): {error}")

        if not failures:
            streams = {
                s["name"]: (
                    "colour+depth" if self._wants_depth(s["name"]) else "colour"
                )
                for s in self._camera_specs
                if is_realsense_device(s["device"])
            }
            logger.info(f"Camera pre-flight OK: {streams}")
            return

        holders = video_node_holders()
        held_by = (
            "\nProcesses currently holding a /dev/video* node:\n  "
            + "\n  ".join(holders)
            if holders
            else "\nNo other process holds a /dev/video* node, and a power-cycle "
            "did not help. A camera that starts but never delivers a frame is "
            "usually USB bandwidth: two D435s cannot both stream depth through "
            "one controller. Depth is already limited to --primary-camera and "
            "calibrated cameras; move one camera to a different USB controller "
            "if you need more, or replug the one that failed."
        )
        raise RuntimeError(
            "Camera pre-flight failed:\n" + "\n".join(failures) + held_by
        )

    def _wants_depth(self, name: str) -> bool:
        """Whether this camera's depth is used by anything.

        Reconstruction needs depth *and* an extrinsic to place it in the robot
        frame, so an uncalibrated camera's depth has no consumer. The primary
        camera keeps depth regardless: it is the one a calibration is normally
        added for, and having it change stream configuration when a calibration
        file appears would be a surprising coupling.
        """
        if name == self._primary_camera:
            return True
        return "extrinsic" in self._calibration.get(name, {})

    def _connect_realsense(self) -> None:
        """Open an SDK stream for every ``rs:<serial>`` camera."""
        from policy_abstraction.envs.ur5.realsense import (
            RealSenseCamera,
            realsense_serial,
        )

        for spec in self._camera_specs:
            if not is_realsense_device(spec["device"]):
                continue
            cam = RealSenseCamera(
                realsense_serial(spec["device"]),
                width=spec["width"],
                height=spec["height"],
                fps=spec["fps"],
                depth=self._wants_depth(spec["name"]),
            )
            cam.connect()
            self._realsense[spec["name"]] = cam

    def _disconnect_realsense(self) -> None:
        for name, cam in list(self._realsense.items()):
            cam.disconnect()
            self._realsense.pop(name, None)

    def _check_calibration_tcp_offset(self) -> None:
        """Warn when a camera was calibrated against a different TCP offset.

        The calibration tool tracks the controller's TCP, so its result is only
        as good as the TCP definition it was made with. A wrist entry is
        expressed *in* that TCP frame, so a changed offset moves the camera.
        """
        configured = np.asarray(self._tcp_offset, dtype=np.float64)
        for name, cam in self._calibration.items():
            meta = cam.get("meta") if isinstance(cam, dict) else None
            recorded = None if not isinstance(meta, dict) else meta.get("tcp_offset")
            if recorded is None:
                continue
            recorded_arr = np.asarray(recorded, dtype=np.float64).reshape(-1)
            if recorded_arr.shape != (6,):
                continue
            if not np.allclose(recorded_arr, configured, atol=1e-4):
                logger.warning(
                    f"Camera {name!r} was calibrated with TCP offset "
                    f"{np.round(recorded_arr, 5).tolist()} but this server runs with "
                    f"--tcp-offset {np.round(configured, 5).tolist()}. "
                    + (
                        "The wrist extrinsic is expressed in that TCP frame, so it "
                        "is now wrong by the difference; recalibrate."
                        if cam.get("parent") == "tcp"
                        else "Recalibrate if the pendant's TCP was changed."
                    )
                )

    def _check_tcp_offset(self, robot: Any) -> None:
        """Warn when ``--tcp-offset`` disagrees with the controller's own TCP.

        The offset is published in the EnvSpec metadata and used by
        :class:`~policy_abstraction.dynamics.ur5_kinematics.Ur5KinematicsDynamicsModel`;
        a mismatch silently shifts every predicted end-effector pose, so it is
        worth surfacing loudly rather than debugging later from the visualiser.
        """
        read_offset = getattr(robot.rtde_ctrl, "getTCPOffset", None)
        if read_offset is None:
            return
        measured = np.asarray(read_offset(), dtype=np.float64)
        configured = np.asarray(self._tcp_offset, dtype=np.float64)
        if measured.shape != configured.shape:
            return
        if not np.allclose(measured, configured, atol=1e-4):
            logger.warning(
                "Configured --tcp-offset "
                f"{np.round(configured, 5).tolist()} differs from the "
                f"controller's TCP offset {np.round(measured, 5).tolist()}. "
                "Predicted end-effector poses will be offset; pass the "
                "controller value to --tcp-offset."
            )

    def _halt(self) -> None:
        """Stop any in-flight servo motion. Safe to call when disconnected."""
        self._next_tick = None
        robot = self._robot
        if robot is None or self._dry_run or robot.rtde_ctrl is None:
            return
        robot.rtde_ctrl.servoStop()

    def shutdown(self) -> None:
        """Stop motion, park the arm, and drop the RTDE session.

        Runs from the signal handler, so the ``finally`` guarantees the RTDE
        session is released even if parking fails — an unreleased session
        leaves the controller refusing the next connection.
        """
        robot = self._robot
        self._robot = None
        if robot is None:
            return
        try:
            self._halt()
            if not self._dry_run:
                robot.home()
        finally:
            self._disconnect_realsense()
            robot.disconnect()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def _home(self) -> None:
        """Move to the home joint configuration, skipping if already there.

        ``EpisodeRunner`` calls ``reset()`` and then ``set_init_state()``, both
        of which home; the tolerance check keeps that from costing two blocking
        ``moveJ`` round-trips per episode.
        """
        if self._dry_run:
            logger.info("[dry-run] skipping home()")
            return
        if self._at_home():
            logger.debug("Already at the home pose; skipping the homing move.")
            return
        logger.info("Homing the arm …")
        self._robot.home()

    def _at_home(self, tol: float = 1e-3) -> bool:
        current = np.asarray(self._robot.rtde_rec.getActualQ(), dtype=np.float64)
        home = np.asarray(UR5E_HOME_JOINT_POSITIONS, dtype=np.float64)
        return bool(np.allclose(current, home, atol=tol))

    def _reset_episode(self) -> dict:
        """Home the arm, optionally wait for the operator, and read obs."""
        self._halt()
        if self._home_on_reset:
            self._home()
        if self._stage_prompt:
            self._wait_for_operator()
        # Let the arm settle and the camera pipeline flush a fresh frame.
        time.sleep(0.5)
        self._next_tick = None
        self._last_step_end = None
        self._late_steps = 0
        self._stall_steps = 0
        self._total_steps = 0
        self._consecutive_clamps = 0
        obs = self._read_obs()
        self._last_gripper_openness = float(obs["gripper_qpos"][0])
        return obs

    def _wait_for_operator(self) -> None:
        """Block until the operator confirms the scene is staged.

        No-op without a TTY (already warned about at startup) so an automated
        run never deadlocks.
        """
        if not self._interactive:
            return
        print("\n>>> Stage the scene, then press ENTER to start the episode.")
        sys.stdin.readline()

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _read_obs(self) -> dict:
        """Read the driver's raw observation and flatten it to the EnvSpec keys."""
        import cv2

        raw = self._robot.get_observation()

        joint_pos = np.array(
            [float(raw[f"joint_{i}"]) for i in range(6)], dtype=np.float64
        )
        gripper_raw = float(raw["gripper"])
        # `lerobot_robot_ur5e` reports 0 = open, 1 = closed. Everything
        # downstream (and the pi0/pi05 UR5e checkpoints) speaks *openness*.
        aperture = 1.0 - gripper_raw if self._flip_gripper else gripper_raw

        obs: dict[str, Any] = {
            "joint_pos": joint_pos,
            "gripper_qpos": np.array([aperture], dtype=np.float64),
            "gripper_raw": gripper_raw,
        }

        rec = self._robot.rtde_rec
        if rec is not None:
            tcp = np.asarray(rec.getActualTCPPose(), dtype=np.float64)
            obs["eef_pos"] = tcp[:3].copy()
            obs["eef_axis_angle"] = tcp[3:6].copy()
            obs["eef_quat"] = rotvec_to_quat(tcp[3:6])
            obs["joint_vel"] = np.asarray(rec.getActualQd(), dtype=np.float64)

        out_h, out_w = self._out_hw or (UR5_CAMERA_HEIGHT, UR5_CAMERA_WIDTH)
        for name in self._camera_names:
            rs_cam = self._realsense.get(name)
            if rs_cam is not None:
                # One grab yields the colour/depth pair for the same instant;
                # `read_depth_m` must not re-poll or the mask and the depth
                # would come from different frames.
                frame = rs_cam.async_read()
                # Cameras perception cannot place in the world are opened
                # colour-only, to stay inside the USB controller's bandwidth.
                if rs_cam.depth:
                    depth_m = rs_cam.read_depth_m()
                    if depth_m.shape[0] != out_h or depth_m.shape[1] != out_w:
                        # Nearest-neighbour: averaging across a depth
                        # discontinuity invents surfaces that are not there.
                        depth_m = cv2.resize(
                            depth_m, (out_w, out_h), interpolation=cv2.INTER_NEAREST
                        )
                    obs[f"{name}_depth"] = np.ascontiguousarray(
                        depth_m.astype(np.float32)
                    )
            else:
                frame = np.asarray(raw[name])
            if frame.shape[0] != out_h or frame.shape[1] != out_w:
                # INTER_AREA is the right kernel for downscaling; the frames are
                # always >= the requested size in practice.
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            # No colour conversion: OpenCVCamera already emits RGB, and the
            # checkpoints were trained on exactly what the driver hands back.
            obs[name] = np.ascontiguousarray(frame.astype(np.uint8))

        return obs

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def _clamp_action(self, action) -> tuple[np.ndarray, float, dict]:
        """Validate + clamp a raw policy action into a safe joint/gripper target.

        Returns ``(joint_targets, gripper_openness, clamp_info)``.
        """
        a = np.asarray(action, dtype=np.float64).reshape(-1)
        if a.shape[0] < 7:
            raise ValueError(
                f"UR5 actions need 6 joint targets + 1 gripper channel; got "
                f"shape {a.shape}."
            )
        if not np.all(np.isfinite(a[:7])):
            raise ValueError(f"Non-finite action rejected: {a[:7]!r}")

        current = np.array(
            [float(v) for v in self._robot.rtde_rec.getActualQ()], dtype=np.float64
        )

        # `EpisodeRunner` emits an all-zeros "dummy action" during its
        # settle phase (``--env.num-steps-wait``). For a Cartesian-delta sim
        # that means "don't move"; read as absolute joint targets it would
        # instead command the fully-extended singular pose. Treat the exact
        # zero vector as a hold so the wait phase is safe on hardware — no
        # policy ever emits it, and the joint clamp below would only turn it
        # into a slow drift toward that pose anyway.
        if not np.any(a[:6]):
            if not self._hold_warned:
                self._hold_warned = True
                logger.warning(
                    "Received an all-zeros joint target (the episode runner's "
                    "dummy action). Holding the current pose. Set "
                    "--env.num-steps-wait 0 to skip the settle phase entirely."
                )
            return (
                current,
                self._last_gripper_openness,
                {
                    "clamped": False,
                    "max_clamp_rad": 0.0,
                    "hold": True,
                },
            )

        target = a[:6].copy()
        gripper = float(np.clip(a[6], 0.0, 1.0))
        raw_target = target.copy()

        # 1) Rate limit relative to the *measured* joints. This is what keeps a
        #    single bad action (or the all-zeros dummy action the episode
        #    runner emits during its wait phase) from throwing the arm.
        target = np.clip(
            target,
            current - self._max_joint_delta,
            current + self._max_joint_delta,
        )
        # 2) Absolute joint limits.
        lo = np.array([lim[0] for lim in UR5E_JOINT_LIMITS])
        hi = np.array([lim[1] for lim in UR5E_JOINT_LIMITS])
        target = np.clip(target, lo, hi)

        max_dev = float(np.max(np.abs(raw_target - target))) if target.size else 0.0
        info = {
            "clamped": bool(max_dev > 1e-9),
            "max_clamp_rad": max_dev,
            "hold": False,
        }
        self._last_gripper_openness = gripper
        return target, gripper, info

    def _step(self, action) -> tuple[dict, float, bool, dict]:
        arrived = time.perf_counter()
        # How long the caller spent between steps. A replan stall shows up here
        # (the evaluation blocks on policy inference), and it is a different
        # problem from this server being too slow to hold the control rate.
        caller_gap = (
            0.0 if self._last_step_end is None else arrived - self._last_step_end
        )

        target, gripper_openness, clamp_info = self._clamp_action(action)

        # Back to the driver's convention (0 = open) before commanding.
        gripper_cmd = 1.0 - gripper_openness if self._flip_gripper else gripper_openness
        action_map = {f"joint_{i}": float(target[i]) for i in range(6)}
        action_map["gripper"] = float(np.clip(gripper_cmd, 0.0, 1.0))

        if self._dry_run:
            if self._total_steps % int(self._control_hz) == 0:
                logger.info(f"[dry-run] would send {action_map}")
        else:
            self._robot.send_action(action_map)

        # Pace to the control rate so the arm tracks the trajectory at the
        # frequency the policy was trained at.
        late = self._pace(caller_gap)
        self._total_steps += 1

        obs = self._read_obs()
        done, reward = self._poll_verdict()
        self._last_step_end = time.perf_counter()
        info = {
            "control_late": late,
            "caller_gap_s": caller_gap,
            "late_step_fraction": self._late_steps / self._total_steps,
            "stall_step_fraction": self._stall_steps / self._total_steps,
            **clamp_info,
        }
        self._log_clamping(clamp_info)
        return obs, reward, done, info

    def _log_clamping(self, clamp_info: dict) -> None:
        """Warn on clamping, escalating when it is clearly systematic.

        One clamped step is a spiky action. Clamping *every* step means the
        policy is commanding a pose the arm cannot reach at this rate — usually
        because the arm started far from where the demonstrations did, so it is
        creeping toward the first chunk at the rate limit.
        """
        if not clamp_info["clamped"]:
            self._consecutive_clamps = 0
            return
        self._consecutive_clamps += 1
        if self._consecutive_clamps == 1:
            logger.warning(
                f"Clamped joint target by up to {clamp_info['max_clamp_rad']:.4f} rad "
                f"(--max-joint-delta {self._max_joint_delta} / joint limits)."
            )
        elif self._consecutive_clamps == int(self._control_hz):
            logger.warning(
                f"Joint targets have been clamped for {self._consecutive_clamps} "
                "consecutive steps. The arm is rate-limited toward a pose far "
                "from where it is now — check that it starts near the pose the "
                "demonstrations began from."
            )

    def _pace(self, caller_gap: float) -> bool:
        """Sleep until the next control tick. Returns True if we ran late.

        ``caller_gap`` separates the two ways the loop can miss its deadline:
        the caller stalling between steps (policy inference) versus this server
        taking too long to service one step. Only the latter is something to
        fix here, so they are counted separately.
        """
        if self._next_tick is None:
            self._next_tick = time.perf_counter()
        self._next_tick += self._control_period
        sleep_s = self._next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
            return False

        # Fell behind — resync instead of accumulating debt.
        self._next_tick = time.perf_counter()
        overrun_ms = -sleep_s * 1000.0
        if caller_gap > self._control_period:
            self._stall_steps += 1
            if self._stall_steps in (1, 10) or self._stall_steps % 100 == 0:
                logger.info(
                    f"Arm stalled {caller_gap * 1000:.0f} ms waiting for the "
                    f"caller ({self._stall_steps} stalls so far). Expected at "
                    "replan boundaries; raise --policy.replan-steps if it is "
                    "happening too often."
                )
            return True

        self._late_steps += 1
        if self._late_steps in (1, 50) or self._late_steps % 500 == 0:
            logger.warning(
                f"Control loop is behind {self._control_hz:.0f} Hz "
                f"({self._late_steps}/{self._total_steps + 1} steps overran by "
                f"{overrun_ms:.1f} ms). Disable the visualizer / episode "
                "recording, or lower --control-hz."
            )
        return True

    def _poll_verdict(self) -> tuple[bool, float]:
        """Check whether the operator has ended the episode. ``(done, reward)``."""
        if self._success_mode != "stdin" or not self._interactive:
            return False, 0.0
        verdict = _poll_stdin_verdict()
        if verdict is None:
            return False, 0.0
        logger.info(
            f"Operator ended the episode: {'SUCCESS' if verdict else 'FAILURE'}"
        )
        return True, (1.0 if verdict else 0.0)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @staticmethod
    def _base_frame_correction(base_frame: str) -> np.ndarray:
        """4x4 that maps a pose expressed in *base_frame* into RTDE's frame.

        Universal Robots ships two base frames 180 degrees apart about z:

        * ``base``      — the controller's own frame. ``getActualTCPPose()`` and
          every other RTDE reading is in this one, so it is what this adapter
          publishes and what reconstruction must agree with.
        * ``base_link`` — the ROS/URDF convention, x pointing the opposite way.
          Calibration tools that build on a URDF (Kalib among them) commonly
          produce extrinsics in this frame.

        Mixing them mirrors the scene in x and y while leaving the table
        perfectly horizontal, so it survives a plane-fit sanity check and only
        shows up when you compare against the robot's own reported TCP.
        """
        if base_frame == "base":
            return np.eye(4)
        if base_frame != "base_link":
            raise ValueError(
                f"base_frame must be 'base' or 'base_link', got {base_frame!r}"
            )
        T = np.eye(4)
        T[:3, :3] = [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]
        return T

    @staticmethod
    def _kalib_camera_to_base(raw: dict) -> np.ndarray:
        """Camera-to-base 4×4 from a Kalib result, with the direction verified.

        Kalib's headline key ``T_cam_base`` is the **opposite** of what its name
        suggests — the file's own convention string says
        ``X_cam = T_cam_base @ X_base``, i.e. it maps base coordinates *into* the
        camera. The camera-to-base pose is therefore ``T_base_cam``. Getting this
        backwards silently relocates every reconstructed object (measured: the
        table plane tilts 63° instead of 3°), so the convention is asserted
        rather than assumed.
        """
        convention = str(raw.get("convention", ""))
        if "X_cam = T_cam_base" not in convention:
            raise ValueError(
                f"Unrecognised Kalib convention {convention!r}. Refusing to guess "
                "which way the transform goes — a mirrored extrinsic silently "
                "misplaces every reconstructed object. Convert it explicitly with "
                "scripts/setup/ur5_calibration_from_kalib.py."
            )
        if "T_base_cam" in raw:
            E = np.asarray(raw["T_base_cam"], dtype=np.float64).reshape(4, 4)
        else:
            E = np.linalg.inv(
                np.asarray(raw["T_cam_base"], dtype=np.float64).reshape(4, 4)
            )
        R = E[:3, :3]
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6) or not np.isclose(
            np.linalg.det(R), 1.0, atol=1e-6
        ):
            raise ValueError(
                f"Kalib extrinsic is not a rigid transform (det={np.linalg.det(R):.6f})."
            )
        return E

    @staticmethod
    def _resolve_base_frame(raw: dict, override: str | None, path: str) -> str:
        """Which base frame the calibration file's extrinsic is expressed in.

        Declared by the file itself (``"base_frame"``), overridable from the
        command line. There is deliberately **no default**: the two frames sit
        180 degrees apart about z, so guessing wrong mirrors the scene in x and
        y while leaving the table perfectly horizontal. That passes a plane-fit
        sanity check, records cleanly, and only shows up much later as guidance
        distances measured in a frame the robot does not use. Refusing to guess
        costs one key in the file.
        """
        declared = raw.get("base_frame")
        frame = override or declared
        if frame is None:
            raise ValueError(
                f"{path} does not declare which base frame its extrinsic is in, "
                "and --calibration-base-frame was not given. Add "
                '\'"base_frame": "base"\' or \'"base_frame": "base_link"\' to the '
                "file. UR ships two base frames 180 degrees apart about z: "
                "`base` is what RTDE reports poses in, `base_link` is the "
                "ROS/URDF convention that Kalib and other URDF-based hand-eye "
                "tools produce. Mixing them mirrors every reconstructed object."
            )
        if declared and override and declared != override:
            logger.warning(
                f"{path} declares base_frame={declared!r} but "
                f"--calibration-base-frame says {override!r}; using the flag."
            )
        return str(frame)

    @staticmethod
    def _load_calibration(
        path: str | None, primary_camera: str = "", base_frame: str | None = None
    ) -> dict:
        """Load camera calibration, accepting either supported layout.

        **Env layout** — one entry per camera::

            {"exterior_image_1_left": {"extrinsic": 4x4, "intrinsic": 3x3}}

        **Kalib layout** — a hand-eye result as Kalib writes it, carrying both
        transform directions plus a ``convention`` string. It describes a single
        camera and names none, so it is attached to *primary_camera*. Accepting
        it directly removes the manual conversion step between calibrating and
        running; ``scripts/setup/ur5_calibration_from_kalib.py`` remains for
        multi-camera cells and for inspecting the result.

        The extrinsic follows the same convention the simulator servers
        publish: a *camera-to-world* 4×4 pose, where "world" is the UR5e base
        frame. Returns ``{}`` when no path is given, which makes
        ``ChunkProjector`` skip projection instead of failing.

        ``intrinsic`` is optional for RealSense cameras — the SDK reports the
        aligned-colour intrinsics at the streaming resolution, which is what the
        published depth is registered to. Kalib's intrinsics are deliberately
        *not* adopted: its calibration resolution (1280×720 by default) need not
        match the stream, and a mismatched K silently skews reconstruction.
        """
        if not path:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        resolved_frame = Ur5EnvServer._resolve_base_frame(raw, base_frame, str(path))
        # Not a camera entry -- strip it before the per-camera loop below.
        raw = {k: v for k, v in raw.items() if k != "base_frame"}

        if "convention" in raw or "T_base_cam" in raw or "T_cam_base" in raw:
            if not primary_camera:
                raise ValueError(
                    "A Kalib calibration names no camera; pass the camera to "
                    "attach it to."
                )
            E = Ur5EnvServer._base_frame_correction(
                resolved_frame
            ) @ Ur5EnvServer._kalib_camera_to_base(raw)
            meta = raw.get("meta", {})
            detail = (
                f", calibrated at {meta['width']}x{meta['height']}, reproj err "
                f"{meta.get('avg_reprojection_err_px', float('nan')):.2f} px"
                if meta.get("width")
                else ""
            )
            logger.info(
                f"Loaded Kalib calibration for {primary_camera!r} from {path} "
                f"(camera origin in base frame {np.round(E[:3, 3], 4).tolist()} m"
                f"{detail}; base_frame={resolved_frame}). Intrinsics come from "
                "the device."
            )
            return {primary_camera: {"extrinsic": E}}

        out: dict[str, dict[str, Any]] = {}
        for name, entry in raw.items():
            if not isinstance(entry, dict) or "extrinsic" not in entry:
                raise ValueError(
                    f"calibration entry {name!r} has no 'extrinsic'. The camera's "
                    "pose in the robot base frame cannot be inferred; only the "
                    "intrinsics can come from the device."
                )
            E = np.asarray(entry["extrinsic"], dtype=np.float64).reshape(4, 4)
            parent = str(entry.get("parent", "base"))
            cam: dict[str, Any]
            if parent == "tcp":
                # A wrist camera: its pose in the TCP frame. The base-frame
                # question does not arise; it is composed with the live TCP pose
                # at every get_camera_params.
                cam = {"extrinsic": E, "parent": "tcp"}
            elif parent == "base":
                cam = {
                    "extrinsic": Ur5EnvServer._base_frame_correction(resolved_frame) @ E
                }
            else:
                raise ValueError(
                    f"calibration entry {name!r} has parent={parent!r}; only 'base' "
                    "(a fixed camera) and 'tcp' (a wrist camera) are supported."
                )
            if entry.get("intrinsic") is not None:
                cam["intrinsic"] = np.asarray(
                    entry["intrinsic"], dtype=np.float64
                ).reshape(3, 3)
            if isinstance(entry.get("meta"), dict):
                cam["meta"] = entry["meta"]
            out[name] = cam
        logger.info(
            f"Loaded camera calibration for {sorted(out)} from {path} "
            f"(base_frame={resolved_frame})"
        )
        return out


__all__ = ["Ur5EnvServer", "parse_camera_arg", "rotvec_to_quat"]
