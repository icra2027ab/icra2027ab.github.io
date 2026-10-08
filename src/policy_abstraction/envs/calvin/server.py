# ruff: noqa: E402
"""CALVIN WebSocket environment server.

Run from the calvin venv (``third_party/calvin/.venv``) via
``scripts/servers/serve_calvin_env.py``. Implements the same wire protocol
as the LIBERO and RoboCasa servers so the existing
:class:`policy_abstraction.envs.WebsocketEnvClient` connects unchanged.

Suite model: each ``task_id`` maps to one CALVIN scene config (e.g.
``calvin_scene_D``). ``get_task_init_states`` pre-samples N initial states by
calling ``env.reset()`` with consecutive seeds and capturing
``{"robot_obs": …, "scene_obs": …}`` dicts compatible with CALVIN's
``env.reset(robot_obs=…, scene_obs=…)`` signature.

Observation flattening: CALVIN's native obs is nested
(``rgb_obs``/``depth_obs`` dicts + flat ``robot_obs`` ndarray). The server
emits a flat dict whose keys match the EnvSpec published by
:func:`policy_abstraction.envs.calvin.config.calvin_env_spec`.

CALVIN ships with Python 3.8; defer annotation evaluation so ``list[str]`` /
``X | Y`` parse as strings rather than executing at class-creation time.
"""

from __future__ import annotations

import os
import sys

# Make policy_abstraction importable when running from the calvin venv.
# server.py is at src/policy_abstraction/envs/calvin/server.py; dirname×4 = src/.
_SRC_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import logging
from typing import Any

import numpy as np

from policy_abstraction.envs.base_server import BaseEnvServer, pack_result
from policy_abstraction.envs.calvin.config import (
    DEFAULT_MAX_STEPS,
    calvin_env_spec,
)

logger = logging.getLogger(__name__)


class CalvinEnvServer(BaseEnvServer):
    """Serves a CALVIN PlayTable env over a WebSocket connection."""

    server_metadata = {"server": "CalvinEnvServer", "version": "1.0"}

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8767,
        scenes: list[str] | None = None,
        suite_name: str = "calvin",
        cameras_cfg: str = "static_and_gripper",
        robot_cfg: str = "panda_longer_finger",
        seed: int = 0,
        num_init_states: int = 10,
        max_steps: int = DEFAULT_MAX_STEPS,
        primary_camera: str = "static",
        include_gripper: bool = True,
        show_gui: bool = False,
        use_egl: bool = True,
        enable_segmentation: bool = True,
    ) -> None:
        super().__init__(host=host, port=port)
        if not scenes:
            raise ValueError("CalvinEnvServer requires a non-empty scenes list.")
        self._scenes: list[str] = list(scenes)
        self._suite_name = suite_name
        self._cameras_cfg = cameras_cfg
        self._robot_cfg = robot_cfg
        self._seed = seed
        self._num_init_states = int(num_init_states)
        self._max_steps = int(max_steps)
        self._primary_camera = primary_camera
        self._include_gripper = include_gripper
        self._show_gui = show_gui
        self._use_egl = use_egl
        self._enable_segmentation = enable_segmentation

        # Cached lazily.
        self._task_languages: list[str] | None = None
        self._init_state_cache: dict[int, list[dict]] = {}

    # ------------------------------------------------------------------
    # Per-connection state
    # ------------------------------------------------------------------

    def _initial_state(self) -> dict:
        return {"env": None, "task_id": None}

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, cmd: str, args: list, kwargs: dict, state: dict) -> tuple:
        env = state["env"]

        # ------------------------------------------------------------------ #
        # Suite-level commands                                                 #
        # ------------------------------------------------------------------ #

        if cmd == "setup_suite":
            languages = (
                self._scenes if self._task_languages is None else (self._task_languages)
            )
            return {
                "num_tasks": len(self._scenes),
                "task_languages": list(languages),
                "task_suite_name": self._suite_name,
                "max_steps": self._max_steps,
                "evaluation_cfg": {},
                "env_spec": calvin_env_spec(
                    primary_camera=self._primary_camera,
                    include_gripper=self._include_gripper,
                    enable_segmentation=self._enable_segmentation,
                ).to_dict(),
            }, state

        if cmd == "get_task_init_states":
            task_id = int(kwargs["task_id"])
            return pack_result(self._get_or_build_init_states(task_id)), state

        # ------------------------------------------------------------------ #
        # Environment lifecycle commands                                       #
        # ------------------------------------------------------------------ #

        if cmd == "make_env":
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            task_id = int(kwargs["task_id"])
            camera_heights = int(kwargs.get("camera_heights", 200))
            camera_widths = int(kwargs.get("camera_widths", 200))
            seed = int(kwargs.get("seed", self._seed))
            env = self._make_env(task_id, camera_heights, camera_widths, seed)
            state["env"] = env
            state["task_id"] = task_id
            return {
                "language_instruction": self._scenes[task_id],
                "segmentation_name_mapping": self._segmentation_name_mapping(env),
            }, state

        if cmd == "get_segmentation_name_mapping":
            if env is None:
                raise RuntimeError("No environment is active. Call make_env first.")
            return self._segmentation_name_mapping(env), state

        if cmd == "close":
            if env is not None:
                env.close()
            state["env"] = None
            return None, state

        if cmd == "get_robot_base":
            if env is None:
                raise RuntimeError("No environment is active. Call make_env first.")
            base_pos, base_ori = self._robot_base(env)
            return pack_result(
                {
                    "base_pos": np.asarray(base_pos, dtype=np.float64),
                    "base_ori": np.asarray(base_ori, dtype=np.float64),
                }
            ), state

        if cmd == "get_camera_params":
            if env is None:
                raise RuntimeError("No environment is active. Call make_env first.")
            camera_names = list(kwargs.get("camera_names", ["static", "gripper"]))
            camera_names = [
                n.decode() if isinstance(n, bytes) else n for n in camera_names
            ]
            params: dict = {}
            for cam in env.cameras:
                if cam.name not in camera_names:
                    continue
                K, E = self._camera_intr_extr(cam)
                params[cam.name] = {
                    "intrinsic": K,
                    "extrinsic": E,
                }
            return pack_result(params), state

        # ------------------------------------------------------------------ #
        # Lifecycle methods that need adapter logic                            #
        # ------------------------------------------------------------------ #

        if env is None:
            raise RuntimeError(
                f"No environment is active. Call make_env first (cmd={cmd!r})."
            )

        if cmd == "reset":
            obs = env.reset()
            return pack_result(self._flatten_obs(env, obs)), state

        if cmd == "set_init_state":
            init_state = args[0] if args else kwargs.get("state")
            if init_state is None:
                raise ValueError("set_init_state requires a state arg.")
            obs = self._reset_to(env, init_state)
            return pack_result(self._flatten_obs(env, obs)), state

        if cmd == "step":
            action = args[0] if args else kwargs.get("action")
            if action is None:
                raise ValueError("step requires an action arg.")
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            # CALVIN's robot.apply_action asserts `gripper_action in (-1, 1)`.
            # Continuous gripper outputs from the policy/dummy must be snapped
            # to the discrete open/close convention CALVIN expects.
            if action.shape[0] >= 7:
                action[6] = 1.0 if action[6] >= 0.0 else -1.0
            obs, reward, done, info = env.step(action)
            flat = self._flatten_obs(env, obs)
            return pack_result((flat, reward, done, info)), state

        if cmd == "seed":
            seed = args[0] if args else kwargs.get("seed", 0)
            return pack_result(env.seed(int(seed))), state

        # Generic fallback for any other env method (get_info, get_obs, …).
        method = getattr(env, cmd)
        result = method(*args, **kwargs)
        return pack_result(result), state

    # ------------------------------------------------------------------
    # Suite helpers
    # ------------------------------------------------------------------

    def _get_or_build_init_states(self, task_id: int) -> list[dict]:
        if task_id in self._init_state_cache:
            return self._init_state_cache[task_id]

        env = self._make_env(task_id, 84, 84, self._seed)
        states: list[dict] = []
        try:
            for i in range(self._num_init_states):
                env.seed(self._seed + i)
                obs = env.reset()
                # CALVIN's reset(robot_obs=…, scene_obs=…) accepts these flat
                # arrays directly, so we just round-trip them.
                states.append(
                    {
                        "robot_obs": np.asarray(obs["robot_obs"], dtype=np.float64),
                        "scene_obs": np.asarray(obs["scene_obs"], dtype=np.float64),
                    }
                )
        finally:
            env.close()

        self._init_state_cache[task_id] = states
        return states

    # ------------------------------------------------------------------
    # Env factory + reset_to + obs flattening
    # ------------------------------------------------------------------

    def _make_env(
        self,
        task_id: int,
        camera_heights: int,
        camera_widths: int,
        seed: int,
    ):
        import calvin_env
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate

        scene_name = self._scenes[task_id]

        if not GlobalHydra.instance().is_initialized():
            # calvin's conf lives at <repo>/calvin_env/conf, sibling to the
            # ``calvin_env`` python package — not inside it — so hydra can't
            # discover it via ``initialize_config_module``.
            calvin_env_pkg = os.path.dirname(os.path.abspath(calvin_env.__file__))
            conf_dir = os.path.normpath(os.path.join(calvin_env_pkg, "..", "conf"))
            try:
                initialize_config_dir(config_dir=conf_dir, version_base=None)
            except TypeError:
                # hydra < 1.2 doesn't accept ``version_base``.
                initialize_config_dir(config_dir=conf_dir)

        overrides = [
            f"scene={scene_name}",
            f"cameras={self._cameras_cfg}",
            f"robot={self._robot_cfg}",
            f"seed={seed}",
        ]
        cfg = compose(config_name="config_data_collection", overrides=overrides)
        # Override per-camera resolution dynamically.
        if cfg.get("cameras"):
            for cam_cfg in cfg.cameras.values():
                cam_cfg.width = camera_widths
                cam_cfg.height = camera_heights

        env = instantiate(
            cfg.env,
            show_gui=self._show_gui,
            use_vr=False,
            use_scene_info=True,
            use_egl=self._use_egl,
        )
        env.seed(seed)
        logger.info(
            f"Created calvin env scene={scene_name!r} "
            f"(res={camera_heights}x{camera_widths}, seed={seed})"
        )
        return env

    @staticmethod
    def _reset_to(env, init_state):
        if isinstance(init_state, np.ndarray):
            return env.reset(robot_obs=init_state)
        if not isinstance(init_state, dict):
            raise TypeError(
                f"set_init_state expects dict or ndarray, got {type(init_state)}"
            )
        return env.reset(
            robot_obs=init_state.get("robot_obs"),
            scene_obs=init_state.get("scene_obs"),
        )

    def _flatten_obs(self, env, obs: dict) -> dict:
        """CALVIN's nested obs → flat key dict (matches calvin_env_spec)."""
        import pybullet as p

        flat: dict[str, Any] = {}
        rgb_obs = obs.get("rgb_obs", {}) or {}
        depth_obs = obs.get("depth_obs", {}) or {}

        # Cameras: rgb_<name>/depth_<name> → <name>_image / <name>_depth.
        # (str.removeprefix is 3.9+; calvin's venv is Python 3.8.)
        for key, val in rgb_obs.items():
            name = key[len("rgb_") :] if key.startswith("rgb_") else key
            flat[f"{name}_image"] = np.asarray(val)
        for key, val in depth_obs.items():
            name = key[len("depth_") :] if key.startswith("depth_") else key
            flat[f"{name}_depth"] = np.asarray(val)

        if self._enable_segmentation:
            for cam in env.cameras:
                flat[f"{cam.name}_segmentation"] = self._render_segmentation(
                    cam, env.cid
                )

        robot_obs = np.asarray(obs.get("robot_obs", []))
        if robot_obs.size >= 7:
            tcp_pos = robot_obs[:3]
            # CALVIN's robot has euler_obs=True by default → orn is 3D euler.
            # When euler_obs=False the orn is a 4-D quaternion. Detect by length.
            if robot_obs.size == 16:
                tcp_orn_quat = robot_obs[3:7]
                gripper_width = robot_obs[7]
                arm_joints = robot_obs[8:15]
            else:
                tcp_orn_euler = robot_obs[3:6]
                tcp_orn_quat = np.asarray(p.getQuaternionFromEuler(tcp_orn_euler))
                gripper_width = robot_obs[6]
                arm_joints = robot_obs[7:14]
            flat["tcp_pos"] = np.asarray(tcp_pos, dtype=np.float64)
            flat["tcp_orn_quat"] = np.asarray(tcp_orn_quat, dtype=np.float64)
            flat["gripper_qpos"] = np.array([float(gripper_width)], dtype=np.float64)
            flat["arm_joint_states"] = np.asarray(arm_joints, dtype=np.float64)

        scene_obs = obs.get("scene_obs")
        if scene_obs is not None:
            flat["scene_obs"] = np.asarray(scene_obs)

        # Preserve raw nested obs for any caller that wants it.
        flat["robot_obs"] = robot_obs
        return flat

    # ------------------------------------------------------------------
    # Segmentation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_segmentation(cam, cid) -> np.ndarray:
        """Render an instance-segmentation mask for ``cam`` at its current pose.

        Calvin's ``Camera.process_rgbd`` discards the seg buffer that
        ``getCameraImage`` returns, so we re-call it here with the
        ``ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX`` flag and decode the
        body uid. Background pixels (PyBullet returns -1) become 0; object
        pixels carry ``uid + 1`` so the value can be looked up directly in
        the ``segmentation_name_mapping`` published by the server.

        ``StaticCamera`` exposes ``viewMatrix`` / ``projectionMatrix`` set
        once in ``__init__``; ``GripperCamera`` recomputes them every step
        and exposes them as ``view_matrix`` / ``projection_matrix`` after
        its first ``render()`` call (which the env always issues before
        observations are read).
        """
        import pybullet as p

        view_m = getattr(cam, "viewMatrix", None) or getattr(cam, "view_matrix", None)
        proj_m = getattr(cam, "projectionMatrix", None) or getattr(
            cam, "projection_matrix", None
        )
        if view_m is None or proj_m is None:
            raise RuntimeError(
                f"Camera {getattr(cam, 'name', cam)!r} has no view/projection "
                "matrix set yet — was render() called before reading obs?"
            )

        img = p.getCameraImage(
            width=cam.width,
            height=cam.height,
            viewMatrix=view_m,
            projectionMatrix=proj_m,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=cid,
        )
        seg_buf = np.reshape(img[4], (cam.height, cam.width)).astype(np.int32)
        # Low 24 bits hold the body uid; -1 indicates background.
        obj_uid = np.where(seg_buf < 0, -1, seg_buf & ((1 << 24) - 1))
        return (obj_uid + 1).astype(np.int32)

    @staticmethod
    def _segmentation_name_mapping(env) -> dict[str, str]:
        """Build a ``{str(uid + 1): name}`` registry for the live scene.

        Mirrors the LIBERO/RoboCasa contract: pixel value 0 is background,
        and every other key in this dict appears as a labelled instance in
        the rendered seg masks. Named scene objects + the robot take priority;
        any remaining loaded bodies (plane, etc.) fall back to the URDF base
        name reported by ``pybullet.getBodyInfo``.
        """
        import pybullet as p

        mapping: dict[str, str] = {}
        scene = getattr(env, "scene", None)
        if scene is not None and hasattr(scene, "get_objects"):
            for obj in scene.get_objects():
                uid = getattr(obj, "uid", None)
                name = getattr(obj, "name", None)
                if uid is None or name is None:
                    continue
                mapping[str(int(uid) + 1)] = str(name)
        robot = getattr(env, "robot", None)
        robot_uid = getattr(robot, "robot_uid", None)
        if robot_uid is not None:
            mapping[str(int(robot_uid) + 1)] = "robot"

        cid = getattr(env, "cid", 0)
        for i in range(p.getNumBodies(physicsClientId=cid)):
            uid = p.getBodyUniqueId(i, physicsClientId=cid)
            key = str(int(uid) + 1)
            if key in mapping:
                continue
            try:
                base_name, body_name = p.getBodyInfo(uid, physicsClientId=cid)
            except Exception:
                continue
            label = (
                body_name.decode()
                if isinstance(body_name, bytes)
                else (body_name or "")
            ) or (
                base_name.decode()
                if isinstance(base_name, bytes)
                else (base_name or "")
            )
            mapping[key] = label or f"body_{uid}"
        return mapping

    # ------------------------------------------------------------------
    # Robot base + camera intrinsics/extrinsics
    # ------------------------------------------------------------------

    @staticmethod
    def _robot_base(env):
        import pybullet as p

        pos, orn = p.getBasePositionAndOrientation(
            env.robot.robot_uid, physicsClientId=env.cid
        )
        rot_flat = p.getMatrixFromQuaternion(orn)
        rot = np.asarray(rot_flat, dtype=np.float64).reshape(3, 3)
        return np.asarray(pos, dtype=np.float64), rot

    @staticmethod
    def _camera_intr_extr(cam) -> tuple[np.ndarray, np.ndarray]:
        """Pinhole intrinsic + 4×4 world-to-camera extrinsic from a CALVIN cam.

        PyBullet's view matrix is column-major OpenGL (Y-up, -Z forward); the
        published extrinsic flips Y/Z so it matches the OpenCV convention used
        elsewhere in policy_abstraction.
        """
        fov_rad = float(cam.fov) * np.pi / 180.0
        f = cam.height / (2.0 * np.tan(fov_rad / 2.0))
        K = np.array(
            [[f, 0, cam.width / 2], [0, f, cam.height / 2], [0, 0, 1]],
            dtype=np.float64,
        )

        # StaticCamera exposes `viewMatrix` set in __init__; GripperCamera
        # exposes `view_matrix` set inside render() (it tracks the EE link, so
        # it must be re-rendered each frame to stay current).
        view_seq = getattr(cam, "viewMatrix", None)
        if view_seq is None:
            cam.render()
            view_seq = cam.view_matrix
        view = np.asarray(view_seq, dtype=np.float64).reshape(4, 4).T
        # OpenGL → OpenCV (flip Y and Z of the camera frame).
        gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
        E = gl_to_cv @ view
        return K, E


__all__ = ["CalvinEnvServer"]
