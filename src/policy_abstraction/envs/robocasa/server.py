# ruff: noqa: E402
"""RoboCasa WebSocket environment server.

Run from the robocasa venv (``third_party/robocasa/.venv``) via
``scripts/servers/serve_robocasa_env.py``. Implements the same wire protocol
as :class:`~policy_abstraction.envs.libero.server.LiberoEnvServer` so the
existing :class:`policy_abstraction.envs.WebsocketEnvClient` can talk to it
without changes.

A "suite" here is just a list of registered RoboCasa env-class names; one
``task_id`` maps 1-to-1 to one env class. ``get_task_init_states`` pre-samples
N initial states by calling ``env.reset()`` with consecutive seeds and
capturing ``{states, model, ep_meta}`` dicts compatible with robosuite's
``reset_to``-style restoration.

PandaOmron's native action is 12-D
(eef_pos[3] + eef_rot[3] + gripper[1] + base_motion[4] + control_mode[1]); the
``EnvSpec`` published in ``setup_suite`` advertises a 7-D arm-only action and
this server zero-pads to 12-D before calling ``env.step``.
"""

import os
import sys

# Make policy_abstraction importable when running from the robocasa venv.
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import json
import logging

import numpy as np

from policy_abstraction.envs.base_server import BaseEnvServer, pack_result
from policy_abstraction.envs.robocasa.config import (
    DEFAULT_MAX_STEPS,
    robocasa_env_spec,
)
from policy_abstraction.envs.robosuite_geom_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_gt_object_pointclouds,
)

logger = logging.getLogger(__name__)


# Leaderboard task-suite keys exposed via TASK_SET_REGISTRY. Kept here as a
# single source of truth so the launcher / server-entry CLI can show valid
# choices without importing robocasa (which is only available in the robocasa
# venv).
LEADERBOARD_TASK_SUITES: tuple[str, ...] = (
    "atomic_seen",
    "composite_seen",
    "composite_unseen",
)


def _resolve_task_suite(task_suite: str) -> tuple[list[str], list[int]]:
    """Resolve a leaderboard suite name → ``(task_names, per_task_max_steps)``.

    Imports robocasa lazily because this module is also imported by the
    server-launcher (root venv), where robocasa isn't installed. Only the
    server subprocess actually runs ``__init__`` and thus this resolver.
    """
    from robocasa.utils.dataset_registry import (  # noqa: PLC0415 — lazy by design
        ATOMIC_TASK_DATASETS,
        COMPOSITE_TASK_DATASETS,
        TASK_SET_REGISTRY,
    )

    if task_suite not in TASK_SET_REGISTRY:
        raise ValueError(
            f"Unknown task_suite={task_suite!r}. "
            f"Valid keys: {sorted(TASK_SET_REGISTRY.keys())}"
        )
    names = list(TASK_SET_REGISTRY[task_suite])

    # Merge atomic + composite catalogues — every leaderboard task lives in
    # exactly one of these.
    catalogue: dict = {**ATOMIC_TASK_DATASETS, **COMPOSITE_TASK_DATASETS}
    horizons: list[int] = []
    missing: list[str] = []
    for name in names:
        cfg = catalogue.get(name)
        if cfg is None or "horizon" not in cfg:
            missing.append(name)
            continue
        horizons.append(int(cfg["horizon"]))
    if missing:
        raise ValueError(
            f"task_suite={task_suite!r}: missing horizon for tasks {missing}; "
            "robocasa's ATOMIC_TASK_DATASETS / COMPOSITE_TASK_DATASETS may have "
            "drifted from TASK_SET_REGISTRY."
        )
    return names, horizons


class RobocasaEnvServer(BaseEnvServer):
    """Serves a RoboCasa kitchen env over a WebSocket connection."""

    server_metadata = {"server": "RobocasaEnvServer", "version": "1.0"}

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        task_names: list[str] | None = None,
        suite_name: str = "robocasa_atomic",
        robots: str = "PandaOmron",
        split: str | None = None,
        seed: int = 0,
        num_init_states: int = 20,
        max_steps: int | None = None,
        primary_camera: str = "robot0_agentview_left",
        include_right_view: bool = True,
        task_suite: str | None = None,
    ) -> None:
        super().__init__(host=host, port=port)
        # Exactly one of ``task_names`` / ``task_suite`` must be set. The
        # former is the explicit ad-hoc path; the latter resolves a canonical
        # leaderboard suite via robocasa's TASK_SET_REGISTRY.
        if bool(task_names) == bool(task_suite):
            raise ValueError(
                "RobocasaEnvServer requires exactly one of "
                "task_names=[...] or task_suite=<atomic_seen|composite_seen|composite_unseen>; "
                f"got task_names={task_names!r}, task_suite={task_suite!r}"
            )

        # Per-task horizons; populated alongside task_names. When the user did
        # not pass --max-steps, this list carries the leaderboard's per-task
        # horizon. When --max-steps WAS passed, this is filled with that
        # uniform value (override semantics — see plan).
        self._task_max_steps: list[int]

        if task_suite is not None:
            (
                self._task_names,
                self._task_max_steps,
            ) = _resolve_task_suite(task_suite)
            # Leaderboard always evaluates on obj_instance_split="target".
            if split is None:
                logger.info(
                    f"[RobocasaEnvServer] task_suite={task_suite!r} → "
                    "defaulting split to 'target' to match the leaderboard."
                )
                split = "target"
            # Informative metadata label when user didn't override.
            if suite_name == "robocasa_atomic":  # the dataclass default
                suite_name = f"robocasa_{task_suite}"
            if max_steps is not None:
                # User-provided uniform cap wins (see plan).
                self._task_max_steps = [int(max_steps)] * len(self._task_names)
            logger.info(
                f"[RobocasaEnvServer] task_suite={task_suite!r}: "
                f"{len(self._task_names)} tasks, "
                f"max_steps min={min(self._task_max_steps)} "
                f"max={max(self._task_max_steps)}"
            )
        else:
            assert task_names is not None  # narrowed by the bool() guard above
            self._task_names = list(task_names)
            uniform = int(max_steps) if max_steps is not None else DEFAULT_MAX_STEPS
            self._task_max_steps = [uniform] * len(self._task_names)

        self._task_suite = task_suite
        self._suite_name = suite_name
        self._robots = robots
        self._split = split
        self._seed = seed
        self._num_init_states = int(num_init_states)
        # Kept for back-compat in the setup_suite payload (older clients read a
        # scalar). For task_suite runs we publish max(_task_max_steps) so a
        # client that only reads the scalar still gets a conservative cap.
        self._max_steps = max(self._task_max_steps)
        self._primary_camera = primary_camera
        self._include_right_view = include_right_view

        # Cached lazily on first setup_suite / get_task_init_states call.
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
            self._ensure_task_languages()
            return {
                "num_tasks": len(self._task_names),
                "task_languages": list(self._task_languages or []),
                "task_suite_name": self._suite_name,
                # Scalar kept for back-compat; clients that consume per-task
                # caps should read ``task_max_steps`` (aligned with
                # ``task_languages`` order).
                "max_steps": self._max_steps,
                "task_max_steps": list(self._task_max_steps),
                "evaluation_cfg": {},
                "env_spec": robocasa_env_spec(
                    primary_camera=self._primary_camera,
                    include_right_view=self._include_right_view,
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
            camera_heights = int(kwargs.get("camera_heights", 256))
            camera_widths = int(kwargs.get("camera_widths", 256))
            seed = int(kwargs.get("seed", self._seed))
            env = self._make_env(task_id, camera_heights, camera_widths, seed)
            state["env"] = env
            state["task_id"] = task_id
            language_instruction = self._language_for(env)
            seg_mapping = self._segmentation_name_mapping(env)
            return {
                "language_instruction": language_instruction,
                "segmentation_name_mapping": seg_mapping,
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
            # The visualiser uses base_pos/base_ori to anchor a *standalone arm*
            # MJCF (e.g. menagerie panda) at the position where the real arm's
            # link0 sits. For PandaOmron (mobile-base composite robot),
            # ``robot.base_pos`` returns the mobile-base root at floor level —
            # using that places the rendered arm sunk to the floor while the
            # actual arm is ~0.7 m + torso-slider higher. Robosuite's
            # ``add_mobile_base`` creates a body called ``manipulator_mount``
            # parented to the support body that holds the arm; its
            # forward-kinematics pose accounts for the torso slider, so it is
            # the correct anchor. Prefer it when present, then fall back to
            # other arm-mount bodies, then to ``robot.base_pos``.
            base_pos: np.ndarray | None = None
            base_ori: np.ndarray | None = None
            try:
                # robocasa kitchen envs construct with ``load_model_on_init=False``
                # so ``env.sim`` is None until the first ``env.reset()``.
                # ``get_robot_base`` is called in EpisodeRunner.__init__ — before
                # the first reset — so we need to trigger sim creation here
                # ourselves to read body world-poses. Reset is idempotent: the
                # subsequent ``set_init_state`` re-initialises the sim from the
                # recorded state anyway.
                sim = getattr(env, "sim", None)
                if sim is None:
                    if hasattr(env, "reset"):
                        env.reset()
                    sim = getattr(env, "sim", None)
                if sim is not None:
                    body_names = list(sim.model.body_names)
                    for body_name in (
                        "manipulator_mount",
                        "robot0_manipulator_mount",
                        "mobilebase0_support",
                        "robot0_base",
                    ):
                        if body_name in body_names:
                            bid = sim.model.body_name2id(body_name)
                            base_pos = sim.data.body_xpos[bid].copy()
                            base_ori = sim.data.body_xmat[bid].reshape(3, 3).copy()
                            logger.info(
                                f"[robocasa get_robot_base] anchored on {body_name!r} "
                                f"at xpos={base_pos.tolist()}"
                            )
                            break
            except Exception as exc:
                logger.debug(f"manipulator-mount body lookup failed: {exc}")

            if base_pos is None or base_ori is None:
                try:
                    robot = env.robots[0] if getattr(env, "robots", None) else None
                    if robot is not None:
                        bp = getattr(robot, "base_pos", None)
                        bo = getattr(robot, "base_ori", None)
                        if bp is not None and bo is not None:
                            bo_arr = np.asarray(bo)
                            if bo_arr.ndim == 1 and bo_arr.size == 3:
                                import robosuite.utils.transform_utils as T

                                bo_arr = T.euler2mat(bo_arr)
                            base_pos = np.asarray(bp, dtype=np.float64)
                            base_ori = np.asarray(bo_arr, dtype=np.float64)
                except Exception as exc:
                    logger.debug(f"robot.base_* lookup failed: {exc}")

            if base_pos is None or base_ori is None:
                logger.warning(
                    "robocasa get_robot_base: no arm-mount body found and "
                    "robot.base_* lookup failed; returning identity pose."
                )
                base_pos = np.zeros(3, dtype=np.float64)
                base_ori = np.eye(3, dtype=np.float64)

            return pack_result(
                {
                    "base_pos": np.asarray(base_pos, dtype=np.float64),
                    "base_ori": np.asarray(base_ori, dtype=np.float64),
                }
            ), state

        if cmd == "get_camera_params":
            if env is None:
                raise RuntimeError("No environment is active. Call make_env first.")
            camera_names = list(
                kwargs.get(
                    "camera_names",
                    ["robot0_agentview_left", "robot0_eye_in_hand"],
                )
            )
            camera_names = [
                n.decode() if isinstance(n, bytes) else n for n in camera_names
            ]
            camera_height = int(kwargs.get("camera_height", 256))
            camera_width = int(kwargs.get("camera_width", 256))

            params: dict = {}
            for cam_name in camera_names:
                try:
                    K = get_camera_intrinsic_matrix(
                        env.sim, cam_name, camera_height, camera_width
                    )
                    E = get_camera_extrinsic_matrix(env.sim, cam_name)
                    params[cam_name] = {
                        "intrinsic": np.array(K, dtype=np.float64),
                        "extrinsic": np.array(E, dtype=np.float64),
                    }
                except Exception as cam_exc:
                    logger.warning(
                        f"get_camera_params: skipping {cam_name!r}: {cam_exc}"
                    )
            return pack_result(params), state

        if cmd == "get_gt_object_pointclouds":
            if env is None:
                raise RuntimeError("No environment is active. Call make_env first.")
            n_points = int(kwargs.get("n_points_per_geom", 500))
            return pack_result(get_gt_object_pointclouds(env, n_points)), state

        # ------------------------------------------------------------------ #
        # Lifecycle methods that need adapter logic                            #
        # ------------------------------------------------------------------ #

        if env is None:
            raise RuntimeError(
                f"No environment is active. Call make_env first (cmd={cmd!r})."
            )

        if cmd == "set_init_state":
            init_state = args[0] if args else kwargs.get("state")
            if init_state is None:
                raise ValueError("set_init_state requires a state arg.")
            obs = self._reset_to(env, init_state)
            return pack_result(obs), state

        if cmd == "step":
            action = args[0] if args else kwargs.get("action")
            if action is None:
                raise ValueError("step requires an action arg.")
            padded = self._pad_action(np.asarray(action, dtype=np.float64), env)
            obs, reward, done, info = env.step(padded)
            return pack_result((obs, reward, done, info)), state

        if cmd == "reset":
            return pack_result(env.reset()), state

        # ------------------------------------------------------------------ #
        # Generic env-method dispatch                                          #
        # ------------------------------------------------------------------ #

        method = getattr(env, cmd)
        result = method(*args, **kwargs)
        return pack_result(result), state

    # ------------------------------------------------------------------
    # Suite helpers
    # ------------------------------------------------------------------

    def _ensure_task_languages(self) -> None:
        """Lazily probe each task's language by spinning up a one-shot env."""
        if self._task_languages is not None:
            return
        languages: list[str] = []
        for name in self._task_names:
            try:
                probe = self._make_env_by_name(name, 64, 64, self._seed)
                probe.reset()
                lang = self._language_for(probe)
                probe.close()
            except Exception as exc:
                logger.warning(f"Failed to probe language for {name!r}: {exc}")
                lang = name
            languages.append(lang)
        self._task_languages = languages

    def _get_or_build_init_states(self, task_id: int) -> list[dict]:
        if task_id in self._init_state_cache:
            return self._init_state_cache[task_id]

        env = self._make_env_by_name(self._task_names[task_id], 64, 64, self._seed)
        states: list[dict] = []
        try:
            for i in range(self._num_init_states):
                # In this robosuite version, env.seed is an int attribute set
                # at construction time, not a method. Re-seeding mid-rollout
                # would require a fresh env build; for now we rely on the
                # internal RNG advancing across consecutive resets, giving
                # distinct init states from one constructor seed.
                env.reset()
                xml = env.sim.model.get_xml()
                flat = np.array(env.sim.get_state().flatten())
                ep_meta = env.get_ep_meta() if hasattr(env, "get_ep_meta") else {}
                states.append(
                    {
                        "states": flat,
                        "model": xml,
                        "ep_meta": json.dumps(ep_meta, default=str),
                    }
                )
        finally:
            env.close()

        self._init_state_cache[task_id] = states
        return states

    # ------------------------------------------------------------------
    # Env factory + reset_to + action padding
    # ------------------------------------------------------------------

    def _make_env(
        self,
        task_id: int,
        camera_heights: int,
        camera_widths: int,
        seed: int,
    ):
        return self._make_env_by_name(
            self._task_names[task_id], camera_heights, camera_widths, seed
        )

    def _make_env_by_name(
        self, env_name: str, camera_heights: int, camera_widths: int, seed: int
    ):
        # robocasa.utils.env_utils.create_env hardcodes camera_depths=False and
        # then `**kwargs`-unpacks caller overrides into the same dict literal —
        # Python raises TypeError when both set the same key. Inline the
        # robosuite.make call so we can request depth + segmentation directly.
        # `import robocasa` is required for its side effect: it registers
        # the kitchen atomic envs (CloseDrawer, PnPCounterToCab, …) with
        # robosuite's environment registry.
        import robocasa  # noqa: F401  (env-registration side effect)
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        primary_robot = (
            self._robots if isinstance(self._robots, str) else self._robots[0]
        )
        controller_config = load_composite_controller_config(
            controller=None, robot=primary_robot
        )

        obj_instance_split = None
        layout_ids = None
        style_ids = None
        layout_and_style_ids = None
        if self._split == "target":
            obj_instance_split = "target"
            layout_and_style_ids = list(zip(range(1, 11), range(1, 11)))
        elif self._split == "pretrain":
            obj_instance_split = "pretrain"
            layout_ids = -2
            style_ids = -2
        elif self._split == "all":
            layout_ids = -3
            style_ids = -3
        elif self._split is not None:
            raise ValueError(
                f"split must be one of None/'all'/'pretrain'/'target', "
                f"got {self._split!r}"
            )

        env = robosuite.make(
            env_name=env_name,
            robots=self._robots,
            controller_configs=controller_config,
            camera_names=[
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ],
            camera_widths=camera_widths,
            camera_heights=camera_heights,
            has_renderer=False,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=True,
            camera_depths=True,
            # Kitchen.__init__ does not accept camera_segmentations; the
            # robocasa EnvSpec still advertises a segmentation_key on each
            # camera, but the obs dict will not actually contain it. Real
            # segmentation support requires a kitchen-side patch.
            seed=seed,
            obj_instance_split=obj_instance_split,
            generative_textures=None,
            randomize_cameras=False,
            layout_and_style_ids=layout_and_style_ids,
            layout_ids=layout_ids,
            style_ids=style_ids,
            translucent_robot=False,
        )
        logger.info(
            f"Created robocasa env {env_name!r} "
            f"(res={camera_heights}x{camera_widths}, seed={seed})"
        )
        return env

    def _reset_to(self, env, init_state):
        """Robomimic-style ``reset_to`` mirrored inline.

        ``init_state`` can be a dict with ``states`` / ``model`` / ``ep_meta``
        keys, or a flat numpy array (treated as ``{"states": …}``).
        """
        if isinstance(init_state, np.ndarray):
            init_state = {"states": init_state}
        elif not isinstance(init_state, dict):
            raise TypeError(
                f"set_init_state expects dict or ndarray, got {type(init_state)}"
            )

        if "model" in init_state:
            ep_meta = (
                json.loads(init_state["ep_meta"]) if init_state.get("ep_meta") else {}
            )
            if hasattr(env, "set_ep_meta"):
                env.set_ep_meta(ep_meta)
            elif hasattr(env, "set_attrs_from_ep_meta"):
                env.set_attrs_from_ep_meta(ep_meta)
            env.reset()
            xml = env.edit_model_xml(init_state["model"])
            env.reset_from_xml_string(xml)
            env.sim.reset()

        if "states" in init_state:
            env.sim.set_state_from_flattened(init_state["states"])
            env.sim.forward()
            if hasattr(env, "update_state"):
                env.update_state()
            elif hasattr(env, "update_sites"):
                env.update_sites()
            return env._get_observations(force_update=True)

        return None

    def _pad_action(self, action: np.ndarray, env) -> np.ndarray:
        """Map 7-D arm-only action → robosuite-native action.

        For PandaOmron (12-D), pads ``base_motion = 0`` and ``control_mode = 0``.
        For arms whose native dim equals the published 7-D, returns as-is.
        """
        action_dim = int(env.action_dim)
        action = action.reshape(-1)
        if action.shape[0] == action_dim:
            return action
        if action.shape[0] > action_dim:
            return action[:action_dim]
        out = np.zeros(action_dim, dtype=np.float64)
        out[: action.shape[0]] = action
        return out

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _language_for(env) -> str:
        if hasattr(env, "get_ep_meta"):
            try:
                return env.get_ep_meta().get("lang", "") or ""
            except Exception:
                return ""
        return ""

    @staticmethod
    def _segmentation_name_mapping(env) -> dict[str, str]:
        """Build robosuite's instance-segmentation mask-id → name dict.

        Robosuite's instance segmentation pixel value is
        ``name2id[instance] + 1`` where ``name2id`` is built from
        ``env.model.instances_to_ids.keys()``. We invert that here so the
        client's ``segmentation_name_mapping`` matches the LIBERO format
        (``{str(mask_id): instance_name}``).
        """
        try:
            instance_names = list(env.model.instances_to_ids.keys())
        except AttributeError:
            return {}
        return {str(i + 1): name for i, name in enumerate(instance_names)}


__all__ = ["RobocasaEnvServer"]
