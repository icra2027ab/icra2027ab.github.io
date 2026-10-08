# ruff: noqa: E402
"""LIBERO WebSocket environment server.

Run via the entry-point script in a Python environment that has LIBERO
installed (the "simulator Python"):

    python scripts/servers/serve_libero_env.py --host 0.0.0.0 --port 8765

Wire protocol is defined by :mod:`policy_abstraction.envs.base_server`. The
LIBERO-specific commands handled here are:

setup_suite, get_task_init_states, make_env, get_segmentation_name_mapping,
close, get_robot_base, get_camera_params, get_gt_object_pointclouds, plus a
generic fallback that forwards ``getattr(env, cmd)(*args, **kwargs)`` (covers
``reset``, ``set_init_state``, ``step``, ``seed``, …).
"""

import os
import sys

# ---------------------------------------------------------------------------
# Repo + LIBERO-PRO path bootstrap.
# Resolve four dirnames up: server.py is at
#   src/policy_abstraction/envs/libero/server.py
# so we land on src/. Adding src/ (not src/policy_abstraction/envs/) avoids
# shadowing the real ``libero`` package on PYTHONPATH with this directory's
# local ``libero/`` adapter.
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_REPO_ROOT = os.path.dirname(_SRC_DIR)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

_libero_pro_path = os.getenv("LIBERO_PRO_PATH") or os.path.join(
    _REPO_ROOT, "third_party", "LIBERO-PRO"
)
if not os.path.isdir(_libero_pro_path):
    raise ImportError(
        f"LIBERO-PRO not found at {_libero_pro_path}. "
        "Clone it under third_party/ (see docs/install_libero_pro.md) "
        "or set LIBERO_PRO_PATH to an existing checkout."
    )
sys.path.insert(0, _libero_pro_path)

# Export for downstream consumers:
#  - LIBERO_PRO_PATH:  YAMLs reference it via ${LIBERO_PRO_PATH}/...
#  - LIBERO_CONFIG_PATH:  LIBERO-PRO's libero/libero/__init__.py reads
#    "<LIBERO_CONFIG_PATH>/config.yaml". We ship config.yaml inside this
#    package's configs/ dir.
os.environ.setdefault("LIBERO_PRO_PATH", _libero_pro_path)
_LIBERO_CONFIGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs"
)
os.environ.setdefault("LIBERO_CONFIG_PATH", _LIBERO_CONFIGS_DIR)

import logging
import pathlib

import numpy as np
import yaml
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import SegmentationRenderEnv

from policy_abstraction.envs.base_server import BaseEnvServer, pack_result
from policy_abstraction.envs.libero.config import (
    TASK_MAX_STEPS,
    libero_env_spec,
)
from policy_abstraction.envs.libero.perturbation import (
    PerturbationManager,
)
from policy_abstraction.envs.robosuite_geom_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
    get_gt_object_pointclouds,
)

logger = logging.getLogger(__name__)

print("LiberoEnvServer initialized. Ready to serve connections.\n\n\n")


def _load_evaluation_config(
    evaluation_config_path: str,
    task_suite_name: str,
    use_swap: bool,
    use_object: bool,
    use_language: bool,
    use_task: bool,
    use_environment: bool,
) -> dict:
    """Load the LIBERO-PRO evaluation config and inject suite/perturbation keys."""
    with open(evaluation_config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def _expand(value):
        if isinstance(value, str):
            return os.path.expandvars(os.path.expanduser(value))
        if isinstance(value, dict):
            return {k: _expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_expand(v) for v in value]
        return value

    cfg = _expand(cfg)
    cfg["bddl_files_path"] = cfg.get("bddl_files_path", "") + "/" + task_suite_name
    cfg.update(
        {
            "task_suite_name": task_suite_name,
            "use_swap": use_swap,
            "use_object": use_object,
            "use_language": use_language,
            "use_task": use_task,
            "use_environment": use_environment,
        }
    )
    logger.info(f"Loaded evaluation config:\n{cfg}\n")
    return cfg


def _resolve_max_steps(task_suite_name: str) -> int:
    if task_suite_name in TASK_MAX_STEPS:
        return TASK_MAX_STEPS[task_suite_name]
    for suite_name, max_steps in TASK_MAX_STEPS.items():
        if task_suite_name.startswith(suite_name):
            return max_steps
    raise ValueError(f"Unknown task suite: {task_suite_name}")


class LiberoEnvServer(BaseEnvServer):
    """Serves a LIBERO environment over a WebSocket connection.

    All LIBERO-PRO suite / perturbation setup happens at server startup. The
    effective (possibly perturbation-suffixed) task suite name is computed
    once here and later returned to the client via ``setup_suite``.
    """

    server_metadata = {"server": "LiberoEnvServer", "version": "1.0"}

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        task_suite_name: str = "test",
        evaluation_config_path: str = "",
        use_swap: bool = False,
        use_object: bool = False,
        use_language: bool = False,
        use_task: bool = False,
        use_environment: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(host=host, port=port)
        self._seed = seed

        self._evaluation_cfg = _load_evaluation_config(
            evaluation_config_path,
            task_suite_name,
            use_swap,
            use_object,
            use_language,
            use_task,
            use_environment,
        )

        perturb_manager = PerturbationManager(self._evaluation_cfg)
        self._effective_task_suite_name = perturb_manager.setup_perturbed_environment(
            task_suite_name,
            use_swap,
            use_object,
            use_language,
            use_task,
            use_environment,
        )
        logger.info(
            f"Effective task_suite_name after perturbation: "
            f"{self._effective_task_suite_name!r}"
        )
        self._max_steps = _resolve_max_steps(self._effective_task_suite_name)

    # ------------------------------------------------------------------
    # Per-connection state
    # ------------------------------------------------------------------

    def _initial_state(self) -> dict:
        return {"env": None, "suite": None}

    # ------------------------------------------------------------------
    # Command dispatch
    # ------------------------------------------------------------------

    async def _dispatch(self, cmd: str, args: list, kwargs: dict, state: dict) -> tuple:
        env = state["env"]
        suite = state["suite"]

        # ------------------------------------------------------------------ #
        # Suite-level commands (no active env required)                        #
        # ------------------------------------------------------------------ #

        if cmd == "setup_suite":
            task_suite_name = self._effective_task_suite_name
            benchmark_dict = benchmark.get_benchmark_dict()
            suite = benchmark_dict[task_suite_name]()
            state["suite"] = suite
            num_tasks = suite.n_tasks
            task_languages = [suite.get_task(i).language for i in range(num_tasks)]
            logger.info(f"Suite loaded: {task_suite_name!r} ({num_tasks} tasks)")
            return {
                "num_tasks": num_tasks,
                "task_languages": task_languages,
                "task_suite_name": task_suite_name,
                "max_steps": self._max_steps,
                "evaluation_cfg": self._evaluation_cfg,
                "env_spec": libero_env_spec().to_dict(),
            }, state

        if cmd == "get_task_init_states":
            if suite is None:
                raise RuntimeError("No suite loaded. Call setup_suite first.")
            task_id = int(kwargs["task_id"])
            init_states = suite.get_task_init_states(task_id)
            return pack_result(init_states), state

        # ------------------------------------------------------------------ #
        # Environment lifecycle commands                                       #
        # ------------------------------------------------------------------ #

        if cmd == "make_env":
            if suite is None:
                raise RuntimeError("No suite loaded. Call setup_suite first.")
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
            task_id = int(kwargs["task_id"])
            camera_heights = int(kwargs.get("camera_heights", 256))
            camera_widths = int(kwargs.get("camera_widths", 256))
            seed = int(kwargs.get("seed", 0))
            env = self._make_env(suite, task_id, camera_heights, camera_widths, seed)
            state["env"] = env
            language_instruction = getattr(env, "language_instruction", "")
            seg_mapping: dict = getattr(env, "segmentation_id_mapping", {})
            segmentation_name_mapping = {
                str(seg_id + 1): instance_name
                for seg_id, instance_name in seg_mapping.items()
            }
            return {
                "language_instruction": language_instruction,
                "segmentation_name_mapping": segmentation_name_mapping,
            }, state

        if cmd == "get_segmentation_name_mapping":
            if env is None:
                raise RuntimeError("No environment is active. Call make_env first.")
            seg_mapping = getattr(env, "segmentation_id_mapping", {})
            return {
                str(seg_id + 1): instance_name
                for seg_id, instance_name in seg_mapping.items()
            }, state

        if cmd == "close":
            if env is not None:
                env.close()
            state["env"] = None
            return None, state

        if cmd == "get_robot_base":
            if env is None:
                raise RuntimeError("No environment is active. Call make_env first.")
            robot = env.robots[0]
            return pack_result(
                {
                    "base_pos": np.array(robot.base_pos, dtype=np.float64),
                    "base_ori": np.array(robot.base_ori, dtype=np.float64),
                }
            ), state

        if cmd == "get_camera_params":
            if env is None:
                raise RuntimeError("No environment is active. Call make_env first.")
            camera_names = list(
                kwargs.get("camera_names", ["agentview", "robot0_eye_in_hand"])
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
        # Generic env-method dispatch                                          #
        # ------------------------------------------------------------------ #

        if env is None:
            raise RuntimeError(
                f"No environment is active. Call make_env first (cmd={cmd!r})."
            )
        method = getattr(env, cmd)
        result = method(*args, **kwargs)
        return pack_result(result), state

    # ------------------------------------------------------------------
    # Environment factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_env(
        suite,
        task_id: int,
        camera_heights: int = 256,
        camera_widths: int = 256,
        seed: int = 0,
    ):
        task = suite.get_task(task_id)
        bddl_file = str(
            pathlib.Path(get_libero_path("bddl_files"))
            / task.problem_folder
            / task.bddl_file
        )
        env_args = {
            "bddl_file_name": bddl_file,
            "camera_heights": camera_heights,
            "camera_widths": camera_widths,
            "camera_segmentations": "instance",
            "camera_depths": True,
        }
        env = SegmentationRenderEnv(**env_args)
        env.seed(seed)
        logger.info(
            f"Created env for task {task_id}: {bddl_file!r} "
            f"(res={camera_heights}x{camera_widths}, seed={seed})"
        )
        return env


__all__ = [
    "LiberoEnvServer",
    "get_camera_intrinsic_matrix",
    "get_camera_extrinsic_matrix",
]
