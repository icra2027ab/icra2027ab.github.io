"""Server and client lifecycle for an evaluation run.

:class:`EvalBackends` owns everything that is expensive to build and must
outlive a single episode: the policy inference server, the env server, the
retrieval index, the demo analyzer, the perception models. Building it once and
running many episodes against it is what makes a multi-rollout session possible
without paying the policy's load-and-compile cost per rollout — measured at
~2.5 min load plus ~40 s warmup for the pi05 UR5e checkpoints.

Two entry points share it:

* ``scripts/evaluate.py`` — batch evaluation over a task suite.
* ``scripts/experiments/collect_grid_experiment.py`` — an operator driving
  rollouts one at a time from a GUI.

The split matters for the second one. The GUI must be able to start a rollout,
stop it, record a verdict, and start another, all against the *same* loaded
policy; anything that tore down and rebuilt the backends between rollouts would
make an afternoon of data collection impossible.
"""

from __future__ import annotations

import atexit
import dataclasses
import os
import pathlib
import signal
import time
from typing import Any, Optional

import numpy as np
from loguru import logger

from policy_abstraction.data.recorder import BaseRecorder
from policy_abstraction.dynamics import (
    BaseDynamicsModel,
    IntegrationDynamicsModel,
    Ur5KinematicsDynamicsModel,
)
from policy_abstraction.envs.calvin.utils import start_calvin_env_server
from policy_abstraction.envs.libero.utils import start_libero_env_server
from policy_abstraction.envs.robocasa.utils import start_robocasa_env_server
from policy_abstraction.envs.ur5.utils import start_ur5_env_server
from policy_abstraction.envs.websocket_client import WebsocketEnvClient
from policy_abstraction.evaluation.abstraction import PerceptionMode
from policy_abstraction.evaluation.config import (
    Args,
    CalvinEnvArgs,
    CosmosPolicyArgs,
    GrootPolicyArgs,
    LerobotPolicyArgs,
    LiberoEnvArgs,
    OpenpiPolicyArgs,
    RobocasaEnvArgs,
    Ur5EnvArgs,
    env_kind,
)
from policy_abstraction.evaluation.episode_runner import (
    EpisodeRunner,
    build_policy_input,
    policy_preprocess_imgs,
)
from policy_abstraction.evaluation.obs_encoder import IdentityObsEncoder
from policy_abstraction.evaluation.retrieval import build_retrieval_components
from policy_abstraction.policies.cosmos import CosmosPolicyClient, start_cosmos_server
from policy_abstraction.policies.dummy import CircularPolicy, DummyPolicy
from policy_abstraction.policies.groot import GrootPolicyClient, start_groot_server
from policy_abstraction.policies.lerobot import (
    LerobotPolicyClient,
    start_lerobot_server,
)
from policy_abstraction.policies.openpi import (
    PatientWebsocketClient,
    start_policy_server,
)
from policy_abstraction.utils.servers import terminate_servers
from policy_abstraction.visualization.rerun import RerunVisualizer, viewer_url
from policy_abstraction.vlm.live_object_inferer import LiveObjectInferer

# Fallback shape for the in-process fake policies (--guidance.debug), used
# when nothing better is known: the pi05_libero_grounded chunk geometry the
# fakes were written against.
_FAKE_POLICY_HORIZON = 10
_FAKE_POLICY_ACTION_DIM = 7

# ---------------------------------------------------------------------------
# Small helpers, lifted verbatim from scripts/evaluate.py
# ---------------------------------------------------------------------------


def resolve_flip_u(setting: str, env_spec) -> bool:
    """Resolve ``--perception.flip-u`` against the env's camera transform.

    ``auto`` mirrors u exactly when the abstractor pre-flipped the frames, which
    is what ``CameraSpec.pixels_mirrored`` tells us. That reproduces
    the simulator adapters' long-standing ``flip_u=True`` and gives the real
    UR5e (upright frames) the ``False`` it needs — so the same command works on
    either without a per-env override.
    """
    if setting == "true":
        return True
    if setting == "false":
        return False
    if setting != "auto":
        raise ValueError(
            f"--perception.flip-u must be auto/true/false, got {setting!r}"
        )
    if env_spec is None:
        # No spec to consult (server published none): keep the historical
        # simulator behaviour rather than silently changing geometry.
        return True
    return env_spec.primary_camera.pixels_mirrored


def _ur5_tcp_offset(args: Args, env_spec=None):
    """The controller's TCP offset: the flag first, else what the env published."""
    tcp_offset = args.env.tcp_offset if isinstance(args.env, Ur5EnvArgs) else None
    if tcp_offset is None and env_spec is not None:
        tcp_offset = env_spec.metadata.get("tcp_offset")
    return tcp_offset


def guidance_action_space_for(args: Args) -> str:
    """Resolve ``--policy.guidance-action-space auto`` from the env.

    The UR5e is the one env whose actions are absolute joint targets; every
    simulator emits Cartesian deltas.
    """
    requested = getattr(args.policy, "guidance_action_space", "auto")
    if requested != "auto":
        return str(requested)
    return "ur5e-joints" if isinstance(args.env, Ur5EnvArgs) else "cartesian-delta"


def guidance_gripper_geometry_for(args: Args) -> str | None:
    """The gripper the server's guidance measures finger pairs from.

    ``Args.__post_init__`` already resolves the visualiser's gripper model
    per env (the 2F-140 on the UR5e); the guidance loss must use the same
    one, so it is read from there. ``None`` leaves the server's default.
    """
    if not isinstance(args.env, Ur5EnvArgs):
        return None
    return getattr(args.viz, "gripper_model", None) or "robotiq_2f140"


def _fake_policy_holds_joints(args: Args) -> bool:
    """Whether the selected fake emits the UR5e's absolute joint targets.

    The noop fake starts from the observed joints and lets guidance move them
    (see :class:`DummyPolicy`), so on the UR5e its chunk *is* joint targets.
    The circular fake still traces Cartesian deltas whatever the env is.
    """
    return (
        args.guidance.debug
        and args.guidance.fake_policy == "noop"
        and isinstance(args.env, Ur5EnvArgs)
    )


def dynamics_model_for(args: Args, env_spec=None) -> BaseDynamicsModel:
    """Pick the dynamics model that matches the env's action space.

    Cartesian-delta envs integrate their actions; a UR5e driven with absolute
    joint targets needs forward kinematics instead, otherwise every predicted
    end-effector trajectory is nonsense.
    """
    if args.guidance.debug and not _fake_policy_holds_joints(args):
        # The circular fake (and every fake on a Cartesian env) emits
        # LIBERO-style deltas whatever the env is. Running those through the
        # UR5e's forward kinematics would read them as joint angles and draw
        # the predicted chunk metres from the gripper -- the arm itself is
        # safe (the launcher forces --dry-run), but the 3-D view is nonsense.
        return IntegrationDynamicsModel()
    if isinstance(args.env, Ur5EnvArgs):
        return Ur5KinematicsDynamicsModel(tcp_offset=_ur5_tcp_offset(args, env_spec))
    return IntegrationDynamicsModel()


def build_runner_components(args: Args, env_spec=None) -> dict:
    """Return obs_encoder / dynamics_model / visualizer for EpisodeRunner."""
    visualizer = None
    if args.viz.enable:
        visualizer = RerunVisualizer(
            application_id=f"{env_kind(args.env)}_eval",
            spawn=args.viz.connect is None,
            # Must be None when spawning: the visualiser prefers a connect
            # address over spawning, and an f-string always produces a truthy
            # one -- "rerun+http://None/proxy" silently won, so --viz.connect
            # None connected to a nonexistent viewer instead of launching one.
            connect_addr=(viewer_url(args.viz.connect) if args.viz.connect else None),
            robot_model_path=args.viz.robot_model,
            robot_description_name=args.viz.robot_description,
            gripper_model_name=args.viz.gripper_model,
            gripper_mount_yaw_deg=args.viz.gripper_mount_yaw_deg,
            layout=args.viz.layout,
            serve=args.viz.serve,
            serve_grpc_port=args.viz.serve_grpc_port,
            serve_web_port=args.viz.serve_web_port,
        )
    return dict(
        obs_encoder=IdentityObsEncoder(),
        dynamics_model=dynamics_model_for(args, env_spec),
        visualizer=visualizer,
    )


def relation_filter_for(args: Args) -> Optional[frozenset[str]]:
    """Build the allowed-relation set from the per-relation ablation gates.

    When all three are on, return ``None`` so the guidance builder takes its
    unfiltered path (bit-for-bit identical to the non-ablation behaviour).
    """
    allowed = {
        name
        for name, on in (
            ("positional", args.guidance.relation_positional),
            ("orientation", args.guidance.relation_orientation),
            ("gripper", args.guidance.relation_gripper),
        )
        if on
    }
    return None if len(allowed) == 3 else frozenset(allowed)


def start_env_server(args: Args, run_dir: pathlib.Path):
    """Dispatch to the per-env launcher based on the EnvArgs variant."""
    e = args.env
    if isinstance(e, LiberoEnvArgs):
        return start_libero_env_server(
            e.host,
            e.port,
            task_suite_name=e.task_suite_name,
            evaluation_config_path=e.evaluation_config_path,
            use_swap=e.perturb.swap,
            use_object=e.perturb.object,
            use_language=e.perturb.language,
            use_task=e.perturb.task,
            use_environment=e.perturb.environment,
            seed=args.seed,
            log_dir=run_dir,
        )
    if isinstance(e, RobocasaEnvArgs):
        return start_robocasa_env_server(
            e.host,
            e.port,
            task=e.task,
            tasks=e.tasks,
            task_suite=e.task_suite,
            suite_name=e.suite_name,
            robots=e.robots,
            split=e.split,
            seed=args.seed,
            num_init_states=e.num_init_states,
            max_steps=e.max_steps,
            primary_camera=e.primary_camera,
            include_right_view=e.include_right_view,
        )
    if isinstance(e, CalvinEnvArgs):
        return start_calvin_env_server(
            e.host,
            e.port,
            scenes=e.scenes,
            suite_name=e.suite_name,
            cameras_cfg=e.cameras_cfg,
            robot_cfg=e.robot_cfg,
            seed=args.seed,
            num_init_states=e.num_init_states,
            max_steps=e.max_steps,
            primary_camera=e.primary_camera,
            include_gripper=e.include_gripper,
        )
    if isinstance(e, Ur5EnvArgs):
        return start_ur5_env_server(
            e.host,
            e.port,
            tasks=e.tasks,
            suite_name=e.suite_name,
            robot_ip=e.robot_ip,
            cameras=e.cameras,
            wrist_camera=e.wrist_camera,
            primary_camera=e.primary_camera,
            control_hz=e.control_hz,
            max_steps=e.max_steps,
            num_init_states=e.num_init_states,
            max_joint_delta=e.max_joint_delta,
            flip_gripper=e.flip_gripper,
            home_on_reset=e.home_on_reset,
            home_on_disconnect=e.home_on_disconnect,
            success_mode=e.success_mode,
            dry_run=e.dry_run,
            simulate_hardware=e.simulate_hardware,
            tcp_offset=e.tcp_offset,
            calibration_path=e.calibration_path,
            calibration_base_frame=e.calibration_base_frame,
            sdk_path=e.sdk_path,
            log_dir=run_dir,
        )
    raise TypeError(f"Unknown env args type: {type(e).__name__}")


def cleanup_servers(server_processes: list) -> None:
    """Terminate all server subprocesses, parking hardware on the way out.

    The grace period is shared across the whole list rather than spent per
    server, so a UR5e server that needs a few seconds to park the arm and
    release RTDE gets them without multiplying the wait by the number of
    servers.
    """
    terminate_servers(server_processes)


# ---------------------------------------------------------------------------
# The backends bundle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EvalBackends:
    """Everything an episode needs that outlives the episode.

    Build with :func:`start_backends`; release with :meth:`shutdown`. Between
    those two calls any number of :class:`EpisodeRunner` instances can be
    constructed via :meth:`build_runner` against the same loaded policy.
    """

    args: Args
    run_dir: pathlib.Path
    env: WebsocketEnvClient
    policy: Any
    suite_info: dict
    server_processes: list = dataclasses.field(default_factory=list)
    retriever: Any = None
    analyzer: Any = None
    perception_client: Any = None
    live_object_inferer: Any = None
    demo_segmenter: Any = None
    _closed: bool = False
    action_shape: tuple[int, ...] | None = None
    """Shape of the chunk the policy returned to :meth:`warm_up_policy`
    (``(num_chunks, horizon, action_dim)``), so a later guided warm-up can
    size its request to the steady-state chunk rather than guess."""

    # -- suite introspection -------------------------------------------------

    @property
    def num_tasks(self) -> int:
        return int(self.suite_info["num_tasks"])

    @property
    def task_languages(self) -> list[str]:
        return list(self.suite_info["task_languages"])

    @property
    def task_max_steps(self) -> list[int]:
        """Per-task step caps.

        New servers publish ``task_max_steps`` (e.g. RoboCasa's leaderboard
        horizons); older ones publish only the scalar ``max_steps``, which we
        broadcast uniformly.
        """
        scalar = self.suite_info["max_steps"]
        published = self.suite_info.get("task_max_steps")
        return list(published or [scalar] * self.num_tasks)

    # -- episode construction ------------------------------------------------

    def abstractor_kwargs(self) -> Optional[dict]:
        """Kwargs for the ``RolloutAbstractor``, or ``None`` when disabled."""
        args = self.args
        if not (args.retrieval.enable and args.demo_analysis.enable):
            return None
        kwargs = {
            "vlm_model": args.demo_analysis.vlm_model,
            "perception_mode": args.perception.mode,
            "mta_max_advance": (
                args.alignment.max_advance_chunks * args.policy.replan_steps
                if args.alignment.max_advance_chunks is not None
                else None
            ),
            "subtraj_time_weight": args.alignment.subtraj_time_weight,
            "gripper_speed_threshold": args.subtraj.gripper_speed_threshold,
            "subtraj_min_length": args.subtraj.min_length,
            "alignment_mode": args.alignment.mode,
            "mixed_variation": args.alignment.mixed_variation,
            "temporal_factor_temperature": args.alignment.temporal_factor_temperature,
            "temporal_factor_floor": args.alignment.temporal_factor_floor,
            "sg_switch_persistence": args.alignment.sg_switch_persistence,
            "relevance_near_threshold": args.guidance.relevance_near_threshold,
            "relevance_far_threshold": args.guidance.relevance_far_threshold,
            "record_scene_pointcloud": args.viz.record_scene_pointcloud,
            "scene_pcd_max_points": args.viz.scene_pointcloud_max_points,
            "dynamics_model": dynamics_model_for(args, self.env.env_spec),
        }
        if self.perception_client is not None:
            kwargs["multi_view_segmentation_fn"] = self.perception_client
        if self.live_object_inferer is not None:
            kwargs["live_object_inferer"] = self.live_object_inferer
        return kwargs

    def warm_up_policy(self, task_description: str) -> float | None:
        """Pay the policy's first-inference cost now, without moving the arm.

        A ``torch.compile``d checkpoint (pi05 defaults to
        ``compile_model=true, compile_mode=max-autotune``) autotunes on its
        *first* real query, not at load: measured on the UR5e cell, the server
        reported "ready" after 106 s and then spent a further **449 s** inside
        the first inference. Left to the first rollout, that is a rollout in
        which the arm does not move, the operator cannot tell compiling from
        hung, and any wall-clock timeout is spent before the robot is asked to
        do anything.

        So it is done here, between the servers coming up and the operator
        window opening: one query built exactly as an episode builds it — same
        camera set, same resize, same state layout, so the compiler sees the
        shapes it will see for real — and the returned actions are dropped on
        the floor. ``self.env`` is only *read*: no ``step``, no ``reset``, no
        action ever reaches the controller.

        Returns the seconds it took, or ``None`` when there was nothing to warm
        (no policy, or the env served no observation — a simulator adapter that
        cannot answer ``get_obs`` on demand, say). Never raises: a failed
        warm-up costs the first rollout its old slow start and nothing else.
        """
        policy = self.policy
        if policy is None or not hasattr(policy, "infer"):
            return None
        spec = self.env.env_spec
        if spec is None:
            return None
        try:
            obs = self.env.get_obs()
            images = {
                cam.name: np.ascontiguousarray(
                    cam.apply_image_transform(obs[cam.image_key])
                )
                for cam in spec.cameras
                if cam.image_key in obs
            }
            element = build_policy_input(
                spec,
                obs,
                policy_preprocess_imgs(images, self.args.camera.resize_size),
                task_description=str(task_description),
                rng_seed=0,
            )
            t0 = time.monotonic()
            result = policy.infer(element)
            elapsed = time.monotonic() - t0
            actions = result.get("actions") if isinstance(result, dict) else None
            if actions is not None:
                self.action_shape = tuple(int(n) for n in np.shape(actions))
        except Exception as exc:
            logger.warning(
                f"Policy warm-up failed ({type(exc).__name__}: {exc}); the "
                "first rollout will pay the compile instead. It is not "
                "otherwise affected."
            )
            return None

        # The warm-up is a real inference as far as the server is concerned, so
        # under RTC its chunk sits in the previous-chunk buffer waiting to be
        # inpainted into the first chunk of rollout 0 -- an episode blended
        # with actions computed for a scene nobody staged. `reset` is what
        # clears that, and the runner's own per-episode reset comes too late to
        # be relied on here.
        if getattr(policy, "resets_per_episode", False):
            try:
                policy.reset()
            except Exception as exc:
                logger.warning(f"Policy reset after warm-up failed: {exc}")
        return elapsed

    def warm_up_guided_policy(self, task_description: str) -> float | None:
        """Pay the *guided* path's first-inference cost, without moving the arm.

        :meth:`warm_up_policy` compiles the plain sampling loop. A guided
        request runs a second graph — the guidance loss and its gradient
        through every denoising step — which ``torch.compile`` meets for the
        first time at the first guided replan, with the arm standing still
        for the duration. For a batch run that is a one-off stall at step 0;
        for an operator flipping guidance on under a running policy it is a
        pause at exactly the moment they are showing something.

        So this sends one request shaped like a rollout's: the observation as
        an episode builds it, plus a synthetic single-pair guidance set (the
        end-effector to a point just below it, told to hold its current
        distance) with the same control flags the runner sends. The returned
        actions are dropped and the policy is reset, so nothing reaches the
        controller and RTC's previous-chunk buffer is left empty.

        Returns the seconds it took, or ``None`` when there was nothing to warm
        — guidance not configured, no policy, no observation — or when the
        request could not be built. Never raises: the first guided replan then
        simply pays the compile as before.
        """
        policy = self.policy
        args = self.args
        if policy is None or not hasattr(policy, "infer"):
            return None
        if not args.guidance.enable or args.guidance.debug:
            return None
        spec = self.env.env_spec
        if spec is None:
            return None
        try:
            from policy_abstraction.evaluation.guidance import (
                guidance_request_flags,
                pack_guidance_params,
            )
            from policy_abstraction.evaluation.guidance_builder import (
                build_distance_guidance_params,
            )
            from policy_abstraction.utils.transforms import quat2axisangle
            from policy_abstraction.vlm.types import DistancePair, DistanceType

            obs = self.env.get_obs()
            images = {
                cam.name: np.ascontiguousarray(
                    cam.apply_image_transform(obs[cam.image_key])
                )
                for cam in spec.cameras
                if cam.image_key in obs
            }
            element = build_policy_input(
                spec,
                obs,
                policy_preprocess_imgs(images, args.camera.resize_size),
                task_description=str(task_description),
                rng_seed=0,
            )
            robot = spec.robot
            ee_pos = np.asarray(obs[robot.eef_pos_key], dtype=np.float64)
            ee_ori = quat2axisangle(
                np.asarray(obs[robot.eef_quat_key], dtype=np.float64)
            )
            gripper = np.atleast_1d(
                robot.gripper.aperture_from_qpos(obs[robot.gripper_qpos_key])
            )
            # The runner sizes guidance to the whole chunk once it knows the
            # policy's output shape, and to replan_steps before that.
            if self.action_shape is not None and len(self.action_shape) == 3:
                chunk_size = int(self.action_shape[0] * self.action_shape[1])
            else:
                chunk_size = int(args.policy.replan_steps)
            label = "warmup_ee_to_target"
            pair = DistancePair(
                from_entity="ee",
                to_entity=1,
                label=label,
                distance_type=DistanceType.CENTROID,
                subtraj_idx=0,
            )
            standoff = 0.05
            target = ee_pos + np.array([0.0, 0.0, -standoff])
            control_hz = getattr(args.env, "control_hz", None) or 50.0
            params = build_distance_guidance_params(
                pairs=[pair],
                live_distances={label: standoff},
                entity_centroids={"ee": ee_pos, 1: target},
                object_pointclouds=None,
                ref_traj=np.full((chunk_size, 1), standoff, dtype=np.float32),
                pair_labels=[label],
                aligned_ts=np.zeros(1, dtype=np.int32),
                chunk_size=chunk_size,
                dt=1.0 / float(control_hz),
                guidance_factor=float(args.guidance.factor),
                guidance_weights=np.ones(1, dtype=np.float32),
                ee_ori_0=ee_ori,
                gripper_0=gripper,
            )
            if params is None:
                logger.warning(
                    "Guided warm-up skipped: could not build a guidance request."
                )
                return None
            params = dict(params)
            params.update(
                guidance_request_flags(
                    normalize=args.guidance.normalize,
                    line_search=args.guidance.line_search,
                    max_alpha=args.guidance.max_alpha,
                    multi_alpha=args.guidance.multi_alpha,
                    multi_alpha_combine=args.guidance.multi_alpha_combine,
                )
            )
            element = pack_guidance_params(element, params)
            t0 = time.monotonic()
            policy.infer(element)
            elapsed = time.monotonic() - t0
        except Exception as exc:
            logger.warning(
                f"Guided policy warm-up failed ({type(exc).__name__}: {exc}); "
                "the first guided replan will pay the compile instead."
            )
            return None
        if getattr(policy, "resets_per_episode", False):
            try:
                policy.reset()
            except Exception as exc:
                logger.warning(f"Policy reset after guided warm-up failed: {exc}")
        return elapsed

    def build_runner(
        self,
        *,
        task_description: str,
        initial_state: Any,
        max_steps: int,
        episode_data_path: str | None = None,
        episode_idx: int = 0,
        recorder: BaseRecorder | None = None,
        **overrides: Any,
    ) -> EpisodeRunner:
        """Construct one :class:`EpisodeRunner` against the loaded backends.

        Any :class:`EpisodeRunner` keyword can be overridden via *overrides* —
        the collection GUI uses that to pass a ``stop_signal``.
        """
        args = self.args
        kwargs: dict[str, Any] = dict(
            env=self.env,
            client=self.policy,
            task_description=task_description,
            initial_state=initial_state,
            args=args,
            max_steps=max_steps,
            episode_data_path=episode_data_path,
            depth_stride=args.output.depth_stride,
            demo_segmenter=self.demo_segmenter,
            seed=args.seed,
            episode_idx=episode_idx,
            recorder=recorder,
            retriever=self.retriever,
            analyzer=self.analyzer,
            abstractor_kwargs=self.abstractor_kwargs(),
            robot_model_name=args.viz.robot_description or args.viz.robot_model,
            use_distance_guidance=args.guidance.enable,
            guidance_factor=args.guidance.factor,
            debug_guidance=args.guidance.debug,
            aligned_t_offset=args.guidance.aligned_t_offset,
            normalize_guidance=args.guidance.normalize,
            use_line_search=args.guidance.line_search,
            guidance_max_alpha=args.guidance.max_alpha,
            multi_alpha=args.guidance.multi_alpha,
            multi_alpha_combine=args.guidance.multi_alpha_combine,
            relation_filter=relation_filter_for(args),
            **build_runner_components(args, self.env.env_spec),
        )
        kwargs.update(overrides)
        return EpisodeRunner(**kwargs)

    # -- teardown ------------------------------------------------------------

    def shutdown(self) -> None:
        """Disconnect the env and terminate every spawned server. Idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            self.env.disconnect()
        except Exception:
            pass
        cleanup_servers(self.server_processes)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _start_policy_backend(args: Args, run_dir: pathlib.Path) -> tuple[Any, list]:
    """Launch the policy server (if any) and return ``(handle, processes)``.

    ``handle`` is the subprocess used by the matching client's readiness wait,
    or ``None`` for backends with no spawned server.
    """
    processes: list = []
    if args.guidance.debug:
        # Fake in-process policy; nothing to launch.
        return None, processes

    if isinstance(args.policy, LerobotPolicyArgs):
        # Top-level, not nested under the LIBERO branch: this backend serves the
        # real arm too, and gating it on env:libero would silently skip the
        # launch for env:ur5.
        if args.policy.model_path is None:
            raise ValueError(
                "LeRobot backend requires `policy:lerobot --policy.model-path <path>`."
            )
        if not args.policy.launch_server:
            logger.info(
                f"--policy.no-launch-server: expecting a LeRobot policy server "
                f"already listening on 127.0.0.1:{args.policy.port}"
            )
            return None, processes
        action_space = guidance_action_space_for(args)
        tcp_offset = _ur5_tcp_offset(args) if action_space == "ur5e-joints" else None
        if action_space == "ur5e-joints" and tcp_offset is None:
            logger.warning(
                "[eval] Joint-space guidance without --env.tcp-offset: the "
                "guidance kinematics stop at the flange, ~22 cm short of the "
                "tool the perception stack measures distances from."
            )
        proc = start_lerobot_server(
            port=args.policy.port,
            model_path=args.policy.model_path,
            host="127.0.0.1",
            gpus=args.policy.gpus,
            repo_path=args.policy.repo_path,
            policy_type=args.policy.policy_type,
            device=args.policy.device,
            camera_map=args.policy.camera_map,
            state_adapter=args.policy.state_adapter,
            num_inference_steps=args.policy.num_inference_steps,
            compile_model=args.policy.compile_model,
            guidance=args.policy.guidance,
            rtc=args.policy.rtc,
            rtc_execution_horizon=int(args.policy.rtc_execution_horizon or 10),
            rtc_max_guidance_weight=args.policy.rtc_max_guidance_weight,
            rtc_prefix_attention_schedule=args.policy.rtc_prefix_attention_schedule,
            robot_type=args.policy.robot_type,
            guidance_action_space=action_space,
            tcp_offset=tcp_offset,
            gripper_geometry=guidance_gripper_geometry_for(args),
            log_dir=run_dir,
        )
        processes.append(proc)
        return proc, processes

    if not isinstance(args.env, LiberoEnvArgs):
        raise RuntimeError(
            f"No policy checkpoint exists for env '{env_kind(args.env)}'. "
            "Run with --guidance.debug to use the DummyPolicy"
            + (
                ", or select the real-robot backend with "
                "`policy:lerobot --policy.model-path <checkpoint>`."
                if isinstance(args.env, Ur5EnvArgs)
                else "."
            )
        )

    if isinstance(args.policy, OpenpiPolicyArgs):
        proc = start_policy_server(
            args.policy.port,
            gpus=args.policy.gpus,
            dynamics_model_path=args.policy.dynamics_model_path,
            log_dir=run_dir,
            policy_env=args.policy.env,
            policy_config=args.policy.config,
            policy_dir=args.policy.dir,
        )
    elif isinstance(args.policy, GrootPolicyArgs):
        if args.policy.model_path is None:
            raise ValueError(
                "GR00T backend requires `policy:groot --policy.model-path <path>`."
            )
        proc = start_groot_server(
            port=args.policy.port,
            model_path=args.policy.model_path,
            gpus=args.policy.gpus,
            embodiment_tag=args.policy.embodiment_tag,
            host="127.0.0.1",
            repo_path=args.policy.repo_path,
            log_dir=run_dir,
        )
    elif isinstance(args.policy, CosmosPolicyArgs):
        # Cosmos LIBERO ↔ env:libero; Cosmos RoboCasa ↔ env:robocasa.
        # Mismatched pairings would produce malformed obs (e.g. the RoboCasa
        # checkpoint asserts num_third_person_images=2 and indexes a secondary
        # image that LIBERO's spec doesn't route).
        kind = env_kind(args.env)
        if args.policy.suite != kind:
            raise ValueError(
                f"cosmos.suite={args.policy.suite!r} requires env:"
                f"{args.policy.suite}; got env:{kind}."
            )
        proc = start_cosmos_server(
            port=args.policy.port,
            model_path=args.policy.model_path,
            gpus=args.policy.gpus,
            host="127.0.0.1",
            repo_path=args.policy.repo_path,
            suite=args.policy.suite,
            config=args.policy.config,
            chunk_size=args.policy.chunk_size,
            num_open_loop_steps=args.policy.num_open_loop_steps,
            num_denoising_steps_action=args.policy.num_denoising_steps_action,
            num_third_person_images=args.policy.num_third_person_images,
            seed=args.policy.seed,
            deterministic=args.policy.deterministic,
            t5_embeddings_path=args.policy.t5_embeddings_path,
            dataset_stats_path=args.policy.dataset_stats_path,
            log_dir=run_dir,
        )
    else:
        raise ValueError(
            f"Unsupported policy backend type: {type(args.policy).__name__}"
        )
    processes.append(proc)
    return proc, processes


def _fake_policy_shape(args: Args, env_spec=None) -> tuple[int, int]:
    """``(action_horizon, action_dim)`` a fake policy must emit for this run.

    The horizon has to cover ``--policy.replan-steps``: the runner consumes
    that many actions out of the *first* chunk between inferences, so a
    shorter one leaves it replanning early and warning every time. It never
    shrinks below the class default, so the LIBERO defaults (replan every 5)
    keep the horizon they always had.

    The width comes from the env's own action layout when it is known --
    every env in-tree happens to be 7-D today, but a fake policy that emits
    the wrong width would fail deep inside an env adapter rather than here.
    """
    horizon = max(_FAKE_POLICY_HORIZON, int(args.policy.replan_steps))
    action_dim = _FAKE_POLICY_ACTION_DIM
    if env_spec is not None:
        action_dim = int(env_spec.robot.action_layout.action_dim)
    return horizon, action_dim


def _build_policy_client(
    args: Args, run_dir: pathlib.Path, proc: Any, env_spec=None
) -> Any:
    """Construct the policy client matching the selected backend."""
    if args.guidance.debug:
        horizon, action_dim = _fake_policy_shape(args, env_spec)
        if args.guidance.fake_policy == "circular":
            logger.info(
                "[eval] Using CircularPolicy "
                f"(amplitude={args.guidance.fake_policy_amplitude}, "
                f"period={args.guidance.fake_policy_period_steps} steps)"
            )
            return CircularPolicy(
                amplitude=args.guidance.fake_policy_amplitude,
                period_steps=args.guidance.fake_policy_period_steps,
                gripper_period_steps=args.guidance.fake_policy_gripper_period_steps,
                action_horizon=horizon,
                num_output_actions=action_dim,
            )
        if args.guidance.fake_policy == "noop":
            if _fake_policy_holds_joints(args):
                # Joint-absolute arm: "no-op" is the current joints, not zero
                # (zero is the fully-extended pose), and the guidance loss has
                # to see the chunk through forward kinematics or it reads the
                # joint targets as tool velocities. The live gripper decides
                # where the finger pairs are measured.
                from policy_abstraction.dynamics.ur5_kinematics_jax import (
                    make_ur5e_jax_dynamics,
                )
                from policy_steering.robot_config import load_gripper_geometry

                tcp_offset = _ur5_tcp_offset(args, env_spec)
                geometry_id = (
                    env_spec.robot.gripper.geometry_id
                    if env_spec is not None
                    else args.viz.gripper_model
                )
                logger.info(
                    f"[eval] Using DummyPolicy (no checkpoint loaded); holding "
                    f"the observed joints over {horizon}x{action_dim} per chunk "
                    f"and letting guidance move them through UR5e forward "
                    f"kinematics (tcp_offset={tcp_offset}, gripper={geometry_id})"
                )
                return DummyPolicy(
                    action_horizon=horizon,
                    num_output_actions=action_dim,
                    dynamics_fn=make_ur5e_jax_dynamics(tcp_offset),
                    init_from_state=True,
                    gripper_geometry=(
                        load_gripper_geometry(geometry_id) if geometry_id else None
                    ),
                )
            logger.info(
                f"[eval] Using DummyPolicy (no checkpoint loaded); emitting "
                f"{horizon}x{action_dim} zero actions per chunk, corrected by "
                "guidance when it is enabled"
            )
            return DummyPolicy(
                action_horizon=horizon,
                num_output_actions=action_dim,
            )
        raise ValueError(
            f"Unknown --guidance.fake-policy {args.guidance.fake_policy!r}; "
            "expected 'noop' or 'circular'."
        )

    # Every ZMQ backend below binds loopback by default, so --policy.host
    # (WebSocket-flavoured "0.0.0.0" for openpi) is deliberately ignored.
    # Passing the subprocess handle lets each client's wait-loop bail in
    # seconds if the server died (port collision, missing checkpoint) instead
    # of hanging until its ready timeout expires.
    if isinstance(args.policy, GrootPolicyArgs):
        return GrootPolicyClient(
            host="127.0.0.1",
            port=args.policy.port,
            server_process=proc,
            gripper_remap=args.policy.gripper_remap,
        )
    if isinstance(args.policy, LerobotPolicyArgs):
        return LerobotPolicyClient(
            host="127.0.0.1",
            port=args.policy.port,
            server_process=proc,
            gripper_remap=args.policy.gripper_remap,
            # The runner drains exactly `replan_steps` actions per chunk, so
            # that is the RTC leftover boundary the server needs. Sent
            # unconditionally: the server ignores it when RTC is off, and
            # the client requires a positive int.
            replan_steps=args.policy.replan_steps,
        )
    if isinstance(args.policy, CosmosPolicyArgs):
        return CosmosPolicyClient(
            host="127.0.0.1",
            port=args.policy.port,
            server_process=proc,
            gripper_remap=args.policy.gripper_remap,
            suite=args.policy.suite,
        )
    return PatientWebsocketClient(args.policy.host, args.policy.port)


def start_backends(
    args: Args,
    run_dir: pathlib.Path,
    *,
    install_signal_handlers: bool = True,
) -> EvalBackends:
    """Launch every server, connect every client, and load every model.

    Args:
        args: The parsed evaluation configuration.
        run_dir: Directory that server stdout/stderr logs are written into.
        install_signal_handlers: Register ``atexit`` plus SIGINT/SIGTERM
            handlers that terminate the spawned servers. Leave this on for a
            plain CLI run. A GUI installs its own handlers so it can bring the
            robot home before exiting, and passes ``False``.

    Returns:
        A live :class:`EvalBackends`. The caller owns it and must call
        :meth:`EvalBackends.shutdown`.
    """
    # Bridge the --guidance.attach-objects flag to the PA_ATTACH_OBJECTS env
    # gate read by the guidance builder + HL-relevance (same process, set
    # before any episode runs).
    os.environ["PA_ATTACH_OBJECTS"] = "1" if args.guidance.attach_objects else "0"

    # Armed before the first spawn, not after the last: the handlers close over
    # this list and see whatever is appended later, so arming early also covers
    # a failure *between* two spawns (a port already taken, a missing
    # checkpoint) that would otherwise strand the servers already running.
    server_processes: list = []
    if install_signal_handlers:
        atexit.register(cleanup_servers, server_processes)

        def _signal_handler(signum, frame):
            cleanup_servers(server_processes)
            raise SystemExit(1)

        # SIGHUP matters as much as the other two here: closing the terminal or
        # dropping an SSH session is how a rollout most often ends unplanned,
        # and Python's default disposition for it exits without running atexit.
        for _sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(_sig, _signal_handler)

    policy_proc, policy_processes = _start_policy_backend(args, run_dir)
    server_processes.extend(policy_processes)

    env_proc = None
    if getattr(args.env, "launch_server", True):
        env_proc = start_env_server(args, run_dir)
        server_processes.append(env_proc)
    else:
        logger.info(
            f"--env.no-launch-server: expecting an env server already listening "
            f"on {args.env.host}:{args.env.port}"
        )

    # The handle lets the connect loop fail loudly if the server died at
    # startup instead of retrying against a corpse forever.
    env = WebsocketEnvClient(
        host=args.env.host,
        port=args.env.port,
        server_process=env_proc,
        log_dir=run_dir,
    )
    suite_info = env.setup_suite()
    if isinstance(args.env, LiberoEnvArgs):
        # Server applies perturbation suffixing; mirror its effective name.
        args.env.task_suite_name = suite_info["task_suite_name"]

    policy = _build_policy_client(args, run_dir, policy_proc, env.env_spec)
    retriever, analyzer = build_retrieval_components(args)

    perception_client = None
    flip_u = False
    if args.perception.mode == PerceptionMode.REAL_PERCEPTION:
        # Eager-load SAM3/SAM2/DINO once. Reused across every episode; reset
        # per-episode in RolloutAbstractor.setup().
        from policy_abstraction.perception.open_perception_client import (
            OpenPerceptionClient,
        )

        logger.info("[eval] Initializing OpenPerceptionClient (multi-view, fused)...")
        # env.env_spec was cached during setup_suite() above; the primary
        # view's index in the abstractor's per-view lists must match the
        # `main_camera` index used by ``fuse_views`` re-aliasing.
        primary_view_idx = (
            env.env_spec.primary_camera_idx if env.env_spec is not None else 0
        )
        flip_u = resolve_flip_u(args.perception.flip_u, env.env_spec)
        logger.info(
            f"[eval] Perception back-projection flip_u={flip_u} "
            f"(--perception.flip-u {args.perception.flip_u})"
        )
        perception_client = OpenPerceptionClient(
            reference_hdf5=args.perception.reference_hdf5,
            primary_view_idx=primary_view_idx,
            flip_u=flip_u,
            debug_perception=args.perception.debug,
            detector_backend=args.perception.detector_backend,
            detector_model=args.perception.detector_model,
            vlm_model=args.perception.vlm_model,
        )
        logger.info(
            "[eval] OpenPerceptionClient ready "
            f"(backend={args.perception.detector_backend})."
        )

    # A retrieved demo that carries no ground-truth masks (any real recording)
    # is segmented at analysis time. It gets its **own** perception client:
    # detection is stateful, so sharing the live one would interleave two
    # scenes through a single tracker and scramble the instance ids that the
    # demo->live correspondence step depends on.
    demo_segmenter = None
    if perception_client is not None and args.demo_analysis.enable:
        from policy_abstraction.analysis.demo_segmenter import PerceptionDemoSegmenter
        from policy_abstraction.perception.open_perception_client import (
            OpenPerceptionClient as _OpenPerceptionClient,
        )

        demo_segmenter = PerceptionDemoSegmenter(
            _OpenPerceptionClient(
                reference_hdf5=args.perception.reference_hdf5,
                primary_view_idx=0,
                flip_u=flip_u,
                debug_perception=args.perception.debug,
                detector_backend=args.perception.detector_backend,
                detector_model=args.perception.detector_model,
                vlm_model=args.perception.vlm_model,
            ),
            vlm_model=args.demo_analysis.vlm_model,
            cache_dir=(
                args.demo_analysis.vlm_cache_dir
                if args.demo_analysis.cache_live_objects
                else None
            ),
        )
        logger.info("[eval] Demo segmenter ready (own perception client).")

    live_object_inferer = None
    if args.demo_analysis.enable:
        live_object_inferer = LiveObjectInferer(
            model=args.demo_analysis.vlm_model,
            cache_dir=(
                args.demo_analysis.vlm_cache_dir
                if args.demo_analysis.cache_live_objects
                else None
            ),
        )

    return EvalBackends(
        args=args,
        run_dir=run_dir,
        env=env,
        policy=policy,
        suite_info=suite_info,
        server_processes=server_processes,
        retriever=retriever,
        analyzer=analyzer,
        perception_client=perception_client,
        live_object_inferer=live_object_inferer,
        demo_segmenter=demo_segmenter,
    )


__all__ = [
    "EvalBackends",
    "build_runner_components",
    "cleanup_servers",
    "dynamics_model_for",
    "guidance_action_space_for",
    "guidance_gripper_geometry_for",
    "relation_filter_for",
    "resolve_flip_u",
    "start_backends",
    "start_env_server",
]
