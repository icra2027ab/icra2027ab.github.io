"""Configuration dataclasses for the evaluation entry point.

These live in the library rather than in ``scripts/evaluate.py`` because more
than one entry point needs them. The batch evaluator, the sweep launcher, and
the operator-driven real-world collection GUI in
:mod:`policy_abstraction.experiments` all build the same ``Args`` tree and hand
it to the same :class:`~policy_abstraction.evaluation.backends.EvalBackends`.

``scripts/evaluate.py`` re-exports every name defined here, so its CLI surface
is unchanged: ``tyro.cli(Args)`` still produces the same flags, subcommands and
defaults.
"""
# ruff: noqa: A003  — `object` field on PerturbationArgs intentionally shadows
#                     builtin so the CLI flag stays `--env.perturb.object`
#                     rather than `--env.perturb.object_`.

from __future__ import annotations

import dataclasses
from typing import Annotated, Literal, Optional, Union

import tyro

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.analysis.strategies import StrategyKind
from policy_abstraction.envs.ur5.utils import DEFAULT_ROBOT_IP
from policy_abstraction.evaluation.abstraction import PerceptionMode
from policy_abstraction.evaluation.alignment import AlignmentMode


@dataclasses.dataclass
class _PolicyArgsBase:
    """Fields shared by every policy backend.

    Concrete backend-specific args inherit from this; users select a backend
    via tyro Union dispatch on ``Args.policy``.
    """

    host: str = "0.0.0.0"
    port: Optional[int] = None
    gpus: Optional[str] = None
    replan_steps: int = 5
    # openpi policy launcher selection. Defaults assume the LIBERO checkpoint;
    # other envs run with --guidance.debug today (no checkpoint exists).
    env: str = "LIBERO"
    config: str = "pi05_libero_grounded"
    dir: str = "gs://openpi-assets/checkpoints/pi05_libero"


@dataclasses.dataclass
class OpenpiPolicyArgs(_PolicyArgsBase):
    """OpenPI / pi0_libero backend (the default).

    CLI: ``policy:openpi --policy.<field> ...``.
    """

    dynamics_model_path: Optional[str] = None


@dataclasses.dataclass
class GrootPolicyArgs(_PolicyArgsBase):
    """NVIDIA Isaac GR00T N1.7 backend.

    Spawned from its own venv at ``<repo_path>/.venv``; uses ZMQ rather than
    WebSocket. CLI: ``policy:groot --policy.<field> ...``.
    """

    model_path: Optional[str] = None
    """Required at runtime. Path to a GR00T checkpoint directory (e.g.
    ``checkpoints/GR00T-N1.7-LIBERO/libero_10``)."""
    repo_path: str = "third_party/Isaac-GR00T"
    """Filesystem location of the Isaac-GR00T checkout. The GR00T venv is
    expected at ``<this>/.venv``."""
    embodiment_tag: str = "LIBERO_PANDA"
    gripper_remap: str = "libero_robosuite"
    """Name of the gripper convention to apply to GR00T's ``action.gripper``
    output (see ``policy_abstraction.policies.GRIPPER_REMAPS``). GR00T emits
    RLDS-canonical ``[0, 1]`` (0=close, 1=open); the right remap depends on
    the *target env*, not the policy. ``libero_robosuite`` (default) matches
    LIBERO/robosuite; use ``passthrough`` for envs that already accept the
    canonical form, or extend ``GRIPPER_REMAPS`` for new platforms."""


# Accepted --policy.guidance-action-space values; "auto" is resolved from the
# env by ``backends.guidance_action_space_for``.
GUIDANCE_ACTION_SPACES = frozenset({"auto", "cartesian-delta", "ur5e-joints"})


@dataclasses.dataclass
class LerobotPolicyArgs(_PolicyArgsBase):
    """HuggingFace LeRobot backend (pi0 / pi0.5 / pi0-fast checkpoints).

    Spawned from a venv that has ``lerobot`` installed; uses ZMQ rather than
    WebSocket. CLI: ``policy:lerobot --policy.<field> ...``.

    Guided runs are supported for pi0 / pi0.5 (and smolvla): ``__gp_*__``
    params steer the flow-matching integration server-side, mirroring openpi's
    ``Pi0Grounded``. ``pi0_fast`` has no denoising loop to hook, so pair it
    with ``--guidance.enable False``.
    """

    model_path: Optional[str] = None
    """Required at runtime. Directory holding ``config.json`` +
    ``model.safetensors`` (usually ``checkpoints/<run>/pretrained_model``), or
    a HuggingFace repo id."""
    repo_path: str = "third_party/physical-ai-sdk"
    """Directory whose ``.venv`` has ``lerobot`` installed. Symlink your SDK
    checkout in (``ln -s ~/dev/physical-ai-sdk third_party/physical-ai-sdk``)
    or point this at it directly."""
    policy_type: Optional[str] = None
    """``pi05`` / ``pi0`` / ``pi0_fast`` / ... ``None`` reads the ``type``
    field out of the checkpoint's ``config.json``."""
    device: str = "cuda"
    camera_map: Optional[str] = None
    """``"<view>=<feature_key>,..."`` pinning PA's view names
    (``image``, ``wrist_image``, ...) onto the checkpoint's camera features.
    ``None`` auto-derives them from ``config.json``; the resolved map is
    logged at startup, so check it before trusting a new checkpoint."""
    state_adapter: str = "identity"
    """``identity`` fails when the env's state dim differs from the
    checkpoint's (the usual sign of an env/checkpoint mismatch);
    ``pad_or_truncate`` zero-pads or clips instead."""
    num_inference_steps: Optional[int] = None
    """Override the checkpoint's denoising step count."""
    compile_model: str = "auto"
    """``auto`` honours the checkpoint's ``compile_model`` setting; ``on`` /
    ``off`` force it. Training configs commonly ship
    ``compile_model=true, compile_mode=max-autotune``, which spends minutes
    autotuning inside the first inference (~250 s measured) to buy a ~70 ms
    steady state. Worth it for a full eval, not for a two-episode debug run."""
    guidance: str = "auto"
    """``auto`` steers the denoising loop whenever the run sends ``__gp_*__``
    params (i.e. whenever ``--guidance.enable`` is on); ``off`` makes the
    server refuse guided requests, which pins an unguided baseline even if the
    rest of the pipeline is configured to steer."""
    rtc: bool = True
    """Real-Time Chunking: each new chunk is inpainted to agree with the
    unexecuted tail of the previous one. On by default so every entry point
    (evaluate.py, the experiment runner, the demo) gets chunk-boundary
    continuity; ``--policy.no-rtc`` turns it off. PA steps the env
    synchronously, so continuity is all it buys (there is no inference
    latency to hide). ``rtc_execution_horizon`` must cover ``replan_steps``."""
    rtc_execution_horizon: Optional[int] = None
    """Steps of the new chunk RTC blends with the previous one's leftover.
    ``None`` resolves to ``max(10, replan_steps)``: the blend has to cover
    every step the runner consumes between inferences, and the UR5e scripts
    set both to 25."""
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: str = "linear"
    gripper_remap: str = "passthrough"
    """A LeRobot checkpoint emits actions in its training dataset's
    convention, so a LIBERO-trained one already matches robosuite's
    ``{-1, +1}``. See ``policy_abstraction.policies.GRIPPER_REMAPS``."""
    robot_type: str = ""
    """``robot_type`` handed to LeRobot's inference frame builder for the
    tokenizer step. ``"ur5_follower"`` for the real arm; empty for the
    simulators, whose checkpoints do not condition on it."""
    guidance_action_space: str = "auto"
    """How the server's guidance loss reads an action chunk. ``auto`` picks
    ``ur5e-joints`` for ``env:ur5`` and ``cartesian-delta`` otherwise.
    ``cartesian-delta`` Euler-integrates channels 0-2 as tool velocities (the
    simulators' convention). ``ur5e-joints`` unnormalizes the chunk with the
    checkpoint's action stats and pushes every joint target through the UR5e
    forward kinematics (composed with ``--env.tcp-offset``), so a guided
    chunk is scored where the arm will actually be and starts at the
    end-effector rather than wherever an integrated joint angle lands."""
    launch_server: bool = True
    """Spawn the inference server. Pass ``--policy.no-launch-server`` to attach
    to one you started by hand — which is the usual way to run the real arm, so
    you own the server's terminal and its ~3 minute load is paid once."""

    def __post_init__(self) -> None:
        if self.rtc_execution_horizon is None:
            self.rtc_execution_horizon = max(10, int(self.replan_steps))


# Sentinel defaults for cosmos LIBERO/RoboCasa. Picked off in
# ``CosmosPolicyArgs.__post_init__`` so passing ``--policy.suite robocasa``
# auto-fills the RoboCasa checkpoint/config without forcing the user to
# restate every hyperparameter — but explicit overrides still win.
_COSMOS_LIBERO_DEFAULTS = {
    "model_path": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
    "config": "cosmos_predict2_2b_480p_libero__inference_only",
    "chunk_size": 16,
    "num_open_loop_steps": 16,
    "num_third_person_images": 1,
    "dataset_stats_path": None,
    "t5_embeddings_path": None,
}
_COSMOS_ROBOCASA_DEFAULTS = {
    "model_path": "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B",
    "config": "cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference",
    "chunk_size": 32,
    "num_open_loop_steps": 16,
    "num_third_person_images": 2,
    "dataset_stats_path": (
        "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json"
    ),
    "t5_embeddings_path": (
        "nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl"
    ),
}


@dataclasses.dataclass
class CosmosPolicyArgs(_PolicyArgsBase):
    """NVIDIA Cosmos-Predict2 (Cosmos-Policy) backend.

    Spawned from its own venv at ``<repo_path>/.venv`` (Python 3.10 + torch
    2.7.0+cu128 + flash-attn); uses ZMQ rather than WebSocket. CLI:
    ``policy:cosmos --policy.<field> ...``.

    Pass ``--policy.suite robocasa`` to auto-fill the RoboCasa checkpoint,
    config, chunk size and dataset-stats/T5 paths. Explicit overrides win
    (e.g. ``--policy.suite robocasa --policy.chunk-size 16`` keeps 16).
    """

    suite: Literal["libero", "robocasa"] = "libero"
    """Which Cosmos eval suite this policy targets. Must match the env type
    (``env:libero`` vs ``env:robocasa``). Drives observation schema
    (RoboCasa adds a third "secondary" camera) and the lookup of default
    checkpoint/config/stats paths."""

    model_path: Optional[str] = None
    """HF repo id or local checkpoint dir. ``None`` selects the suite's
    canonical checkpoint."""
    repo_path: str = "third_party/cosmos-policy"
    """Filesystem location of the cosmos-policy checkout. The cosmos venv is
    expected at ``<this>/.venv`` (built with
    ``uv sync --extra cu128 --group libero --python 3.10`` inside the repo,
    plus ``uv pip install msgpack pyzmq tyro``). The libero-group venv also
    serves RoboCasa inference — `cosmos_utils.py` doesn't import
    `robosuite`/`mujoco`/`robocasa` on the inference path."""
    config: Optional[str] = None
    """Cosmos experiment config name. ``None`` selects the suite's default."""
    chunk_size: Optional[int] = None
    """``None`` selects the suite default (LIBERO=16, RoboCasa=32)."""
    num_open_loop_steps: Optional[int] = None
    """``None`` selects the suite default (16 for both)."""
    num_third_person_images: Optional[int] = None
    """How many third-person cameras the server consumes. ``None`` selects
    the suite default (LIBERO=1, RoboCasa=2). The RoboCasa checkpoint
    asserts ``num_third_person_images=2``."""
    num_denoising_steps_action: int = 5
    seed: int = 195
    deterministic: bool = True
    t5_embeddings_path: Optional[str] = None
    """Override the server's default T5 cache pickle path. ``None`` selects
    the suite's bundled cache. Cache misses fall through to on-the-fly T5."""
    dataset_stats_path: Optional[str] = None
    """Override the server's default dataset stats JSON path. ``None``
    selects the suite's bundled stats."""
    gripper_remap: str = "passthrough"
    """Cosmos emits robosuite-style ``{-1, +1}`` gripper actions for both
    LIBERO and RoboCasa, so the default is identity. See
    ``policy_abstraction.policies.groot.GRIPPER_REMAPS``."""

    def __post_init__(self) -> None:
        defaults = (
            _COSMOS_ROBOCASA_DEFAULTS
            if self.suite == "robocasa"
            else _COSMOS_LIBERO_DEFAULTS
        )
        for key, val in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, val)


# Tyro Union dispatch: select with ``policy:openpi``, ``policy:groot``,
# ``policy:cosmos`` or ``policy:lerobot``.
# To add another backend, define a new ``XxxPolicyArgs(_PolicyArgsBase)``
# and append it to this Union with its own ``tyro.conf.subcommand`` name.
PolicyArgs = Union[
    Annotated[OpenpiPolicyArgs, tyro.conf.subcommand(name="openpi")],
    Annotated[GrootPolicyArgs, tyro.conf.subcommand(name="groot")],
    Annotated[CosmosPolicyArgs, tyro.conf.subcommand(name="cosmos")],
    Annotated[LerobotPolicyArgs, tyro.conf.subcommand(name="lerobot")],
]


@dataclasses.dataclass
class PerturbationArgs:
    """LIBERO-PRO perturbation flags."""

    swap: bool = True
    object: bool = False
    language: bool = False
    task: bool = False
    environment: bool = False


@dataclasses.dataclass
class LiberoEnvArgs:
    host: str = "localhost"
    port: Optional[int] = None
    num_steps_wait: int = 10
    task_suite_name: str = "libero_goal"
    evaluation_config_path: str = str(
        PROJECT_ROOT
        / "src/policy_abstraction/envs/libero/configs/evaluation_config.yaml"
    )
    perturb: PerturbationArgs = dataclasses.field(default_factory=PerturbationArgs)


@dataclasses.dataclass
class RobocasaEnvArgs:
    host: str = "localhost"
    port: Optional[int] = None
    num_steps_wait: int = 10
    task: Optional[str] = None
    tasks: Optional[list[str]] = None
    # Leaderboard task suite — one of ``atomic_seen``, ``composite_seen``,
    # ``composite_unseen``. Mutually exclusive with --task and --tasks.
    # Resolves the canonical task list (and per-task horizons) server-side
    # from ``robocasa.utils.dataset_registry.TASK_SET_REGISTRY``. Also
    # defaults --split to ``target`` to match the leaderboard.
    task_suite: Optional[str] = None
    suite_name: str = "robocasa_atomic"
    robots: str = "PandaOmron"
    split: Optional[str] = None
    num_init_states: int = 20
    primary_camera: str = "robot0_agentview_left"
    include_right_view: bool = True
    # Per-episode step cap; ``None`` keeps the server default (per-task
    # leaderboard horizons when --task-suite is set, else a uniform fallback).
    # When explicitly set, this value uniformly caps every task.
    max_steps: Optional[int] = None


@dataclasses.dataclass
class CalvinEnvArgs:
    host: str = "localhost"
    port: Optional[int] = None
    num_steps_wait: int = 10
    scenes: list[str] = dataclasses.field(default_factory=lambda: ["calvin_scene_D"])
    suite_name: str = "calvin"
    cameras_cfg: str = "static_and_gripper"
    robot_cfg: str = "panda_longer_finger"
    num_init_states: int = 10
    primary_camera: str = "static"
    include_gripper: bool = True
    # Per-episode step cap; ``None`` keeps the server default.
    max_steps: Optional[int] = None


@dataclasses.dataclass
class Ur5EnvArgs:
    """Real UR5e cell served by ``scripts/servers/serve_ur5_env.py``.

    This is the only non-simulated env: ``step`` drives a physical arm through
    ``servoJ``. Read ``docs/ur5.md`` before the first run — in particular the
    ``--dry-run`` / ``--simulate-hardware`` bring-up path and the note about
    keeping the per-step overhead low enough to hold 50 Hz.
    """

    host: str = "localhost"
    port: Optional[int] = None
    num_steps_wait: int = 0
    """Settle steps before the policy takes over. Zero on hardware: the
    runner's dummy action is an all-zeros *joint target*, which the server has
    to special-case as "hold" rather than obey."""
    tasks: list[str] = dataclasses.field(
        default_factory=lambda: ["Put all cables in the bag."]
    )
    """Instructions, one per task_id. Must match what the checkpoint was
    trained on — this string is the policy's prompt."""
    suite_name: str = "ur5"
    robot_ip: str = DEFAULT_ROBOT_IP
    cameras: Optional[list[str]] = None
    """``name=/dev/videoN[:WxH][@FPS]`` specs. Names must match the
    checkpoint's camera keys. ``None`` uses the UR5e defaults."""
    wrist_camera: Optional[str] = None
    primary_camera: Optional[str] = None
    control_hz: Optional[float] = None
    """Rate the env server paces ``step`` at. Must match the policy's training
    rate (50 Hz for the pi0/pi05 UR5e checkpoints)."""
    max_steps: Optional[int] = None
    num_init_states: int = 10
    max_joint_delta: Optional[float] = None
    """Safety clamp, radians per step, relative to the measured joints."""
    flip_gripper: bool = True
    """Invert the Robotiq 0=open convention into openness. Almost always right
    for the pi0/pi05 UR5e checkpoints."""
    home_on_reset: bool = True
    home_on_disconnect: bool = True
    """Park the arm at home when the evaluation disconnects. Keep this on: the
    shutdown path only allows the server a few seconds to exit, which is not
    enough for a blocking homing move."""
    success_mode: str = "never"
    """``never`` or ``stdin``. ``stdin`` only works with
    ``--env.no-launch-server`` and a hand-started server."""
    dry_run: bool = False
    """Connect to the arm and read sensors, but never command motion."""
    simulate_hardware: bool = False
    """Don't touch the robot; serve synthetic observations. Pipeline smoke
    tests with the arm powered off."""
    tcp_offset: Optional[list[float]] = None
    """``x y z rx ry rz`` TCP offset from the UR teach pendant."""
    calibration_path: Optional[str] = None
    calibration_base_frame: Optional[str] = None
    """``base`` (RTDE's frame) or ``base_link`` (ROS/URDF, 180 deg about z).
    Kalib-style calibrations are usually ``base_link``; using the wrong one
    mirrors the reconstruction in x/y without tilting the table."""
    sdk_path: str = "~/dev/physical-ai-sdk"
    launch_server: bool = True
    """Spawn the env server. Pass ``--env.no-launch-server`` to attach to one
    you started by hand (recommended on hardware, so the operator owns the
    arm's terminal and can use ``--success-mode stdin``)."""


EnvArgs = Union[LiberoEnvArgs, RobocasaEnvArgs, CalvinEnvArgs, Ur5EnvArgs]


def env_kind(env_args) -> str:
    if isinstance(env_args, LiberoEnvArgs):
        return "libero"
    if isinstance(env_args, RobocasaEnvArgs):
        return "robocasa"
    if isinstance(env_args, CalvinEnvArgs):
        return "calvin"
    if isinstance(env_args, Ur5EnvArgs):
        return "ur5"
    raise TypeError(f"Unknown env args type: {type(env_args).__name__}")


def eval_type_for(env_args) -> str:
    if isinstance(env_args, LiberoEnvArgs):
        parts = [
            f.name
            for f in dataclasses.fields(env_args.perturb)
            if getattr(env_args.perturb, f.name)
        ]
        return "_".join(parts)
    if isinstance(env_args, RobocasaEnvArgs):
        if env_args.task is not None:
            return env_args.task
        if env_args.task_suite is not None:
            return env_args.task_suite
        return "_".join(env_args.tasks or [])
    if isinstance(env_args, CalvinEnvArgs):
        return "_".join(env_args.scenes)
    if isinstance(env_args, Ur5EnvArgs):
        # Instructions are free-form sentences and this ends up in a path;
        # slugify so "Put all cables in the bag." stays one readable segment.
        return (
            "_".join(
                "".join(ch if ch.isalnum() else "_" for ch in task).strip("_")
                for task in env_args.tasks
            )
            or "ur5"
        )
    raise TypeError(f"Unknown env args type: {type(env_args).__name__}")


@dataclasses.dataclass
class CameraArgs:
    height: int = 256
    width: int = 256
    resize_size: int = 224


@dataclasses.dataclass
class TaskIterArgs:
    num_trials_per_task: int = 1
    start_trial_idx: int = 0
    start_task_idx: int = 0
    max_tasks: Optional[int] = None


@dataclasses.dataclass
class RetrievalArgs:
    """Retrieval — needs *_embeddings.npz caches (see scripts/encode_datasets.py)."""

    enable: bool = True
    dataset_dirs: Optional[list[str]] = None
    auto_encode: bool = True
    """Encode any dataset dir that has demos but no ``*_embeddings.npz`` cache,
    instead of failing the run. The encoding is mechanical and uses the same
    DINOv2/CLIP models retrieval is about to load anyway, so the default is to
    just do it. ``--retrieval.no-auto-encode`` restores the hard failure, with
    the exact ``scripts/encode_datasets.py`` command in the message."""
    dino_model: str = "facebook/dinov2-base"
    clip_model: str = "openai/clip-vit-base-patch16"


@dataclasses.dataclass
class IterativeAnalysisArgs:
    """Knobs for the iterative pre/post-condition strategy.

    Only consulted when ``DemoAnalysisArgs.strategy == StrategyKind.ITERATIVE``.
    """

    max_iterations: int = 3
    consistency_metric: str = "std_after_align"
    """Either ``"std_after_align"`` (post-DTW std, default) or ``"mean_dtw"``
    (mean DTW path distance)."""
    top_k_for_verification: int = 5
    min_segment_length: int = 5
    accept_score_threshold: Optional[float] = None
    """Early-stop the iteration loop if a proposal scores at or below this
    aggregate threshold. ``None`` disables early-stop."""


@dataclasses.dataclass
class DemoAnalysisArgs:
    """Demo analysis — requires retrieval.enable=True and OPENAI_API_KEY."""

    enable: bool = True
    vlm_model: str = "gpt-5.4"
    vlm_cache_dir: Optional[str] = None
    cache_pairs: bool = True
    """If False, distance-pair selection bypasses the VLM cache (always re-queries)."""
    cache_live_objects: bool = True
    """If False, live object inference bypasses the VLM cache (always re-queries)."""
    strategy: StrategyKind = StrategyKind.SINGLE_TURN
    """Which pair/constraint proposal strategy to use. Default ``single_turn``
    preserves today's behaviour bit-for-bit; ``iterative`` enables the
    pre/post + DTW verification loop."""
    iterative: IterativeAnalysisArgs = dataclasses.field(
        default_factory=IterativeAnalysisArgs
    )


@dataclasses.dataclass
class GuidanceArgs:
    enable: bool = True
    factor: float = 1.0
    # When true, DISABLES THE POLICY AND ONLY USES THE GUIDANCE, for testing
    # and visualization purposes.
    debug: bool = False
    aligned_t_offset: int = 5
    # Guidance-correction step: L2-normalize the per-sample gradient to a unit
    # ascent direction before applying. Disable with ``--guidance.no-normalize``
    # to use the raw gradient (magnitude scales with the loss surface, so
    # ``factor`` then behaves as a classic gradient-descent step size).
    normalize: bool = True
    # Golden-section line search along the (normalized or raw) descent
    # direction. Disable with ``--guidance.no-line-search`` to take a fixed
    # step of size ``factor`` (or ``factor * ||grad||`` when normalize=False).
    line_search: bool = True
    # Initial upper bound for the golden-section line search bracket. The
    # search may further expand the bracket (up to ~32x) before narrowing.
    # Raise above 1.0 when ``normalize=False`` so the search can reach a
    # meaningful step size despite the small raw-gradient magnitude.
    max_alpha: float = 1.0
    # Per-pair multi-alpha guidance (GR00T/torch head only). When True, each
    # distance pair gets its own descent direction + line-searched step instead
    # of a single shared alpha. Default False = unchanged single-alpha path.
    multi_alpha: bool = False
    multi_alpha_combine: Literal["sequential", "sum", "weighted", "per_timestep"] = (
        "sequential"
    )
    # Object-to-object (EEF-attached) distance pairs. When True (default), an
    # object-to-object pair (e.g. bowl->plate) is kept by treating obj1 as
    # rigidly attached to the EEF so gradients flow to the EEF pose. Disable
    # with ``--guidance.no-attach-objects`` to drop such pairs (legacy
    # behavior) — implemented via the ``PA_ATTACH_OBJECTS`` env gate read by
    # both the guidance builder and the HL-relevance computation.
    attach_objects: bool = True
    # Per-relation ablation gates. Each toggles whether one relation category
    # contributes to the guidance gradient: positional (centroid/surface/axis/
    # plane distance pairs), orientation (EE approach-direction projection), and
    # gripper (aperture / finger pairs). All True (default) keeps the unfiltered
    # path bit-for-bit. Disable one with e.g. ``--guidance.no-relation-orientation``
    # to measure that category's contribution.
    relation_positional: bool = True
    relation_orientation: bool = True
    relation_gripper: bool = True
    # High-level relevance thresholds (m) used by compute_hl_relevances.
    # Relevance is 0 at/below `relevance_near_threshold`, ramps linearly to 1
    # by `relevance_far_threshold`, and is clamped to [0, 1].
    relevance_near_threshold: float = 0.05
    relevance_far_threshold: float = 0.15
    # Selects the in-process fake policy when ``debug`` is set:
    # ``"noop"`` returns zero actions (default), ``"circular"`` returns
    # 7-D deltas tracing a circle in the xy-plane plus a periodic gripper
    # toggle — useful as a visualization sanity check for new env adapters.
    fake_policy: str = "noop"
    # Normalized amplitude in [0, 1] for the circular fake policy. Each env's
    # controller scales this to physical motion (robosuite OSC default is
    # 0.05 m/step at amplitude 1; CALVIN's cartesian-rel controller is
    # similar), so this knob is unitless and portable across envs.
    fake_policy_amplitude: float = 0.3
    fake_policy_period_steps: int = 80
    fake_policy_gripper_period_steps: int = 40


@dataclasses.dataclass
class AlignmentArgs:
    mode: AlignmentMode = AlignmentMode.MIXED
    # max aligned_t advance per step, in action chunks
    max_advance_chunks: Optional[float] = 1.2
    # weight for sub-trajectory time dimension in alignment
    subtraj_time_weight: float = 0.0
    mixed_variation: int = 5
    # softmax temperature for per-feature guidance weights
    temporal_factor_temperature: float = 5.0
    # minimum guidance weight as fraction of base_factor
    temporal_factor_floor: float = 0.05
    # consecutive timesteps sub-goal must be true/false before switching
    sg_switch_persistence: int = 10


@dataclasses.dataclass
class SubtrajArgs:
    """Sub-trajectory segmentation (shared between DemoAnalyzer & RolloutAbstractor)."""

    gripper_speed_threshold: float = 0.005
    min_length: int = 20


@dataclasses.dataclass
class PerceptionArgs:
    mode: PerceptionMode = PerceptionMode.GT_SEG_RECONSTRUCT
    # Whether `reconstruct`/`fuse_views` mirror the u axis when back-projecting.
    # It compensates for the abstractor pre-flipping frames: the simulator
    # adapters publish 180°-rotated RGB/depth (CameraSpec.image_transform
    # "flip_vh") and the abstractor `[::-1, ::-1]`s them before perception sees
    # them, so u must be mirrored to land on the right pixels. Envs that publish
    # upright frames (image_transform "none", e.g. the real UR5e) must not flip
    # — doing so mirrors every reconstructed object about the optical axis,
    # measured at ~0.8 m of displacement on the UR5 cell.
    #   "auto"  — derive from the primary camera's image_transform (default;
    #             correct for every env in-tree, and keeps sim behaviour
    #             bit-for-bit)
    #   "true"/"false" — force it, for a camera whose transform doesn't imply
    #             the right answer.
    flip_u: Literal["auto", "true", "false"] = "auto"
    # Box-proposal transport for REAL_PERCEPTION's detect.gemini stage:
    #   "openrouter" (default) — OpenAI-wire model via OpenRouter
    #     (needs OPENROUTER_API_KEY); model id from --perception.vlm-model.
    #   "gemini" — Google native API (needs GEMINI_API_KEY/GOOGLE_API_KEY);
    #     required for robotics-ER models, which are NOT on OpenRouter.
    detector_backend: Literal["gemini", "openrouter"] = "openrouter"
    # Box-detector model id for the chosen backend. None -> backend default
    # (openrouter: --perception.vlm-model; gemini: gemini-robotics-er-1.6-preview).
    # For the gemini backend pass the NATIVE id, e.g. "gemini-robotics-er-1.6-preview".
    detector_model: Optional[str] = None
    # OpenRouter model id for the open_perception VLM box-detection stage
    # (detect.gemini, used by REAL_PERCEPTION when detector_backend=openrouter).
    # Passed verbatim; requires OPENROUTER_API_KEY. Only the box detector uses
    # this — demo analysis has its own --demo-analysis.vlm-model.
    vlm_model: str = "google/gemini-3-flash-preview"
    # Override for the DINO reference bank used by open_perception's
    # disambiguate.dino stage. Defaults to <project_root>/datasets/all_objects.hdf5.
    # Unused by the default detect.gemini → sam2 pipeline (kept for legacy paths).
    reference_hdf5: Optional[str] = None
    # Render the per-stage open_perception debug panel (matplotlib rendering of
    # instances/detections/scores per pipeline stage) and surface it on the
    # rerun Perception tab. Off by default — the panel is heavy to render.
    debug: bool = False


@dataclasses.dataclass
class VisualizerArgs:
    enable: bool = True
    connect: Optional[str] = "127.0.0.1:9876"
    # Fail fast when no viewer is listening at ``connect``. Off by default:
    # rr.connect_grpc never raises, so a headless run logs into the void and
    # completes normally — which is fine for the simulator suites that run
    # unattended, and infuriating for a robot rollout you are standing next to.
    # Opt in for the latter (scripts/run_ur5_rollout.sh --viz does).
    require_viewer: bool = False
    robot_model: Optional[str] = None
    robot_description: Optional[str] = "panda_mj_description"
    # Menagerie publishes its arms bare, so a UR5e renders as a tool flange
    # with nothing on it unless a gripper model is grafted on. Auto-selected
    # for env:ur5 below; pass ``None`` to render the arm alone.
    gripper_model: Optional[str] = None
    gripper_mount_yaw_deg: float = 0.0
    """How the gripper is bolted onto the flange, as a rotation about the tool
    axis. Menagerie's default is only one of the ways a gripper can be mounted;
    rendered at the wrong one, its fingers open across the real ones. It has
    to agree with the ``finger_axis`` of the gripper geometry YAML, which is
    where the fingertips guidance measures distances from come from: measured
    against the cell (2026-09-02), the UR5e's 2F-140 opens along tool x, which
    is menagerie's default mount (0°) and ``finger_axis: x``. The previous
    90° here matched a y-axis FK, and both were a quarter turn off the real
    gripper."""
    # Record the full-scene fused point cloud (every valid-depth pixel, not
    # just the relevant objects) per replan so it can be shown as a separate
    # ``scene/full_pcd`` overlay when replaying the .vla. Off by default — it
    # back-projects the whole depth image and noticeably enlarges recordings.
    record_scene_pointcloud: bool = False
    # Stride-subsample cap for the recorded scene cloud (points per replan).
    scene_pointcloud_max_points: int = 30000
    serve: bool = False
    """Serve the recording from this process — a gRPC server plus the Rerun
    web viewer — instead of connecting to or spawning a viewer. Needs no
    ``rerun`` binary and gives a URL the demo window embeds as its 3-D panel.
    Wins over ``connect``."""
    serve_grpc_port: Optional[int] = 9876
    serve_web_port: Optional[int] = 9090
    layout: Literal["full", "scene"] = "full"
    """Which viewer layout to install: ``full`` is the analysis layout (3-D
    scene, cameras, retrieval, charts); ``scene`` is the 3-D scene alone with
    every panel collapsed — the robot model, object clouds, distance pairs and
    action chunks filling the window, for a projector. The demo GUI switches
    between the two at runtime."""


@dataclasses.dataclass
class OutputArgs:
    run: str = "test_mapping"
    exp_name: Optional[str] = None
    summary_out_path: Optional[str] = None
    video_out_path: Optional[str] = None
    episode_data_path: Optional[str] = None
    log_output_path: Optional[str] = None
    save_episodes: bool = True
    depth_stride: int = 5
    """Record metric depth into the .vla every N steps (0 disables).

    Depth is what lets a recorded rollout be re-used as a retrieval demo: the
    guidance pipeline needs 3-D object positions, which need depth plus the
    camera's intrinsic and extrinsic. It is stored uncompressed (~0.6 MB per
    camera-frame at 640x480 against ~4 KB for the h264 colour stream), so it
    is written on a stride rather than every step. A camera with no depth in
    the observation, or with no extrinsic (i.e. no --env.calibration-path),
    contributes nothing."""
    # When True, resume from any existing summary at ``summary_out_path`` and
    # skip task IDs already recorded there. When False (default), ignore any
    # existing summary and re-run every task from scratch (overwriting it).
    skip_completed: bool = False


@dataclasses.dataclass
class Args:
    """Arguments for multi-env evaluation (LIBERO-PRO / RoboCasa / CALVIN / UR5e)."""

    instance: int = 2  # evaluation instance for parallel runs
    seed: int = 7
    guide: bool = True
    """Master switch for the guidance pipeline. ``--no-guide`` disables
    retrieval, demo analysis, and guidance in one go (equivalent to
    ``--retrieval.no-enable --demo-analysis.no-enable --guidance.no-enable``)."""

    policy: PolicyArgs = dataclasses.field(default_factory=OpenpiPolicyArgs)
    env: Union[
        Annotated[LiberoEnvArgs, tyro.conf.subcommand("libero")],
        Annotated[RobocasaEnvArgs, tyro.conf.subcommand("robocasa")],
        Annotated[CalvinEnvArgs, tyro.conf.subcommand("calvin")],
        Annotated[Ur5EnvArgs, tyro.conf.subcommand("ur5")],
    ] = dataclasses.field(default_factory=LiberoEnvArgs)
    camera: CameraArgs = dataclasses.field(default_factory=CameraArgs)
    iter: TaskIterArgs = dataclasses.field(default_factory=TaskIterArgs)
    retrieval: RetrievalArgs = dataclasses.field(default_factory=RetrievalArgs)
    demo_analysis: DemoAnalysisArgs = dataclasses.field(
        default_factory=DemoAnalysisArgs
    )
    guidance: GuidanceArgs = dataclasses.field(default_factory=GuidanceArgs)
    alignment: AlignmentArgs = dataclasses.field(default_factory=AlignmentArgs)
    subtraj: SubtrajArgs = dataclasses.field(default_factory=SubtrajArgs)
    perception: PerceptionArgs = dataclasses.field(default_factory=PerceptionArgs)
    viz: VisualizerArgs = dataclasses.field(default_factory=VisualizerArgs)
    output: OutputArgs = dataclasses.field(default_factory=OutputArgs)

    def __post_init__(self):
        def get(val, default):
            return val if val is not None else default

        if not self.guide:
            self.retrieval.enable = False
            self.demo_analysis.enable = False
            self.guidance.enable = False

        # Defense-in-depth mutual-exclusion check for RoboCasa task selectors.
        # The launcher and server-entry CLI each re-validate, but failing here
        # short-circuits the heavy env-server spin-up.
        if isinstance(self.env, RobocasaEnvArgs):
            _robocasa_selectors = sum(
                x is not None
                for x in (self.env.task, self.env.tasks, self.env.task_suite)
            )
            if _robocasa_selectors != 1:
                raise ValueError(
                    "RoboCasa needs exactly one of --env.task, --env.tasks, or "
                    "--env.task-suite "
                    "<atomic_seen|composite_seen|composite_unseen>; "
                    f"got task={self.env.task!r}, tasks={self.env.tasks!r}, "
                    f"task_suite={self.env.task_suite!r}"
                )

        eval_type = eval_type_for(self.env)
        kind = env_kind(self.env)

        self.policy.port = get(self.policy.port, 8000 + self.instance)
        self.policy.gpus = get(self.policy.gpus, str(self.instance))
        # Default the GR00T checkpoint to the task-suite-matched directory
        # under ``checkpoints/GR00T-N1.7-LIBERO/`` when running a libero suite.
        # The user can still override via ``--policy.model-path``.
        if (
            isinstance(self.policy, GrootPolicyArgs)
            and self.policy.model_path is None
            and self.env.task_suite_name.startswith("libero")
        ):
            self.policy.model_path = (
                f"checkpoints/GR00T-N1.7-LIBERO/{self.env.task_suite_name}"
            )
        self.env.port = get(self.env.port, 8765 + self.instance)

        if isinstance(self.policy, LerobotPolicyArgs):
            if self.policy.guidance_action_space not in GUIDANCE_ACTION_SPACES:
                raise ValueError(
                    "--policy.guidance-action-space must be one of "
                    f"{sorted(GUIDANCE_ACTION_SPACES)}; got "
                    f"{self.policy.guidance_action_space!r}."
                )

        if isinstance(self.policy, LerobotPolicyArgs) and self.policy.rtc:
            horizon = self.policy.rtc_execution_horizon
            assert horizon is not None  # resolved in LerobotPolicyArgs.__post_init__
            if horizon < self.policy.replan_steps:
                raise ValueError(
                    "RTC blends the new chunk over "
                    "--policy.rtc-execution-horizon "
                    f"({horizon}) steps, which must "
                    f"cover the --policy.replan-steps "
                    f"({self.policy.replan_steps}) the runner consumes between "
                    "inferences."
                )

        if isinstance(self.env, Ur5EnvArgs):
            # The pi0/pi05 UR5e checkpoints were trained on native 640x480
            # frames. Keeping the wire resolution native means the single
            # resize-with-pad down to `resize_size` matches training exactly,
            # instead of squeezing through a square intermediate first.
            if (self.camera.height, self.camera.width) == (
                CameraArgs.height,
                CameraArgs.width,
            ):
                self.camera.height, self.camera.width = 480, 640
            if self.viz.robot_description == VisualizerArgs.robot_description:
                self.viz.robot_description = "ur5e_mj_description"
            if self.viz.gripper_model == VisualizerArgs.gripper_model:
                self.viz.gripper_model = "robotiq_2f140"
            # gripper_mount_yaw_deg stays at menagerie's default: that is how
            # the 2F-140 sits on this cell's flange (jaws along tool x, see
            # robotiq_2f140.yaml's finger_axis).
            if isinstance(self.policy, LerobotPolicyArgs):
                # `robot_type` only: it is the one field whose class default
                # ("") is never a deliberate choice, so overriding it cannot
                # surprise anyone. The rest of the real-arm tuning
                # (--policy.rtc, --policy.replan-steps,
                # --policy.rtc-execution-horizon) is set explicitly by
                # scripts/run_ur5_rollout.sh instead of inferred here: a
                # "differs from the class default" test cannot tell "unset"
                # from "explicitly set to the default", so auto-applying `rtc`
                # would silently ignore --policy.no-rtc.
                if self.policy.robot_type == LerobotPolicyArgs.robot_type:
                    self.policy.robot_type = "ur5_follower"
            if self.perception.mode != PerceptionMode.REAL_PERCEPTION and (
                self.demo_analysis.enable or self.guidance.enable
            ):
                raise ValueError(
                    f"--perception.mode {self.perception.mode.name} needs depth "
                    "*and* ground-truth segmentation from a simulator. The UR5e "
                    "publishes neither; use --perception.mode REAL_PERCEPTION."
                )

        if isinstance(self.env, LiberoEnvArgs):
            default_exp_name = f"{self.env.task_suite_name}/{eval_type}"
        else:
            default_exp_name = f"{kind}/{eval_type}"
        self.output.exp_name = get(self.output.exp_name, default_exp_name)
        run, exp_name = self.output.run, self.output.exp_name
        self.output.summary_out_path = get(
            self.output.summary_out_path, f"data/{run}/{exp_name}/summary.json"
        )
        self.output.video_out_path = get(
            self.output.video_out_path, f"data/{run}/{exp_name}/videos"
        )
        self.output.episode_data_path = get(
            self.output.episode_data_path, f"data/{run}/{exp_name}/episodes"
        )
        self.output.log_output_path = get(
            self.output.log_output_path, f"data/{run}/{exp_name}/evaluation.log"
        )
        # Per-env retrieval defaults: point at the right source root when
        # the user hasn't overridden --retrieval.dataset-dirs. The encoder
        # writes ``*_embeddings.npz`` next to the source data and the
        # DatasetIndex globs recursively.
        if self.retrieval.dataset_dirs is None:
            if isinstance(self.env, RobocasaEnvArgs):
                self.retrieval.dataset_dirs = [
                    str(PROJECT_ROOT / "datasets/robocasa/v1.0/pretrain/atomic")
                ]
            elif isinstance(self.env, CalvinEnvArgs):
                self.retrieval.dataset_dirs = [
                    str(PROJECT_ROOT / "datasets/calvin/calvin_debug_dataset/training")
                ]
            # libero falls through to the default in
            # evaluation/retrieval.build_retrieval_components.
        self.demo_analysis.vlm_cache_dir = get(
            self.demo_analysis.vlm_cache_dir, f"data/{run}/vlm_cache/"
        )


__all__ = [
    "AlignmentArgs",
    "Args",
    "CalvinEnvArgs",
    "CameraArgs",
    "CosmosPolicyArgs",
    "DemoAnalysisArgs",
    "EnvArgs",
    "GrootPolicyArgs",
    "GuidanceArgs",
    "IterativeAnalysisArgs",
    "LerobotPolicyArgs",
    "LiberoEnvArgs",
    "OpenpiPolicyArgs",
    "OutputArgs",
    "PerceptionArgs",
    "PerturbationArgs",
    "PolicyArgs",
    "RetrievalArgs",
    "RobocasaEnvArgs",
    "SubtrajArgs",
    "TaskIterArgs",
    "Ur5EnvArgs",
    "VisualizerArgs",
    "env_kind",
    "eval_type_for",
]
