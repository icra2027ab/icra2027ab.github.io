"""ZMQ REQ/REP server wrapping a LeRobot policy (pi0 / pi0.5 / pi0-fast).

Runs **inside** a venv that has ``lerobot`` installed (e.g.
``third_party/physical-ai-sdk/.venv``), because lerobot pins
``transformers>=5.3,<5.4`` plus its own torch build, which collide with the
project root venv. The wire protocol mirrors
[`scripts/servers/serve_cosmos.py`](./serve_cosmos.py): msgpack frames with a
custom ndarray codec, REQ/REP, endpoints ``ping`` / ``info`` / ``get_action`` /
``reset``.

The host venv needs ``pyzmq``, ``msgpack`` and ``tyro`` on top of ``lerobot``,
plus ``packages/policy_steering/src`` on ``PYTHONPATH`` for the guidance kernel
(``start_lerobot_server`` arranges that). Nothing from ``policy_abstraction``
itself is imported, so no further bridging is needed.

Expected message shapes::

  Request : {"endpoint": "get_action",
             "data": {"observation": {
                 "images": {"<view>": uint8 HxWx3, ...},  # e.g. image, wrist_image
                 "state":  float32 (D,),
                 "task":   str,
                 "consumed_steps": int,   # actions PA executes per chunk (RTC)
             },
             "options": {"__gp_*__": ...} | None}}   # guidance params
  Reply   : {"actions": float32 (T, action_dim),
             "policy_timing": {"infer_ms": float}}

The ``images`` keys are PA's *logical* view names. This server maps them onto
the checkpoint's own ``observation.images.*`` feature keys — automatically from
``config.json`` (wrist-vs-third-person by name), or explicitly via
``--camera-map "image=observation.images.exterior_image_1_left,..."``. The
resolved map is printed at first inference and echoed by the ``info`` endpoint,
so a wrong pairing is visible in the logs rather than silently degrading the
rollout.

Real-Time Chunking (``--rtc``) is served statefully: the raw previous chunk is
kept and, on the next request, its unexecuted tail (everything past
``consumed_steps``) is passed as ``prev_chunk_left_over`` so the new chunk is
inpainted to agree with it. ``inference_delay`` is 0 because PA holds the
simulator still while the policy runs. ``reset`` clears that buffer and must be
called between episodes.

Guidance
--------
PA's ``__gp_*__`` params ride in ``options`` and steer the flow-matching
integration, mirroring openpi's ``Pi0Grounded.sample_actions`` (see
``third_party/openpi/src/openpi/models/pi0grounded.py``): the guidance loss and
gradient kernels are hoisted once per request, then every denoising step
replaces ``v_t`` with ``compute_guidance_velocity(...)`` before the Euler
update.

Rather than copying lerobot's denoising loop the way the GR00T backend copies
GR00T's, this hooks the seam lerobot already provides: ``sample_actions`` routes
``v_t`` through ``self.rtc_processor.denoise_step(...)`` when an ``RTCConfig``
is present, handing it the un-guided ``original_denoise_step_partial``. A
:class:`GuidedRTCProcessor` subclass therefore sees every step of pi0, pi0.5 and
smolvla alike with no per-model copy, and composes with RTC instead of
competing with it. ``sample_actions`` is decorated ``@torch.no_grad()`` rather
than ``inference_mode``, so the guidance kernel's ``torch.enable_grad()`` works
inside it — the same property that forced ``groot_grounded_policy.py`` to swap
GR00T's decorator.

Guidance is applied to the model's *internal* action tensor, i.e. before the
post-processor unnormalizes it — matching both existing grounded backends. See
``ACTION normalization`` in the startup log: with anything other than
``IDENTITY``, ``--guidance.factor`` is expressed in normalized units and may
need re-tuning per checkpoint.

How the loss turns that tensor into a tool trajectory is ``--action-space``.
The default, ``cartesian-delta``, Euler-integrates the first three channels as
tool velocities — the simulators' convention. The UR5e's actions are absolute
joint targets, so ``--action-space ur5e-joints`` instead unnormalizes the
chunk with the checkpoint's own action statistics and pushes every joint
target through a differentiable UR5e forward kinematics (composed with
``--tcp-offset``), and ``--gripper-geometry`` names the gripper whose fingers
the finger-sourced pairs are measured from. Read joint radians as velocities
and the correction lands on the wrong channels, front-loaded onto the first
action of every chunk; ``--action-space ur5e-joints`` is what keeps a guided
chunk starting at the end-effector.
"""

from __future__ import annotations

import io
import time
import traceback
from dataclasses import dataclass
from typing import Any, Literal

import msgpack
import numpy as np
import torch
import tyro
import zmq

# ``lerobot`` is imported lazily inside ``LerobotRuntime`` (and nowhere else)
# so this module stays importable from the project root venv, which has no
# lerobot. That keeps the pure observation-mapping helpers below unit-testable
# without the heavy dependency — mirroring the lazy-import rationale in
# ``policy_abstraction.policies.openpi``.

# Lerobot's canonical state feature key. Every pi0-family checkpoint declares
# its proprioception under this name (see ``lerobot.constants.OBS_STATE``).
STATE_FEATURE_KEY = "observation.state"

# ``lerobot.configs.types.FeatureType`` is a ``str`` Enum, so comparing its
# ``.value`` against this literal matches both the enum and a plain string.
_VISUAL_FEATURE_TYPE = "VISUAL"

# Camera features named like this are placeholders the model pads internally
# (``PI0Config.empty_cameras``); they must never be matched to a real view.
_EMPTY_CAMERA_MARKER = "empty_camera"

# Policy types whose ``predict_action_chunk`` actually reads the RTC kwargs.
# Others (notably ``pi0_fast``) declare ``**kwargs: Unpack[ActionSelectKwargs]``
# with a *different* kwargs TypedDict and drop ours on the floor — no
# TypeError, no warning — so ``--rtc`` would report as enabled while being
# completely inert. Derived by grepping lerobot for ``prev_chunk_left_over``.
RTC_CAPABLE_POLICY_TYPES = frozenset({"pi0", "pi05", "smolvla"})


# ---------------------------------------------------------------------------
# msgpack ndarray codec (identical to the one used by cosmos' server/client).
# ---------------------------------------------------------------------------


def _encode_custom(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        buf = io.BytesIO()
        np.save(buf, obj, allow_pickle=False)
        return {"__ndarray_class__": True, "as_npy": buf.getvalue()}
    return obj


def _decode_custom(obj: Any) -> Any:
    if isinstance(obj, dict) and "__ndarray_class__" in obj:
        return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
    return obj


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------


@dataclass
class ServerConfig:
    """CLI args for the LeRobot policy server."""

    model_path: str
    """Local checkpoint dir (holding config.json + model.safetensors, usually
    ``.../<run>/pretrained_model``) or a HuggingFace repo id."""

    policy_type: str | None = None
    """``pi05`` / ``pi0`` / ``pi0_fast`` / ... Defaults to the ``type`` field
    recorded in the checkpoint's config.json."""

    device: str = "cuda"

    camera_map: str = ""
    """``"<view>=<feature_key>,..."`` pinning PA view names onto checkpoint
    camera features, e.g.
    ``image=observation.images.exterior_image_1_left,wrist_image=observation.images.wrist_image_left``.
    Empty means auto-derive from config.json (see ``resolve_camera_map``)."""

    state_adapter: Literal["identity", "pad_or_truncate"] = "identity"
    """``identity`` requires the incoming state dim to equal the checkpoint's
    (the default — a mismatch means the env and the checkpoint disagree about
    the robot, which is a bug worth failing on). ``pad_or_truncate`` zero-pads
    or clips instead; only use it when you know the leading dims line up."""

    robot_type: str = ""
    """Optional robot-type tag forwarded to the preprocessor pipeline."""

    num_inference_steps: int | None = None
    """Override the checkpoint's denoising step count (pi0/pi05 only)."""

    compile_model: Literal["auto", "on", "off"] = "auto"
    """``torch.compile`` of the sampling loop. ``auto`` honours the value
    recorded in the checkpoint's config.json. Training configs commonly set
    ``compile_model=true, compile_mode=max-autotune``, which costs minutes of
    autotuning on the first inference (measured: ~250 s for
    ``lerobot/pi0_libero_finetuned``) in exchange for a much faster steady
    state. Pass ``off`` when you want a server that is usable immediately —
    e.g. short debug runs — and ``on`` to force compilation for a checkpoint
    whose config left it disabled."""

    guidance: Literal["auto", "off"] = "auto"
    """``auto`` steers the flow-matching integration whenever a request carries
    ``__gp_*__`` params, and runs unguided otherwise. ``off`` refuses guided
    requests outright — use it to serve a checkpoint whose policy type has no
    denoising hook, or to guarantee an unguided baseline."""

    rtc: bool = False
    """Enable Real-Time Chunking (blend each chunk into the previous one's
    unexecuted tail)."""

    rtc_execution_horizon: int = 10
    rtc_max_guidance_weight: float = 10.0
    rtc_prefix_attention_schedule: Literal["linear", "exp", "ones", "zeros"] = "linear"

    action_space: Literal["cartesian-delta", "ur5e-joints"] = "cartesian-delta"
    """How guidance reads the action chunk. ``cartesian-delta`` (the
    simulators): channels 0-2 are tool velocities, Euler-integrated from the
    current pose. ``ur5e-joints`` (the real arm): channels 0-5 are absolute
    joint targets, unnormalized with the checkpoint's action stats and pushed
    through the UR5e forward kinematics, so a chunk that holds the current
    joints is scored at the current tool pose."""

    tcp_offset: tuple[float, float, float, float, float, float] | None = None
    """``[x, y, z, rx, ry, rz]`` TCP offset configured on the UR controller,
    composed onto the flange pose so ``ur5e-joints`` guidance measures the
    same point RTDE's ``getActualTCPPose()`` reports. Ignored otherwise."""

    gripper_geometry: str | None = None
    """``policy_steering`` robot config naming the gripper being guided
    (``robotiq_2f140`` on the UR5e). Finger-sourced distance pairs are
    measured from its pads. ``None`` keeps the default (the Panda's)."""

    host: str = "0.0.0.0"
    port: int = 5557


# ---------------------------------------------------------------------------
# Checkpoint introspection
# ---------------------------------------------------------------------------


def camera_feature_keys(input_features: dict) -> list[str]:
    """Real (non-placeholder) camera feature keys, in declaration order."""
    return [
        key
        for key, feat in input_features.items()
        if str(getattr(feat.type, "value", feat.type)) == _VISUAL_FEATURE_TYPE
        and _EMPTY_CAMERA_MARKER not in key
    ]


def resolve_camera_map(
    view_names: list[str],
    feature_keys: list[str],
    explicit: str = "",
) -> tuple[dict[str, str], list[str], list[str]]:
    """Pair PA view names with checkpoint camera features.

    Args:
        view_names: PA's logical views, in priority order (primary first).
        feature_keys: Checkpoint camera features, in declaration order.
        explicit: ``"<view>=<feature_key>,..."``. When non-empty it is used
            verbatim and every entry is validated against ``feature_keys``.

    Returns:
        ``(mapping, unmapped_views, unused_features)``. Unmapped views are ones
        the checkpoint has no camera for (the extra image is dropped); unused
        features are cameras the env can't supply (the model pads them).

    A view whose name mentions ``wrist`` binds to a wrist-named feature when
    one is free; everything else takes the first free non-wrist feature. That
    reproduces the usual DROID/UR5e layout (``exterior_*`` + ``wrist_*``) and
    the LIBERO one (``image`` + ``image2``) without per-checkpoint config.
    """
    if explicit:
        mapping: dict[str, str] = {}
        for entry in explicit.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise ValueError(
                    f"--camera-map entry {entry!r} is not '<view>=<feature_key>'."
                )
            view, key = (part.strip() for part in entry.split("=", 1))
            if key not in feature_keys:
                raise ValueError(
                    f"--camera-map points {view!r} at {key!r}, which is not a "
                    f"camera feature of this checkpoint. Available: {feature_keys}."
                )
            # Both collisions below would otherwise resolve silently — the
            # second binding wins and one camera's image is dropped, which
            # looks exactly like a legitimately spare camera downstream. Since
            # checkpoints routinely carry near-identical feature names
            # (``exterior_image_1_left``/``..._2_left``, ``image``/``image2``),
            # a copy-paste typo here would quietly corrupt every rollout.
            if view in mapping:
                raise ValueError(
                    f"--camera-map binds view {view!r} twice "
                    f"({mapping[view]!r} then {key!r}); each view must appear once."
                )
            duplicate_of = next((v for v, k in mapping.items() if k == key), None)
            if duplicate_of is not None:
                raise ValueError(
                    f"--camera-map points both {duplicate_of!r} and {view!r} at "
                    f"{key!r}. Each camera feature can take only one view — "
                    f"otherwise one image silently overwrites the other and the "
                    f"remaining camera is padded as empty."
                )
            mapping[view] = key
        unmapped = [v for v in view_names if v not in mapping]
        unused = [k for k in feature_keys if k not in mapping.values()]
        return mapping, unmapped, unused

    free = list(feature_keys)
    mapping = {}

    # Pass 1 — name match. A wrist view claims a wrist-named feature before
    # anything else can, so feature declaration order can't hand the wrist
    # camera to the third-person view. Strictly no fallback here: checkpoints
    # with generic camera names (LIBERO's ``image``/``image2``) must reach the
    # positional pass instead.
    for view in view_names:
        if "wrist" not in view.lower():
            continue
        match = next((k for k in free if "wrist" in k.lower()), None)
        if match is not None:
            mapping[view] = match
            free.remove(match)

    # Pass 2 — positional, in view priority order (primary first). A view
    # still prefers a feature whose name doesn't contradict it, so a lone
    # third-person view won't land on a wrist camera.
    for view in view_names:
        if view in mapping or not free:
            continue
        want_wrist = "wrist" in view.lower()
        match = next((k for k in free if ("wrist" in k.lower()) == want_wrist), free[0])
        mapping[view] = match
        free.remove(match)

    unmapped = [v for v in view_names if v not in mapping]
    return mapping, unmapped, free


def adapt_state(state: np.ndarray, expected_dim: int, mode: str) -> np.ndarray:
    """Fit the incoming proprio vector to the checkpoint's state dim."""
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] == expected_dim:
        return state
    if mode == "identity":
        raise ValueError(
            f"State dim mismatch: env sent {state.shape[0]}D, checkpoint "
            f"expects {expected_dim}D. The env and the checkpoint disagree "
            f"about the robot's proprioception layout. Pass "
            f"--state-adapter pad_or_truncate if the leading dims really do "
            f"line up and the rest is padding."
        )
    if mode != "pad_or_truncate":
        raise ValueError(f"Unknown state adapter {mode!r}.")
    out = np.zeros(expected_dim, dtype=np.float32)
    n = min(expected_dim, state.shape[0])
    out[:n] = state[:n]
    return out


# ---------------------------------------------------------------------------
# Guidance
# ---------------------------------------------------------------------------


# Combine-rule codes as packed by ``EpisodeRunner._query_policy``.
_MULTI_ALPHA_COMBINE = {0: "sequential", 1: "sum", 2: "weighted", 3: "per_timestep"}


def split_guidance_options(
    options: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Split ``__gp_*__`` keys out of a request's ``options``.

    Returns ``(guidance_params, cleaned_options)``; ``(None, options)`` when the
    sentinel key is absent, which is what keeps the un-guided path byte-for-byte
    unchanged. Mirrors ``groot_grounded_head._pop_guidance_from_options`` but is
    kept local so this server needs no ``policy_abstraction`` import.
    """
    from policy_steering.params import _BOOL_KEYS, GP_KEYS, GP_SENTINEL_KEY

    if options is None or GP_SENTINEL_KEY not in options:
        return None, options
    guidance_params: dict[str, Any] = {}
    cleaned: dict[str, Any] = {}
    for key, val in options.items():
        if key in GP_KEYS:
            guidance_params[key] = bool(val) if key in _BOOL_KEYS else val
        else:
            cleaned[key] = val
    return guidance_params, (cleaned or None)


class GuidanceKernel:
    """Loss/gradient functions plus settings, hoisted once per inference.

    openpi builds these outside the denoising loop so ``jax.grad`` isn't
    retraced per step; here the motivation is the same in spirit — binding the
    parameter tensors once instead of re-uploading them ten times per chunk.

    Deliberately free of any ``lerobot`` import so it can be unit-tested from
    the project root venv.
    """

    def __init__(
        self,
        guidance_params: dict[str, Any],
        *,
        device: Any,
        dtype: Any,
        action_horizon: int,
        num_steps: int,
        dynamics_fn: Any | None = None,
        gripper_geometry: Any | None = None,
    ) -> None:
        """
        Args:
            dynamics_fn: How the loss recovers a tool trajectory from the
                chunk. ``None`` Euler-integrates Cartesian deltas; the UR5e
                passes :func:`policy_steering.ur5e_kinematics_torch.make_ur5e_torch_dynamics`.
            gripper_geometry: The gripper whose fingers finger-sourced pairs
                are measured from. ``None`` is the Panda.
        """
        try:
            from policy_steering.guidance_fns_torch import (
                build_guidance_fns,
                guidance_capture_enabled,
                guidance_capture_next_inference,
            )
        except ImportError as e:  # pragma: no cover - environment wiring
            raise ImportError(
                "Guidance requires `policy_steering` on PYTHONPATH inside the "
                "lerobot venv. start_lerobot_server adds "
                "packages/policy_steering/src automatically; if you launched "
                "the server by hand, set PYTHONPATH yourself."
            ) from e

        self.action_horizon = int(action_horizon)
        self.num_steps = int(num_steps)
        # dt matches lerobot's ``dt = -1.0 / num_steps``: time runs 1 → 0, the
        # same sign convention openpi uses, so the shared kernel's sign
        # handling applies unchanged.
        self.dt = -1.0 / float(num_steps)

        self.guidance_factor = float(guidance_params.get("__gp_guidance_factor__", 1.0))
        self.use_line_search = bool(guidance_params.get("__gp_line_search__", True))
        self.use_normalization = bool(guidance_params.get("__gp_normalize__", True))
        self.max_alpha = float(guidance_params.get("__gp_max_alpha__", 1.0))
        self.use_multi_alpha = bool(guidance_params.get("__gp_multi_alpha__", False))
        self.combine = _MULTI_ALPHA_COMBINE.get(
            int(guidance_params.get("__gp_multi_alpha_combine__", 0)), "sequential"
        )
        # ``__gp_debug_guidance__`` zeroes the network velocity so the guidance
        # field can be inspected in isolation, exactly as in pi0grounded.
        self.debug_guidance = bool(guidance_params.get("__gp_debug_guidance__", False))

        # Guidance is defined over the first chunk only; the reference
        # truncates the expected distances to the action horizon before
        # building the kernels, so a longer demo horizon can't reach past the
        # actions this call will actually produce.
        params = dict(guidance_params)
        if "__gp_expected_dists__" in params:
            params["__gp_expected_dists__"] = params["__gp_expected_dists__"][
                :, : self.action_horizon
            ]
        # A NaN or inf anywhere in the parameters becomes a NaN in every action
        # channel of the chunk, which the env server then rejects with a
        # message that points at the policy. Refuse it here, naming the key.
        bad = _nonfinite_param_keys(params)
        if bad:
            raise ValueError(
                f"Guidance params carry non-finite values in {bad}; the client "
                "must filter these before sending (see guidance_builder)."
            )
        self.params = params
        # Denoising steps whose guided velocity came out non-finite and were
        # served unguided instead; reported back to the client per inference.
        self.nonfinite_steps = 0
        self.guidance_weights = params.get("__gp_guidance_weights__")

        self.dynamics_fn = dynamics_fn
        self.gripper_geometry = gripper_geometry
        self.batch_loss_fn, self.batch_grad_fn = build_guidance_fns(
            params,
            dynamics_fn=dynamics_fn,
            device=device,
            dtype=dtype,
            gripper_geometry=gripper_geometry,
        )
        self.per_pair_loss_fn = None
        self.per_pair_grad_fn = None
        self.per_timestep_loss_fn = None
        self.per_timestep_grad_fn = None
        self.capture_inference_idx = None

        # Diagnostic landscape capture only applies to the single-alpha
        # line-search path (same restriction as the GR00T head).
        self._capture = guidance_capture_enabled() and not self.use_multi_alpha
        if self.use_multi_alpha:
            (
                _,
                _,
                self.per_pair_loss_fn,
                self.per_pair_grad_fn,
                self.per_timestep_loss_fn,
                self.per_timestep_grad_fn,
            ) = build_guidance_fns(
                params,
                dynamics_fn=dynamics_fn,
                device=device,
                dtype=dtype,
                with_per_pair=True,
                gripper_geometry=gripper_geometry,
            )
        elif self._capture:
            # The landscape sweep runs in float32 regardless of the model
            # dtype so true optima aren't quantized away.
            import torch as _torch

            (
                _,
                _,
                self.per_pair_loss_fn,
                self.per_pair_grad_fn,
                self.per_timestep_loss_fn,
                self.per_timestep_grad_fn,
            ) = build_guidance_fns(
                params,
                dynamics_fn=dynamics_fn,
                device=device,
                dtype=_torch.float32,
                with_per_pair=True,
                gripper_geometry=gripper_geometry,
            )
            self.capture_inference_idx = guidance_capture_next_inference()

    def apply(self, x_t: Any, v_t: Any, step_idx: int) -> Any:
        """Return ``v_t`` corrected by the guidance gradient.

        A correction that is not finite is dropped for this step and the
        network's own velocity is returned, so one bad gradient does not turn
        the whole chunk into NaN and halt the arm. The step is counted in
        :attr:`nonfinite_steps`, logged with the gradient/loss state that
        produced it, and reported back to the client.
        """
        if self.debug_guidance:
            v_t = torch.zeros_like(v_t)
        guided = self._apply(x_t, v_t, step_idx)
        if bool(torch.isfinite(guided).all()):
            return guided
        self.nonfinite_steps += 1
        if self.nonfinite_steps == 1:
            print(
                "[lerobot server] guidance produced a non-finite velocity at "
                f"denoising step {step_idx}; serving this step unguided. "
                f"{self._nonfinite_diagnostics(x_t, v_t)}",
                flush=True,
            )
        return v_t

    def _nonfinite_diagnostics(self, x_t: Any, v_t: Any) -> str:
        """Which stage went non-finite: the input state, the loss, or the gradient."""
        try:
            x_first = x_t[:, : self.action_horizon, :]
            loss = self.batch_loss_fn(x_first)
            grad = self.batch_grad_fn(x_first)
            return (
                f"finite(x_t)={bool(torch.isfinite(x_t).all())} "
                f"finite(v_t)={bool(torch.isfinite(v_t).all())} "
                f"finite(loss)={bool(torch.isfinite(loss).all())} "
                f"finite(grad)={bool(torch.isfinite(grad).all())} "
                f"n_pairs={int(self.params['__gp_expected_dists__'].shape[0])} "
                f"dtype={x_t.dtype}"
            )
        except Exception as e:  # noqa: BLE001 - diagnostics must not raise
            return f"(diagnostics failed: {type(e).__name__}: {e})"

    def _apply(self, x_t: Any, v_t: Any, step_idx: int) -> Any:
        from policy_steering.guidance_fns_torch import (
            compute_guidance_velocity,
            compute_guidance_velocity_multi_alpha,
        )

        if self.use_multi_alpha:
            assert (
                self.per_pair_loss_fn is not None and self.per_pair_grad_fn is not None
            ), "multi-alpha requires the per-pair kernels built in __init__"
            return compute_guidance_velocity_multi_alpha(
                self.batch_loss_fn,
                self.per_pair_loss_fn,
                self.per_pair_grad_fn,
                x_t=x_t,
                v_t=v_t,
                dt=self.dt,
                action_horizon=self.action_horizon,
                guidance_factor=self.guidance_factor,
                use_normalization=self.use_normalization,
                max_alpha=self.max_alpha,
                combine=self.combine,
                guidance_weights=self.guidance_weights,
                per_timestep_loss_fn=self.per_timestep_loss_fn,
                per_timestep_grad_fn=self.per_timestep_grad_fn,
            )
        return compute_guidance_velocity(
            self.batch_loss_fn,
            self.batch_grad_fn,
            x_t=x_t,
            v_t=v_t,
            dt=self.dt,
            action_horizon=self.action_horizon,
            guidance_factor=self.guidance_factor,
            use_line_search=self.use_line_search,
            use_normalization=self.use_normalization,
            max_alpha=self.max_alpha,
            per_pair_loss_fn=self.per_pair_loss_fn,
            per_pair_grad_fn=self.per_pair_grad_fn,
            capture_inference_idx=self.capture_inference_idx,
            capture_denoise_step=step_idx,
            capture_num_steps=self.num_steps,
            capture_guidance_params=self.params if self._capture else None,
        )


def _nonfinite_param_keys(params: dict[str, Any]) -> list[str]:
    """``__gp_*__`` keys whose numeric payload holds a NaN or inf."""
    bad: list[str] = []
    for key, val in params.items():
        if isinstance(val, bool) or val is None:
            continue
        to_cpu = getattr(val, "cpu", None)
        arr = np.asarray(to_cpu() if callable(to_cpu) else val)
        if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
            bad.append(key)
    return sorted(bad)


# Which stats each lerobot normalization mode is an affine map of, as
# ``(scale, offset)`` -> ``x * scale + offset``. Mirrors
# ``lerobot.processor.normalize_processor._NormalizationMixin._apply_transform``
# with ``inverse=True``: MEAN_STD is ``x * std + mean``; the range modes map
# ``[-1, 1]`` onto ``[lo, hi]``, i.e. ``x * (hi - lo) / 2 + (hi + lo) / 2``.
_RANGE_NORMALIZATION_STATS = {
    "MIN_MAX": ("min", "max"),
    "QUANTILES": ("q01", "q99"),
    "QUANTILE10": ("q10", "q90"),
}


def action_unnormalization_affine(
    postprocess: Any, mode: str, action_dim: int, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray] | None:
    """The checkpoint's action unnormalization as ``(scale, offset)`` arrays.

    Read off the ``unnormalizer_processor`` step of the postprocess pipeline
    (its ``stats["action"]``), so the guidance kinematics undo exactly what
    the model's output would be put through. ``None`` for ``IDENTITY``.

    Raises when the stats a non-identity ``mode`` needs are missing: guidance
    would otherwise read normalized units as radians and move the wrong
    joints, silently.
    """
    mode = str(mode).upper()
    if mode == "IDENTITY":
        return None
    stats = None
    for step in getattr(postprocess, "steps", postprocess) or []:
        step_stats = getattr(step, "stats", None)
        if isinstance(step_stats, dict) and "action" in step_stats:
            stats = step_stats["action"]
            break
    if stats is None:
        raise ValueError(
            f"ACTION normalization is {mode} but the postprocess pipeline "
            "carries no action stats to undo it with; joint-space guidance "
            "cannot map the model's actions back to radians."
        )

    def _stat(name: str) -> np.ndarray:
        if name not in stats:
            raise ValueError(
                f"ACTION normalization {mode} needs the {name!r} stat, which the "
                f"checkpoint's postprocessor lacks (has {sorted(stats)})."
            )
        arr = np.asarray(stats[name], dtype=np.float64).reshape(-1)
        if arr.shape[0] != action_dim:
            raise ValueError(
                f"action stat {name!r} has {arr.shape[0]} entries; the checkpoint "
                f"declares action_dim={action_dim}."
            )
        return arr

    if mode == "MEAN_STD":
        return _stat("std"), _stat("mean")
    if mode in _RANGE_NORMALIZATION_STATS:
        lo_key, hi_key = _RANGE_NORMALIZATION_STATS[mode]
        lo, hi = _stat(lo_key), _stat(hi_key)
        span = hi - lo
        # lerobot substitutes eps for a zero span so the channel still maps.
        span = np.where(span == 0, eps, span)
        return span / 2.0, lo + span / 2.0
    raise ValueError(f"Unsupported ACTION normalization mode for guidance: {mode}")


def guided_rtc_processor_class():
    """Build the ``RTCProcessor`` subclass that injects guidance.

    Defined behind a function because subclassing requires importing lerobot,
    which this module otherwise avoids at import time.
    """
    from lerobot.policies.rtc.modeling_rtc import RTCProcessor

    class GuidedRTCProcessor(RTCProcessor):
        """``RTCProcessor`` that also applies PA guidance to each velocity.

        lerobot routes every denoising step through this object when an
        ``RTCConfig`` is attached, which makes it the one seam where ``v_t``
        can be corrected without copying ``sample_actions``. Guidance and RTC
        compose: the parent produces the (optionally RTC-inpainted) velocity
        and guidance is applied on top.
        """

        def __init__(self, rtc_config, *, rtc_requested: bool) -> None:
            super().__init__(rtc_config)
            # The lerobot-level ``enabled`` flag may be on purely to expose
            # this hook. Only when the *user* asked for RTC may a previous
            # chunk actually steer the current one.
            self.rtc_requested = bool(rtc_requested)
            self.guidance: GuidanceKernel | None = None
            self._step_idx = 0

        def begin_inference(self, guidance: GuidanceKernel | None) -> None:
            """Arm (or disarm) guidance for the next ``predict_action_chunk``."""
            self.guidance = guidance
            self._step_idx = 0

        def denoise_step(
            self,
            x_t,
            prev_chunk_left_over,
            inference_delay,
            time,
            original_denoise_step_partial,
            execution_horizon=None,
        ):
            if not self.rtc_requested:
                # Guidance-only mode: the hook is live but RTC must stay inert,
                # otherwise enabling guidance would silently turn on chunk
                # inpainting too.
                prev_chunk_left_over = None
            v_t = super().denoise_step(
                x_t=x_t,
                prev_chunk_left_over=prev_chunk_left_over,
                inference_delay=inference_delay,
                time=time,
                original_denoise_step_partial=original_denoise_step_partial,
                execution_horizon=execution_horizon,
            )
            if self.guidance is None:
                return v_t
            v_t = self.guidance.apply(x_t, v_t, self._step_idx)
            self._step_idx += 1
            return v_t

    return GuidedRTCProcessor


# ---------------------------------------------------------------------------
# Policy runtime
# ---------------------------------------------------------------------------


class LerobotRuntime:
    """Loads the checkpoint and turns PA observations into action chunks."""

    def __init__(self, cfg: ServerConfig) -> None:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.policies.factory import (
            get_policy_class,
            make_pre_post_processors,
        )
        from lerobot.policies.rtc.configuration_rtc import RTCConfig

        self.cfg = cfg

        policy_cfg = PreTrainedConfig.from_pretrained(cfg.model_path)
        policy_type = cfg.policy_type or policy_cfg.type
        if cfg.policy_type is not None and cfg.policy_type != policy_cfg.type:
            # The config object always comes from the checkpoint, so an
            # overriding type would pair (say) PI0Policy with a PI05Config and
            # fail somewhere deep in __init__. Treat the flag as an assertion.
            raise ValueError(
                f"--policy-type {cfg.policy_type!r} contradicts the checkpoint, "
                f"whose config.json declares type {policy_cfg.type!r}. The "
                f"policy config is always read from the checkpoint, so the two "
                f"cannot differ; drop the flag to use {policy_cfg.type!r}."
            )
        policy_cfg.device = cfg.device
        # Gradient checkpointing is a training-memory tradeoff with no
        # inference benefit; always off here.
        policy_cfg.gradient_checkpointing = False
        if cfg.compile_model != "auto":
            policy_cfg.compile_model = cfg.compile_model == "on"
        if cfg.num_inference_steps is not None:
            if not hasattr(policy_cfg, "num_inference_steps"):
                raise ValueError(
                    f"--num-inference-steps is not supported by policy type "
                    f"{policy_type!r}."
                )
            policy_cfg.num_inference_steps = cfg.num_inference_steps
        # OpenPI-exported pi0-fast checkpoints record the upstream tokenizer
        # repo, whose tokenizer.json sits at the repo root; Transformers loads
        # the BPE from the ``bpe_tokenizer/`` subfolder LeRobot's mirror uses.
        # Same content, different layout.
        if getattr(policy_cfg, "action_tokenizer_name", None) == (
            "physical-intelligence/fast"
        ):
            policy_cfg.action_tokenizer_name = "lerobot/fast-action-tokenizer"

        # Guidance and RTC both reach the denoising loop through the same
        # seam: lerobot only calls ``rtc_processor.denoise_step`` when an
        # enabled ``RTCConfig`` is attached. So the config is switched on
        # whenever *either* feature is wanted, and ``rtc_requested`` records
        # which one actually gets to steer (see ``GuidedRTCProcessor``).
        self.rtc_requested = bool(cfg.rtc)
        self.guidance_enabled = cfg.guidance != "off"
        needs_hook = self.rtc_requested or self.guidance_enabled
        if needs_hook and policy_type not in RTC_CAPABLE_POLICY_TYPES:
            what = "--rtc" if self.rtc_requested else "guidance"
            raise ValueError(
                f"{what} is not supported for policy type {policy_type!r}: its "
                f"predict_action_chunk ignores the RTC kwargs, so the denoising "
                f"loop never routes through the processor this feature hooks "
                f"and the run would report the feature as active while it did "
                f"nothing. Supported types: {sorted(RTC_CAPABLE_POLICY_TYPES)}. "
                f"Pass --guidance off to serve this checkpoint unguided."
            )
        self.rtc_config: Any | None = None
        if needs_hook:
            self.rtc_config = RTCConfig(
                enabled=True,
                execution_horizon=cfg.rtc_execution_horizon,
                max_guidance_weight=cfg.rtc_max_guidance_weight,
                prefix_attention_schedule=RTCAttentionSchedule(
                    cfg.rtc_prefix_attention_schedule.upper()
                ),
            )
        policy_cfg.rtc_config = self.rtc_config
        if needs_hook:
            # Guidance differentiates through the compiled denoising step, which
            # a captured CUDA graph cannot survive. Must run before load.
            _disable_cudagraphs_for_rtc(policy_cfg)

        policy_class = get_policy_class(policy_type)
        # Older checkpoints name processor steps that lerobot has since renamed;
        # from_pretrained resolves them by registry name and would KeyError.
        _register_legacy_processor_aliases()
        # Constructing the policy runs every module's random init -- ~3 B
        # parameters through kaiming_uniform_/normal_ -- before a single
        # checkpoint weight is read, and lerobot does it on the CPU: ~100 s
        # of a ~110 s startup for a pi0.5 UR5e checkpoint (profiled on a
        # 128-core box; the state-dict load itself is ~3 s). Making the target
        # device the default for tensor creation moves that throwaway init
        # onto the GPU, where it takes ~2 s, and turns the policy's own
        # ``.to(device)`` into a no-op. The loaded weights are identical; only
        # the scratch values differ. Construction peaks at roughly twice the
        # final footprint (fp32 before the bf16 cast: ~18 GB for pi0.5), so
        # the surplus is handed back to the driver straight after.
        with torch.device(cfg.device):
            self.policy = policy_class.from_pretrained(
                cfg.model_path, config=policy_cfg
            )
        if torch.device(cfg.device).type == "cuda":
            torch.cuda.empty_cache()
        # A config patched in after load leaves ``rtc_processor`` stale on
        # policies that build it in ``__init__``; re-init is idempotent.
        if hasattr(self.policy, "init_rtc_processor"):
            self.policy.init_rtc_processor()
        self.policy.eval()

        # Swap in the guided processor. ``init_rtc_processor`` binds the same
        # instance onto both the policy and its inner model, so both are
        # replaced here to keep them in sync.
        self._processor = None
        if needs_hook:
            self._processor = guided_rtc_processor_class()(
                self.rtc_config, rtc_requested=self.rtc_requested
            )
            self.policy.rtc_processor = self._processor
            inner = getattr(self.policy, "model", None)
            if inner is not None:
                inner.rtc_processor = self._processor

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            cfg.model_path,
            preprocessor_overrides={"device_processor": {"device": cfg.device}},
        )

        self.device = torch.device(cfg.device)
        self.feature_keys = camera_feature_keys(self.policy.config.input_features)
        if not self.feature_keys:
            raise ValueError(
                f"Checkpoint {cfg.model_path} declares no camera features; "
                f"input_features={list(self.policy.config.input_features)}."
            )
        state_feature = self.policy.config.input_features.get(STATE_FEATURE_KEY)
        self.state_dim = int(state_feature.shape[0]) if state_feature else 0
        self.action_dim = int(self.policy.config.output_features["action"].shape[0])
        self.chunk_size = int(getattr(self.policy.config, "chunk_size", 0))
        self.compile_enabled = bool(getattr(self.policy.config, "compile_model", False))
        # Guidance steers the model's internal (still-normalized) actions, so
        # how far ``guidance_factor`` moves the end-effector depends on this.
        # Surfaced in ``info`` so a normalization that isn't IDENTITY is
        # visible rather than being an invisible scale factor on the results.
        # ``normalization_mapping`` may be keyed by ``FeatureType`` enums whose
        # hash is the member name, not the string, so match on the value.
        self.action_normalization = _action_normalization(self.policy.config)
        # How the guidance loss turns the (normalized) chunk into a tool
        # trajectory, built once: the kinematics and the affine that undoes
        # the checkpoint's action normalization in front of it.
        self.guidance_dynamics_fn: Any | None = None
        self.guidance_gripper_geometry: Any | None = None
        if self.guidance_enabled:
            self._build_guidance_kinematics()

        # Resolved lazily on the first request, when the arriving view names
        # are known; cached and logged once.
        self.camera_map: dict[str, str] | None = None
        # Raw (pre-postprocess) previous chunk, kept for RTC inpainting.
        self._prev_chunk: torch.Tensor | None = None

    # -- info ----------------------------------------------------------

    def info(self) -> dict:
        return {
            "policy_type": self.policy.config.type,
            "model_path": self.cfg.model_path,
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "camera_features": self.feature_keys,
            "camera_map": self.camera_map or {},
            # The lerobot-level RTCConfig may be enabled purely to expose the
            # denoising hook; report what the *user* asked for.
            "rtc_enabled": self.rtc_requested,
            "guidance_enabled": self.guidance_enabled,
            "action_normalization": self.action_normalization,
            "compile_enabled": self.compile_enabled,
            "guidance_action_space": self.cfg.action_space,
            "guidance_tcp_offset": (
                list(self.cfg.tcp_offset) if self.cfg.tcp_offset else None
            ),
            "guidance_gripper_geometry": self.cfg.gripper_geometry,
        }

    def reset(self) -> None:
        """Clear everything that is scoped to one episode."""
        self._prev_chunk = None
        # Some processor steps are stateful across calls — notably the
        # relative/absolute action pair, which caches a reference state for
        # checkpoints with ``use_relative_actions``. Carrying that into the
        # next episode would offset its actions against a stale origin.
        for pipeline in (self.preprocess, self.postprocess):
            reset = getattr(pipeline, "reset", None)
            if callable(reset):
                reset()

    # -- inference -----------------------------------------------------

    def _ensure_camera_map(self, view_names: list[str]) -> dict[str, str]:
        if self.camera_map is not None:
            return self.camera_map
        mapping, unmapped, unused = resolve_camera_map(
            view_names, self.feature_keys, self.cfg.camera_map
        )
        if not mapping:
            raise ValueError(
                f"Could not map any of the env's views {view_names} onto the "
                f"checkpoint's cameras {self.feature_keys}. Pin the pairing "
                f"with --camera-map."
            )
        print(f"[lerobot server] camera map: {mapping}", flush=True)
        if unmapped:
            print(
                f"[lerobot server] WARNING: views {unmapped} have no matching "
                f"camera in this checkpoint and are dropped.",
                flush=True,
            )
        if unused:
            print(
                f"[lerobot server] WARNING: checkpoint cameras {unused} get no "
                f"image from the env; the model will pad them as empty.",
                flush=True,
            )
        self.camera_map = mapping
        return mapping

    def _build_batch(self, obs: dict) -> dict:
        images = obs.get("images")
        if not isinstance(images, dict) or not images:
            raise ValueError(
                "get_action: observation['images'] must be a non-empty dict of "
                f"{{view_name: HxWx3 uint8}}; got {type(images).__name__}."
            )
        views: dict[str, Any] = {str(name): img for name, img in images.items()}
        mapping = self._ensure_camera_map(list(views))

        frame: dict[str, np.ndarray] = {}
        for view, feature_key in mapping.items():
            if view not in views:
                raise ValueError(
                    f"get_action: view {view!r} is in the resolved camera map "
                    f"but missing from this request (have {sorted(views)}). "
                    f"The env must send the same views on every step."
                )
            img = np.asarray(views[view])
            if img.ndim != 3 or img.shape[-1] != 3:
                raise ValueError(
                    f"get_action: view {view!r} must be HxWx3; got {img.shape}."
                )
            frame[feature_key] = np.ascontiguousarray(img, dtype=np.uint8)

        if self.state_dim:
            state = obs.get("state")
            if state is None:
                raise ValueError(
                    f"get_action: checkpoint expects a {self.state_dim}D "
                    f"'state' but the request carries none."
                )
            frame[STATE_FEATURE_KEY] = adapt_state(
                state, self.state_dim, self.cfg.state_adapter
            )

        task = obs.get("task")
        if not isinstance(task, str):
            raise ValueError(
                f"get_action: 'task' must be a str; got {type(task).__name__}."
            )

        from lerobot.policies.utils import prepare_observation_for_inference

        # Converts to tensors, scales images to [0,1] CHW, adds the batch dim,
        # and attaches task/robot_type for the tokenizer step.
        observation = prepare_observation_for_inference(
            frame, self.device, task=task, robot_type=self.cfg.robot_type
        )
        return self.preprocess(observation)

    def _rtc_kwargs(self, consumed_steps: int) -> dict:
        # Gated on what the user asked for, not on ``rtc_config.enabled`` —
        # the latter is also switched on to expose the guidance hook.
        if not self.rtc_requested or self.rtc_config is None:
            return {}
        leftover = None
        if self._prev_chunk is not None:
            consumed = max(0, min(consumed_steps, self._prev_chunk.shape[1]))
            tail = self._prev_chunk[:, consumed:]
            if tail.shape[1] > 0:
                leftover = tail
        return {
            # PA blocks the simulator during inference, so no actions are
            # consumed while sampling: the leftover starts at "now".
            "inference_delay": 0,
            "prev_chunk_left_over": leftover,
            "execution_horizon": self.rtc_config.execution_horizon,
        }

    def _build_guidance_kinematics(self) -> None:
        """Bind ``--action-space`` / ``--tcp-offset`` / ``--gripper-geometry``."""
        if self.cfg.gripper_geometry:
            from policy_steering.robot_config import load_gripper_geometry

            self.guidance_gripper_geometry = load_gripper_geometry(
                self.cfg.gripper_geometry
            )
        if self.cfg.action_space != "ur5e-joints":
            return
        from policy_steering.ur5e_kinematics_torch import (
            make_affine_unnormalizer,
            make_ur5e_torch_dynamics,
        )

        affine = action_unnormalization_affine(
            self.postprocess, self.action_normalization, self.action_dim
        )
        unnormalize = None
        if affine is not None:
            scale, offset = affine
            unnormalize = make_affine_unnormalizer(
                torch.as_tensor(scale, dtype=torch.float32, device=self.device),
                torch.as_tensor(offset, dtype=torch.float32, device=self.device),
            )
        self.guidance_dynamics_fn = make_ur5e_torch_dynamics(
            self.cfg.tcp_offset, unnormalize=unnormalize
        )
        print(
            "[lerobot server] guidance reads the chunk as UR5e joint targets: "
            f"tcp_offset={list(self.cfg.tcp_offset) if self.cfg.tcp_offset else None} "
            f"unnormalize={self.action_normalization} "
            f"gripper={self.cfg.gripper_geometry or 'default'}",
            flush=True,
        )

    def _build_guidance(self, options: dict | None) -> Any:
        """Hoist the guidance kernels for one inference, or ``None``."""
        guidance_params, _ = split_guidance_options(options)
        if guidance_params is None:
            return None
        if not self.guidance_enabled:
            raise ValueError(
                "This server was started with --guidance off but the request "
                "carries __gp_*__ guidance params. Refusing rather than "
                "running unguided and reporting the result as guided."
            )
        if self._processor is None:  # pragma: no cover - guarded at startup
            raise RuntimeError("Guidance requested but no denoising hook is installed.")
        # ``num_inference_steps`` sets lerobot's ``dt = -1/num_steps``, which
        # the kernel needs to size its position-space step.
        num_steps = int(getattr(self.policy.config, "num_inference_steps", 10))
        return GuidanceKernel(
            guidance_params,
            device=self.device,
            dtype=self._guidance_dtype(),
            action_horizon=self.chunk_size,
            num_steps=num_steps,
            dynamics_fn=self.guidance_dynamics_fn,
            gripper_geometry=self.guidance_gripper_geometry,
        )

    def _guidance_dtype(self) -> Any:
        """Match the dtype ``sample_actions`` denoises in (its noise dtype)."""
        dtype = getattr(self.policy.config, "dtype", None)
        if isinstance(dtype, str):
            return getattr(torch, dtype, torch.float32)
        return dtype if isinstance(dtype, torch.dtype) else torch.float32

    def get_action(self, obs: dict, options: dict | None = None) -> dict:
        batch = self._build_batch(obs)
        consumed_steps = int(obs.get("consumed_steps", 1) or 1)
        guidance = self._build_guidance(options)
        if self._processor is not None:
            self._processor.begin_inference(guidance)

        t0 = time.monotonic()
        # pi0/pi05 already decorate ``predict_action_chunk`` with no_grad; this
        # covers policy types that don't. Neither RTC's correction nor the
        # guidance gradient is disabled by it — both re-enable grad internally
        # via ``torch.enable_grad``.
        with torch.no_grad():
            raw = self.policy.predict_action_chunk(
                batch, **self._rtc_kwargs(consumed_steps)
            )
            if self.rtc_requested:
                self._prev_chunk = raw.detach().clone()
            actions = self.postprocess(raw)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        actions_np = actions[0].to("cpu").to(torch.float32).numpy()
        if actions_np.ndim != 2:
            raise RuntimeError(
                f"Expected a (T, action_dim) chunk from the policy; got "
                f"{actions_np.shape}."
            )
        nonfinite_steps = int(guidance.nonfinite_steps) if guidance is not None else 0
        if not np.all(np.isfinite(actions_np)):
            # Guidance falls back to the raw velocity when its correction is
            # non-finite (see GuidanceKernel.apply), so a NaN here came from
            # the network or RTC itself. Say so, rather than letting the env
            # server's "non-finite action rejected" point at the wrong place.
            raise RuntimeError(
                "Policy returned a non-finite action chunk "
                f"(guided={guidance is not None}, rtc={self.rtc_requested}, "
                f"compile={self.compile_enabled}, "
                f"nan={int(np.isnan(actions_np).sum())}/{actions_np.size}). "
                "Guidance was not the source: its non-finite steps are served "
                "unguided. Suspect the model forward pass or the RTC blend; "
                "try --compile-model off to rule out the compiled graph."
            )
        # The full chunk goes back: PA executes ``replan_steps`` of it and
        # keeps the tail for the chunk visualiser, so truncating here would
        # only lose information.
        reply: dict[str, Any] = {
            "actions": np.ascontiguousarray(actions_np, dtype=np.float32),
            "policy_timing": {"infer_ms": float(elapsed_ms)},
        }
        if nonfinite_steps:
            reply["guidance_nonfinite_steps"] = nonfinite_steps
        return reply


# ---------------------------------------------------------------------------
# Server loop
# ---------------------------------------------------------------------------


def _action_normalization(policy_config) -> str:
    """The ACTION normalization mode declared by a policy config, or UNKNOWN.

    ``normalization_mapping`` may be keyed by ``FeatureType`` enums whose hash
    is the member name rather than the string, so a plain ``["ACTION"]`` lookup
    misses. Compare on the value instead.
    """
    norm_map = getattr(policy_config, "normalization_mapping", {}) or {}
    for key, mode in norm_map.items():
        if str(getattr(key, "value", key)) == "ACTION":
            return str(getattr(mode, "value", mode))
    return "UNKNOWN"


# --- UR5e checkpoint/runtime compatibility ------------------------------
# Ported from the UR5e branch's server, which this file replaces.


# torch.compile modes that capture CUDA graphs, mapped to their nearest
# equivalent that does not. See `_disable_cudagraphs_for_rtc`.
_CUDAGRAPH_COMPILE_MODES = {
    "max-autotune": "max-autotune-no-cudagraphs",
    "reduce-overhead": "default",
}


def _disable_cudagraphs_for_rtc(cfg) -> None:
    """Drop CUDA-graph capture from the checkpoint's ``compile_mode`` under RTC.

    RTC's guidance term differentiates through the denoising step: it re-enters
    the compiled ``denoise_step`` inside ``torch.enable_grad()`` and calls
    ``torch.autograd.grad``. That is a second, grad-enabled graph for a function
    inductor has already captured into a CUDA-graph tree, and adding it after
    the graph pool is live trips
    ``apply_checkpoint_execution_state_in_allocator`` ("Expected
    curr_block->next == nullptr to be true"). The failure lands on the *second*
    inference — the first one has no previous chunk to blend into — so it looks
    like a mid-rollout crash rather than a setup problem.

    ``max-autotune-no-cudagraphs`` keeps the autotuned kernels and drops only
    the graph capture. Kernel-level speed is unaffected; per-call launch
    overhead goes up slightly.
    """
    if not getattr(cfg, "compile_model", False):
        return
    mode = getattr(cfg, "compile_mode", None)
    if mode not in _CUDAGRAPH_COMPILE_MODES:
        return
    replacement = _CUDAGRAPH_COMPILE_MODES[mode]
    cfg.compile_mode = replacement
    print(
        f"RTC enabled: switching torch.compile mode {mode!r} -> {replacement!r} "
        "(CUDA graphs are incompatible with RTC's autograd guidance step).",
        flush=True,
    )


def _register_legacy_processor_aliases() -> None:
    """Re-register processor steps under the names older checkpoints saved.

    ``PolicyProcessorPipeline.from_pretrained`` resolves each saved step by its
    ``registry_name``, so a checkpoint written before a step was renamed fails
    to load with a bare ``KeyError``. The classes themselves are unchanged (and
    the config keys still line up), so mapping the old name onto the current
    class is a faithful load rather than a workaround.

    Known rename: ``relative_actions_processor`` -> ``delta_actions_processor``
    (still the ``RelativeActionsProcessorStep`` class). The
    ``pi05_*_a100`` UR5e checkpoints predate it.
    """
    from lerobot.processor.pipeline import ProcessorStepRegistry
    from lerobot.processor.relative_action_processor import (
        RelativeActionsProcessorStep,
    )

    aliases = {"relative_actions_processor": RelativeActionsProcessorStep}
    registered = set(ProcessorStepRegistry.list())
    for name, cls in aliases.items():
        if name in registered:
            continue  # A current lerobot already provides it.
        ProcessorStepRegistry.register(name)(cls)
        print(
            f"Registered legacy processor alias {name!r} -> {cls.__name__} "
            "(checkpoint predates the rename).",
            flush=True,
        )


def main(cfg: ServerConfig) -> None:
    print("Starting LeRobot inference server...")
    print(f"  Model path : {cfg.model_path}")
    print(f"  Policy type: {cfg.policy_type or '(from config.json)'}")
    print(f"  Device     : {cfg.device}")
    print(f"  RTC        : {cfg.rtc}")
    print(f"  Guidance   : {cfg.guidance} (action space {cfg.action_space})")
    print(f"  Host:Port  : {cfg.host}:{cfg.port}")

    runtime = LerobotRuntime(cfg)
    print(f"Model ready: {runtime.info()}")
    if runtime.guidance_enabled and runtime.action_normalization != "IDENTITY":
        where = (
            "the loss is measured in metres through the UR5e kinematics, but "
            "the correction still lands on the normalized tensor"
            if cfg.action_space == "ur5e-joints"
            else "guidance steers normalized actions rather than metres"
        )
        print(
            f"  NOTE: ACTION normalization is {runtime.action_normalization}, so "
            f"{where}. --guidance.factor and --guidance.max-alpha are in those "
            "units and may need re-tuning for this checkpoint.",
            flush=True,
        )
    if runtime.compile_enabled:
        print(
            "  NOTE: torch.compile is on for this checkpoint "
            f"(mode={getattr(runtime.policy.config, 'compile_mode', '?')}). "
            "The first inference pays the autotuning cost — minutes, not "
            "seconds. Pass --compile-model off to skip it.",
            flush=True,
        )

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{cfg.host}:{cfg.port}")
    print(f"\n✓ Server ready — listening on {cfg.host}:{cfg.port}\n", flush=True)

    try:
        while True:
            raw = socket.recv()
            try:
                request = msgpack.unpackb(raw, object_hook=_decode_custom)
                endpoint = request.get("endpoint")
                data = request.get("data") or {}

                if endpoint == "ping":
                    reply: Any = {"ok": True}
                elif endpoint == "info":
                    reply = runtime.info()
                elif endpoint == "reset":
                    runtime.reset()
                    reply = {"ok": True}
                elif endpoint == "get_action":
                    obs = data.get("observation")
                    if not isinstance(obs, dict):
                        raise ValueError(
                            "get_action: 'observation' must be a dict, got "
                            f"{type(obs).__name__}"
                        )
                    # ``options`` carries the __gp_*__ guidance params when
                    # PA is steering; absent, the un-guided path runs.
                    reply = runtime.get_action(obs, data.get("options"))
                else:
                    reply = {"error": f"unknown endpoint: {endpoint!r}"}
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[lerobot server] error: {e}\n{tb}", flush=True)
                reply = {"error": f"{type(e).__name__}: {e}"}

            socket.send(msgpack.packb(reply, default=_encode_custom))
    except KeyboardInterrupt:
        print("\nShutting down LeRobot server...")
    finally:
        socket.close(linger=0)


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
