"""In-process dummy policy for debug_guidance mode.

Mimics the flow-matching integration loop from
``openpi.models.pi0grounded.Pi0Grounded.sample_actions`` but drops the
PaliGemma forward pass (``v_t = 0``). When a guidance gradient is supplied
via ``__gp_*__`` params, it is applied every denoising step exactly as in
the real model, so guidance behaviour can be tested in isolation without
the cost of loading or running the policy network.

Used directly in-process by ``evaluate.py`` (no websocket server):
the ``infer`` signature matches the websocket client's, so
``EpisodeRunner`` works unchanged.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger

from policy_steering import (
    build_guidance_fns,
    compute_guidance_velocity,
    extract_guidance_params,
)

_RNG_SEED_KEY = "__rng_seed__"


def _denoise_loop_impl(
    x_init,
    guidance_params,
    guidance_factor,
    num_denoise_steps: int,
    action_horizon: int,
    use_line_search: bool,
    dynamics_fn,
    gripper_geometry=None,
):
    """Flow-matching denoising loop with guidance — traced once per shape/flag combo."""
    batch_loss_fn, batch_grad_fn = build_guidance_fns(
        guidance_params, dynamics_fn=dynamics_fn, gripper_geometry=gripper_geometry
    )
    dt = -1.0 / num_denoise_steps

    def step(carry):
        x, t = carry
        v = jnp.zeros_like(x)
        v = compute_guidance_velocity(
            batch_loss_fn,
            batch_grad_fn,
            x,
            v,
            dt,
            action_horizon,
            guidance_factor,
            use_line_search,
        )
        return x + dt * v, t + dt

    def cond(carry):
        _, t = carry
        return t >= -dt / 2

    x_0, _ = jax.lax.while_loop(cond, step, (x_init, 1.0))
    return x_0


_run_denoise_loop: Callable[..., Any] = jax.jit(
    _denoise_loop_impl,
    static_argnames=(
        "num_denoise_steps",
        "action_horizon",
        "use_line_search",
        "dynamics_fn",
        "gripper_geometry",
    ),
)


class DummyPolicy:
    """Drop-in replacement for ``WebsocketClientPolicy`` under debug_guidance.

    The defaults match ``pi05_libero_grounded`` so the returned action tensor
    has shape ``[num_action_chunks, action_horizon, num_output_actions]`` —
    identical to what ``LiberoOutputs`` emits from the real policy.

    Denoising starts from a zero chunk, which for a Cartesian-delta env means
    "don't move" and lets the guidance correction be read directly off the
    output. For a robot that reads its actions as *absolute joint targets*
    (the UR5e), zero is a real pose — the fully-extended one — so pass
    ``init_from_state=True`` to start from the observed ``observation/state``
    (joints + gripper) tiled over the horizon instead: "don't move" for that
    action space. Pair it with a ``dynamics_fn`` that runs forward kinematics
    (:func:`policy_abstraction.dynamics.ur5_kinematics_jax.make_ur5e_jax_dynamics`)
    so the guidance loss measures the chunk where the arm would actually go;
    the default Euler integration would read joint targets as velocities.
    ``gripper_geometry`` names the live gripper the finger-position pairs are
    measured on (default: the Panda).
    """

    def __init__(
        self,
        *,
        num_action_chunks: int = 2,
        action_horizon: int = 10,
        action_dim: int = 32,
        num_output_actions: int = 7,
        num_denoise_steps: int = 10,
        dynamics_fn=None,
        init_from_state: bool = False,
        gripper_geometry=None,
    ) -> None:
        self._num_action_chunks = num_action_chunks
        self._action_horizon = action_horizon
        self._action_dim = action_dim
        self._num_output_actions = num_output_actions
        self._num_denoise_steps = num_denoise_steps
        self._dynamics_fn = dynamics_fn
        self._init_from_state = bool(init_from_state)
        self._gripper_geometry = gripper_geometry
        self._missing_state_warned = False
        self._rng = jax.random.key(0)

    def reset(self) -> None:
        pass

    def _hold_chunk(self, obs: dict, x_init):
        """``x_init`` with the observed state written into every step.

        Only the leading ``min(state_dim, action_dim)`` channels are set; for
        the UR5e that is all seven (six joints + gripper). Without a state in
        the observation there is nothing to hold, so the zero chunk stands and
        a warning says so once — the warm-up query legitimately has no state.
        """
        state = obs.get("observation/state")
        if state is None:
            if not self._missing_state_warned:
                self._missing_state_warned = True
                logger.warning(
                    "[DummyPolicy] init_from_state requested but the observation "
                    "carries no 'observation/state'; starting from a zero chunk."
                )
            return x_init
        s = jnp.asarray(np.asarray(state, dtype=np.float32).reshape(-1))
        n = min(int(s.shape[0]), self._action_dim)
        return x_init.at[:, :, :n].set(s[:n])

    def infer(self, obs: dict) -> dict:
        start_time = time.monotonic()

        if _RNG_SEED_KEY in obs:
            rng = jax.random.key(int(obs[_RNG_SEED_KEY]))
            obs = {k: v for k, v in obs.items() if k != _RNG_SEED_KEY}
        else:
            self._rng, rng = jax.random.split(self._rng)

        guidance_params, cleaned_obs = extract_guidance_params(obs)
        if guidance_params is not None:
            # extract_guidance_params now returns numpy arrays (backend-agnostic);
            # convert to jnp here so downstream JIT'd guidance_fns operate on
            # the right tensor type. Bool keys stay as Python bools.
            guidance_params = {
                k: (v if isinstance(v, bool) else jnp.asarray(v))
                for k, v in guidance_params.items()
            }

        num_chunks = self._num_action_chunks
        gp_arrays: dict | None = None
        guidance_factor = 0.0
        use_line_search = False
        if guidance_params is not None:
            num_chunks = int(guidance_params.get("__gp_num_chunks__", num_chunks))
            gp = dict(guidance_params)
            # Guidance is computed on the first chunk only — match the
            # slicing in ``Pi0Grounded.sample_actions`` so shapes line up.
            gp["__gp_expected_dists__"] = gp["__gp_expected_dists__"][
                :, : self._action_horizon
            ]
            guidance_factor = float(
                gp.get(
                    "__gp_guidance_factor__",
                    gp.get("guidance_factor", 1.0),
                )
            )
            use_line_search = bool(gp.get("__gp_line_search__", True))
            # Strip Python-bool keys so the dict is a pure pytree of jnp arrays
            # (otherwise jit would turn them into weak-typed tracers).
            gp_arrays = {k: v for k, v in gp.items() if not isinstance(v, bool)}

        batch_size = 1
        total_horizon = num_chunks * self._action_horizon
        # debug_guidance always starts from zero (matches pi0grounded's
        # ``jnp.where(debug_guidance, noise * 0.0, noise)``).
        _ = rng  # noise is zero; rng kept for API parity / future use
        x_init = jnp.zeros((batch_size, total_horizon, self._action_dim))
        if self._init_from_state:
            x_init = self._hold_chunk(cleaned_obs, x_init)

        if gp_arrays is not None:
            x_0 = _run_denoise_loop(
                x_init,
                gp_arrays,
                guidance_factor,
                num_denoise_steps=self._num_denoise_steps,
                action_horizon=self._action_horizon,
                use_line_search=use_line_search,
                dynamics_fn=self._dynamics_fn,
                gripper_geometry=self._gripper_geometry,
            )
        else:
            # No guidance -> v=0 every step, so x stays at x_init (zeros).
            x_0 = x_init
        x_0 = x_0.reshape(
            batch_size, num_chunks, self._action_horizon, self._action_dim
        )
        actions_out = np.asarray(x_0[0, :, :, : self._num_output_actions])

        state = cleaned_obs.get("observation/state")
        state = (
            np.asarray(state)
            if state is not None
            else np.zeros(self._num_output_actions, dtype=np.float32)
        )

        return {
            "state": state,
            "actions": actions_out,
            "policy_timing": {"infer_ms": (time.monotonic() - start_time) * 1000},
        }


class CircularPolicy:
    """Fake policy that drives the end-effector in a circle (xy-plane).

    Used as a visualization sanity check: the EE traces a known shape and
    the gripper cycles open/closed on a known period, so any rendering or
    action-interpretation bug (axis swap, finger handedness, doubled
    aperture, frame mismatch) is immediately obvious.

    The action chunks emitted are 7-D ``[dx, dy, dz, drx, dry, drz, g]``,
    matching the wire format used by every env adapter (LIBERO / RoboCasa /
    CALVIN). The first six dims are interpreted by the env's controller as
    *normalized* deltas in roughly ``[-1, 1]`` — robosuite OSC defaults
    scale them by ``output_max=0.05 m`` per step, and CALVIN's relative
    cartesian controller treats them similarly. Emitting raw physical
    metres (e.g. ``radius * omega``) therefore produces near-invisible
    motion because the controller scales it down further; this policy
    instead emits ``amplitude * sin/cos(theta)`` directly so the visible
    circle radius is roughly ``amplitude * controller_output_max *
    period_steps / (2 * pi)``.

    Each call to :meth:`infer` returns one full multi-chunk plan; an
    internal phase counter is advanced by ``action_horizon`` per call so
    consecutive plans stitch into a continuous circle.

    Parameters
    ----------
    amplitude:
        Normalized x/y signal magnitude in ``[0, 1]``. Default 0.3 — yields
        a clearly visible circle on OSC-default robosuite envs without
        leaving a typical tabletop workspace.
    period_steps:
        Number of env steps per full circle.
    gripper_period_steps:
        Period (in env steps) of the open/close gripper square wave.
    """

    def __init__(
        self,
        *,
        num_action_chunks: int = 2,
        action_horizon: int = 10,
        num_output_actions: int = 7,
        amplitude: float = 0.3,
        period_steps: int = 80,
        gripper_period_steps: int = 40,
    ) -> None:
        self._num_action_chunks = num_action_chunks
        self._action_horizon = action_horizon
        self._num_output_actions = num_output_actions
        self._amplitude = float(amplitude)
        self._period_steps = max(int(period_steps), 1)
        self._gripper_period_steps = max(int(gripper_period_steps), 1)
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def infer(self, obs: dict) -> dict:
        start_time = time.monotonic()
        chunks = self._num_action_chunks
        h = self._action_horizon
        d = self._num_output_actions
        actions_out = np.zeros((chunks, h, d), dtype=np.float32)

        omega = 2.0 * np.pi / self._period_steps
        a = self._amplitude
        g_omega = 2.0 * np.pi / self._gripper_period_steps

        for c in range(chunks):
            for t in range(h):
                step_idx = self._step + c * h + t
                theta = omega * step_idx
                actions_out[c, t, 0] = -a * np.sin(theta)
                actions_out[c, t, 1] = a * np.cos(theta)
                if d > 6:
                    actions_out[c, t, 6] = (
                        1.0 if np.sin(g_omega * step_idx) >= 0.0 else -1.0
                    )

        self._step += h

        state = obs.get("observation/state")
        state = (
            np.asarray(state) if state is not None else np.zeros(d, dtype=np.float32)
        )
        return {
            "state": state,
            "actions": actions_out,
            "policy_timing": {"infer_ms": (time.monotonic() - start_time) * 1000},
        }
