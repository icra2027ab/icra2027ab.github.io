"""ZMQ REQ/REP server wrapping cosmos-policy `get_action` for LIBERO eval.

Runs **inside** the cosmos-policy isolated venv (`third_party/cosmos-policy/.venv`)
because cosmos pins torch==2.7.0+cu128 / flash-attn / megatron-core which would
collide with the project root venv. The wire protocol mirrors
`scripts/servers/serve_groot.py`: msgpack frames with a custom ndarray codec,
REQ/REP, three endpoints (`ping`, `get_action`, `reset`).

Expected message shapes:
  Request : {"endpoint": "get_action",
             "data": {
                 "observation": {
                     "primary_image": uint8 HxWxC,
                     "wrist_image":   uint8 HxWxC,
                     "proprio":       float64 (9,),  # [gripper(2), pos(3), quat_xyzw(4)]
                     "task_description": str,
                 },
                 "options": dict | None,  # __gp_*__ guidance params (optional)
             }}
  Reply   : {"actions": float32 (T, 7), "policy_timing": {"infer_ms": float}}

This server consumes the PA libero env's images *already preprocessed by
`CameraSpec.apply_image_transform("flip_vh")` and de-mirrored client-side*
(see `policy_abstraction.policies.cosmos`). It therefore disables cosmos's
own `flip_images` step (`get_action(... flip_images=False)`), matching the
training-distribution orientation.
"""

from __future__ import annotations

import io
import os
import time
import traceback
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import msgpack
import numpy as np
import torch
import tyro
import zmq
from cosmos_policy.experiments.robot.cosmos_utils import (
    get_action,
    get_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)

# These imports require ``policy_abstraction`` + ``policy_steering`` to be on
# PYTHONPATH inside the cosmos venv. ``start_cosmos_server`` already arranges
# this; see ``src/policy_abstraction/policies/cosmos.py``.
from policy_abstraction.policies.cosmos_grounded_model import (
    CosmosPolicyVideo2WorldModelGrounded,
    _pop_guidance_from_options,
)
from policy_abstraction.policies.cosmos_grounded_runtime import run_guided_get_action

# ---------------------------------------------------------------------------
# msgpack ndarray codec (identical to the one used by groot's server/client).
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
    """CLI args for the cosmos-policy server. Defaults mirror the LIBERO.md
    reference invocation (chkpt45000--5stepAct--seed195--deterministic)."""

    model_path: str = "nvidia/Cosmos-Policy-LIBERO-Predict2-2B"
    """HF repo id or local checkpoint dir. Auto-downloads from HF on first use."""

    config: str = "cosmos_predict2_2b_480p_libero__inference_only"
    """Cosmos experiment config name (passed to load_model_from_checkpoint)."""

    config_file: str = "cosmos_policy/config/config.py"
    """Path to the cosmos config registry, relative to the cosmos-policy repo."""

    dataset_stats_path: str = (
        "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json"
    )
    """HF file path or local path to dataset stats JSON (for proprio/action norm)."""

    t5_text_embeddings_path: str = (
        "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl"
    )
    """HF file path or local path to precomputed T5 embeddings pickle. Cache
    misses fall through to on-the-fly T5 encoding via `get_text_embedding`,
    which is what enables LIBERO-PRO language perturbations to work."""

    suite: str = "libero"
    """One of {libero, robocasa, aloha}. Determines camera/proprio layout."""

    chunk_size: int = 16
    num_open_loop_steps: int = 16
    num_denoising_steps_action: int = 5

    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_proprio: bool = True
    normalize_proprio: bool = True
    unnormalize_actions: bool = True
    trained_with_image_aug: bool = True
    use_jpeg_compression: bool = True
    use_variance_scale: bool = False

    seed: int = 195
    randomize_seed: bool = False
    deterministic: bool = True

    host: str = "0.0.0.0"
    port: int = 5556


# ---------------------------------------------------------------------------
# Inference glue
# ---------------------------------------------------------------------------


def _build_cfg(cfg: ServerConfig) -> SimpleNamespace:
    """Translate the server's CLI args into the attribute bag that cosmos's
    `get_action` / `get_model` read off `cfg`."""
    return SimpleNamespace(
        ckpt_path=cfg.model_path,
        config=cfg.config,
        config_file=cfg.config_file,
        suite=cfg.suite,
        chunk_size=cfg.chunk_size,
        num_open_loop_steps=cfg.num_open_loop_steps,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        use_wrist_image=cfg.use_wrist_image,
        num_wrist_images=cfg.num_wrist_images,
        use_third_person_image=cfg.use_third_person_image,
        num_third_person_images=cfg.num_third_person_images,
        use_proprio=cfg.use_proprio,
        normalize_proprio=cfg.normalize_proprio,
        unnormalize_actions=cfg.unnormalize_actions,
        trained_with_image_aug=cfg.trained_with_image_aug,
        use_jpeg_compression=cfg.use_jpeg_compression,
        use_variance_scale=cfg.use_variance_scale,
        seed=cfg.seed,
        randomize_seed=cfg.randomize_seed,
        deterministic=cfg.deterministic,
    )


def _stack_actions(raw: Any, chunk_size: int) -> np.ndarray:
    """`get_action` returns a Python list of length `chunk_size`, each element
    a 1D ndarray of size action_dim, when batch_size=1. Stack back to (T, D)."""
    if isinstance(raw, np.ndarray):
        return raw
    if isinstance(raw, (list, tuple)):
        return np.stack([np.asarray(a) for a in raw], axis=0)
    raise TypeError(f"Unexpected actions container: {type(raw).__name__}")


def _run_inference(
    cfg_ns: SimpleNamespace,
    model: torch.nn.Module,
    dataset_stats: dict,
    obs: dict,
    options: dict | None = None,
) -> dict:
    """Run one cosmos action inference and return a serializable dict.

    Routes to the guided helper when ``options`` carries ``__gp_*__``
    guidance keys; otherwise calls upstream ``get_action`` verbatim so the
    unguided path stays bit-for-bit identical to pre-guidance behaviour.
    """
    task_description = obs.pop("task_description", None)
    if task_description is None:
        raise ValueError(
            "cosmos server: observation is missing 'task_description' string."
        )
    if not isinstance(task_description, str):
        raise TypeError(
            f"cosmos server: 'task_description' must be str, got "
            f"{type(task_description).__name__}."
        )

    # Cosmos expects float64 proprio (rescale_proprio operates in float).
    if "proprio" in obs:
        obs["proprio"] = np.asarray(obs["proprio"], dtype=np.float64)
    # Images must be uint8 HxWxC (check_images_format inside cosmos).
    for k in ("primary_image", "wrist_image"):
        if k in obs and obs[k].dtype != np.uint8:
            obs[k] = obs[k].astype(np.uint8)

    guidance_params, _ = _pop_guidance_from_options(options)

    t0 = time.monotonic()
    if guidance_params is None:
        result = get_action(
            cfg_ns,
            model,
            dataset_stats,
            obs,
            task_label_or_embedding=task_description,
            seed=cfg_ns.seed,
            randomize_seed=cfg_ns.randomize_seed,
            num_denoising_steps_action=cfg_ns.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
            batch_size=1,
        )
    else:
        result = run_guided_get_action(
            cfg_ns,
            model,
            dataset_stats,
            obs,
            task_label_or_embedding=task_description,
            guidance_params=guidance_params,
            seed=cfg_ns.seed,
            randomize_seed=cfg_ns.randomize_seed,
            num_denoising_steps_action=cfg_ns.num_denoising_steps_action,
            generate_future_state_and_value_in_parallel=False,
            batch_size=1,
        )
    elapsed_ms = (time.monotonic() - t0) * 1000.0

    actions = _stack_actions(result["actions"], cfg_ns.chunk_size).astype(np.float32)
    # Trim to num_open_loop_steps if smaller; pad if larger (rare).
    n = cfg_ns.num_open_loop_steps
    if actions.shape[0] > n:
        actions = actions[:n]
    return {"actions": actions, "policy_timing": {"infer_ms": float(elapsed_ms)}}


# ---------------------------------------------------------------------------
# Server loop
# ---------------------------------------------------------------------------


def main(cfg: ServerConfig) -> None:
    print("Starting Cosmos-Policy inference server...")
    print(f"  Model path      : {cfg.model_path}")
    print(f"  Config          : {cfg.config}")
    print(f"  Chunk size      : {cfg.chunk_size} (open-loop {cfg.num_open_loop_steps})")
    print(f"  Denoising steps : {cfg.num_denoising_steps_action}")
    print(f"  Seed            : {cfg.seed} (deterministic={cfg.deterministic})")
    print(f"  Host:Port       : {cfg.host}:{cfg.port}")

    if cfg.deterministic:
        # Mirror run_libero_eval.py's deterministic mode toggle.
        os.environ["DETERMINISTIC"] = "true"

    cfg_ns = _build_cfg(cfg)

    print("Loading model + dataset stats + T5 cache (first run downloads ~5 GB)...")
    model, _model_config = get_model(cfg_ns)
    # Rebind to the grounded subclass so ``options``-carried ``__gp_*__``
    # guidance params steer the FlowUniPC denoising loop. The subclass is
    # a pure superset: when no guidance is active it delegates to the
    # parent, so this rebind is safe even on requests with options=None.
    model.__class__ = CosmosPolicyVideo2WorldModelGrounded
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    print("Model ready.")

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
                elif endpoint == "reset":
                    # Cosmos is stateless across calls; nothing to reset.
                    reply = {"ok": True}
                elif endpoint == "get_action":
                    obs = data.get("observation")
                    if not isinstance(obs, dict):
                        raise ValueError(
                            "get_action: 'observation' must be a dict, got "
                            f"{type(obs).__name__}"
                        )
                    # 'options' may carry __gp_*__ guidance keys (see
                    # _pop_guidance_from_options). When absent, _run_inference
                    # takes the unguided fast path that's bit-identical to the
                    # pre-guidance behaviour.
                    options = data.get("options")
                    reply = _run_inference(
                        cfg_ns, model, dataset_stats, obs, options=options
                    )
                else:
                    reply = {"error": f"unknown endpoint: {endpoint!r}"}
            except Exception as e:  # noqa: BLE001
                tb = traceback.format_exc()
                print(f"[cosmos server] error: {e}\n{tb}", flush=True)
                reply = {"error": f"{type(e).__name__}: {e}"}

            socket.send(msgpack.packb(reply, default=_encode_custom))
    except KeyboardInterrupt:
        print("\nShutting down cosmos server...")
    finally:
        socket.close(linger=0)


if __name__ == "__main__":
    main(tyro.cli(ServerConfig))
