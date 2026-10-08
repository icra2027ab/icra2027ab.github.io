"""Single Gr00t server entry that always installs the grounded action head.

Mirrors ``third_party/Isaac-GR00T/gr00t/eval/run_gr00t_server.py`` but, before
policy instantiation, swaps two class bindings so the resulting model and
policy carry the guidance hook:

- ``Gr00tN1d7ActionHead`` → ``Gr00tN1d7GroundedActionHead`` (extends
  ``get_action_with_features`` to apply guidance velocity correction)
- ``Gr00tPolicy`` → ``Gr00tGroundedPolicy`` (forwards ``options`` into the
  model and uses ``torch.no_grad`` instead of ``torch.inference_mode`` so
  autograd works inside the head's denoising loop)

When an inbound request carries no ``__gp_*__`` keys in ``options``, both
overrides delegate to their parents and behaviour is byte-identical to the
unguarded path.
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# --- Class swaps (must happen before policy instantiation) ------------------
import gr00t.model.gr00t_n1d7.gr00t_n1d7 as _gr00t_mod

from policy_abstraction.policies.groot_grounded_head import (
    Gr00tN1d7GroundedActionHead,
)
from policy_abstraction.policies.groot_grounded_policy import Gr00tGroundedPolicy

_gr00t_mod.Gr00tN1d7ActionHead = Gr00tN1d7GroundedActionHead
# Note: the upstream module also re-exports Gr00tN1d7ActionHead at package
# level; the only consumer is the model __init__ at line 530 of the upstream
# file, which resolves the name from this module's globals — so this single
# binding suffices.

# --- Mirror of run_gr00t_server.py ------------------------------------------
import tyro
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.policy.replay_policy import ReplayPolicy
from gr00t.policy.server_client import PolicyServer

DEFAULT_MODEL_SERVER_PORT = 5555


@dataclass
class ServerConfig:
    """Configuration for running the (grounded) GR00T inference server."""

    model_path: str | None = None
    """Path to the model checkpoint directory"""

    embodiment_tag: str = "new_embodiment"
    """Embodiment tag (name or value, case-insensitive). Run with --help to see known tags."""

    device: str = "cuda"
    """Device to run the model on"""

    dataset_path: str | None = None
    """Path to the dataset for replay trajectory"""

    modality_config_path: str | None = None
    """Path to the modality configuration file"""

    execution_horizon: int | None = None
    """Policy execution horizon during inference. Required when --dataset-path is set (ReplayPolicy)."""

    host: str = "0.0.0.0"
    """Host address for the server"""

    port: int = DEFAULT_MODEL_SERVER_PORT
    """Port number for the server"""

    strict: bool = True
    """Whether to enforce strict input and output validation"""

    use_sim_policy_wrapper: bool = False
    """Whether to use the sim policy wrapper"""


def main(config: ServerConfig):
    config.embodiment_tag = EmbodimentTag.resolve(config.embodiment_tag)
    print("Starting GR00T (grounded) inference server...")
    print(f"  Embodiment tag: {config.embodiment_tag}")
    print(f"  Model path: {config.model_path}")
    print(f"  Device: {config.device}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(
        "  Action head: Gr00tN1d7GroundedActionHead "
        "(falls through to upstream when no __gp_*__ options are present)"
    )

    if config.model_path is not None:
        if config.model_path.startswith("/") and not os.path.exists(config.model_path):
            raise FileNotFoundError(f"Model path {config.model_path} does not exist")
        policy = Gr00tGroundedPolicy(
            embodiment_tag=config.embodiment_tag,
            model_path=config.model_path,
            device=config.device,
            strict=config.strict,
        )
    elif config.dataset_path is not None:
        if config.execution_horizon is None:
            raise ValueError(
                "--execution-horizon is required when --dataset-path is set "
                "(ReplayPolicy needs a positive integer to advance episodes)."
            )
        if config.execution_horizon <= 0:
            raise ValueError(
                f"--execution-horizon must be positive; got {config.execution_horizon}."
            )

        modality_configs: dict[str, ModalityConfig] | None = None
        if config.modality_config_path is not None:
            config_path = Path(config.modality_config_path)
            if config_path.suffix == ".py":
                sys.path.append(str(config_path.parent))
                importlib.import_module(config_path.stem)
                print(f"Loaded modality config: {config_path}")
            elif config_path.suffix == ".json":
                with open(config.modality_config_path, "r") as f:
                    raw = json.load(f)
                modality_configs = {k: ModalityConfig(**v) for k, v in raw.items()}
            else:
                raise ValueError(
                    f"Unsupported modality config format: {config_path.suffix}. Use .py or .json"
                )

        if modality_configs is None:
            from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

            modality_configs = MODALITY_CONFIGS.get(config.embodiment_tag.value)
            if modality_configs is None:
                raise ValueError(
                    f"No built-in modality config for embodiment tag "
                    f"'{config.embodiment_tag.name}' (value='{config.embodiment_tag.value}'). "
                    f"Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
                    f"Please provide --modality-config-path (JSON or .py) "
                    f"when using this tag with ReplayPolicy."
                )
        policy = ReplayPolicy(
            dataset_path=config.dataset_path,
            modality_configs=modality_configs,
            execution_horizon=config.execution_horizon,
            strict=config.strict,
        )
    else:
        raise ValueError("Either model_path or dataset_path must be provided")

    if config.use_sim_policy_wrapper:
        from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper

        policy = Gr00tSimPolicyWrapper(policy)

    server = PolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
    )

    print(f"\n✓ Server ready — listening on {config.host}:{config.port}\n")

    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down server...")


if __name__ == "__main__":
    cfg = tyro.cli(ServerConfig)
    main(cfg)
