"""Entry-point script for the LIBERO environment WebSocket server.

Run this in the Python environment that has LIBERO (the "simulator Python")
*before* launching the evaluation script:

    # Terminal 1 – simulator Python (may be a different venv / conda env)
    python scripts/servers/serve_libero_env.py --host 0.0.0.0 --port 8765

    # Terminal 2 – policy Python
    python scripts/evaluate.py \\
        --env-host localhost --env-port 8765 \\
        --host 0.0.0.0 --port 8000

The server will block and handle one client connection at a time.
Press Ctrl-C to stop.
"""

import dataclasses
import logging
import sys

import tyro

sys.path.append("./src")  # Ensure the root directory is in the Python path
from policy_abstraction import PROJECT_ROOT
from policy_abstraction.envs.libero.server import LiberoEnvServer


@dataclasses.dataclass
class Args:
    """Arguments for the LIBERO environment server."""

    host: str = "0.0.0.0"
    """Interface to bind to (0.0.0.0 accepts all incoming connections)."""

    port: int = 8765
    """WebSocket port to listen on."""

    task_suite_name: str = "test"
    """LIBERO-PRO task suite name (before perturbation suffixing)."""

    evaluation_config_path: str = str(
        PROJECT_ROOT
        / "src/policy_abstraction/envs/libero/configs/evaluation_config.yaml"
    )
    """Path to the LIBERO-PRO evaluation_config.yaml file."""

    use_swap: bool = False
    use_object: bool = False
    use_language: bool = False
    use_task: bool = False
    use_environment: bool = False

    seed: int = 0
    """Random seed used by the env; affects object positions."""


def main(args: Args) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = LiberoEnvServer(
        host=args.host,
        port=args.port,
        task_suite_name=args.task_suite_name,
        evaluation_config_path=args.evaluation_config_path,
        use_swap=args.use_swap,
        use_object=args.use_object,
        use_language=args.use_language,
        use_task=args.use_task,
        use_environment=args.use_environment,
        seed=args.seed,
    )
    server.serve_forever()


if __name__ == "__main__":
    main(tyro.cli(Args))
