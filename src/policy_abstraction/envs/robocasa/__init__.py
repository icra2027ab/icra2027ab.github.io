"""RoboCasa env adapter — server-side, runs in third_party/robocasa/.venv.

Mirrors the structure of :mod:`policy_abstraction.envs.libero`. Importing the
top-level package does NOT pull in robosuite/MuJoCo; the actual server lives
in :mod:`policy_abstraction.envs.robocasa.server` and is only imported by
``scripts/servers/serve_robocasa_env.py`` from inside the robocasa venv.
"""

__all__: list[str] = []
