"""Smoke tests for every script under ``examples/``.

Three tiers, cheapest first:

1. ``test_example_compiles`` — byte-compile every example. Catches syntax
   errors only. Microseconds per file.
2. ``test_example_imports`` — import each file as a module, which runs its
   top-level ``from open_perception import ...`` lines. Catches renamed
   or removed public API without invoking ``main()``.
3. ``test_example_runs`` — execute each script end-to-end as a subprocess.
   Loads real models and hits the GPU, so it's marked ``slow`` and only
   runs under ``pytest -m slow``. Examples that require an external API
   key are skipped when the key is missing.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_ROOT = Path(__file__).resolve().parent.parent / "examples"
PACKAGE_ROOT = EXAMPLES_ROOT.parent
EXAMPLE_FILES: list[Path] = sorted(
    p for p in EXAMPLES_ROOT.rglob("*.py") if p.name != "__init__.py"
)

# Tier-3 subprocess examples that short-circuit without a specific env var.
# Skipped when the var is unset so contributors without the key can still
# run the smoke suite.
REQUIRED_ENV: dict[str, str] = {
    "06_disambiguate_vlm.py": "OPENAI_API_KEY",
    "13_detect_gemini_sam2.py": "GEMINI_API_KEY",
}

RUN_TIMEOUT_S = 600


def _id(path: Path) -> str:
    return path.relative_to(EXAMPLES_ROOT).as_posix()


def test_examples_discovered() -> None:
    assert EXAMPLE_FILES, f"no example scripts found under {EXAMPLES_ROOT}"


@pytest.mark.parametrize("example", EXAMPLE_FILES, ids=_id)
def test_example_compiles(example: Path) -> None:
    compile(example.read_text(), str(example), "exec")


@pytest.mark.parametrize("example", EXAMPLE_FILES, ids=_id)
def test_example_imports(example: Path) -> None:
    spec = importlib.util.spec_from_file_location(f"op_example_{example.stem}", example)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(getattr(module, "main", None)), (
        f"{_id(example)}: expected a top-level main() callable"
    )


@pytest.mark.slow
@pytest.mark.parametrize("example", EXAMPLE_FILES, ids=_id)
def test_example_runs(example: Path) -> None:
    env_var = REQUIRED_ENV.get(example.name)
    if env_var and not os.environ.get(env_var):
        pytest.skip(f"{_id(example)} requires {env_var}")

    result = subprocess.run(
        [sys.executable, str(example)],
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{_id(example)} failed (exit {result.returncode}):\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
