"""Every runnable ``scripts/evaluate.py`` command in the docs must parse.

tyro's subcommand dispatch requires ``--env.*`` flags to come *after*
``env:<name>`` and ``--policy.*`` after ``policy:<name>``. Getting that wrong
produces a command that looks right, is copied into a terminal, and exits 2 —
which is easy to miss when writing docs and annoying to hit at the robot. This
sweeps the docs and runs each command through the real ``Args`` parser.

Parsing only: no servers are started and no hardware is touched.
"""

from __future__ import annotations

import re
import shlex
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

import pytest
import tyro

from policy_abstraction import PROJECT_ROOT

# `scripts/` isn't a package; import evaluate.py by path.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import evaluate  # noqa: E402

DOC_DIR = PROJECT_ROOT / "docs"
# Lines carrying an ellipsis are illustrative fragments, not runnable commands.
_PLACEHOLDERS = ("...", "…", "<")

# Docs whose commands predate the env/policy subcommand split and pass
# `--env.*` flags without naming `env:libero` first. They exit 2 today. Not
# fixed here because they belong to the LIBERO workflow, not this one — but
# they are the same defect this test exists to catch, so they are listed
# rather than filtered out silently. Delete an entry once its doc is fixed.
_KNOWN_STALE = {
    "SETUP.md",
    "install_openpi.md",
    "libero_pro_eval_commands.md",
    "libero_pro_setup.md",
}


def _commands() -> list[tuple[str, str]]:
    """Every runnable ``evaluate.py`` invocation in a fenced bash block."""
    found: list[tuple[str, str]] = []
    for doc in sorted(DOC_DIR.glob("*.md")):
        for block in re.findall(r"```bash\n(.*?)```", doc.read_text(), re.S):
            # Fold shell line-continuations into one line first.
            for line in block.replace("\\\n", " ").splitlines():
                line = line.strip()
                if "scripts/evaluate.py" not in line or line.startswith("#"):
                    continue
                if any(p in line for p in _PLACEHOLDERS):
                    continue
                found.append((doc.name, line))
    return found


COMMANDS = _commands()


def test_docs_contain_runnable_commands():
    """Guard against the extractor silently matching nothing."""
    assert COMMANDS, f"no runnable evaluate.py commands found under {DOC_DIR}"


@pytest.mark.parametrize(
    "doc, command",
    COMMANDS,
    ids=[f"{doc}:{i}" for i, (doc, _) in enumerate(COMMANDS)],
)
def test_doc_command_parses(doc: str, command: str):
    if doc in _KNOWN_STALE:
        pytest.xfail(
            f"{doc} predates the env/policy subcommand split — see _KNOWN_STALE"
        )

    argv = shlex.split(command)
    argv = argv[argv.index("scripts/evaluate.py") + 1 :]

    buf = StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            tyro.cli(evaluate.Args, args=argv)
    except SystemExit as exc:
        pytest.fail(
            f"{doc}: command exits {exc.code} instead of parsing.\n"
            f"  {command}\n"
            f"tyro output:\n{buf.getvalue()[:1500]}"
        )
