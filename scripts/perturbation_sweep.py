"""Sweep ``evaluate.py`` over perturbation flags and task suites.

For each (task_suite, perturbation) pair we launch
``scripts/evaluate.py`` with that perturbation flag set to True and
the others (including ``environment``) set to False. Any extra CLI args are
forwarded verbatim to the eval script.

If multiple GPUs are available, runs are dispatched in parallel — one per GPU.
The ``instance`` argument is set to the GPU id so policy/env ports don't clash
(see ``Args.__post_init__`` in ``scripts/evaluate.py``).

Usage::

    python scripts/perturbation_sweep.py [sweep flags] [forwarded eval args]
    # 4 perturbations x 1 default suite, 4 GPUs:
    python scripts/perturbation_sweep.py --gpus 0,1,2,3
    # 4 perturbations x 3 suites = 12 runs, dispatched 4-at-a-time:
    python scripts/perturbation_sweep.py --gpus 0,1,2,3 \\
        --task-suites libero_object libero_spatial libero_goal
    # Forward eval-only args (anything we don't recognize):
    python scripts/perturbation_sweep.py --output.run my_sweep \\
        --iter.num-trials-per-task 5
    # GR00T backend (anything after the sweep flags is forwarded verbatim;
    # `--instance` is set per-GPU automatically, so policy/env ports don't
    # clash. GR00T needs the LIBERO native 256x256 resolution, not pi05's
    # 224, hence the explicit --camera.resize-size override. Backend is
    # selected via the tyro subcommand `policy:groot`; backend-specific
    # flags like --policy.model-path must follow that token.):
    python scripts/perturbation_sweep.py --gpus 0,1,2,3 \\
        --camera.resize-size 256 \\
        policy:groot \\
        --policy.model-path checkpoints/GR00T-N1.7-LIBERO/libero_10

    # Config-driven sweep (--config <yaml>): cartesian matrix of axes (e.g.
    # policy backend × perturbation × task_suite × guidance.factor) with
    # exclude rules and per-cell output naming templates. Each cell becomes
    # one ``evaluate.py`` invocation; cells share the GPU pool just like
    # the CLI mode. See ``configs/sweeps/overnight_example.yaml``.
    python scripts/perturbation_sweep.py --gpus 0,1,2,3 \\
        --config configs/sweeps/overnight_example.yaml

Outputs (under ``--log-dir``):
  * ``<suite>__<perturbation>_gpu<id>.log`` — per-run stdout/stderr
  * ``<suite>__<perturbation>_gpu<id>.sh`` — exact command launched
  * ``sweep_summary.json``                  — per-run rc + error, plus an
    ``error_summary`` aggregating loguru ERROR-level messages from every
    run's log (count + which runs they fired in)
"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import itertools
import json
import os
import queue
import re
import shutil
import signal
import socket
import string
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from policy_abstraction.utils.seeding import seed_everything

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = "scripts/evaluate.py"

PERTURBATIONS = ("swap", "task", "object", "language")
ALL_PERTURBATION_FLAGS = (*PERTURBATIONS, "environment")

# Base ports — must match the defaults in
# ``evaluate.Args.__post_init__``:
#   policy.port = POLICY_PORT_BASE + instance
#   env.port    = ENV_PORT_BASE    + instance
POLICY_PORT_BASE = 8000
ENV_PORT_BASE = 8765

# Loguru's default record header: `... | ERROR    | name:func:line - message`.
# Anchored to ` | ERROR ` so we don't false-match the literal string elsewhere.
_LOGURU_ERROR_RE = re.compile(
    r"\|\s*ERROR\s*\|\s*[\w\.]+:[\w_<>\.]+:\d+\s*-\s*(.+?)\s*$"
)


# ---------------------------------------------------------------------------
# Tiny ANSI colour helper. Auto-disabled when stdout isn't a TTY (e.g.
# redirected to a file) or NO_COLOR is set.
# ---------------------------------------------------------------------------
class C:
    enabled = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    @classmethod
    def _w(cls, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if cls.enabled else s

    @classmethod
    def bold(cls, s: str) -> str:
        return cls._w("1", s)

    @classmethod
    def dim(cls, s: str) -> str:
        return cls._w("2", s)

    @classmethod
    def red(cls, s: str) -> str:
        return cls._w("31", s)

    @classmethod
    def green(cls, s: str) -> str:
        return cls._w("32", s)

    @classmethod
    def yellow(cls, s: str) -> str:
        return cls._w("33", s)

    @classmethod
    def blue(cls, s: str) -> str:
        return cls._w("34", s)

    @classmethod
    def magenta(cls, s: str) -> str:
        return cls._w("35", s)

    @classmethod
    def cyan(cls, s: str) -> str:
        return cls._w("36", s)


# Flags the sweep manages itself — drop them from forwarded extras to avoid
# silently overriding the per-run perturbation/instance/gpu we set.
_MANAGED_BOOL_FLAGS = {f"--env.perturb.{p}" for p in ALL_PERTURBATION_FLAGS} | {
    f"--env.perturb.no-{p}" for p in ALL_PERTURBATION_FLAGS
}
_MANAGED_VALUE_FLAGS = {
    "--instance",
    "--policy.gpus",
    "--policy.port",
    "--env.port",
    "--iter.start-trial-idx",
    "--iter.start_trial_idx",
    "--iter.start-task-idx",
    "--iter.start_task_idx",
}


# ---------------------------------------------------------------------------
# Live-subprocess tracker — so Ctrl-C tears down every child eval (and, via
# the eval's own SIGTERM handler, the policy + env servers it spawned).
# ---------------------------------------------------------------------------
_live_procs: dict[int, subprocess.Popen] = {}
_live_lock = threading.Lock()
_shutdown = threading.Event()


def _register(proc: subprocess.Popen) -> None:
    with _live_lock:
        _live_procs[proc.pid] = proc


def _unregister(proc: subprocess.Popen) -> None:
    with _live_lock:
        _live_procs.pop(proc.pid, None)


def _kill_pgroup(proc: subprocess.Popen, sig: int) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _kill_all_children(reason: str, grace_s: float = 10.0) -> None:
    """SIGTERM every tracked child group, escalate to SIGKILL after grace_s."""
    with _live_lock:
        procs = list(_live_procs.values())
    if not procs:
        return
    print(
        C.yellow(
            f"\n[sweep] {reason}: terminating {len(procs)} child process group(s)…"
        ),
        file=sys.stderr,
        flush=True,
    )
    for p in procs:
        _kill_pgroup(p, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline and any(p.poll() is None for p in procs):
        time.sleep(0.2)
    for p in procs:
        if p.poll() is None:
            print(
                C.red(f"[sweep] force-killing PID {p.pid} (SIGKILL)"),
                file=sys.stderr,
                flush=True,
            )
            _kill_pgroup(p, signal.SIGKILL)


def _signal_handler(signum: int, frame) -> None:
    name = signal.Signals(signum).name
    if _shutdown.is_set():
        # Second signal — the user wants out *now*.
        print(
            C.red(f"\n[sweep] second {name} — forcing exit"),
            file=sys.stderr,
            flush=True,
        )
        # Best-effort SIGKILL on whatever's still alive, then bail.
        with _live_lock:
            procs = list(_live_procs.values())
        for p in procs:
            _kill_pgroup(p, signal.SIGKILL)
        os._exit(130)
    _shutdown.set()
    _kill_all_children(f"received {name}")


def _is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if ``port`` is already bound on ``host``.

    Uses a SO_REUSEADDR-disabled bind probe — matches what the policy/env
    server child processes do at startup, so a False here is a strong signal
    the launch will succeed (and a True is a near-certain conflict).
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def required_ports(gpus: list[int], port_offset: int) -> list[tuple[int, int, int]]:
    """Return ``(gpu, policy_port, env_port)`` for every GPU in the sweep."""
    return [
        (
            g,
            POLICY_PORT_BASE + g + port_offset,
            ENV_PORT_BASE + g + port_offset,
        )
        for g in gpus
    ]


def find_port_conflicts(
    gpus: list[int], port_offset: int
) -> list[tuple[int, list[int]]]:
    """Return ``(gpu, [busy_ports])`` for any GPU whose policy/env port is taken."""
    conflicts: list[tuple[int, list[int]]] = []
    for gpu, policy_port, env_port in required_ports(gpus, port_offset):
        busy = [p for p in (policy_port, env_port) if _is_port_in_use(p)]
        if busy:
            conflicts.append((gpu, busy))
    return conflicts


def suggest_port_offset(
    gpus: list[int], start: int = 1, max_search: int = 200
) -> int | None:
    """Search for the smallest offset >= ``start`` that frees all required ports.

    Returns None if no clean offset is found within ``max_search`` attempts —
    in that case the caller should ask the user to pick one manually.
    """
    for offset in range(start, start + max_search):
        if not find_port_conflicts(gpus, offset):
            return offset
    return None


def query_gpu_memory() -> list[tuple[int, int, int]]:
    """Return ``(index, memory_used_mib, memory_total_mib)`` for each GPU.

    Empty list if ``nvidia-smi`` is missing or fails.
    """
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    rows: list[tuple[int, int, int]] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
        except ValueError:
            continue
    return rows


def select_idle_gpus(
    rows: list[tuple[int, int, int]], max_used_mib: int
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Split GPUs into (idle_indices, busy_rows) by memory used."""
    idle: list[int] = []
    busy: list[tuple[int, int, int]] = []
    for idx, used, total in rows:
        (idle.append(idx) if used <= max_used_mib else busy.append((idx, used, total)))
    return idle, busy


def confirm(prompt: str) -> bool:
    """Prompt y/N; treat EOF/SIGINT as 'no'."""
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return ans in ("y", "yes")


def extract_extra_value(extras: list[str], flag: str) -> str | None:
    """Read the value of a forwarded ``--flag VALUE`` / ``--flag=VALUE`` arg.

    Non-destructive: ``extras`` is still forwarded verbatim to the eval script.
    Returns the *last* occurrence (mirrors argparse semantics).
    """
    found: str | None = None
    i = 0
    while i < len(extras):
        tok = extras[i]
        if tok == flag and i + 1 < len(extras):
            found = extras[i + 1]
            i += 2
            continue
        if tok.startswith(f"{flag}="):
            found = tok.split("=", 1)[1]
        i += 1
    return found


def filter_extras(extras: list[str], extra_value_flags: set[str]) -> list[str]:
    """Drop sweep-managed flags from forwarded args, with a warning.

    ``extra_value_flags`` are additional ``--flag value`` style args (e.g.
    ``--env.task-suite-name`` when ``--task-suites`` is in use) to strip.
    """
    value_flags = _MANAGED_VALUE_FLAGS | extra_value_flags
    filtered: list[str] = []
    skip_next = False
    for i, tok in enumerate(extras):
        if skip_next:
            skip_next = False
            continue
        head = tok.split("=", 1)[0]
        if head in _MANAGED_BOOL_FLAGS:
            print(
                f"[sweep] dropping conflicting forwarded flag: {tok}",
                file=sys.stderr,
                flush=True,
            )
            continue
        if head in value_flags:
            print(
                f"[sweep] dropping conflicting forwarded flag: {tok}",
                file=sys.stderr,
                flush=True,
            )
            if "=" not in tok and i + 1 < len(extras):
                skip_next = True
            continue
        filtered.append(tok)
    return filtered


def _split_policy_block(extras: list[str]) -> tuple[list[str], list[str]]:
    """Partition ``extras`` into ``(top_level_tokens, policy_block_tokens)``.

    tyro requires top-level args (``--seed``, ``--iter.*``, ``--guidance.*``,
    …) to appear before any ``policy:X`` subcommand token. But sweep extras
    are assembled by concatenating multiple sources (base_args, axis tokens,
    user CLI extras), so a top-level arg from a later source can end up
    after the policy block if we don't reorder.

    This split preserves relative order within each bucket. Policy-block
    tokens are ``policy:X`` plus any ``--policy.<field>`` flag with its
    value (a value is "the next token that isn't itself a flag or a
    subcommand selector"). Everything else lands in the top-level bucket.
    """
    top: list[str] = []
    pol: list[str] = []
    i = 0
    while i < len(extras):
        tok = extras[i]
        if tok.startswith("policy:"):
            pol.append(tok)
            i += 1
        elif tok.startswith("--policy."):
            pol.append(tok)
            if "=" not in tok and i + 1 < len(extras):
                nxt = extras[i + 1]
                if (
                    not nxt.startswith("--")
                    and not nxt.startswith("policy:")
                    and not nxt.startswith("env:")
                ):
                    pol.append(nxt)
                    i += 2
                    continue
            i += 1
        else:
            top.append(tok)
            i += 1
    return top, pol


def build_cmd(
    perturbation: str | None,
    instance: int,
    extras: list[str],
    task_suite: str | None = None,
    port_offset: int = 0,
    cell_run: str | None = None,
    exp_name: str | None = None,
) -> list[str]:
    """Build the ``uv run python …`` command for one sweep cell.

    ``port_offset`` shifts only the policy/env ports — it is *not* added to
    ``--instance``, since the eval script also derives ``policy.gpus`` from
    ``--instance`` and the sweep already sets ``CUDA_VISIBLE_DEVICES`` to the
    chosen GPU. So we pass ``--instance gpu`` and override the two ports
    explicitly when an offset is in effect.

    ``cell_run`` / ``exp_name`` are config-mode overrides — when set they are
    emitted as ``--output.run`` / ``--output.exp-name`` so each cell gets a
    dedicated output directory. In legacy CLI mode both are None and
    ``evaluate.py``'s default exp_name (``{task_suite}/{perturbation}``) is
    used unchanged.

    ``perturbation`` may be None in config mode when the sweep doesn't vary
    perturbation — in that case the perturbation flag bundle is omitted and
    evaluate.py's defaults apply.

    ``evaluate.py`` exposes both ``policy`` and ``env`` as tyro subcommand
    unions. tyro enforces the order
    ``[top-level args] [policy:X [policy args]] env:libero [env args]``:

      * ``--policy.port`` and any ``--policy.<field>`` must follow a
        ``policy:{openpi,groot,cosmos,lerobot}`` token. If the user didn't supply
        one but we need ``--policy.port``, we inject ``policy:openpi``.
      * ``--env.port`` / ``--env.task-suite-name`` / ``--env.perturb.<x>``
        must follow the ``env:libero`` token. The sweep is libero-specific,
        so we always inject that token.
    """
    cmd = [
        "uv",
        "run",
        "python",
        EVAL_SCRIPT,
        "--instance",
        str(instance),
    ]
    if cell_run is not None:
        cmd += ["--output.run", cell_run]
    if exp_name is not None:
        cmd += ["--output.exp-name", exp_name]

    # Split extras into [top-level tokens] + [policy:X block]. tyro requires
    # top-level args before the policy subcommand token, but our extras come
    # from multiple concatenated sources so a top-level arg from a later
    # source may sit after the policy block — the split restores the
    # required order.
    top_extras, policy_extras = _split_policy_block(extras)
    cmd.extend(top_extras)

    if port_offset:
        policy_port_args = [
            "--policy.port",
            str(POLICY_PORT_BASE + instance + port_offset),
        ]
        if policy_extras and policy_extras[0].startswith("policy:"):
            # Insert immediately after the policy:X token so the new flag
            # is unambiguously scoped to that backend.
            policy_extras = [policy_extras[0], *policy_port_args, *policy_extras[1:]]
        else:
            policy_extras = ["policy:openpi", *policy_port_args, *policy_extras]
    cmd.extend(policy_extras)

    cmd.append("env:libero")
    if port_offset:
        cmd += ["--env.port", str(ENV_PORT_BASE + instance + port_offset)]
    if task_suite is not None:
        cmd += ["--env.task-suite-name", task_suite]
    if perturbation is not None:
        for p in ALL_PERTURBATION_FLAGS:
            cmd.append(
                f"--env.perturb.{p}" if p == perturbation else f"--env.perturb.no-{p}"
            )
    return cmd


@dataclasses.dataclass
class Cell:
    """One sweep cell: a single ``evaluate.py`` invocation.

    ``axis_values`` maps axis_name -> picked value_name. ``extra_tokens`` is
    the concatenation of CLI tokens contributed by every axis value, in axis
    declaration order. ``cell_run`` / ``exp_name`` are rendered from the
    config's templates and passed to ``evaluate.py`` via ``--output.run`` /
    ``--output.exp-name``. ``task_suite`` and ``perturbation`` are extracted
    from the picked axis values when the corresponding convenience axes are
    present (None if the user didn't sweep over them).
    """

    axis_values: dict[str, str]
    extra_tokens: list[str]
    cell_run: str | None  # None in legacy CLI mode (lets evaluate.py default)
    exp_name: str | None
    task_suite: str | None
    perturbation: str | None
    label: str  # filename-safe identifier for logs


def _coerce_axis(axis_name: str, raw: object) -> dict[str, list[str]]:
    """Normalise an axis spec into ``{value_name: [cli tokens]}``.

    Accepted shapes:
      * ``{value_name: [tokens]}``  — explicit, preferred.
      * ``[value_name, ...]``        — shorthand; tokens default to ``[]``
        (caller must combine with a known flag template, e.g. for the
        convenience ``perturbations`` / ``task_suites`` axes).
    """
    if isinstance(raw, dict):
        out: dict[str, list[str]] = {}
        for k, v in raw.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"axis {axis_name!r}: value name must be a string, got {k!r}"
                )
            if not isinstance(v, list) or not all(isinstance(t, str) for t in v):
                raise ValueError(
                    f"axis {axis_name!r}, value {k!r}: must be a list of "
                    f"strings (CLI tokens), got {v!r}"
                )
            out[k] = [str(t) for t in v]
        return out
    if isinstance(raw, list):
        return {str(v): [] for v in raw}
    raise ValueError(
        f"axis {axis_name!r}: expected dict or list, got {type(raw).__name__}"
    )


def _required_template_keys(template: str) -> set[str]:
    """Return the names of all ``{key}`` substitutions in ``template``."""
    return {field for _, field, _, _ in string.Formatter().parse(template) if field}


def expand_matrix(
    cfg: dict, user_extras: list[str]
) -> tuple[list[Cell], list[str], str]:
    """Expand a sweep config into a list of cells.

    Returns ``(cells, axis_names_in_order, output_run)``. ``user_extras`` are
    appended verbatim to every cell's CLI (after base_args and axis tokens)
    so command-line overrides take precedence over the config.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Sweep config root must be a mapping (YAML dict).")

    output_run = cfg.get("output_run")
    if not isinstance(output_run, str) or not output_run:
        raise ValueError("Sweep config must define a non-empty 'output_run' string.")

    base_args = cfg.get("base_args", []) or []
    if not isinstance(base_args, list) or not all(
        isinstance(t, str) for t in base_args
    ):
        raise ValueError("'base_args' must be a list of strings (CLI tokens).")

    # Reserved flags: the sweep injects these per-cell, so neither base_args
    # nor any matrix axis value may emit them. Fail loudly at config-load
    # rather than silently overriding.
    _SWEEP_RESERVED = {"--output.run", "--output.exp-name", "--output.exp_name"}

    def _check_reserved(tokens: list[str], origin: str) -> None:
        for t in tokens:
            head = t.split("=", 1)[0]
            if head in _SWEEP_RESERVED:
                raise ValueError(
                    f"{origin} contains reserved flag {head!r}. The sweep "
                    "injects --output.run and --output.exp-name per cell "
                    "from 'cell_run_template' / 'cell_exp_name_template'."
                )

    _check_reserved(base_args, "'base_args'")

    # Convenience axes: fold ``perturbations`` / ``task_suites`` into the
    # matrix so they participate in the cartesian product alongside any
    # user-defined axes. Their CLI tokens are filled in below (after the
    # cartesian product), since perturbations need the full mutual-exclusion
    # flag bundle (the picked one True, all others False).
    axes: dict[str, dict[str, list[str]]] = {}
    perturbations = cfg.get("perturbations")
    if perturbations is not None:
        if not isinstance(perturbations, list) or not perturbations:
            raise ValueError("'perturbations' must be a non-empty list.")
        for p in perturbations:
            if p not in PERTURBATIONS:
                raise ValueError(
                    f"'perturbations' contains unknown value {p!r}; "
                    f"valid: {list(PERTURBATIONS)}"
                )
        axes["perturbation"] = {str(p): [] for p in perturbations}
    task_suites = cfg.get("task_suites")
    if task_suites is not None:
        if not isinstance(task_suites, list) or not task_suites:
            raise ValueError("'task_suites' must be a non-empty list.")
        axes["task_suite"] = {str(s): [] for s in task_suites}

    raw_matrix = cfg.get("matrix", {}) or {}
    if not isinstance(raw_matrix, dict):
        raise ValueError("'matrix' must be a mapping of axis_name -> values.")
    for axis_name, raw in raw_matrix.items():
        if axis_name in axes:
            raise ValueError(
                f"axis {axis_name!r} is defined both via the convenience "
                f"'{axis_name}s' shorthand and under 'matrix'. Pick one."
            )
        if not isinstance(axis_name, str) or not axis_name:
            raise ValueError(
                f"matrix axis name must be a non-empty string; got {axis_name!r}"
            )
        axes[axis_name] = _coerce_axis(axis_name, raw)
        for vname, vtokens in axes[axis_name].items():
            _check_reserved(vtokens, f"matrix.{axis_name}.{vname}")

    if not axes:
        raise ValueError(
            "Sweep config produced no axes. Define at least one of "
            "'perturbations', 'task_suites', or 'matrix'."
        )

    axis_names = list(axes.keys())

    # Default templates: cell_run defaults to output_run (everyone shares one
    # top-level run dir); exp_name joins value names by '__' in axis order.
    cell_run_template = cfg.get("cell_run_template", "{output_run}")
    cell_exp_name_template = cfg.get(
        "cell_exp_name_template",
        "__".join("{" + n + "}" for n in axis_names),
    )
    if not isinstance(cell_run_template, str) or not isinstance(
        cell_exp_name_template, str
    ):
        raise ValueError(
            "'cell_run_template' and 'cell_exp_name_template' must be strings."
        )

    allowed_keys = set(axis_names) | {"output_run", "cell_index"}
    for label, tpl in (
        ("cell_run_template", cell_run_template),
        ("cell_exp_name_template", cell_exp_name_template),
    ):
        unknown = _required_template_keys(tpl) - allowed_keys
        if unknown:
            raise ValueError(
                f"{label!r} references unknown placeholder(s) {sorted(unknown)}. "
                f"Available: {sorted(allowed_keys)}."
            )

    # Exclude rules: list of partial {axis: value} maps; cell drops if any
    # rule fully matches its axis_values.
    exclude_rules = cfg.get("exclude", []) or []
    if not isinstance(exclude_rules, list) or not all(
        isinstance(r, dict) for r in exclude_rules
    ):
        raise ValueError("'exclude' must be a list of {axis: value} dicts.")
    for rule in exclude_rules:
        for k, v in rule.items():
            if k not in axes:
                raise ValueError(
                    f"exclude rule references unknown axis {k!r}; known: {axis_names}"
                )
            if v not in axes[k]:
                raise ValueError(
                    f"exclude rule references unknown value {v!r} on axis {k!r}; "
                    f"valid: {list(axes[k].keys())}"
                )

    def _is_excluded(values: dict[str, str]) -> bool:
        for rule in exclude_rules:
            if all(values.get(k) == v for k, v in rule.items()):
                return True
        return False

    cells: list[Cell] = []
    seen_names: dict[tuple[str, str], dict[str, str]] = {}

    for combo in itertools.product(*(axes[a].items() for a in axis_names)):
        axis_values = {a: vname for a, (vname, _) in zip(axis_names, combo)}
        if _is_excluded(axis_values):
            continue

        tokens: list[str] = list(base_args)
        for axis, (_, axis_tokens) in zip(axis_names, combo):
            tokens.extend(axis_tokens)

        suite = axis_values.get("task_suite")
        perturbation = axis_values.get("perturbation")

        cell_index = len(cells)
        fmt_vars = {**axis_values, "output_run": output_run, "cell_index": cell_index}
        cell_run = cell_run_template.format(**fmt_vars)
        exp_name = cell_exp_name_template.format(**fmt_vars)

        if "/" in exp_name:
            raise ValueError(
                f"cell_exp_name {exp_name!r} contains '/'. Use 'cell_run_template' "
                "for sub-directories; 'cell_exp_name_template' must render a single "
                "path component."
            )

        key = (cell_run, exp_name)
        if key in seen_names:
            raise ValueError(
                f"Two cells render to the same output path "
                f"data/{cell_run}/{exp_name}: {seen_names[key]} vs {axis_values}. "
                "Add more axes to the templates so each cell gets a unique name."
            )
        seen_names[key] = axis_values

        label = exp_name  # also used for log filenames
        tokens.extend(user_extras)

        cells.append(
            Cell(
                axis_values=axis_values,
                extra_tokens=tokens,
                cell_run=cell_run,
                exp_name=exp_name,
                task_suite=suite,
                perturbation=perturbation,
                label=label,
            )
        )

    return cells, axis_names, output_run


def _run_label(perturbation: str, task_suite: str | None) -> str:
    """Filename-safe label identifying a sweep cell."""
    return f"{task_suite}__{perturbation}" if task_suite else perturbation


def _extract_errors_from_log(log_path: Path) -> list[str]:
    """Pull loguru ERROR-level message lines from a child eval log.

    One entry per ERROR record; multi-line tracebacks attached via
    ``exc_info=True`` are collapsed to the header line (the message portion).
    Returns ``[]`` if the log is missing or unreadable — extraction failures
    must not break the sweep summary.
    """
    if not log_path.exists():
        return []
    out: list[str] = []
    try:
        with log_path.open("r", errors="replace") as f:
            for line in f:
                m = _LOGURU_ERROR_RE.search(line)
                if m:
                    out.append(m.group(1))
    except OSError:
        pass
    return out


def _read_cell_metrics(cell_run: str | None, exp_name: str | None) -> dict | None:
    """Read ``data/<cell_run>/<exp_name>/summary.json`` if it exists.

    Returns ``{success_rate, episodes, time, ...}`` from the ``total`` block,
    or None when the file is missing / unreadable. Used to fold per-cell
    success metrics into the sweep summary so an overnight run is directly
    comparable across experiments.
    """
    if not cell_run or not exp_name:
        return None
    summary_path = REPO_ROOT / "data" / cell_run / exp_name / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    total = payload.get("total") if isinstance(payload, dict) else None
    if not isinstance(total, dict):
        return None
    return {
        "success_rate": total.get("success_rate"),
        "episodes": total.get("episodes"),
        "successes": total.get("successes"),
        "time": total.get("time"),
        "summary_path": str(summary_path),
    }


def run_one(
    perturbation: str | None,
    gpu: int,
    extras: list[str],
    log_dir: Path,
    timeout_s: int,
    task_suite: str | None = None,
    port_offset: int = 0,
    cell_run: str | None = None,
    exp_name: str | None = None,
    label: str | None = None,
    axis_values: dict[str, str] | None = None,
) -> dict:
    """Launch one sweep cell. Captures stdout+stderr to a log.

    Returns a result dict with ``rc`` and ``error`` (None on success). When
    ``cell_run`` / ``exp_name`` are set (config mode), reads the child run's
    ``summary.json`` after completion and copies ``total`` metrics into the
    result under ``metrics``.
    """
    cmd = build_cmd(
        perturbation,
        gpu,
        extras,
        task_suite=task_suite,
        port_offset=port_offset,
        cell_run=cell_run,
        exp_name=exp_name,
    )
    env = os.environ.copy()
    # Ensure the *parent* eval process (perception/retrieval models) lands on
    # the same GPU as the policy server it spawns.
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    if label is None:
        label = _run_label(perturbation or "default", task_suite)
    log_path = log_dir / f"{label}_gpu{gpu}.log"
    cmd_path = log_dir / f"{label}_gpu{gpu}.sh"
    cmd_path.write_text(" ".join(cmd) + "\n")

    if _shutdown.is_set():
        return {
            "task_suite": task_suite,
            "perturbation": perturbation,
            "cell_run": cell_run,
            "exp_name": exp_name,
            "axis_values": axis_values or {},
            "gpu": gpu,
            "rc": -1,
            "elapsed_s": 0.0,
            "error": "skipped: sweep shutting down before this run started",
            "log": None,
            "log_errors": [],
            "metrics": None,
            "cmd": cmd,
        }

    print(
        C.cyan("[sweep] START ") + f"{label:>40s} gpu={gpu} -> {log_path.name}",
        flush=True,
    )
    t0 = time.time()
    rc = -1
    error: str | None = None
    try:
        with log_path.open("w") as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _register(proc)
            try:
                rc = proc.wait(timeout=timeout_s)
            except subprocess.TimeoutExpired:
                error = f"timed out after {timeout_s}s (see {log_path})"
                _kill_pgroup(proc, signal.SIGTERM)
                try:
                    rc = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _kill_pgroup(proc, signal.SIGKILL)
                    rc = proc.wait()
            finally:
                _unregister(proc)
        if error is None and rc != 0:
            error = (
                "killed by sweep (Ctrl-C / shutdown)"
                if _shutdown.is_set()
                else f"non-zero exit code {rc} (see {log_path})"
            )
    except (OSError, subprocess.SubprocessError) as exc:
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

    elapsed = time.time() - t0
    ok = error is None
    status = C.green("OK") if ok else C.red("FAIL")
    log_errors = _extract_errors_from_log(log_path)
    metrics = _read_cell_metrics(cell_run, exp_name)
    err_suffix = (
        f" {C.yellow(f'({len(log_errors)} ERROR log entr' + ('y' if len(log_errors) == 1 else 'ies') + ')')}"
        if log_errors
        else ""
    )
    sr_value = (metrics or {}).get("success_rate")
    sr_suffix = (
        f" {C.cyan(f'sr={sr_value:.1f}%')}"
        if isinstance(sr_value, (int, float))
        else ""
    )
    print(
        (C.green("[sweep] DONE  ") if ok else C.red("[sweep] DONE  "))
        + f"{label:>40s} gpu={gpu} {status} rc={rc} ({elapsed:.0f}s)"
        + sr_suffix
        + err_suffix,
        flush=True,
    )
    if error is not None:
        print(C.red(f"[sweep]   error: {error}"), file=sys.stderr, flush=True)

    return {
        "task_suite": task_suite,
        "perturbation": perturbation,
        "cell_run": cell_run,
        "exp_name": exp_name,
        "axis_values": axis_values or {},
        "gpu": gpu,
        "rc": rc,
        "elapsed_s": elapsed,
        "error": error,
        "log": str(log_path),
        "log_errors": log_errors,
        "metrics": metrics,
        "cmd": cmd,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML sweep config. When provided, the matrix in the "
        "YAML (axes + experiments + exclude rules) drives the sweep; "
        "--perturbations and --task-suites are forbidden in this mode. "
        "Forwarded CLI args (anything after the sweep flags) still flow "
        "through to every cell, with last-wins precedence over base_args.",
    )
    parser.add_argument(
        "--perturbations",
        nargs="+",
        choices=list(PERTURBATIONS),
        default=None,
        help="Perturbation flags to sweep, one True at a time. "
        f"Default (when --config is not set): {list(PERTURBATIONS)}.",
    )
    parser.add_argument(
        "--task-suites",
        nargs="+",
        default=None,
        help="Task suite names to sweep (cartesian-product with "
        "--perturbations). When omitted, the eval script's default suite is "
        "used (any --env.task-suite-name in forwarded args is honoured).",
    )

    def _gpu_list(s: str) -> list[int]:
        # Accept "0,1,2,3" (preferred) and tolerate stray whitespace.
        # Single-token form avoids argparse ``nargs="+"`` greedily eating
        # forwarded tyro subcommands like ``policy:openpi``.
        try:
            return [int(tok) for tok in s.replace(" ", ",").split(",") if tok]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"--gpus expects a comma-separated list of ints (got {s!r})"
            ) from exc

    parser.add_argument(
        "--gpus",
        type=_gpu_list,
        default=None,
        help="Comma-separated GPU ids to use (e.g. --gpus 0,1,2,3). "
        "Defaults to all detected via nvidia-smi.",
    )
    parser.add_argument(
        "--gpu-mem-used-max-mib",
        type=int,
        default=500,
        help="When auto-detecting GPUs, only pick those with memory.used "
        "below this threshold (MiB). Ignored if --gpus is explicit. "
        "Default: 500 MiB (effectively idle).",
    )
    parser.add_argument(
        "--per-run-timeout-s",
        type=int,
        default=24 * 60 * 60,
        help="Timeout per evaluation run (default: 24h).",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Where to write per-run logs and the sweep summary. Defaults "
        "to ``data/<run>/perturbation_sweep_logs`` when --run is forwarded "
        "to the eval script, else ``data/perturbation_sweep_logs``.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print every cell's full command and exit without launching any "
        "evaluate.py subprocess. Skips port-conflict probing too — useful "
        "for sanity-checking a sweep config before committing to a long "
        "overnight run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for the dispatcher itself. Each child run is seeded "
        "independently via the forwarded --seed arg to evaluate.py.",
    )
    parser.add_argument(
        "--port-offset",
        type=int,
        default=0,
        help="Offset added to the default policy/env ports per GPU. Use this "
        "to avoid conflicts when another sweep / leftover server is already "
        f"using {POLICY_PORT_BASE}+gpu or {ENV_PORT_BASE}+gpu. With "
        "--port-offset N: policy.port=8000+gpu+N, env.port=8765+gpu+N. "
        "Default: 0.",
    )
    args, extras = parser.parse_known_args()
    seed_everything(args.seed)

    # ----- GPU selection ---------------------------------------------------
    gpu_rows = query_gpu_memory()
    if args.gpus is not None:
        # User-specified list wins; report memory for context but don't filter.
        gpus = list(args.gpus)
        gpu_report = [
            f"GPU {idx}: {used} / {total} MiB used"
            for idx, used, total in gpu_rows
            if idx in gpus
        ]
    elif gpu_rows:
        idle, busy = select_idle_gpus(gpu_rows, args.gpu_mem_used_max_mib)
        gpus = idle
        gpu_report = [
            C.green(f"GPU {idx}: {used} / {total} MiB used  (idle, selected)")
            for idx, used, total in gpu_rows
            if idx in idle
        ] + [
            C.yellow(
                f"GPU {idx}: {used} / {total} MiB used  "
                f"(busy, > {args.gpu_mem_used_max_mib} MiB; skipped)"
            )
            for idx, used, total in busy
        ]
    else:
        gpus = []
        gpu_report = [C.yellow("nvidia-smi unavailable or returned no GPUs")]

    if not gpus:
        print(
            C.red(
                "[sweep] No idle GPUs detected. Either pass --gpus explicitly "
                "or raise --gpu-mem-used-max-mib. Detected:"
            ),
            file=sys.stderr,
            flush=True,
        )
        for line in gpu_report:
            print(f"  {line}", file=sys.stderr, flush=True)
        return 2

    # ----- Port-conflict check --------------------------------------------
    # Refuse to launch if any policy/env port we'd default to is already
    # bound (e.g. an earlier sweep left servers running, or another user is
    # on the box). Suggest a clean --port-offset so the user can rerun.
    # Skipped in --dry-run since no subprocess will actually bind a port.
    conflicts = [] if args.dry_run else find_port_conflicts(gpus, args.port_offset)
    if conflicts:
        print(
            C.red(
                f"[sweep] Port conflict detected (--port-offset={args.port_offset}):"
            ),
            file=sys.stderr,
            flush=True,
        )
        for gpu, busy in conflicts:
            ports_str = ", ".join(str(p) for p in busy)
            print(
                C.red(f"  GPU {gpu}: port(s) in use -> {ports_str}"),
                file=sys.stderr,
                flush=True,
            )
        suggested = suggest_port_offset(gpus, start=max(1, args.port_offset + 1))
        if suggested is not None:
            print(
                C.yellow(
                    f"[sweep] Suggested: rerun with --port-offset {suggested} "
                    "(first offset that frees all required ports)."
                ),
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                C.yellow(
                    "[sweep] Could not find a free --port-offset within 200 "
                    "attempts. Free the busy ports (e.g. kill leftover policy "
                    "/ env server processes) or pick a large offset manually."
                ),
                file=sys.stderr,
                flush=True,
            )
        return 2

    # ----- Mode dispatch: --config (YAML matrix) vs. legacy CLI -----------
    config_mode = args.config is not None
    if config_mode:
        if args.perturbations is not None or args.task_suites is not None:
            print(
                C.red(
                    "[sweep] --config is incompatible with --perturbations / "
                    "--task-suites; declare those axes inside the YAML "
                    "('perturbations:' / 'task_suites:' shorthand or "
                    "'matrix:') instead."
                ),
                file=sys.stderr,
                flush=True,
            )
            return 2
        try:
            cfg = yaml.safe_load(Path(args.config).read_text())
        except (OSError, yaml.YAMLError) as exc:
            print(
                C.red(f"[sweep] Failed to load --config {args.config!r}: {exc}"),
                file=sys.stderr,
                flush=True,
            )
            return 2
    else:
        cfg = None
        if args.perturbations is None:
            args.perturbations = list(PERTURBATIONS)

    # In config mode the sweep injects --output.run / --output.exp-name per
    # cell, so strip any conflicting user-forwarded copies. In CLI mode the
    # user owns --output.run, so leave it alone.
    extra_value_flags: set[str] = set()
    if config_mode:
        extra_value_flags |= {"--output.run", "--output.exp-name", "--output.exp_name"}
    if (not config_mode) and args.task_suites:
        # CLI sweep owns the task-suite axis; drop any forwarded copy.
        extra_value_flags.add("--env.task-suite-name")
    extras = filter_extras(extras, extra_value_flags)

    # Sweep-wide default flags (applied to every cell). Each block is a
    # last-wins guard: skip injection if the user explicitly overrode.
    _vis_flags = {"--viz.enable", "--viz.no-enable"}
    if not any(t.split("=", 1)[0] in _vis_flags for t in extras):
        extras = ["--viz.no-enable", *extras]
    _skip_flags = {"--output.skip-completed", "--output.no-skip-completed"}
    if not any(t.split("=", 1)[0] in _skip_flags for t in extras):
        extras = ["--output.skip-completed", *extras]
    _pair_cache_flags = {
        "--demo-analysis.cache-pairs",
        "--demo-analysis.no-cache-pairs",
    }
    if not any(t.split("=", 1)[0] in _pair_cache_flags for t in extras):
        extras = ["--demo-analysis.no-cache-pairs", *extras]
    _live_cache_flags = {
        "--demo-analysis.cache-live-objects",
        "--demo-analysis.no-cache-live-objects",
    }
    if not any(t.split("=", 1)[0] in _live_cache_flags for t in extras):
        extras = ["--demo-analysis.no-cache-live-objects", *extras]

    # Force start-trial/start-task to 0 — sweep cells are independent and must
    # not inherit the eval script's default start_trial_idx=1. _MANAGED_VALUE_FLAGS
    # already stripped any user-supplied override during filter_extras above.
    extras = [
        "--iter.start-trial-idx",
        "0",
        "--iter.start-task-idx",
        "0",
        *extras,
    ]

    # ----- Build the cell list (unified across both modes) ----------------
    if config_mode:
        try:
            cells, axis_names, output_run = expand_matrix(cfg or {}, extras)
        except ValueError as exc:
            print(
                C.red(f"[sweep] Invalid sweep config {args.config!r}: {exc}"),
                file=sys.stderr,
                flush=True,
            )
            return 2
    else:
        suites: list[str | None] = (
            list(args.task_suites) if args.task_suites else [None]
        )
        cells = [
            Cell(
                axis_values={
                    "task_suite": s if s is not None else "<eval-script-default>",
                    "perturbation": p,
                },
                extra_tokens=extras,
                cell_run=None,  # legacy: defer to evaluate.py default
                exp_name=None,
                task_suite=s,
                perturbation=p,
                label=_run_label(p, s),
            )
            for s in suites
            for p in args.perturbations
        ]
        axis_names = ["task_suite", "perturbation"]
        output_run = extract_extra_value(extras, "--output.run")

    if not cells:
        print(
            C.red("[sweep] No cells to run after applying exclude rules."),
            file=sys.stderr,
            flush=True,
        )
        return 2

    # ----- Log directory --------------------------------------------------
    if args.log_dir is not None:
        log_dir = Path(args.log_dir)
    elif output_run:
        log_dir = Path(f"data/{output_run}/perturbation_sweep_logs")
    else:
        log_dir = Path("data/perturbation_sweep_logs")

    # ----- Plan printout + confirmation -----------------------------------
    def _kv(label: str, value: str) -> str:
        return f"{C.bold(label + ':'):<24} {value}"

    print("\n" + C.bold(C.cyan("=== Sweep plan ===")), flush=True)
    print(C.bold("GPUs:"), flush=True)
    for line in gpu_report:
        print(f"  {line}", flush=True)
    print(f"  -> using GPU(s): {C.bold(C.green(str(gpus)))}", flush=True)

    print(_kv("Mode", C.magenta("config" if config_mode else "cli")), flush=True)
    if config_mode:
        print(_kv("Config file", C.blue(args.config)), flush=True)
        print(_kv("Output run", C.magenta(str(output_run))), flush=True)
        for ax in axis_names:
            uniq = sorted({c.axis_values[ax] for c in cells})
            print(_kv(f"Axis '{ax}' ({len(uniq)})", C.magenta(str(uniq))), flush=True)
    else:
        print(
            _kv(
                f"Perturbations ({len(args.perturbations)})",
                C.magenta(str(args.perturbations)),
            ),
            flush=True,
        )
        suites_disp = (
            args.task_suites if args.task_suites else ["<eval script default>"]
        )
        print(
            _kv(f"Task suites ({len(suites_disp)})", C.magenta(str(suites_disp))),
            flush=True,
        )
    print(
        _kv(
            "Total cells",
            f"{C.bold(str(len(cells)))}  "
            f"{C.dim(f'(parallelism: up to {min(len(gpus), len(cells))})')}",
        ),
        flush=True,
    )
    print(_kv("Per-run timeout", f"{args.per_run_timeout_s}s"), flush=True)
    port_pairs = ", ".join(
        f"gpu{g}:policy={pp},env={ep}"
        for g, pp, ep in required_ports(gpus, args.port_offset)
    )
    print(_kv(f"Ports (offset={args.port_offset})", C.dim(port_pairs)), flush=True)
    print(_kv("Log dir", C.blue(str(log_dir))), flush=True)
    if not config_mode:
        # In CLI mode all cells share `extras` verbatim; in config mode each
        # cell carries its own (already-merged) tokens, so dumping a single
        # 'forwarded args' line would be misleading.
        print(
            _kv(
                "Forwarded eval args",
                (str(extras) if extras else C.dim("<none>")),
            ),
            flush=True,
        )

    # Show the first command so the user sees exactly what will be launched.
    sample = cells[0]
    sample_cmd = build_cmd(
        sample.perturbation,
        instance=gpus[0],
        extras=sample.extra_tokens,
        task_suite=sample.task_suite,
        port_offset=args.port_offset,
        cell_run=sample.cell_run,
        exp_name=sample.exp_name,
    )
    print("\n" + C.bold(f"Example command (first cell: {sample.label}):"), flush=True)
    print(
        C.dim(f"  CUDA_VISIBLE_DEVICES={gpus[0]} \\\n    {' '.join(sample_cmd)}"),
        flush=True,
    )

    if args.dry_run:
        # Print every cell's command and exit. Cell→GPU assignment in
        # production is by queue (first-free-wins), so we round-robin here
        # purely for display; the actual assignment may differ.
        print(
            "\n" + C.bold(C.yellow("=== DRY RUN: commands that would be launched ===")),
            flush=True,
        )
        for i, c in enumerate(cells):
            display_gpu = gpus[i % len(gpus)]
            cmd = build_cmd(
                c.perturbation,
                instance=display_gpu,
                extras=c.extra_tokens,
                task_suite=c.task_suite,
                port_offset=args.port_offset,
                cell_run=c.cell_run,
                exp_name=c.exp_name,
            )
            print(
                f"\n{C.cyan(f'[{i + 1}/{len(cells)}]')} {C.bold(c.label)}  "
                f"{C.dim(f'(gpu={display_gpu}, axes={c.axis_values})')}",
                flush=True,
            )
            print(
                f"  CUDA_VISIBLE_DEVICES={display_gpu} \\\n    {' '.join(cmd)}",
                flush=True,
            )
        print(
            "\n"
            + C.green(f"[dry-run] {len(cells)} cell(s) would run. Nothing launched."),
            flush=True,
        )
        return 0

    if not args.yes:
        if not confirm("\n" + C.bold(C.yellow("Proceed?"))):
            print(C.yellow("[sweep] aborted by user."), flush=True)
            return 130

    log_dir.mkdir(parents=True, exist_ok=True)

    # From here on, Ctrl-C / SIGTERM tears down all child eval processes
    # (and via the eval's own cleanup, the policy + env servers it spawned).
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    atexit.register(_kill_all_children, "atexit", 5.0)

    gpu_q: queue.Queue[int] = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    def task(cell: Cell) -> dict:
        gpu = gpu_q.get()
        try:
            return run_one(
                cell.perturbation,
                gpu,
                cell.extra_tokens,
                log_dir,
                args.per_run_timeout_s,
                task_suite=cell.task_suite,
                port_offset=args.port_offset,
                cell_run=cell.cell_run,
                exp_name=cell.exp_name,
                label=cell.label,
                axis_values=cell.axis_values,
            )
        finally:
            gpu_q.put(gpu)

    sweep_t0 = time.time()
    results: list[dict] = []
    workers = max(1, min(len(gpus), len(cells)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(task, c): c for c in cells}
        for fut in as_completed(futures):
            c = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001 — driver-side safety net
                print(
                    f"[sweep] {c.label} crashed in driver: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                results.append(
                    {
                        "task_suite": c.task_suite,
                        "perturbation": c.perturbation,
                        "cell_run": c.cell_run,
                        "exp_name": c.exp_name,
                        "axis_values": c.axis_values,
                        "gpu": None,
                        "rc": -1,
                        "elapsed_s": 0.0,
                        "error": f"driver exception: {exc}\n{traceback.format_exc()}",
                        "log": None,
                        "log_errors": [],
                        "metrics": None,
                        "cmd": None,
                    }
                )

    # ---------------------------- summary ----------------------------------
    total_elapsed = time.time() - sweep_t0
    print(
        "\n" + C.bold(C.cyan(f"=== SWEEP SUMMARY ({total_elapsed / 60:.1f} min) ===")),
        flush=True,
    )

    def _result_label(r: dict) -> str:
        return r.get("exp_name") or _run_label(
            r.get("perturbation") or "default", r.get("task_suite")
        )

    failures: list[dict] = []
    for r in sorted(results, key=_result_label):
        ok = r["error"] is None
        tag = C.green("[ OK ]") if ok else C.red("[FAIL]")
        label = _result_label(r)
        n_errs = len(r.get("log_errors") or [])
        err_suffix = C.yellow(f"  ({n_errs} ERROR)") if n_errs else ""
        m = r.get("metrics") or {}
        sr = m.get("success_rate")
        sr_suffix = C.cyan(f"  sr={sr:.1f}%") if isinstance(sr, (int, float)) else ""
        print(
            f"  {tag} {label:>40s} gpu={r['gpu']} rc={r['rc']} "
            f"{r['elapsed_s']:.0f}s{sr_suffix}{err_suffix}"
        )
        if not ok:
            failures.append(r)

    # ---------------------- per-axis pivot --------------------------------
    # For each sweep axis, list the mean success rate across cells that share
    # each value. Lets the user spot at a glance which policy / suite /
    # perturbation drove the wins or losses overnight.
    axis_pivot: dict[str, dict[str, dict]] = {}
    for r in results:
        m = r.get("metrics") or {}
        sr = m.get("success_rate")
        if not isinstance(sr, (int, float)):
            continue
        for ax, val in (r.get("axis_values") or {}).items():
            bucket = axis_pivot.setdefault(ax, {}).setdefault(
                val, {"n_cells": 0, "sr_sum": 0.0, "episodes": 0}
            )
            bucket["n_cells"] += 1
            bucket["sr_sum"] += float(sr)
            ep = m.get("episodes")
            if isinstance(ep, int):
                bucket["episodes"] += ep

    if axis_pivot:
        print("\n" + C.bold(C.cyan("=== Per-axis pivot (mean success_rate) ===")))
        for ax in sorted(axis_pivot):
            print(f"  {C.bold(ax)}:")
            rows = []
            for val, b in axis_pivot[ax].items():
                mean_sr = b["sr_sum"] / b["n_cells"] if b["n_cells"] else 0.0
                rows.append((val, mean_sr, b["n_cells"], b["episodes"]))
            for val, mean_sr, n, eps in sorted(rows, key=lambda x: -x[1]):
                print(
                    f"    {val:>24s}  "
                    f"{C.cyan(f'sr={mean_sr:5.1f}%')}  "
                    f"{C.dim(f'(n_cells={n}, episodes={eps})')}"
                )

    # ---------------------- cross-tab(s) ---------------------------------
    # When 2+ sweep axes vary, print 2D table(s) of mean success_rate. The
    # two axes with the most distinct values become rows/cols; any extra
    # varying axis becomes a "group" (one table per combination of group
    # values). Skipped when only 0 or 1 axes vary.
    def _axis_sort_key(s: str) -> tuple:
        try:
            return (0, float(s))
        except (TypeError, ValueError):
            return (1, str(s))

    active_axes = [ax for ax, vals in axis_pivot.items() if len(vals) > 1]
    crosstab_payload: list[dict] = []
    if len(active_axes) >= 2:
        # Pick two axes with the most distinct values for rows/cols; tie-break
        # alphabetically. Rows = axis with more values (taller tables read
        # better in a terminal than wider ones); cols = the other one.
        ranked = sorted(
            active_axes,
            key=lambda ax: (-len(axis_pivot[ax]), ax),
        )
        row_ax, col_ax = ranked[0], ranked[1]
        group_axes = ranked[2:]

        # Bucket results by (group_values, row_value, col_value).
        buckets: dict[tuple, dict[tuple[str, str], list[float]]] = {}
        for r in results:
            m = r.get("metrics") or {}
            sr = m.get("success_rate")
            if not isinstance(sr, (int, float)):
                continue
            av = r.get("axis_values") or {}
            rv, cv = av.get(row_ax), av.get(col_ax)
            if rv is None or cv is None:
                continue
            gkey = tuple(av.get(ga, "") for ga in group_axes)
            buckets.setdefault(gkey, {}).setdefault((rv, cv), []).append(float(sr))

        group_order = sorted(
            buckets.keys(), key=lambda k: tuple(_axis_sort_key(v) for v in k)
        )
        for gkey in group_order:
            cell_sr = buckets[gkey]
            if not cell_sr:
                continue
            row_vals = sorted({rv for rv, _ in cell_sr}, key=_axis_sort_key)
            col_vals = sorted({cv for _, cv in cell_sr}, key=_axis_sort_key)
            col_w = max(8, max(len(v) for v in col_vals) + 2)
            row_w = max(len(row_ax), max(len(v) for v in row_vals)) + 2

            group_label = (
                "  ["
                + ", ".join(f"{ga}={gv}" for ga, gv in zip(group_axes, gkey))
                + "]"
                if group_axes
                else ""
            )
            print(
                "\n"
                + C.bold(
                    C.cyan(
                        f"=== Cross-tab: rows={row_ax}, cols={col_ax} — "
                        f"mean success_rate (%){group_label} ==="
                    )
                ),
                flush=True,
            )
            header = " " * row_w + "".join(f"{cv:>{col_w}}" for cv in col_vals)
            print(C.bold(header), flush=True)
            for rv in row_vals:
                cells_str = ""
                for cv in col_vals:
                    srs = cell_sr.get((rv, cv))
                    if srs:
                        cells_str += f"{sum(srs) / len(srs):>{col_w}.1f}"
                    else:
                        cells_str += f"{'—':>{col_w}}"
                print(f"{rv:>{row_w}}{cells_str}", flush=True)

            crosstab_payload.append(
                {
                    "row_axis": row_ax,
                    "col_axis": col_ax,
                    "group_axes": list(group_axes),
                    "group_values": list(gkey),
                    "row_values": row_vals,
                    "col_values": col_vals,
                    "cells": {
                        f"{rv}|{cv}": sum(srs) / len(srs)
                        for (rv, cv), srs in cell_sr.items()
                    },
                }
            )

    # ---------------------- error-frequency aggregation -------------------
    # ``log_errors`` are loguru ERROR-level lines pulled from each child's log.
    # A run can complete with rc=0 (all episodes returned a success bool) but
    # still have errors logged — e.g. abstractor setup failed for some episodes
    # and the runner counted them as failed-but-non-fatal. Surface those here
    # so they don't get silently buried in per-run logs.
    msg_counter: Counter[str] = Counter()
    msg_runs: dict[str, list[str]] = {}
    for r in results:
        label = _result_label(r)
        for msg in r.get("log_errors") or []:
            msg_counter[msg] += 1
            msg_runs.setdefault(msg, []).append(label)
    total_errs = sum(msg_counter.values())
    runs_with_errs = sum(1 for r in results if r.get("log_errors"))
    if total_errs:
        print(
            "\n"
            + C.bold(C.yellow("=== Errors logged during sweep ==="))
            + f"  {C.dim(f'({total_errs} entr' + ('y' if total_errs == 1 else 'ies') + f' across {runs_with_errs} run(s))')}",
            flush=True,
        )
        for msg, count in msg_counter.most_common():
            run_counts = Counter(msg_runs[msg])
            run_summary = ", ".join(
                f"{lbl} ({n}x)" for lbl, n in run_counts.most_common()
            )
            display_msg = msg if len(msg) <= 200 else msg[:197] + "..."
            print(
                f"  {C.red(f'{count:>4d}x')}  {display_msg}",
                flush=True,
            )
            print(f"        {C.dim('in: ' + run_summary)}", flush=True)
    else:
        print("\n" + C.green("No ERROR-level log entries detected."), flush=True)

    summary_path = log_dir / "sweep_summary.json"
    summary_payload = {
        "results": results,
        "axis_pivot": {
            ax: {
                val: {
                    "n_cells": b["n_cells"],
                    "mean_success_rate": (
                        b["sr_sum"] / b["n_cells"] if b["n_cells"] else None
                    ),
                    "total_episodes": b["episodes"],
                }
                for val, b in values.items()
            }
            for ax, values in axis_pivot.items()
        },
        "crosstab": crosstab_payload,
        "error_summary": {
            "total": total_errs,
            "runs_with_errors": runs_with_errs,
            "by_message": [
                {
                    "message": msg,
                    "count": count,
                    "by_run": dict(Counter(msg_runs[msg])),
                }
                for msg, count in msg_counter.most_common()
            ],
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=str))
    print("\n" + C.bold("Sweep summary JSON: ") + C.blue(str(summary_path)), flush=True)

    if failures:
        print(
            "\n" + C.red(C.bold(f"{len(failures)} run(s) failed:")),
            file=sys.stderr,
            flush=True,
        )
        for r in failures:
            print(
                f"  - {_result_label(r)}: {r['error']}\n      log: {r.get('log')}",
                file=sys.stderr,
                flush=True,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
