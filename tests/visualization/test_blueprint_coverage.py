"""Every entity the visualiser logs must be reachable from some blueprint view.

An entity with no view is not a loud failure. It is logged, recorded, streamed,
and invisible — which in the viewer is indistinguishable from a value that was
never computed. That is how the live-vs-reference distance charts, the thing the
whole method turns on, went missing while ``send_blueprint``'s own docstring
claimed to show them.

This test reads the entity paths out of the source rather than running a
rollout, because the point is to catch a *new* ``rr.log`` whose view nobody
added.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import policy_abstraction.visualization.rerun as rerun_module

SOURCE = Path(rerun_module.__file__).read_text()

# 3-D geometry: covered by the Spatial3DView rooted at "/". Listed explicitly
# so adding a new 3-D entity is a deliberate choice rather than a silent pass.
_COVERED_BY_3D_VIEW = (
    "scene",
    "action_chunks",
    "robot/body",
    "robot/site",
    "robot/eef_frame",
    "dynamics/trajectory_3d",
    "camera",
)

# Logged for the record, not for a panel: tensors and per-camera pinholes that
# the 3-D view or the .vla consumes.
_DELIBERATELY_UNVIEWED = {
    "robot/state",
    "robot/action",
    "dynamics/predicted_states",
    "episode/done",
    # Written by log_abstraction_setup into the recording for replay tooling.
    "abstraction/live_scene",
    "abstraction/live_seg_overlay",
    "abstraction/object_mapping",
    "abstraction/ref_traj",
    "abstraction/ref_demo",
    "constraints/metadata",
}


def _logged_entities() -> set[str]:
    """Entity path prefixes passed to ``rr.log`` / ``_log_image``."""
    out: set[str] = set()
    for m in re.finditer(
        r'(?:rr\.log|self\._log_image)\(\s*f?["\'](.*?)["\']', SOURCE, re.S
    ):
        path = m.group(1)
        # Truncate at the first f-string hole: everything before it is literal.
        literal = path.split("{")[0].rstrip("/")
        if literal:
            out.add(literal)
    return out


def _blueprint_targets() -> list[str]:
    """Origins and explicit contents named anywhere in the blueprint."""
    start = SOURCE.index("    def send_blueprint")
    end = SOURCE.rindex("    def close")
    tab = SOURCE.index("    def _build_perception_tab")
    blob = SOURCE[start:end] + SOURCE[tab:start]

    targets = re.findall(r'origin=\s*\(?\s*f?["\'](.*?)["\']', blob, re.S)
    targets += re.findall(r"contents=\[([^\]]*)\]", blob, re.S)
    flat: list[str] = []
    for t in targets:
        flat.extend(re.findall(r'["\']([^"\']+)["\']', t) or [t])
    # An f-string origin such as "perception/{n}/overlay" matches any camera
    # name; reduce it to its literal prefix.
    return [t.split("{")[0].rstrip("/") for t in flat if t]


def _is_covered(path: str, targets: list[str]) -> bool:
    if path.startswith(_COVERED_BY_3D_VIEW) or path in _DELIBERATELY_UNVIEWED:
        return True
    if any(path.startswith(p) for p in _DELIBERATELY_UNVIEWED):
        return True
    for t in targets:
        if t in ("", "/"):
            continue
        stem = t.rstrip("*").rstrip("/")
        if not stem:
            continue
        if path == stem or path.startswith(stem + "/") or stem.startswith(path + "/"):
            return True
    return False


def test_every_logged_entity_has_somewhere_to_appear():
    targets = _blueprint_targets()
    assert targets, "no blueprint origins parsed — the parser is out of date"

    orphans = sorted(p for p in _logged_entities() if not _is_covered(p, targets))
    assert not orphans, (
        "these entities are logged but appear in no blueprint view, so they are "
        f"invisible in the viewer: {orphans}. Add a view in send_blueprint(), or "
        "list the path in _DELIBERATELY_UNVIEWED with a reason."
    )


@pytest.mark.parametrize(
    "entity",
    [
        "abstraction/live",
        "abstraction/ref",
        "abstraction/error",
        "constraints",
        "guidance",
    ],
)
def test_the_core_signals_are_charted(entity):
    """The measured-vs-reference distances and their error are the method's
    output; they were logged with no view for the whole of its development."""
    assert _is_covered(entity, _blueprint_targets())


def test_blueprint_builds_for_a_real_env_spec():
    """A malformed view only raises when the blueprint is actually sent."""
    import rerun as rr

    from policy_abstraction.envs.ur5.config import ur5_env_spec

    viz = rerun_module.RerunVisualizer(spawn=False)
    viz.set_env_spec(ur5_env_spec(["exterior_image_1_left", "wrist_image_left"]))
    rr.init("blueprint_coverage_test", spawn=False)
    viz.send_blueprint()
