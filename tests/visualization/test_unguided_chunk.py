"""The unguided counterfactual chunk is drawn next to the executed one.

Both are trimmed as the robot advances and each lives under its own entity
path in its own colour, so the viewer shows where guidance moved the plan.
"""

from __future__ import annotations

import numpy as np
import pytest
import rerun as rr

from policy_abstraction.visualization.rerun import RerunVisualizer


@pytest.fixture
def viz(monkeypatch):
    logged: list[tuple[str, object]] = []
    monkeypatch.setattr(
        rr,
        "log",
        lambda path, *args, **kw: logged.append((path, args[0] if args else None)),
    )
    v = RerunVisualizer(
        application_id="test",
        spawn=False,
        connect_addr=None,
        log_images=False,
        log_predicted_states=False,
    )
    v.logged = logged
    return v


def chunk(x0: float, n: int = 10) -> np.ndarray:
    pts = np.zeros((1, n, 7), dtype=np.float32)
    pts[0, :, 0] = x0 + np.arange(n) * 0.01
    return pts


def strips(viz, prefix: str):
    return {
        path: arch
        for path, arch in viz.logged
        if path.startswith(prefix) and isinstance(arch, rr.LineStrips3D)
    }


def test_both_chunks_are_logged_under_their_own_roots(viz):
    viz.update(
        timestep=0,
        robot_state=np.zeros(7, dtype=np.float32),
        action_chunk_states=chunk(0.0),
        unguided_chunk_states=chunk(0.5),
    )
    assert "action_chunks/chunk_0" in strips(viz, "action_chunks/chunk")
    assert "action_chunks/unguided/chunk_0" in strips(viz, "action_chunks/unguided")


def test_the_unguided_chunk_is_trimmed_as_the_robot_advances(viz):
    viz.update(
        timestep=0,
        robot_state=np.zeros(7, dtype=np.float32),
        action_chunk_states=chunk(0.0),
        unguided_chunk_states=chunk(0.5),
    )
    viz.logged.clear()
    viz.update(timestep=4, robot_state=np.zeros(7, dtype=np.float32))
    paths = [p for p, _ in viz.logged if p.startswith("action_chunks")]
    assert set(paths) == {"action_chunks/chunk_0", "action_chunks/unguided/chunk_0"}
    viz.logged.clear()
    viz.update(timestep=9, robot_state=np.zeros(7, dtype=np.float32))
    cleared = [p for p, a in viz.logged if isinstance(a, rr.Clear)]
    assert "action_chunks/unguided/chunk_0" in cleared


def test_the_two_chunks_have_different_colours():
    assert RerunVisualizer._UNGUIDED_CHUNK_COLOR[:3] != RerunVisualizer._CHUNK_COLOR[:3]
