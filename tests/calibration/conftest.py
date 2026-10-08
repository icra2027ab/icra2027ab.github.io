"""Shared fixtures: a fake cell (arm + camera + tracker) and a driven session."""

from __future__ import annotations

from typing import Optional, TypeVar

import numpy as np
import pytest

from policy_abstraction.calibration.fake_hardware import (
    FakeArm,
    FakeCamera,
    FakeTracker,
    default_truth,
)
from policy_abstraction.calibration.geometry import CalibrationMode
from policy_abstraction.calibration.session import (
    CalibrationSession,
    SessionConfig,
    Step,
)

TCP_OFFSET = [0.00052, 0.00109, 0.22347, 0.0, 0.0, 0.0]

T = TypeVar("T")


def nn(value: Optional[T]) -> T:
    """Assert-not-None, for values the type checker cannot know are set."""
    assert value is not None
    return value


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_session(tmp_path, mode: CalibrationMode, *, target=20, noise=0.3, seed=1):
    arm = FakeArm(
        TCP_OFFSET,
        seed=seed,
        spread_rad=0.45 if mode is CalibrationMode.EXTERIOR else 0.22,
    )
    truth = default_truth(mode, arm.tcp_pose())
    camera = FakeCamera(arm, truth, pixel_noise=noise, seed=seed)
    clock = FakeClock()
    cfg = SessionConfig(
        camera_name="cam",
        camera_device="fake",
        mode=mode,
        dataset_root=tmp_path / "capture" / mode.value,
        output_path=tmp_path / "calib.json",
        tcp_offset=TCP_OFFSET,
        target_keyframes=target,
        max_frames=300,
    )
    session = CalibrationSession(
        cfg, arm=arm, camera=camera, tracker=FakeTracker(pixel_noise=noise), clock=clock
    )
    return session, arm, camera, truth, clock


def drive_capture(session, arm, clock, *, max_polls=6000):
    """Freedrive the fake arm until the session has its target of still poses."""
    n = 0
    while not session.status.target_reached and n < max_polls:
        arm.tick()
        clock.advance(0.05)
        session.poll()
        n += 1
    assert session.status.target_reached, "fake freedrive never reached the target"
    return n


def run_to_verify(session, arm, camera, truth, clock):
    """Capture, click the true marker, track, solve. Returns the result."""
    from policy_abstraction.calibration.fake_hardware import find_marker

    if session.mode is CalibrationMode.WRIST:
        # Stand in for the operator touching the mark with the fingertips.
        session.reference_point_base = np.asarray(truth.reference_point_base)
        session.step = Step.CAPTURE
    session.start_capture()
    drive_capture(session, arm, clock)
    session.finish_capture()
    click = find_marker(session.first_frame())
    assert click is not None
    session.set_click(*click)
    return session.run_tracking(log=lambda _s: None)


@pytest.fixture(
    params=[CalibrationMode.EXTERIOR, CalibrationMode.WRIST], ids=["exterior", "wrist"]
)
def mode(request):
    return request.param


def make_suite(tmp_path, *, target=20, noise=0.3, seed=1):
    """Exterior + wrist stages on one fake cell. The exterior camera also shows
    the wrist stage's table mark, in cyan, so it can be clicked."""
    from policy_abstraction.calibration.suite import CalibrationSuite

    arm = FakeArm(TCP_OFFSET, seed=seed, spread_rad=0.22)
    home = arm.tcp_pose()
    truths = {
        CalibrationMode.EXTERIOR: default_truth(CalibrationMode.EXTERIOR, home),
        CalibrationMode.WRIST: default_truth(CalibrationMode.WRIST, home),
    }
    mark = nn(truths[CalibrationMode.WRIST].reference_point_base)
    clock = FakeClock()
    tracker = FakeTracker(pixel_noise=noise)
    output = tmp_path / "calib.json"
    stages = []
    cameras = {}
    for _i, mode in enumerate((CalibrationMode.EXTERIOR, CalibrationMode.WRIST)):
        cam = FakeCamera(
            arm,
            truths[mode],
            pixel_noise=noise,
            seed=seed,
            table_marks_base=[mark] if mode is CalibrationMode.EXTERIOR else None,
        )
        cameras[mode] = cam
        cfg = SessionConfig(
            camera_name="ext_cam" if mode is CalibrationMode.EXTERIOR else "wrist_cam",
            camera_device="fake",
            mode=mode,
            dataset_root=tmp_path / "capture" / mode.value,
            output_path=output,
            tcp_offset=TCP_OFFSET,
            target_keyframes=target,
            max_frames=300,
        )
        stages.append(
            CalibrationSession(
                cfg, arm=arm, camera=cam, tracker=tracker, clock=clock, mover=arm
            )
        )
    suite = CalibrationSuite(stages, calibration_path=output)
    return suite, arm, cameras, truths, clock
