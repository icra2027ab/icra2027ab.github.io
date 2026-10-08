"""Planned motion inside the safe box, and the box check that guards it."""

import numpy as np
import pytest

from policy_abstraction.calibration.geometry import (
    CalibrationMode,
    pose_to_matrix,
    rotation_angle_deg,
)
from policy_abstraction.calibration.motion import (
    MAX_SPEED,
    MotionConfig,
    SafeBox,
    plan_targets,
)
from policy_abstraction.calibration.session import Step
from tests.calibration.conftest import make_session

NOMINAL = np.array([-0.49, -0.13, 0.26, 2.221, 2.221, 0.0])


class TestSafeBox:
    def test_around_and_contains(self):
        box = SafeBox.around([0.1, 0.2, 0.3], half_xy=0.1, below=0.05, above=0.2)
        assert box.x == (0.0, 0.2) and box.z == (0.25, 0.5)
        # By default the box never reaches below where the arm was parked.
        assert SafeBox.around([0.1, 0.2, 0.3]).z[0] == 0.3
        assert box.contains([0.1, 0.2, 0.3])
        assert not box.contains([0.25, 0.2, 0.3])
        assert box.contains([0.21, 0.2, 0.3], margin=0.02)
        np.testing.assert_allclose(box.center, [0.1, 0.2, 0.375])

    def test_rejects_inverted_ranges(self):
        with pytest.raises(ValueError):
            SafeBox((0.2, 0.1), (0, 1), (0, 1))

    def test_roundtrip(self):
        box = SafeBox((-1, 0), (0, 1), (0.1, 0.4))
        assert SafeBox.from_dict(box.to_dict()) == box


class TestPlanTargets:
    def test_exterior_targets_fill_the_box_and_keep_the_orientation(self):
        box = SafeBox.around(NOMINAL[:3])
        cfg = MotionConfig(box=box, n_targets=40, tilt_deg=15, roll_deg=20, seed=3)
        targets = plan_targets(cfg, NOMINAL, CalibrationMode.EXTERIOR)
        assert len(targets) == 40
        P = np.stack([t[:3] for t in targets])
        assert all(box.contains(p) for p in P)
        # Stratified: every octant of the box gets visited.
        mid = box.center
        octants = {tuple((p > mid).astype(int)) for p in P}
        assert len(octants) == 8
        R0 = pose_to_matrix(NOMINAL)[:3, :3]
        for t in targets:
            dR = R0.T @ pose_to_matrix(t)[:3, :3]
            # Two tilts of <= 15 deg and a roll of <= 20 deg compose to less than that sum.
            assert rotation_angle_deg(dR) <= 15 + 15 + 20 + 1e-6

    def test_wrist_targets_aim_the_tool_at_the_mark(self):
        ref = NOMINAL[:3] + np.array([0.0, 0.0, -0.35])
        box = SafeBox.around(NOMINAL[:3], half_xy=0.1, below=0.02, above=0.1)
        cfg = MotionConfig(box=box, n_targets=20, tilt_deg=10, roll_deg=30, seed=1)
        targets = plan_targets(
            cfg, NOMINAL, CalibrationMode.WRIST, reference_point_base=ref
        )
        for t in targets:
            T = pose_to_matrix(t)
            z_axis = T[:3, 2]
            to_mark = ref - T[:3, 3]
            to_mark /= np.linalg.norm(to_mark)
            ang = np.degrees(np.arccos(np.clip(z_axis @ to_mark, -1, 1)))
            assert ang < 25.0, ang

    def test_wrist_needs_the_reference(self):
        cfg = MotionConfig(box=SafeBox.around(NOMINAL[:3]), n_targets=12)
        with pytest.raises(ValueError, match="reference"):
            plan_targets(cfg, NOMINAL, CalibrationMode.WRIST)

    def test_first_target_is_near_the_start(self):
        box = SafeBox.around(NOMINAL[:3])
        cfg = MotionConfig(box=box, n_targets=30, seed=0)
        targets = plan_targets(cfg, NOMINAL, CalibrationMode.EXTERIOR)
        first = np.linalg.norm(targets[0][:3] - NOMINAL[:3])
        far = max(np.linalg.norm(t[:3] - NOMINAL[:3]) for t in targets)
        assert first < far

    def test_speed_is_capped(self):
        cfg = MotionConfig(box=SafeBox.around(NOMINAL[:3]), speed=5.0, accel=99.0)
        assert cfg.speed == MAX_SPEED and cfg.accel <= 0.8


def _autonomous(tmp_path, mode, **kw):
    session, arm, camera, truth, clock = make_session(tmp_path, mode)
    camera._on_read = arm.tick  # the arm advances one step per frame
    session.mover = arm
    if mode is CalibrationMode.WRIST:
        session.reference_point_base = np.asarray(truth.reference_point_base)
        session.step = Step.CAPTURE
    tcp = arm.tcp_pose()
    box = SafeBox.around(tcp[:3], half_xy=0.15, below=0.05, above=0.2)
    cfg = MotionConfig(
        box=box, n_targets=kw.pop("n", 20), tilt_deg=15, roll_deg=20, **kw
    )
    run = session.start_autonomous_capture(cfg)
    return session, arm, truth, clock, run


def test_autonomous_capture_visits_the_poses_and_stays_in_the_box(tmp_path, mode):
    session, arm, truth, clock, run = _autonomous(tmp_path, mode)
    assert 12 <= len(run.targets) <= 20
    worst = 0.0
    n = 0
    while run.active and n < 20000:
        clock.advance(0.05)
        s = session.poll()
        worst = max(
            worst,
            float(
                np.max(
                    np.abs(s.tcp_pose[:3] - run.config.box.center)
                    - run.config.box.size / 2
                )
            ),
        )
        n += 1
    assert run.phase == "done", (run.phase, run.message)
    assert worst <= run.config.margin_m
    assert session.status.n_keyframes >= len(run.targets) - run.skipped - 1
    assert arm.commands == len(run.targets)
    session.finish_capture()
    assert session.step is Step.SELECT_POINT


def test_autonomous_capture_solves_correctly(tmp_path):
    from policy_abstraction.calibration.fake_hardware import find_marker
    from policy_abstraction.calibration.solve import pose_difference

    session, arm, truth, clock, run = _autonomous(
        tmp_path, CalibrationMode.EXTERIOR, n=30
    )
    n = 0
    while run.active and n < 20000:
        clock.advance(0.05)
        session.poll()
        n += 1
    session.finish_capture()
    click = find_marker(session.first_frame())
    assert click is not None
    session.set_click(*click)
    result = session.run_tracking(log=lambda _s: None)
    d = pose_difference(result.solution.T_parent_cam, truth.T_parent_cam)
    assert d.translation_m < 0.01 and d.rotation_deg < 0.6, d


def test_leaving_the_box_stops_the_arm(tmp_path):
    session, arm, truth, clock, run = _autonomous(tmp_path, CalibrationMode.EXTERIOR)
    # Sabotage: teleport the TCP outside the box between polls.
    q_far = arm.solve_ik(np.concatenate([run.config.box.hi + 0.1, arm.tcp_pose()[3:]]))
    if q_far is None:
        pytest.skip("fake IK could not place the arm outside the box")
    arm._driver.set_joints(q_far)
    clock.advance(0.05)
    session.poll()
    assert run.phase == "fault"
    assert "left the safe box" in run.message
    assert not arm.is_moving()
    # Capture itself is still open so the operator decides what to do.
    assert session.capturing


def test_stop_motion_halts_the_arm(tmp_path):
    session, arm, truth, clock, run = _autonomous(tmp_path, CalibrationMode.EXTERIOR)
    for _ in range(5):
        clock.advance(0.05)
        session.poll()
    assert arm.is_moving() or run.phase in ("moving", "dwell")
    session.stop_motion()
    assert run.phase == "stopped" and not arm.is_moving()
    clock.advance(0.05)
    session.poll()
    assert not arm.is_moving()


def test_cannot_start_outside_the_box(tmp_path):
    session, arm, camera, truth, clock = make_session(
        tmp_path, CalibrationMode.EXTERIOR
    )
    session.mover = arm
    box = SafeBox((5.0, 5.2), (5.0, 5.2), (5.0, 5.2))
    with pytest.raises(RuntimeError, match="outside the safe box"):
        session.start_autonomous_capture(MotionConfig(box=box, n_targets=12))
    assert not session.capturing


def test_no_mover_means_manual_only(tmp_path):
    session, arm, camera, truth, clock = make_session(
        tmp_path, CalibrationMode.EXTERIOR
    )
    box = SafeBox.around(arm.tcp_pose()[:3])
    with pytest.raises(RuntimeError, match="no motion interface"):
        session.start_autonomous_capture(MotionConfig(box=box, n_targets=12))


def test_wrist_box_must_clear_the_mark(tmp_path):
    session, arm, camera, truth, clock = make_session(tmp_path, CalibrationMode.WRIST)
    session.mover = arm
    tcp = arm.tcp_pose()
    # A mark just 2 cm under the box's floor: refused.
    session.reference_point_base = np.array([tcp[0], tcp[1], tcp[2] - 0.02])
    session.step = Step.CAPTURE
    box = SafeBox.around(tcp[:3])
    with pytest.raises(RuntimeError, match="above the mark"):
        session.start_autonomous_capture(MotionConfig(box=box, n_targets=12))
    assert not session.capturing
