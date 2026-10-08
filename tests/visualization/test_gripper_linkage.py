"""Rendering the Robotiq 2F-140 on the UR5e.

Menagerie publishes its arms bare, so the gripper is a second model grafted
onto the tool flange, and it is a *closed* chain: each finger's four-bar loop is
held shut by ``connect`` equality constraints that ``mj_kinematics`` — all the
visualiser ever calls — does not solve. Writing a driver angle and stopping
there renders four disconnected links. These tests pin the two things that
makes correct: that the solved free-air pose is the one the mechanism actually
takes, and that the drawn pads sit where the *aperture* says, not where a joint
angle says.
"""

from __future__ import annotations

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("scipy.optimize")

from policy_abstraction.visualization.robot_assembly import (  # noqa: E402
    VENDORED_MENAGERIE,
    _pad_gap,
    _pad_geoms,
    build_linkage,
    compose_arm_with_gripper,
    gripper_mjcf_path,
    menagerie_checkout,
)

ARM = "ur5e_mj_description"
GRIPPER = "robotiq_2f140"

# Taught on the real arm: the TCP sits 223.47 mm from the flange along the tool
# axis. The model's fingertip is the pad mesh's tip, a couple of mm shy of the
# taught contact plane, so this is a sanity bound and not an equality.
TAUGHT_TCP_Z_M = 0.22347


@pytest.fixture(scope="module")
def assembled():
    return compose_arm_with_gripper(ARM, GRIPPER)


@pytest.fixture(scope="module")
def gripper_only():
    """The gripper on its own, so linkage claims aren't confounded by the arm."""
    model = mujoco.MjModel.from_xml_path(str(gripper_mjcf_path(GRIPPER)))
    return model, build_linkage(model, mujoco)


class TestVendoredModel:
    def test_the_mjcf_resolves_and_loads(self):
        path = gripper_mjcf_path(GRIPPER)
        assert path.is_file()
        model = mujoco.MjModel.from_xml_path(str(path))
        # 4 joints per finger, and the closure that makes them dependent.
        assert model.njnt == 8
        assert model.neq == 3

    def test_unknown_gripper_names_the_place_it_looked(self):
        with pytest.raises(FileNotFoundError, match="robotiq_2f140"):
            gripper_mjcf_path("definitely_not_a_gripper")

    def test_the_2f140_comes_from_the_vendored_copy(self):
        """It is not upstream yet. If this starts failing because menagerie now
        ships one, delete the vendored directory — nothing else needs changing."""
        assert VENDORED_MENAGERIE in gripper_mjcf_path(GRIPPER).parents

    def test_an_upstream_model_is_preferred_over_a_vendored_one(self):
        """The escape hatch for when PR #161 merges. Exercised with the 2F-85,
        which menagerie *does* ship — otherwise this branch is never taken and
        can rot unnoticed, which is exactly what happened once already."""
        checkout = menagerie_checkout()
        if checkout is None or not (checkout / "robotiq_2f85").is_dir():
            pytest.skip("no mujoco_menagerie checkout to resolve against")
        assert gripper_mjcf_path("robotiq_2f85").is_relative_to(checkout)

    def test_guidance_treats_this_gripper_as_a_four_bar(self):
        """The rendering path and the guidance geometry must agree that this
        gripper's pads swing rather than slide.

        This used to compare ``arc_pivot_y`` against the model's knuckle body,
        which checked a parameter that did not describe the pad path -- it held
        while the arc was wrong by 26 mm. The quantitative check now lives in
        ``tests/policy_steering/test_four_bar_gripper.py``, which measures this
        same MJCF; all that is useful here is that the two sides agree the
        linkage exists.
        """
        from policy_steering.robot_config import load_gripper_geometry

        assert load_gripper_geometry(GRIPPER).is_four_bar
        model = mujoco.MjModel.from_xml_path(str(gripper_mjcf_path(GRIPPER)))
        assert build_linkage(model, mujoco) is not None


class TestLinkage:
    def test_a_prismatic_gripper_has_no_linkage_to_solve(self):
        """The Panda's fingers really do slide, so LIBERO/RoboCasa rendering
        must go on using the plain qpos path."""
        loader = pytest.importorskip("robot_descriptions.loaders.mujoco")
        panda = loader.load_robot_description("panda_mj_description")
        assert build_linkage(panda, mujoco) is None

    def test_spans_the_hardware_stroke(self, gripper_only):
        """The datasheet stroke is 140 mm. The model's *authored* driver range
        stops at a 128.6 mm aperture, so a solve confined to that range could
        not draw a fully-open gripper at all."""
        _, linkage = gripper_only
        lo, hi = linkage.aperture_range
        assert lo <= 0.0, "must be able to draw a shut gripper"
        assert hi >= 0.140, f"only spans {hi * 1000:.1f} mm, short of the 140 mm stroke"

    def test_aperture_round_trips(self, gripper_only):
        """Ask for an aperture, draw it, measure it back."""
        model, linkage = gripper_only
        data = mujoco.MjData(model)
        gl, gr = _pad_geoms(model, mujoco)
        for want in (0.0, 0.01, 0.04, 0.07, 0.10, 0.128, 0.140):
            q = np.array(model.qpos0)
            linkage.write_into(q, want)
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            assert _pad_gap(model, data, gl, gr) == pytest.approx(want, abs=5e-5)

    def test_closes_the_loop_it_is_there_to_close(self, gripper_only):
        """The whole point: the four-bar's ends must coincide at every pose.
        A naive `qpos[driver] = x` leaves them centimetres apart."""
        from policy_abstraction.visualization.robot_assembly import _closure_residual

        model, linkage = gripper_only
        data = mujoco.MjData(model)
        for want in np.linspace(0.0, 0.140, 15):
            q = np.array(model.qpos0)
            linkage.write_into(q, float(want))
            assert np.abs(_closure_residual(model, data, mujoco, q)).max() < 1e-6

    def test_naive_posing_would_tear_the_linkage_apart(self, gripper_only):
        """Guards the premise of the test above — if writing the driver angle
        alone happened to be fine, none of this machinery would be needed."""
        from policy_abstraction.visualization.robot_assembly import _closure_residual

        model, _ = gripper_only
        data = mujoco.MjData(model)
        q = np.array(model.qpos0)
        for j in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if name.endswith("driver_joint"):
                q[model.jnt_qposadr[j]] = -0.6
        assert np.abs(_closure_residual(model, data, mujoco, q)).max() > 0.01

    def test_apertures_are_ascending(self, gripper_only):
        """``np.interp`` returns nonsense, silently, on an unsorted table."""
        _, linkage = gripper_only
        assert np.all(np.diff(linkage.apertures) > 0)

    def test_out_of_range_requests_clamp(self, gripper_only):
        model, linkage = gripper_only
        lo, hi = linkage.aperture_range
        np.testing.assert_allclose(
            linkage.qpos_for_aperture(hi + 1.0), linkage.qpos_for_aperture(hi)
        )
        np.testing.assert_allclose(
            linkage.qpos_for_aperture(lo - 1.0), linkage.qpos_for_aperture(lo)
        )

    @pytest.mark.slow
    def test_solved_pose_matches_a_full_dynamics_settle(self, gripper_only):
        """The free-air model is "the coupler rests on its stop". That is an
        assertion about the hardware, so check it against MuJoCo actually
        simulating the spring-loaded mechanism to rest."""
        model, linkage = gripper_only
        data = mujoco.MjData(model)
        gl, gr = _pad_geoms(model, mujoco)
        for ctrl in (0.0, 64.0, 128.0, 192.0):
            mujoco.mj_resetData(model, data)
            data.ctrl[0] = ctrl
            for _ in range(6000):
                mujoco.mj_step(model, data)
            settled = data.qpos.copy()
            settled_gap = _pad_gap(model, data, gl, gr)

            solved = np.array(model.qpos0)
            linkage.write_into(solved, settled_gap)
            data.qpos[:] = solved
            mujoco.mj_kinematics(model, data)

            assert np.abs(settled - solved).max() < 5e-3, f"ctrl={ctrl}"
            assert _pad_gap(model, data, gl, gr) == pytest.approx(settled_gap, abs=1e-4)


class TestAssembly:
    def test_gripper_is_appended_leaving_arm_indices_alone(self, assembled):
        """Every existing caller writes ``qpos[:n_arm]``. If attaching the
        gripper renumbered the arm, they would all silently drive the wrong
        joints."""
        assert assembled.arm_nq == 6
        assert assembled.model.nq == 14
        names = [
            mujoco.mj_id2name(assembled.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(6)
        ]
        assert names == [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        assert all(a in assembled.linkage.qpos_adr for a in range(6, 14))

    def test_arm_kinematics_are_unchanged_by_the_gripper(self, assembled):
        """Cross-checked against the FK that was verified on the real arm."""
        from policy_abstraction.dynamics.ur5_kinematics import ur5e_forward_kinematics

        model = assembled.model
        data = mujoco.MjData(model)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        rng = np.random.default_rng(0)
        for q in [np.zeros(6), *[rng.uniform(-2.0, 2.0, 6) for _ in range(5)]]:
            data.qpos[:] = model.qpos0
            data.qpos[:6] = q
            mujoco.mj_kinematics(model, data)
            expected, _ = ur5e_forward_kinematics(q, None)
            # 2 mm: menagerie rounds every UR5e link offset to millimetres.
            np.testing.assert_allclose(data.site_xpos[sid], expected, atol=2e-3)

    def test_tool_length_agrees_with_the_taught_tcp(self, assembled):
        """An independent check that this is the right gripper: the model's
        fingertip must land where the TCP was taught on the real cell."""
        model = assembled.model
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        origin, rot = data.site_xpos[sid], data.site_xmat[sid].reshape(3, 3)

        reach = -np.inf
        for g in range(model.ngeom):
            body = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]
            )
            if not (body or "").startswith(f"{GRIPPER}/"):
                continue
            if (
                model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH
                or model.geom_group[g] != 2
            ):
                continue
            mid = model.geom_dataid[g]
            va, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            verts = model.mesh_vert[va : va + nv].astype(np.float64)
            world = (data.geom_xmat[g].reshape(3, 3) @ verts.T).T + data.geom_xpos[g]
            reach = max(reach, ((world - origin) @ rot)[:, 2].max())
        assert reach == pytest.approx(TAUGHT_TCP_Z_M, abs=5e-3)

    def test_missing_mount_site_says_which_sites_exist(self):
        with pytest.raises(ValueError, match="no site 'nope'"):
            compose_arm_with_gripper(ARM, GRIPPER, site="nope")


class TestRenderPath:
    """The visualiser's own hook, which is where the aperture is interpreted."""

    @staticmethod
    def _viz(assembled):
        from policy_abstraction.visualization.rerun import RerunVisualizer
        from policy_steering.robot_config import load_gripper_geometry

        viz = RerunVisualizer.__new__(RerunVisualizer)
        viz._mj = mujoco
        viz._mj_model = assembled.model
        viz._mj_data = mujoco.MjData(assembled.model)
        viz._gripper_linkage = assembled.linkage
        viz._arm_nq = assembled.arm_nq
        viz._gripper_geometry = load_gripper_geometry(GRIPPER)
        viz._aperture_clamped = False
        return viz

    def _drawn_gap(self, viz):
        mujoco.mj_kinematics(viz._mj_model, viz._mj_data)
        gl, gr = _pad_geoms(viz._mj_model, mujoco)
        return _pad_gap(viz._mj_model, viz._mj_data, gl, gr)

    @pytest.mark.parametrize("openness", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_openness_is_read_as_openness_not_metres(self, assembled, openness):
        """The env reports a normalised [0, 1] opening; the gripper config maps
        it onto the 140 mm stroke. Feeding the raw number through as metres
        would draw an essentially shut gripper at every value."""
        viz = self._viz(assembled)
        viz._pose_arm_and_gripper(viz._mj_data, np.append(np.zeros(6), openness))
        assert self._drawn_gap(viz) == pytest.approx(openness * 0.140, abs=5e-5)

    def test_arm_only_observations_leave_the_fingers_alone(self, assembled):
        """A 6-vector carries no aperture. Treating the missing entry as zero
        would snap the gripper shut."""
        viz = self._viz(assembled)
        viz._pose_arm_and_gripper(viz._mj_data, np.append(np.zeros(6), 1.0))
        before = self._drawn_gap(viz)
        viz._pose_arm_and_gripper(viz._mj_data, np.zeros(6))
        assert self._drawn_gap(viz) == pytest.approx(before, abs=1e-9)

    def test_impossible_aperture_clamps_and_warns_once(self, assembled):
        viz = self._viz(assembled)
        for _ in range(3):
            viz._pose_arm_and_gripper(viz._mj_data, np.append(np.zeros(6), 5.0))
        assert viz._aperture_clamped
        assert self._drawn_gap(viz) == pytest.approx(
            assembled.linkage.aperture_range[1], abs=5e-5
        )

    def test_arm_joints_still_move_the_arm(self, assembled):
        from policy_abstraction.dynamics.ur5_kinematics import ur5e_forward_kinematics

        viz = self._viz(assembled)
        q = np.array([0.3, -1.2, 1.4, -1.7, -1.5, 0.2])
        viz._pose_arm_and_gripper(viz._mj_data, np.append(q, 0.5))
        mujoco.mj_kinematics(viz._mj_model, viz._mj_data)
        sid = mujoco.mj_name2id(
            viz._mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        expected, _ = ur5e_forward_kinematics(q, None)
        np.testing.assert_allclose(viz._mj_data.site_xpos[sid], expected, atol=2e-3)


class TestOtherRobotiqGrippers:
    """The 2F-85 has the same topology with the driver's sign reversed. Solving
    it too is what shows the linkage code reads the model rather than hardcoding
    the one gripper it was written for."""

    @pytest.fixture(scope="class")
    def gripper_2f85(self):
        checkout = menagerie_checkout()
        if checkout is None or not (checkout / "robotiq_2f85").is_dir():
            pytest.skip("no mujoco_menagerie checkout")
        model = mujoco.MjModel.from_xml_path(str(gripper_mjcf_path("robotiq_2f85")))
        return model, build_linkage(model, mujoco)

    def test_solves_despite_the_opposite_driver_sign(self, gripper_2f85):
        model, linkage = gripper_2f85
        assert linkage is not None
        assert model.jnt_range[0][1] > 0, "premise: 2F-85 opens the other way"
        lo, hi = linkage.aperture_range
        assert lo <= 0.0 < 0.080 <= hi, f"spans {lo * 1000:.1f}-{hi * 1000:.1f} mm"

    def test_aperture_round_trips(self, gripper_2f85):
        model, linkage = gripper_2f85
        data = mujoco.MjData(model)
        gl, gr = _pad_geoms(model, mujoco)
        for want in (0.0, 0.02, 0.05, 0.08):
            q = np.array(model.qpos0)
            linkage.write_into(q, want)
            data.qpos[:] = q
            mujoco.mj_kinematics(model, data)
            assert _pad_gap(model, data, gl, gr) == pytest.approx(want, abs=5e-5)
