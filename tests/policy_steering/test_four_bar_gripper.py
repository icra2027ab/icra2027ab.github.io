"""Four-bar finger kinematics for the Robotiq 2F-140.

A parallel-jaw gripper's tips sit at a fixed height. The 2F-140's pads hang off
a four-bar linkage and ride an arc, so the pad retracts monotonically by ~30 mm
across the stroke. Guidance measures distances from those tips, so pinning them
at a constant z is a systematic error exactly where a grasp is being made.

The reference here is the **vendored menagerie MJCF**, solved as a closed chain
— not the formula's own constants. An earlier version of this file asserted the
excursion against a number derived from `arc_radius`/`arc_pivot_y`, which made
it pass while the model was wrong by 26.8 mm with an inverted slope. Any test
that can be satisfied without consulting the linkage is not testing the model.
"""

import numpy as np
import pytest

from policy_steering.robot_config import GripperGeometry, load_gripper_geometry

torch = pytest.importorskip("torch")
mujoco = pytest.importorskip("mujoco")

from policy_abstraction import PROJECT_ROOT  # noqa: E402
from policy_steering.kinematics_torch import (  # noqa: E402
    torch_finger_positions,
    torch_finger_positions_vmap,
)

MJCF = (
    PROJECT_ROOT
    / "src/policy_abstraction/assets/mujoco_menagerie/robotiq_2f140/2f140.xml"
)
STROKE = 0.140
# Worst-case residual of the fitted arc against the solved linkage, over the
# shipping stroke. Measured at 0.119 mm; the bound leaves room for solver noise
# without admitting a regression (the broken model was off by 26.8 mm).
TOLERANCE_M = 0.0005


@pytest.fixture(scope="module")
def measured():
    """(aperture, pad z relative to closed) sampled from the real linkage."""
    from policy_abstraction.visualization.robot_assembly import (
        _pad_geoms,
        build_linkage,
    )

    if not MJCF.exists():
        pytest.skip(f"vendored 2F-140 MJCF not present at {MJCF}")
    model = mujoco.MjModel.from_xml_path(str(MJCF))
    link = build_linkage(model, mujoco)
    assert link is not None, "2f140.xml should solve as a closed chain"

    gl, gr = _pad_geoms(model, mujoco)
    data = mujoco.MjData(model)
    ap, z = [], []
    for i, a in enumerate(link.apertures):
        data.qpos[link.qpos_adr] = link.table[i]
        mujoco.mj_kinematics(model, data)
        ap.append(float(a))
        z.append(float(0.5 * (data.geom_xpos[gl] + data.geom_xpos[gr])[2]))
    ap, z = np.array(ap), np.array(z)
    z = z - z[np.argmin(ap)]  # zero at closed, matching finger_z_from_grip
    keep = (ap >= 0.0) & (ap <= STROKE)
    order = np.argsort(ap[keep])
    return ap[keep][order], z[keep][order]


class TestAgainstTheLinkage:
    """The tests that would have caught the 26.8 mm error."""

    def test_model_tracks_the_solved_linkage(self, measured):
        ap, z = measured
        g = load_gripper_geometry("robotiq_2f140")
        pred = np.array([g.finger_z_offset(a / 2.0) for a in ap])
        err = np.abs(pred - z).max()
        assert err < TOLERANCE_M, f"worst error {err * 1000:.2f} mm vs the MJCF"

    def test_pad_retracts_monotonically(self, measured):
        """The broken model rose while the real pad fell. Sign of the slope is
        the property that matters for guidance, more than its magnitude."""
        ap, z = measured
        g = load_gripper_geometry("robotiq_2f140")
        pred = np.array([g.finger_z_offset(a / 2.0) for a in ap])
        assert np.all(np.diff(z) <= 1e-9), "measured curve should fall"
        assert np.all(np.diff(pred) <= 1e-9), "model should fall too"

    def test_excursion_matches_the_linkage(self, measured):
        """~30 mm, read off the model rather than recomputed from the config."""
        ap, z = measured
        g = load_gripper_geometry("robotiq_2f140")
        pred = np.array([g.finger_z_offset(a / 2.0) for a in ap])
        assert (pred.max() - pred.min()) == pytest.approx(
            z.max() - z.min(), abs=TOLERANCE_M
        )

    def test_a_constant_offset_would_not_do(self, measured):
        """Justifies the arc existing at all: pinning z loses ~30 mm."""
        _ap, z = measured
        assert (z.max() - z.min()) > 0.02


class TestGeometryLoading:
    def test_2f140_is_a_four_bar(self):
        g = load_gripper_geometry("robotiq_2f140")
        assert g.is_four_bar
        assert g.arc_radius == pytest.approx(0.09648)

    def test_panda_stays_prismatic(self):
        # The Panda's fingers really do slide, so it must not gain an arc.
        g = load_gripper_geometry("franka_panda")
        assert not g.is_four_bar
        assert g.finger_z_offset(0.02) == g.finger_z_from_grip


class TestArc:
    @staticmethod
    def _g(**kw):
        base = dict(
            name="probe",
            finger_x_rest=0.0,
            finger_z_from_grip=0.0,
            action_open=1.0,
            action_close=0.0,
            metric_open=STROKE,
            metric_close=0.0,
            qpos_open=1.0,
            qpos_close=0.0,
            arc_radius=0.09648,
        )
        base.update(kw)
        return GripperGeometry(**base)

    def test_closed_pose_is_the_zero_reference(self):
        # finger_z_from_grip keeps meaning "the offset when shut", so the TCP
        # taught at the closed fingertip plane stays correct.
        assert self._g().finger_z_offset(0.0) == pytest.approx(0.0, abs=1e-12)

    def test_beyond_reach_saturates_instead_of_going_nan(self):
        # A commanded aperture wider than the linkage can reach must not put a
        # NaN into the guidance gradient.
        z = self._g().finger_z_offset(0.5)
        assert np.isfinite(z)

    def test_beyond_reach_keeps_the_gradient_finite_too(self):
        # The forward value was always clamped; the gradient was not. Guidance
        # denoises from pure noise, so apertures past the linkage's reach are
        # the common case there, and a NaN in this derivative propagates to
        # every action channel of the chunk (observed as an all-NaN chunk that
        # the UR5 env server rejected).
        g = self._g()
        for half_aperture in (0.5, -0.5, g.arc_radius, 0.02):
            x = torch.tensor(half_aperture, dtype=torch.float64, requires_grad=True)
            z = g.finger_z_offset(x)
            (dz,) = torch.autograd.grad(z, x)
            assert torch.isfinite(z), half_aperture
            assert torch.isfinite(dz), half_aperture
        # Inside the arc the derivative is the circle's slope, not the clamp's.
        x = torch.tensor(0.02, dtype=torch.float64, requires_grad=True)
        (dz,) = torch.autograd.grad(g.finger_z_offset(x), x)
        expected = -0.02 / np.sqrt(g.arc_radius**2 - 0.02**2)
        assert float(dz) == pytest.approx(expected, rel=1e-9)

    def test_offset_is_even_in_the_half_aperture(self):
        g = self._g()
        assert g.finger_z_offset(0.03) == pytest.approx(g.finger_z_offset(-0.03))


class TestKinematicsIntegration:
    EE_POS = torch.tensor([0.4, 0.0, 0.3], dtype=torch.float64)
    EE_ORI = torch.zeros(3, dtype=torch.float64)

    def test_tips_descend_as_the_gripper_opens(self):
        g = load_gripper_geometry("robotiq_2f140")
        zs = [
            float(
                torch_finger_positions(self.EE_POS, self.EE_ORI, torch.tensor(a), g)[0][
                    2
                ]
            )
            for a in (0.0, 0.06, 0.14)
        ]
        assert zs[0] > zs[1] > zs[2]

    def test_vmap_matches_the_scalar_path(self):
        g = load_gripper_geometry("robotiq_2f140")
        ap = torch.linspace(0.0, STROKE, 5, dtype=torch.float64)
        left, _ = torch_finger_positions_vmap(
            self.EE_POS.expand(5, 3).clone(), self.EE_ORI.expand(5, 3).clone(), ap, g
        )
        for i in range(5):
            one, _ = torch_finger_positions(self.EE_POS, self.EE_ORI, ap[i], g)
            assert torch.allclose(left[i], one)

    def test_gradient_flows_through_the_arc(self):
        # Guidance differentiates the tip position w.r.t. the action chunk; a
        # detached or NaN gradient here would silently disable gripper steering.
        g = load_gripper_geometry("robotiq_2f140")
        a = torch.tensor(0.08, dtype=torch.float64, requires_grad=True)
        left, _ = torch_finger_positions(self.EE_POS, self.EE_ORI, a, g)
        left[2].backward()
        assert a.grad is not None and torch.isfinite(a.grad)
        assert float(a.grad) < 0  # opening the gripper lowers the pad


class TestFingerAxisAcrossImplementations:
    """numpy, JAX and torch FK all put the 2F-140's jaws on the same axis.

    Guidance differentiates through the JAX/torch paths while the abstractor
    and the visualiser read the numpy one; a finger_axis honoured by only some
    of them would steer the fingers towards a point the overlay never shows.
    """

    EE_POS = np.array([0.4, -0.1, 0.3])
    EE_ORI = np.array([0.3, -0.5, 1.1])  # a generic tilt, so axes mix

    @pytest.mark.parametrize("name", ["robotiq_2f140", "franka_panda"])
    def test_jax_and_torch_match_numpy(self, name):
        import jax.numpy as jnp

        from policy_abstraction.utils.gripper import compute_finger_tip_positions
        from policy_steering.kinematics import jax_finger_positions

        g = load_gripper_geometry(name)
        aperture = 0.05 if name == "franka_panda" else 0.10
        # numpy reads raw qpos, whose layout is per-gripper: the Panda
        # publishes metric [q1, q2], the UR5 adapter a normalised openness.
        if name == "franka_panda":
            gs = np.array([[aperture / 2, -aperture / 2]])
        else:
            gs = np.array([[aperture / STROKE]])
        np_l, np_r = compute_finger_tip_positions(
            self.EE_POS[None], self.EE_ORI[None], gs, g
        )
        jl, jr = jax_finger_positions(
            jnp.asarray(self.EE_POS), jnp.asarray(self.EE_ORI), jnp.asarray(aperture), g
        )
        tl, tr = torch_finger_positions(
            torch.tensor(self.EE_POS),
            torch.tensor(self.EE_ORI),
            torch.tensor(aperture),
            g,
        )
        # The jaw line (right minus left) is what the finger axis sets, and
        # it is the same in every implementation. Absolute positions are only
        # compared for the prismatic Panda: numpy's path still holds the
        # 2F-140's pads at a constant z while JAX/torch swing them on the
        # four-bar arc, a pre-existing difference this test does not own.
        np.testing.assert_allclose(
            np.asarray(jr) - np.asarray(jl), np_r[0] - np_l[0], atol=1e-6
        )
        np.testing.assert_allclose((tr - tl).numpy(), np_r[0] - np_l[0], atol=1e-9)
        np.testing.assert_allclose(np.asarray(jl), tl.numpy(), atol=1e-6)
        np.testing.assert_allclose(np.asarray(jr), tr.numpy(), atol=1e-6)
        if not g.is_four_bar:
            np.testing.assert_allclose(np.asarray(jl), np_l[0], atol=1e-6)
            np.testing.assert_allclose(tl.numpy(), np_l[0], atol=1e-9)

    def test_2f140_jaws_open_along_tool_x_in_every_implementation(self):
        import jax.numpy as jnp

        from policy_steering.kinematics import jax_finger_positions

        g = load_gripper_geometry("robotiq_2f140")
        ee_pos = np.zeros(3)
        ee_ori = np.zeros(3)
        expected_l = np.array([-0.05, 0.0, g.finger_z_offset(0.05)])
        expected_r = np.array([0.05, 0.0, g.finger_z_offset(0.05)])
        jl, jr = jax_finger_positions(
            jnp.asarray(ee_pos), jnp.asarray(ee_ori), jnp.asarray(0.10), g
        )
        tl, tr = torch_finger_positions(
            torch.tensor(ee_pos), torch.tensor(ee_ori), torch.tensor(0.10), g
        )
        tvl, tvr = torch_finger_positions_vmap(
            torch.zeros(2, 3, dtype=torch.float64),
            torch.zeros(2, 3, dtype=torch.float64),
            torch.full((2,), 0.10, dtype=torch.float64),
            g,
        )
        for left, right in ((jl, jr), (tl, tr), (tvl[1], tvr[1])):
            np.testing.assert_allclose(np.asarray(left), expected_l, atol=1e-9)
            np.testing.assert_allclose(np.asarray(right), expected_r, atol=1e-9)
