"""Guidance plumbing for the UR5 LeRobot backend.

Loaded by path because the server script lives under ``scripts/`` and is
normally run from the physical-ai-sdk venv. Everything exercised here is
deliberately lerobot-free, so it runs in the project venv.
"""

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

_SERVER = (
    pathlib.Path(__file__).resolve().parents[2]
    / "scripts"
    / "servers"
    / "serve_lerobot.py"
)


def _load_server_module():
    spec = importlib.util.spec_from_file_location("_serve_lerobot", _SERVER)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # `dataclasses` resolves annotations via `sys.modules[cls.__module__]`, so
    # the module has to be registered before its body runs or the @dataclass on
    # `Args` raises.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


srv = _load_server_module()


class TestSplitGuidanceOptions:
    def test_no_options_is_the_unguided_path(self):
        assert srv.split_guidance_options(None) == (None, None)

    def test_options_without_the_sentinel_pass_through_untouched(self):
        # Anything that isn't guidance must reach the server unchanged, so an
        # unguided request stays byte-for-byte what it was.
        opts = {"something_else": 1}
        assert srv.split_guidance_options(opts) == (None, opts)

    def test_sentinel_splits_gp_keys_out(self):
        from policy_steering.params import GP_SENTINEL_KEY

        opts = {GP_SENTINEL_KEY: True, "__gp_guidance_factor__": 2.0, "other": 5}
        params, cleaned = srv.split_guidance_options(opts)
        assert params is not None
        assert params["__gp_guidance_factor__"] == 2.0
        assert cleaned == {"other": 5}

    def test_bool_keys_are_coerced(self):
        # They arrive over msgpack as numpy scalars; the kernel branches on
        # them with plain `if`, so they have to be real bools.
        from policy_steering.params import GP_SENTINEL_KEY

        opts = {GP_SENTINEL_KEY: True, "__gp_line_search__": np.bool_(True)}
        params, _ = srv.split_guidance_options(opts)
        assert params is not None
        assert params["__gp_line_search__"] is True


class TestActionNormalization:
    class _Cfg:
        def __init__(self, mapping):
            self.normalization_mapping = mapping

    def test_reads_the_action_entry(self):
        assert srv._action_normalization(self._Cfg({"ACTION": "QUANTILES"})) == (
            "QUANTILES"
        )

    def test_matches_on_enum_value_not_hash(self):
        # FeatureType is a str Enum whose hash is the member name, so a plain
        # dict lookup on "ACTION" is unreliable — hence the value comparison.
        class _Key:
            value = "ACTION"

        class _Mode:
            value = "MEAN_STD"

        assert srv._action_normalization(self._Cfg({_Key(): _Mode()})) == "MEAN_STD"

    def test_unknown_when_absent(self):
        assert srv._action_normalization(self._Cfg({})) == "UNKNOWN"


class TestGuidanceKernel:
    """The kernel binds parameters once per request; check what it binds."""

    @staticmethod
    def _params(horizon):
        return {
            "__gp_ee_pos_0__": np.zeros(3, dtype=np.float32),
            "__gp_ee_ori_0__": np.zeros(3, dtype=np.float32),
            "__gp_dt__": np.float32(0.02),
            "__gp_expected_dists__": np.zeros((1, horizon), dtype=np.float32),
            "__gp_expected_stds__": np.ones((1, horizon), dtype=np.float32),
            "__gp_dist_types__": np.zeros(1, dtype=np.int64),
            "__gp_waypoints__": np.zeros((1, 3), dtype=np.float32),
            "__gp_has_axis__": np.zeros(1, dtype=np.int64),
            "__gp_axes__": np.zeros((1, 3), dtype=np.float32),
            "__gp_has_plane__": np.zeros(1, dtype=np.int64),
            "__gp_plane_normals__": np.zeros((1, 3), dtype=np.float32),
            "__gp_surface_counts__": np.zeros(1, dtype=np.int64),
            "__gp_surface_pts__": np.zeros((1, 1, 3), dtype=np.float32),
            "__gp_guidance_weights__": np.ones(1, dtype=np.float32),
            "__gp_source_type__": np.zeros(1, dtype=np.uint8),
        }

    def test_dt_matches_lerobots_negative_step(self):
        # lerobot integrates t: 1 -> 0 with dt = -1/num_steps. The shared
        # kernel's sign handling depends on matching that exactly.
        torch = pytest.importorskip("torch")
        k = srv.GuidanceKernel(
            self._params(50),
            device=torch.device("cpu"),
            dtype=torch.float32,
            action_horizon=50,
            num_steps=10,
        )
        assert k.dt == pytest.approx(-0.1)

    def test_expected_dists_are_truncated_to_the_action_horizon(self):
        # A demo longer than the chunk must not steer actions this call will
        # never produce.
        torch = pytest.importorskip("torch")
        k = srv.GuidanceKernel(
            self._params(120),
            device=torch.device("cpu"),
            dtype=torch.float32,
            action_horizon=50,
            num_steps=10,
        )
        assert k.params["__gp_expected_dists__"].shape == (1, 50)

    def test_rejects_non_finite_params(self):
        # A NaN in any parameter ends up in every action channel; the client
        # is supposed to filter these, so the server names the key and refuses.
        torch = pytest.importorskip("torch")
        params = self._params(50)
        params["__gp_expected_dists__"][0, 3] = np.nan
        with pytest.raises(ValueError, match="__gp_expected_dists__"):
            srv.GuidanceKernel(
                params,
                device=torch.device("cpu"),
                dtype=torch.float32,
                action_horizon=50,
                num_steps=10,
            )

    def test_finite_correction_passes_through(self):
        torch = pytest.importorskip("torch")
        k = srv.GuidanceKernel(
            self._params(50),
            device=torch.device("cpu"),
            dtype=torch.float32,
            action_horizon=50,
            num_steps=10,
        )
        torch.manual_seed(0)
        x_t = torch.randn(1, 50, 7)
        v_t = torch.randn(1, 50, 7)
        out = k.apply(x_t, v_t, 0)
        assert out.shape == v_t.shape
        assert torch.isfinite(out).all()
        assert k.nonfinite_steps == 0

    def test_non_finite_correction_is_served_unguided(self, capsys):
        # One bad gradient must not turn the chunk into NaN and halt the arm:
        # the step falls back to the network's velocity, is counted, and is
        # logged once with the stage that went non-finite.
        torch = pytest.importorskip("torch")
        k = srv.GuidanceKernel(
            self._params(50),
            device=torch.device("cpu"),
            dtype=torch.float32,
            action_horizon=50,
            num_steps=10,
        )
        k._apply = lambda x_t, v_t, step_idx: torch.full_like(v_t, float("nan"))
        v_t = torch.ones(1, 50, 7)
        for step in range(3):
            out = k.apply(torch.zeros(1, 50, 7), v_t, step)
            assert torch.equal(out, v_t)
        assert k.nonfinite_steps == 3
        logged = capsys.readouterr().out
        assert logged.count("non-finite velocity") == 1
        assert "finite(grad)=True" in logged


# The cell's controller TCP (docs/ur5.md): a 223 mm tool along the flange z.
_TCP = [0.00052, 0.00109, 0.22347, 0.0, 0.0, 0.0]


class TestActionUnnormalizationAffine:
    """The affine the joint-space kinematics undo the checkpoint's stats with."""

    @staticmethod
    def _pipeline(stats):
        from types import SimpleNamespace

        return SimpleNamespace(
            steps=[SimpleNamespace(), SimpleNamespace(stats={"action": stats})]
        )

    def test_identity_needs_nothing(self):
        assert (
            srv.action_unnormalization_affine(self._pipeline({}), "IDENTITY", 7) is None
        )

    def test_quantiles_map_minus_one_to_q01_and_one_to_q99(self):
        q01 = np.arange(7, dtype=np.float64) - 1.0
        q99 = np.arange(7, dtype=np.float64) + 1.0
        scale, offset = srv.action_unnormalization_affine(
            self._pipeline({"q01": q01, "q99": q99}), "QUANTILES", 7
        )
        np.testing.assert_allclose(-1.0 * scale + offset, q01)
        np.testing.assert_allclose(1.0 * scale + offset, q99)

    def test_mean_std_is_the_plain_affine(self):
        mean, std = np.full(7, 0.5), np.full(7, 2.0)
        scale, offset = srv.action_unnormalization_affine(
            self._pipeline({"mean": mean, "std": std}), "MEAN_STD", 7
        )
        np.testing.assert_allclose(scale, std)
        np.testing.assert_allclose(offset, mean)

    def test_min_max_and_quantile10_use_their_own_stats(self):
        lo, hi = np.zeros(7), np.ones(7)
        for mode, keys in (("MIN_MAX", ("min", "max")), ("QUANTILE10", ("q10", "q90"))):
            scale, offset = srv.action_unnormalization_affine(
                self._pipeline({keys[0]: lo, keys[1]: hi}), mode, 7
            )
            np.testing.assert_allclose(scale, 0.5)
            np.testing.assert_allclose(offset, 0.5)

    def test_a_zero_span_channel_falls_back_to_eps_like_lerobot(self):
        lo = np.zeros(7)
        hi = np.ones(7)
        hi[6] = 0.0  # a gripper that never moved in the training data
        scale, _ = srv.action_unnormalization_affine(
            self._pipeline({"q01": lo, "q99": hi}), "QUANTILES", 7
        )
        assert scale[6] == pytest.approx(0.5e-8)

    def test_missing_stats_are_refused_rather_than_guessed(self):
        # Reading normalized units as radians would move the wrong joints
        # silently; the error has to name what is missing.
        with pytest.raises(ValueError, match="q99"):
            srv.action_unnormalization_affine(
                self._pipeline({"q01": np.zeros(7)}), "QUANTILES", 7
            )
        with pytest.raises(ValueError, match="no action stats"):
            srv.action_unnormalization_affine(SimpleNamespace(steps=[]), "QUANTILES", 7)

    def test_stats_must_match_the_declared_action_dim(self):
        with pytest.raises(ValueError, match="action_dim=7"):
            srv.action_unnormalization_affine(
                self._pipeline({"q01": np.zeros(32), "q99": np.ones(32)}),
                "QUANTILES",
                7,
            )


class TestServerConfigActionSpace:
    def test_the_default_is_the_simulators_convention(self):
        cfg = srv.ServerConfig(model_path="/ckpt")
        assert cfg.action_space == "cartesian-delta"
        assert cfg.tcp_offset is None
        assert cfg.gripper_geometry is None


class TestJointSpaceGuidance:
    """What ``--action-space ur5e-joints`` changes about the correction.

    The regression this guards: on the real arm every guided chunk started
    2-5 cm from the tool, because the loss integrated the base/shoulder/elbow
    angles as x/y/z velocities and the cumsum put the whole correction on the
    first action.
    """

    HORIZON = 12

    @classmethod
    def _home(cls):
        from policy_abstraction.envs.ur5.config import UR5E_HOME_JOINT_POSITIONS

        return np.asarray(UR5E_HOME_JOINT_POSITIONS, dtype=np.float64)

    @classmethod
    def _stats(cls):
        """Quantile stats centred on HOME so the held chunk is normalized 0."""
        home = cls._home()
        q01 = np.r_[home - 0.5, 0.0]
        q99 = np.r_[home + 0.5, 1.0]
        return q01, q99

    def _kernel(self, expected, *, joint_space: bool, torch, line_search=False):
        from policy_abstraction.dynamics.ur5_kinematics import ur5e_forward_kinematics

        home = self._home()
        ee0, ori0 = ur5e_forward_kinematics(home, _TCP)
        params = TestGuidanceKernel._params(self.HORIZON)
        params["__gp_ee_pos_0__"] = ee0.astype(np.float32)
        params["__gp_ee_ori_0__"] = ori0.astype(np.float32)
        params["__gp_waypoints__"] = (ee0 + np.array([0.0, 0.0, -0.3])).astype(
            np.float32
        )[None]
        params["__gp_expected_dists__"] = np.asarray(expected, dtype=np.float32)[None]
        params["__gp_line_search__"] = line_search
        dynamics_fn = None
        geometry = None
        if joint_space:
            from policy_steering.robot_config import load_gripper_geometry
            from policy_steering.ur5e_kinematics_torch import (
                make_affine_unnormalizer,
                make_ur5e_torch_dynamics,
            )

            q01, q99 = self._stats()
            unnormalize = make_affine_unnormalizer(
                torch.as_tensor((q99 - q01) / 2, dtype=torch.float32),
                torch.as_tensor((q99 + q01) / 2, dtype=torch.float32),
            )
            dynamics_fn = make_ur5e_torch_dynamics(_TCP, unnormalize=unnormalize)
            geometry = load_gripper_geometry("robotiq_2f140")
        return srv.GuidanceKernel(
            params,
            device=torch.device("cpu"),
            dtype=torch.float32,
            action_horizon=self.HORIZON,
            num_steps=10,
            dynamics_fn=dynamics_fn,
            gripper_geometry=geometry,
        )

    def _held_chunk(self, torch):
        """HOME held for the horizon, in the normalized 32-wide layout."""
        x = torch.zeros(1, self.HORIZON, 32)
        x[..., 6] = 1.0  # gripper open = q99 of its window
        return x

    def test_a_held_chunk_at_the_expected_distance_is_left_alone(self):
        torch = pytest.importorskip("torch")
        # The waypoint sits 0.3 m below the tool; ask for exactly that.
        k = self._kernel(
            [0.3] * self.HORIZON, joint_space=True, torch=torch, line_search=True
        )
        x = self._held_chunk(torch)
        assert float(k.batch_loss_fn(x)) == pytest.approx(0.0, abs=1e-6)
        assert float(k.batch_grad_fn(x).abs().max()) < 1e-4
        # The unit-normalized direction is noise here; the line search is
        # what turns "nothing to fix" into a zero step (its do-no-harm rule).
        v = torch.zeros_like(x)
        assert torch.allclose(k.apply(x, v, 0), v, atol=1e-6)

    def test_the_same_chunk_read_as_velocities_is_not(self):
        torch = pytest.importorskip("torch")
        # Under Euler integration the normalized joints are tool velocities,
        # so a chunk that means "hold still" is scored as a moving tool.
        k = self._kernel([0.3] * self.HORIZON, joint_space=False, torch=torch)
        x = self._held_chunk(torch)
        x[..., :6] = 0.4  # any non-zero normalized joint reads as a velocity
        assert float(k.batch_loss_fn(x)) > 1e-3

    def test_the_correction_is_not_front_loaded_onto_the_first_action(self):
        torch = pytest.importorskip("torch")
        # Pull the tool 5 cm toward the waypoint over the chunk.
        expected = np.linspace(0.3, 0.25, self.HORIZON)
        x = self._held_chunk(torch)

        k_joint = self._kernel(expected, joint_space=True, torch=torch)
        per_step = k_joint.batch_grad_fn(x)[0, :, :6].norm(dim=-1)
        assert bool(torch.isfinite(per_step).all())
        assert float(per_step[0]) < 2.0 * float(per_step.mean())

        k_euler = self._kernel(expected, joint_space=False, torch=torch)
        per_step_euler = k_euler.batch_grad_fn(x)[0, :, :6].norm(dim=-1)
        # The cumsum makes step 0 carry every later step's error.
        assert float(per_step_euler[0]) > 3.0 * float(per_step_euler[-1])

    def test_the_gradient_moves_the_joints_that_lower_the_tool(self):
        torch = pytest.importorskip("torch")
        expected = np.full(self.HORIZON, 0.25)
        k = self._kernel(expected, joint_space=True, torch=torch)
        g = k.batch_grad_fn(self._held_chunk(torch))[0]
        # Only the real channels can carry a gradient: the padding is dead.
        assert float(g[:, 7:].abs().max()) == 0.0
        # The gripper does not move the tool.
        assert float(g[:, 6].abs().max()) < 1e-6
        # Shoulder lift and elbow do.
        assert float(g[:, 1].abs().max()) > 1e-3
        assert float(g[:, 2].abs().max()) > 1e-3
