"""The backend wiring shared by evaluate.py and the experiment GUI.

The policy-client construction is checked against each client's *real*
signature. It has to be: a stale keyword there is invisible until
someone runs that backend, and the one place it surfaces is at a
powered-on robot cell, three minutes into a checkpoint load. A
``log_dir=`` left on the LeRobot branch did exactly that.
"""

from __future__ import annotations

import dataclasses
import inspect
import pathlib

import numpy as np
import pytest

from policy_abstraction.evaluation import backends
from policy_abstraction.evaluation.config import (
    Args,
    CosmosPolicyArgs,
    GrootPolicyArgs,
    LerobotPolicyArgs,
    LiberoEnvArgs,
    OpenpiPolicyArgs,
    Ur5EnvArgs,
    env_kind,
)
from policy_abstraction.policies.cosmos import CosmosPolicyClient
from policy_abstraction.policies.groot import GrootPolicyClient
from policy_abstraction.policies.lerobot import LerobotPolicyClient
from policy_abstraction.policies.openpi import PatientWebsocketClient


def _args(policy, env=None) -> Args:
    """A parsed Args for guidance off.

    ``guide=False`` because ``env:ur5`` refuses the default GT
    perception mode outright, and these tests are about the
    backend/client wiring rather than about perception.
    """
    return Args(policy=policy, env=env or LiberoEnvArgs(), guide=False)


# (policy args, the client class the branch must build)
POLICY_BACKENDS = [
    pytest.param(OpenpiPolicyArgs(), PatientWebsocketClient, id="openpi"),
    pytest.param(GrootPolicyArgs(model_path="/ckpt"), GrootPolicyClient, id="groot"),
    pytest.param(
        LerobotPolicyArgs(model_path="/ckpt"), LerobotPolicyClient, id="lerobot"
    ),
    pytest.param(CosmosPolicyArgs(), CosmosPolicyClient, id="cosmos"),
]


class _RecordingPolicy:
    """A policy client that records what it was asked, and answers nothing."""

    def __init__(self, resets_per_episode: bool = False) -> None:
        self.calls: list[dict] = []
        self.resets = 0
        self.resets_per_episode = resets_per_episode

    def infer(self, element: dict) -> dict:
        self.calls.append(element)
        return {"actions": np.zeros((1, 10, 7), dtype=np.float32)}

    def reset(self) -> None:
        self.resets += 1


@pytest.fixture
def ur5_env_stub():
    """An env client that serves observations and counts any command."""
    from policy_abstraction.envs.ur5.config import ur5_env_spec

    spec = ur5_env_spec()

    class _Env:
        env_spec = spec
        steps = 0
        resets = 0

        def get_obs(self):
            obs = {
                cam.image_key: np.full((480, 640, 3), 90, dtype=np.uint8)
                for cam in spec.cameras
            }
            obs["eef_pos"] = np.zeros(3, dtype=np.float32)
            obs["eef_quat"] = np.array([0, 0, 0, 1], dtype=np.float32)
            obs["gripper_qpos"] = np.zeros(1, dtype=np.float32)
            obs["joint_pos"] = np.zeros(6, dtype=np.float32)
            return obs

        def step(self, _action):
            type(self).steps += 1
            return self.get_obs(), 0.0, False, {}

        def reset(self):
            type(self).resets += 1

    _Env.steps = 0
    _Env.resets = 0
    return _Env()


class TestPolicyClientConstruction:
    """Every branch of ``_build_policy_client`` must actually be callable."""

    @pytest.mark.parametrize("policy,client_cls", POLICY_BACKENDS)
    def test_no_unknown_keyword_reaches_the_client(
        self, policy, client_cls, monkeypatch, tmp_path
    ):
        """Bind the kwargs against the signature without constructing.

        This is the check that would have caught ``log_dir=`` on the
        LeRobot branch, and it names the offending keyword.
        """
        seen: dict = {}

        def _capture(*a, **k):
            seen["a"], seen["k"] = a, k
            return object()

        monkeypatch.setattr(backends, client_cls.__name__, _capture)
        backends._build_policy_client(_args(policy), tmp_path, None)

        signature = inspect.signature(client_cls.__init__)
        # Raises TypeError naming the bad keyword if one does not fit.
        signature.bind(object(), *seen["a"], **seen["k"])

    def test_lerobot_replan_steps_is_always_a_positive_int(self, monkeypatch, tmp_path):
        """The client rejects ``None``; RTC only changes what the server does."""
        seen: dict = {}
        monkeypatch.setattr(
            backends,
            "LerobotPolicyClient",
            lambda **k: seen.update(k) or object(),
        )
        for rtc in (True, False):
            policy = LerobotPolicyArgs(model_path="/ckpt", rtc=rtc)
            backends._build_policy_client(_args(policy), tmp_path, None)
            assert isinstance(seen["replan_steps"], int)
            assert seen["replan_steps"] >= 1

    def test_debug_uses_an_in_process_fake_and_loads_nothing(self, tmp_path):
        args = _args(LerobotPolicyArgs(model_path="/ckpt"))
        args.guidance.debug = True
        args.guidance.fake_policy = "noop"
        assert backends._build_policy_client(args, tmp_path, None) is not None
        # ... and no server is launched for it.
        proc, processes = backends._start_policy_backend(args, tmp_path)
        assert proc is None and processes == []

    def test_the_fake_chunk_covers_replan_steps(self, tmp_path):
        """A shorter chunk than the runner consumes replans early and warns.

        The real-arm launcher replans every 25 steps, where the fakes were
        written for LIBERO's 10-step chunks.
        """
        args = _args(LerobotPolicyArgs(replan_steps=25), env=Ur5EnvArgs())
        args.guidance.debug = True
        for kind in ("noop", "circular"):
            args.guidance.fake_policy = kind
            client = backends._build_policy_client(args, tmp_path, None)
            actions = client.infer({})["actions"]
            assert actions.shape[1] >= args.policy.replan_steps, kind

    def test_the_fake_chunk_is_unchanged_at_the_libero_defaults(self, tmp_path):
        """Sizing from replan_steps must not shrink the historical shape."""
        args = _args(OpenpiPolicyArgs())  # replan_steps=5
        args.guidance.debug = True
        assert backends._fake_policy_shape(args, None) == (10, 7)

    def test_the_fake_width_follows_the_env_action_layout(self, tmp_path):
        """A fake emitting the wrong width fails inside an env adapter."""
        from policy_abstraction.envs.ur5.config import ur5_env_spec

        spec = ur5_env_spec()
        args = _args(LerobotPolicyArgs(), env=Ur5EnvArgs())
        args.guidance.debug = True
        _, action_dim = backends._fake_policy_shape(args, spec)
        assert action_dim == spec.robot.action_layout.action_dim

    def test_unknown_fake_policy_is_named(self, tmp_path):
        args = _args(OpenpiPolicyArgs())
        args.guidance.debug = True
        args.guidance.fake_policy = "teleport"
        with pytest.raises(ValueError, match="teleport"):
            backends._build_policy_client(args, tmp_path, None)


class TestPolicyBackendLaunch:
    def test_lerobot_without_a_checkpoint_is_a_clear_error(self, tmp_path):
        args = _args(LerobotPolicyArgs(), env=Ur5EnvArgs())
        with pytest.raises(ValueError, match="model-path"):
            backends._start_policy_backend(args, tmp_path)

    def test_no_launch_server_spawns_nothing(self, tmp_path):
        args = _args(
            LerobotPolicyArgs(model_path="/ckpt", launch_server=False),
            env=Ur5EnvArgs(),
        )
        proc, processes = backends._start_policy_backend(args, tmp_path)
        assert proc is None and processes == []

    def test_a_sim_env_without_a_checkpoint_says_what_to_do(self, tmp_path):
        """RoboCasa/CALVIN have no checkpoint; the message must say so."""
        from policy_abstraction.evaluation.config import RobocasaEnvArgs

        args = _args(OpenpiPolicyArgs(), env=RobocasaEnvArgs(task="x"))
        with pytest.raises(RuntimeError, match="guidance.debug"):
            backends._start_policy_backend(args, tmp_path)

    def test_the_ur5_message_points_at_the_lerobot_backend(self, tmp_path):
        args = _args(OpenpiPolicyArgs(), env=Ur5EnvArgs())
        with pytest.raises(RuntimeError, match="policy:lerobot"):
            backends._start_policy_backend(args, tmp_path)


class TestHelpers:
    def test_relation_filter_is_none_when_nothing_is_ablated(self):
        args = _args(OpenpiPolicyArgs())
        assert backends.relation_filter_for(args) is None

    def test_relation_filter_names_the_kept_categories(self):
        args = _args(OpenpiPolicyArgs())
        args.guidance.relation_orientation = False
        assert backends.relation_filter_for(args) == frozenset(
            {"positional", "gripper"}
        )

    def test_flip_u_follows_the_camera_transform(self):
        """Must agree with experiments.overlay.preprocessed_for."""
        from policy_abstraction.envs.libero.config import libero_env_spec

        spec = libero_env_spec()
        assert backends.resolve_flip_u("auto", spec) is True
        assert backends.resolve_flip_u("false", spec) is False
        # No spec published: keep the historical simulator behaviour.
        assert backends.resolve_flip_u("auto", None) is True

    def test_flip_u_rejects_a_typo_rather_than_guessing(self):
        with pytest.raises(ValueError, match="auto/true/false"):
            backends.resolve_flip_u("yes", None)

    def test_ur5_gets_forward_kinematics_not_integration(self):
        from policy_abstraction.dynamics import (
            IntegrationDynamicsModel,
            Ur5KinematicsDynamicsModel,
        )

        sim = backends.dynamics_model_for(_args(OpenpiPolicyArgs()))
        assert isinstance(sim, IntegrationDynamicsModel)
        real = backends.dynamics_model_for(
            _args(LerobotPolicyArgs(model_path="/c"), env=Ur5EnvArgs())
        )
        assert isinstance(real, Ur5KinematicsDynamicsModel)

    def test_the_circular_fake_integrates_even_on_the_ur5(self):
        """It emits Cartesian deltas; read as joint angles they draw the
        predicted chunk metres from the gripper."""
        from policy_abstraction.dynamics import IntegrationDynamicsModel

        args = _args(LerobotPolicyArgs(model_path="/c"), env=Ur5EnvArgs())
        args.guidance.debug = True
        args.guidance.fake_policy = "circular"
        assert isinstance(backends.dynamics_model_for(args), IntegrationDynamicsModel)

    def test_the_noop_fake_on_the_ur5_is_drawn_through_forward_kinematics(self):
        """Its chunk is the observed joints plus the guidance correction —
        absolute joint targets, so the predicted trajectory must be their FK.
        Integrating them instead starts the drawn chunk at the end-effector
        and then shoots off at "joint radians per second"."""
        from policy_abstraction.dynamics import (
            IntegrationDynamicsModel,
            Ur5KinematicsDynamicsModel,
        )

        args = _args(LerobotPolicyArgs(model_path="/c"), env=Ur5EnvArgs())
        args.guidance.debug = True
        args.guidance.fake_policy = "noop"
        assert isinstance(backends.dynamics_model_for(args), Ur5KinematicsDynamicsModel)
        # ... and a Cartesian env keeps integrating.
        sim = _args(OpenpiPolicyArgs())
        sim.guidance.debug = True
        assert isinstance(backends.dynamics_model_for(sim), IntegrationDynamicsModel)

    def test_the_noop_fake_on_the_ur5_holds_the_observed_joints(self, tmp_path):
        """Without guidance the fake must return the joints it was shown, not
        zeros: zero is the fully-extended pose on a joint-absolute arm."""
        import numpy as np

        args = _args(LerobotPolicyArgs(replan_steps=25), env=Ur5EnvArgs())
        args.guidance.debug = True
        args.guidance.fake_policy = "noop"
        client = backends._build_policy_client(args, tmp_path, None)
        state = np.array([0.1, -1.2, 1.4, -1.8, -1.5, 0.3, 0.7], dtype=np.float32)
        actions = client.infer({"observation/state": state})["actions"]
        assert actions.shape[1] >= 25 and actions.shape[2] == 7
        np.testing.assert_allclose(
            actions, np.broadcast_to(state, actions.shape), atol=1e-6
        )

    def test_the_noop_fake_on_a_sim_env_still_emits_zeros(self, tmp_path):
        import numpy as np

        args = _args(OpenpiPolicyArgs())
        args.guidance.debug = True
        args.guidance.fake_policy = "noop"
        client = backends._build_policy_client(args, tmp_path, None)
        state = np.ones(8, dtype=np.float32)
        actions = client.infer({"observation/state": state})["actions"]
        assert not np.any(actions)

    def test_the_ur5_gripper_sits_at_menagerie_default_mount(self):
        """The 2F-140 opens along tool x on this cell, which is how menagerie
        bolts it on; the FK's finger_axis says the same (test_gripper_mount
        checks the two agree). A 90° here used to match a y-axis FK, and both
        were a quarter turn off the real gripper."""
        args = _args(LerobotPolicyArgs(model_path="/c"), env=Ur5EnvArgs())
        assert args.viz.gripper_mount_yaw_deg == 0.0
        assert args.viz.gripper_model == "robotiq_2f140"
        assert _args(OpenpiPolicyArgs()).viz.gripper_mount_yaw_deg == 0.0


class TestEvalBackendsBundle:
    """The suite accessors the episode loop reads."""

    def _bundle(self, suite_info) -> backends.EvalBackends:
        return backends.EvalBackends(
            args=_args(OpenpiPolicyArgs()),
            run_dir=__import__("pathlib").Path("."),
            env=object(),
            policy=object(),
            suite_info=suite_info,
        )

    def test_per_task_horizons_are_used_when_published(self):
        bundle = self._bundle(
            {
                "num_tasks": 3,
                "task_languages": ["a", "b", "c"],
                "max_steps": 100,
                "task_max_steps": [10, 20, 30],
            }
        )
        assert bundle.task_max_steps == [10, 20, 30]

    def test_a_scalar_horizon_is_broadcast(self):
        """Older servers publish only the scalar."""
        bundle = self._bundle(
            {"num_tasks": 3, "task_languages": ["a", "b", "c"], "max_steps": 42}
        )
        assert bundle.task_max_steps == [42, 42, 42]

    def test_shutdown_is_idempotent(self):
        class _Env:
            disconnects = 0

            def disconnect(self):
                _Env.disconnects += 1

        bundle = backends.EvalBackends(
            args=_args(OpenpiPolicyArgs()),
            run_dir=__import__("pathlib").Path("."),
            env=_Env(),
            policy=object(),
            suite_info={"num_tasks": 0, "task_languages": [], "max_steps": 1},
        )
        bundle.shutdown()
        bundle.shutdown()
        assert _Env.disconnects == 1

    def test_abstractor_kwargs_are_none_when_guidance_is_off(self):
        args = _args(OpenpiPolicyArgs())
        args.retrieval.enable = False
        bundle = backends.EvalBackends(
            args=args,
            run_dir=__import__("pathlib").Path("."),
            env=type("E", (), {"env_spec": None})(),
            policy=object(),
            suite_info={"num_tasks": 1, "task_languages": ["a"], "max_steps": 1},
        )
        assert bundle.abstractor_kwargs() is None


class TestPolicyWarmUp:
    """The compile is paid at start-up, and the arm is never commanded for it.

    A torch.compile'd pi05 checkpoint autotunes on its *first real query*, not
    at load: measured on the UR5e cell, 106 s to "server ready" and a further
    449 s inside the first inference. Left to the first rollout that is a
    rollout in which nothing moves and any wall-clock timeout has already
    expired.
    """

    def _bundle(self, policy, env):
        return backends.EvalBackends(
            args=_args(LerobotPolicyArgs(), env=Ur5EnvArgs()),
            run_dir=pathlib.Path("."),
            env=env,
            policy=policy,
            suite_info={"num_tasks": 1, "task_languages": ["a"], "max_steps": 1},
        )

    def test_it_queries_the_policy_and_never_steps_the_env(self, ur5_env_stub):
        policy = _RecordingPolicy()
        elapsed = self._bundle(policy, ur5_env_stub).warm_up_policy("do the thing")

        assert elapsed is not None and elapsed >= 0.0
        assert len(policy.calls) == 1
        assert ur5_env_stub.steps == 0, "the warm-up commanded the arm"
        assert ur5_env_stub.resets == 0

    def test_the_query_carries_the_prompt_and_every_policy_input(self, ur5_env_stub):
        """Same shapes as a real step, or the compiler recompiles for those."""
        policy = _RecordingPolicy()
        self._bundle(policy, ur5_env_stub).warm_up_policy("do the thing")

        element = policy.calls[0]
        io_keys = ur5_env_stub.env_spec.policy_input_keys
        assert element[io_keys.prompt_key] == "do the thing"
        assert io_keys.state_key in element
        for _cam, policy_key in io_keys.images:
            assert element[policy_key].shape == (224, 224, 3)

    def test_rtc_state_is_reset_so_it_cannot_bleed_into_rollout_zero(
        self, ur5_env_stub
    ):
        """The warm-up chunk is a real chunk in the server's RTC buffer.

        Left there, it is inpainted into the opening chunk of the first
        episode — blended with actions computed for a scene nobody staged.
        """
        policy = _RecordingPolicy(resets_per_episode=True)
        self._bundle(policy, ur5_env_stub).warm_up_policy("do the thing")
        assert policy.resets == 1

    def test_a_broken_env_costs_the_warm_up_and_nothing_else(self):
        class _Env:
            env_spec = None

        policy = _RecordingPolicy()
        assert self._bundle(policy, _Env()).warm_up_policy("x") is None
        assert policy.calls == []

    def test_a_raising_policy_is_swallowed(self, ur5_env_stub):
        class _Boom:
            def infer(self, _element):
                raise RuntimeError("server went away")

        assert self._bundle(_Boom(), ur5_env_stub).warm_up_policy("x") is None


class TestEnvKind:
    @pytest.mark.parametrize(
        "env,expected",
        [
            (LiberoEnvArgs(), "libero"),
            (Ur5EnvArgs(), "ur5"),
        ],
    )
    def test_names_each_env(self, env, expected):
        assert env_kind(env) == expected

    def test_unknown_env_raises(self):
        with pytest.raises(TypeError):
            env_kind(object())


class TestArgsPostInit:
    def test_no_guide_turns_off_all_three_stages(self):
        args = Args(guide=False)
        assert not args.retrieval.enable
        assert not args.demo_analysis.enable
        assert not args.guidance.enable

    def test_ur5_rejects_gt_perception_with_guidance(self):
        """The UR5e publishes neither depth nor GT masks."""
        with pytest.raises(ValueError, match="REAL_PERCEPTION"):
            Args(env=Ur5EnvArgs(), policy=LerobotPolicyArgs(model_path="/c"))

    def test_ur5_defaults_to_native_camera_resolution(self):
        args = Args(
            env=Ur5EnvArgs(),
            policy=LerobotPolicyArgs(model_path="/c"),
            guide=False,
        )
        assert (args.camera.height, args.camera.width) == (480, 640)
        assert args.viz.robot_description == "ur5e_mj_description"
        assert args.viz.gripper_model == "robotiq_2f140"
        assert args.policy.robot_type == "ur5_follower"

    def test_rtc_horizon_must_cover_replan_steps(self):
        with pytest.raises(ValueError, match="rtc-execution-horizon"):
            Args(
                policy=LerobotPolicyArgs(
                    model_path="/c",
                    rtc=True,
                    replan_steps=25,
                    rtc_execution_horizon=10,
                )
            )

    def test_output_paths_are_derived_from_run_and_exp_name(self):
        args = Args(guide=False)
        assert args.output.summary_out_path is not None
        assert args.output.run in args.output.summary_out_path

    def test_dataclasses_asdict_round_trips(self):
        """Recorded verbatim into every run's metadata."""
        assert isinstance(dataclasses.asdict(Args(guide=False)), dict)


class TestGuidanceActionSpace:
    """What the LeRobot server is told about the arm's action space.

    The UR5e's actions are absolute joint targets; served with the
    simulators' Euler integration, guided chunks started centimetres from
    the tool. ``auto`` must resolve to the kinematics for env:ur5 and to
    the integration for everything else, and the launcher must be told.
    """

    def test_auto_follows_the_env(self):
        policy = LerobotPolicyArgs(model_path="/ckpt")
        assert backends.guidance_action_space_for(_args(policy, Ur5EnvArgs())) == (
            "ur5e-joints"
        )
        assert backends.guidance_action_space_for(_args(policy, LiberoEnvArgs())) == (
            "cartesian-delta"
        )

    def test_an_explicit_choice_wins(self):
        policy = LerobotPolicyArgs(
            model_path="/ckpt", guidance_action_space="cartesian-delta"
        )
        assert backends.guidance_action_space_for(_args(policy, Ur5EnvArgs())) == (
            "cartesian-delta"
        )

    def test_unknown_values_are_refused_at_parse_time(self):
        with pytest.raises(ValueError, match="guidance-action-space"):
            _args(LerobotPolicyArgs(model_path="/ckpt", guidance_action_space="euler"))

    def test_the_gripper_follows_the_env(self):
        policy = LerobotPolicyArgs(model_path="/ckpt")
        assert backends.guidance_gripper_geometry_for(_args(policy, Ur5EnvArgs())) == (
            "robotiq_2f140"
        )
        assert (
            backends.guidance_gripper_geometry_for(_args(policy, LiberoEnvArgs()))
            is None
        )

    @staticmethod
    def _launch(monkeypatch, args, tmp_path) -> dict:
        seen: dict = {}

        def _capture(**kwargs):
            seen.update(kwargs)
            return object()

        monkeypatch.setattr(backends, "start_lerobot_server", _capture)
        backends._start_policy_backend(args, tmp_path)
        return seen

    def test_the_ur5_launch_carries_kinematics_tcp_and_gripper(
        self, monkeypatch, tmp_path
    ):
        tcp = [0.00052, 0.00109, 0.22347, 0.0, 0.0, 0.0]
        args = _args(LerobotPolicyArgs(model_path="/ckpt"), Ur5EnvArgs(tcp_offset=tcp))
        seen = self._launch(monkeypatch, args, tmp_path)
        assert seen["guidance_action_space"] == "ur5e-joints"
        assert list(seen["tcp_offset"]) == tcp
        assert seen["gripper_geometry"] == "robotiq_2f140"

    def test_a_simulator_launch_keeps_the_integration(self, monkeypatch, tmp_path):
        args = _args(LerobotPolicyArgs(model_path="/ckpt"), LiberoEnvArgs())
        seen = self._launch(monkeypatch, args, tmp_path)
        assert seen["guidance_action_space"] == "cartesian-delta"
        assert seen["tcp_offset"] is None
        assert seen["gripper_geometry"] is None
