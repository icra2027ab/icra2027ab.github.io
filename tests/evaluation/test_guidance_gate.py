"""The runner's operator hooks: the guidance gate and the step observer.

Both exist for the live demo, where guidance is switched on and off under a
running policy. What matters is pinned here without a policy or a robot: a
closed gate sends the request bare and records that it did, an open gate sends
the ``__gp_*__`` params, and the observer sees every step with the flags that
say which it was.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from policy_abstraction.evaluation.episode_runner import EpisodeRunner
from policy_abstraction.evaluation.guidance import GP_SENTINEL_KEY
from tests.experiments.conftest import IMAGE_KEY, make_env_spec


class _Env:
    """Serves a constant scene; counts the steps it was asked for."""

    def __init__(self) -> None:
        self.env_spec = make_env_spec(image_transform="none")
        self.steps = 0

    def _obs(self) -> dict:
        return {
            IMAGE_KEY: np.zeros((32, 32, 3), dtype=np.uint8),
            "robot0_eef_pos": np.array([0.4, 0.0, 0.3], dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.02, -0.02], dtype=np.float32),
            "robot0_joint_pos": np.zeros(7, dtype=np.float32),
        }

    def reset(self):
        return self._obs()

    def set_init_state(self, state):
        return self._obs()

    def step(self, action):
        self.steps += 1
        return self._obs(), 0.0, False, {}

    def get_robot_base(self):
        return {"base_pos": [0, 0, 0], "base_ori": [0, 0, 0, 1]}

    def get_obs(self):
        """The real cell serves frames on demand; a paused loop streams them."""
        return self._obs()

    def get_camera_params(self, **kwargs):
        return {}


class _Policy:
    """Remembers whether each request carried guidance params."""

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self.guided_requests: list[bool] = []

    def infer(self, element: dict) -> dict:
        self.guided_requests.append(GP_SENTINEL_KEY in element)
        return {"actions": np.zeros((1, self.horizon, 7), dtype=np.float32)}


@dataclasses.dataclass
class _Camera:
    height: int = 32
    width: int = 32
    resize_size: int = 32


@dataclasses.dataclass
class _PolicyArgs:
    replan_steps: int = 5


@dataclasses.dataclass
class _EnvArgs:
    num_steps_wait: int = 0


@dataclasses.dataclass
class _Args:
    camera: _Camera = dataclasses.field(default_factory=_Camera)
    policy: _PolicyArgs = dataclasses.field(default_factory=_PolicyArgs)
    env: _EnvArgs = dataclasses.field(default_factory=_EnvArgs)
    seed: int = 0


def _runner(env, policy, **kwargs) -> EpisodeRunner:
    static_guidance = {GP_SENTINEL_KEY: np.zeros((1, 5), dtype=np.float32)}
    return EpisodeRunner(
        env,
        policy,
        "hold still",
        None,
        _Args(),
        max_steps=10,
        guidance_fn=static_guidance,
        **kwargs,
    )


def test_an_open_gate_sends_guidance_and_a_closed_one_withholds_it():
    env, policy = _Env(), _Policy(horizon=5)
    switch = {"on": True}
    seen: list[tuple[int, bool, bool]] = []

    def observe(*, t, replan, guided, **_):
        seen.append((t, replan, guided))
        # Flip the switch after the first chunk has been requested.
        if t == 0:
            switch["on"] = False

    runner = _runner(env, policy, guidance_gate=lambda: switch["on"], on_step=observe)
    runner.run()

    # Ten steps at replan_steps=5: two policy queries, the second unguided.
    assert policy.guided_requests == [True, False]
    replans = [(t, guided) for t, replan, guided in seen if replan]
    assert replans == [(0, True), (5, False)]
    # Between replans the observer keeps reporting the chunk's own mode.
    assert all(guided for t, replan, guided in seen if t < 5)
    assert not any(guided for t, replan, guided in seen if t >= 5)
    assert runner.guidance_applied is False


def test_no_gate_means_the_old_behaviour():
    env, policy = _Env(), _Policy(horizon=5)
    runner = _runner(env, policy)
    runner.run()
    assert policy.guided_requests == [True, True]
    assert runner.guidance_applied is True


def test_the_observer_cannot_end_the_episode():
    env, policy = _Env(), _Policy(horizon=5)

    def explode(**_):
        raise RuntimeError("display bug")

    runner = _runner(env, policy, on_step=explode)
    done, inferences = runner.run()
    assert inferences == 2
    assert env.steps == 10


def test_guidance_available_is_false_without_a_retrieved_demo():
    runner = _runner(_Env(), _Policy(horizon=5), use_distance_guidance=True)
    assert runner.guidance_available is False


@pytest.mark.parametrize("gate", [True, False])
def test_query_policy_respects_use_guidance(gate):
    env, policy = _Env(), _Policy(horizon=5)
    runner = _runner(env, policy)
    obs = env.reset()
    images = {"agentview": obs[IMAGE_KEY]}
    runner._query_policy(obs, images, use_guidance=gate)
    assert policy.guided_requests == [gate]
    assert runner.guidance_applied is gate


def test_a_pause_holds_between_steps_and_resumes_the_same_chunk():
    env, policy = _Env(), _Policy(horizon=5)
    pause = {"on": False}
    seen: list[tuple[int, bool]] = []

    def observe(*, t, paused=False, **_):
        seen.append((t, paused))
        # Pause once, after the third step; a background timer resumes.
        if t == 2 and not paused and not pause.get("done"):
            pause["on"] = True
            pause["done"] = True
            import threading

            threading.Timer(0.35, lambda: pause.__setitem__("on", False)).start()

    runner = _runner(env, policy, on_step=observe, pause_signal=lambda: pause["on"])
    runner.run()
    assert env.steps == 10, "the pause cost no steps"
    assert policy.guided_requests == [True, True], "the pause queried nothing"
    paused_frames = [t for t, p in seen if p]
    assert paused_frames and set(paused_frames) == {3}, "held before step 3, streaming"
    assert runner.paused is False


def test_a_stop_while_paused_ends_the_episode():
    env, policy = _Env(), _Policy(horizon=5)
    flags = {"pause": False, "stop": False}

    def observe(*, t, paused=False, **_):
        if t == 1 and not paused:
            flags["pause"] = True
            import threading

            threading.Timer(0.3, lambda: flags.__setitem__("stop", True)).start()

    runner = _runner(
        env,
        policy,
        on_step=observe,
        pause_signal=lambda: flags["pause"],
        stop_signal=lambda: flags["stop"],
    )
    runner.run()
    assert runner.stopped_early
    assert env.steps == 2
