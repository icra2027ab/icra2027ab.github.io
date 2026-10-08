"""Hardware-free stand-ins for the session's collaborators.

The session's job is orchestration — start a rollout on a thread, stop it
on request, record the verdict, home the arm — and all of that is
testable without a simulator, a policy, or a display. These fakes
implement only the surface :class:`ExperimentSession` actually touches.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from typing import Optional

import numpy as np
import pytest

from policy_abstraction.envs.spec import (
    ActionLayout,
    CameraSpec,
    EnvSpec,
    GripperSpec,
    PolicyIOKeys,
    RobotSpec,
    StateField,
    StateLayout,
)

IMAGE_KEY = "agentview_image"
WRIST_IMAGE_KEY = "wrist_image"


def make_env_spec(image_transform: str = "flip_vh", wrist: bool = False) -> EnvSpec:
    cameras = [
        CameraSpec(
            name="agentview",
            role="world",
            image_key=IMAGE_KEY,
            image_transform=image_transform,
            is_primary=True,
        )
    ]
    if wrist:
        cameras.append(
            CameraSpec(name="wrist", role="wrist", image_key=WRIST_IMAGE_KEY)
        )
    return EnvSpec(
        cameras=tuple(cameras),
        robot=RobotSpec(
            eef_pos_key="robot0_eef_pos",
            eef_quat_key="robot0_eef_quat",
            gripper_qpos_key="robot0_gripper_qpos",
            joint_pos_key="robot0_joint_pos",
            state_layout=StateLayout(
                fields=(StateField(op="pos", obs_key="robot0_eef_pos"),)
            ),
            gripper=GripperSpec(geometry_id="panda"),
            action_layout=ActionLayout(action_dim=7, eef_pos_slice=(0, 3)),
        ),
        policy_input_keys=PolicyIOKeys(),
    )


class FakeEnv:
    """Records the lifecycle calls the session makes on the env."""

    def __init__(self, image_transform: str = "flip_vh", wrist: bool = False) -> None:
        self.env_spec = make_env_spec(image_transform, wrist=wrist)
        self.language_instruction = "Put the cable in the box."
        self.homed = 0
        self.resets = 0
        self.steps = 0
        self.made = []
        self.camera_params: dict = {
            "agentview": {
                "intrinsic": np.array(
                    [[300.0, 0, 128.0], [0, 300.0, 128.0], [0, 0, 1.0]]
                ),
                "extrinsic": np.eye(4),
            }
        }

    def make_env(self, **kwargs):
        self.made.append(kwargs)
        return {"language_instruction": self.language_instruction}

    def get_task_init_states(self, task_id=0):
        return [{"i": i} for i in range(4)]

    def get_camera_params(self, **kwargs):
        return self.camera_params

    def get_obs(self):
        obs = {IMAGE_KEY: np.full((32, 32, 3), 90, dtype=np.uint8)}
        if self.env_spec.wrist_camera is not None:
            obs[WRIST_IMAGE_KEY] = np.full((24, 32, 3), 40, dtype=np.uint8)
        return obs

    def step(self, action):
        """Counted so a test can assert the arm was never commanded."""
        self.steps += 1
        return self.get_obs(), 0.0, False, {}

    def home(self):
        self.homed += 1

    def reset(self):
        self.resets += 1

    def disconnect(self):
        pass


class FakeEnvWithoutHome(FakeEnv):
    """A simulator: no ``home`` command, so the session must fall back."""

    def home(self):
        raise RuntimeError("home is not a command this env implements")


class FakeRunner:
    """Runs for *steps* ticks, polling ``stop_signal`` like the real one."""

    def __init__(
        self,
        *,
        stop_signal=None,
        steps: int = 20,
        tick: float = 0.005,
        warmup: float = 0.0,
        done: bool = False,
        raises: Optional[Exception] = None,
        episode_data_path=None,
        **_kwargs,
    ) -> None:
        self._stop_signal = stop_signal
        self._steps = steps
        self._tick = tick
        # Stands in for the first inference against a torch.compile'd
        # checkpoint, which autotunes for minutes before the arm moves at all.
        self._warmup = warmup
        self._done = done
        self._raises = raises
        self.episode_data_path = episode_data_path
        self.steps_executed = 0
        self.stopped_early = False
        self.saved_success: Optional[bool] = None
        self.saved_video = False

    def _record_timestep(self, **kwargs):
        return None

    def run(self):
        if self._raises is not None:
            raise self._raises
        for _ in range(self._steps):
            if self._stop_signal is not None and self._stop_signal():
                self.stopped_early = True
                return False, self.steps_executed
            if self.steps_executed == 0 and self._warmup:
                time.sleep(self._warmup)
            time.sleep(self._tick)
            self.steps_executed += 1
            # The real runner reports every step through this hook, which is
            # what the session taps for the live step counter.
            self._record_timestep(t=self.steps_executed, images=None)
        return self._done, self.steps_executed

    def save_episode(self, success=None):
        self.saved_success = success

    def save_video(self, *_args, **_kwargs):
        self.saved_video = True


@dataclasses.dataclass
class _Camera:
    height: int = 32
    width: int = 32
    resize_size: int = 32


@dataclasses.dataclass
class _Output:
    save_episodes: bool = False


@dataclasses.dataclass
class _FakeArgs:
    camera: _Camera = dataclasses.field(default_factory=_Camera)
    output: _Output = dataclasses.field(default_factory=_Output)
    seed: int = 0


class FakeBackends:
    """Stands in for :class:`EvalBackends`."""

    def __init__(
        self,
        env: Optional[FakeEnv] = None,
        *,
        runner_kwargs: Optional[dict] = None,
    ) -> None:
        self.env = env or FakeEnv()
        self.args = _FakeArgs()
        self.task_languages = ["Put the cable in the box."]
        self.task_max_steps = [500]
        self.runner_kwargs = runner_kwargs or {}
        self.built: list[dict] = []
        self.shutdown_calls = 0
        self.last_runner: Optional[FakeRunner] = None
        self.building = threading.Event()
        self.warm_ups: list[str] = []
        self.warm_up_seconds: Optional[float] = 0.01
        self.warming = threading.Event()
        self.release_warm_up: Optional[threading.Event] = None

    def warm_up_policy(self, task_description: str):
        """Records the prompt it was warmed with; never touches the env."""
        self.warm_ups.append(task_description)
        self.warming.set()
        if self.release_warm_up is not None:
            self.release_warm_up.wait(timeout=5)
        return self.warm_up_seconds

    def build_runner(self, **kwargs):
        self.built.append(kwargs)
        runner = FakeRunner(**{**kwargs, **self.runner_kwargs})
        self.last_runner = runner
        self.building.set()
        return runner

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.fixture
def fake_env():
    return FakeEnv()


@pytest.fixture
def backends(fake_env):
    return FakeBackends(fake_env)
