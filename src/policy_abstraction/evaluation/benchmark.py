"""Benchmark evaluation across multiple tasks and trials."""

import json
import logging
from pathlib import Path
from typing import Optional

import tqdm


class Benchmark:
    """Run full benchmark evaluation across all tasks and trials.

    Parameters
    ----------
    env_client:
        A WebsocketEnvClient instance connected to the env server.
    policy_client:
        Any object implementing .infer(obs_dict) -> dict.
    task_suite_name:
        Name of the LIBERO task suite to evaluate on.
    num_trials_per_task:
        Number of episodes to attempt per task.
    max_steps:
        Maximum steps per episode (looked up from suite if not provided).
    episode_data_dir:
        Directory to save recorded episode data (.vla files).
    video_dir:
        Directory to save evaluation videos.
    summary_path:
        Path to write the JSON summary file.
    """

    def __init__(
        self,
        env_client,
        policy_client,
        task_suite_name: str,
        num_trials_per_task: int = 3,
        max_steps: Optional[int] = None,
        episode_data_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        summary_path: Optional[str] = None,
        episode_runner_kwargs: Optional[dict] = None,
    ):
        self.env_client = env_client
        self.policy_client = policy_client
        self.task_suite_name = task_suite_name
        self.num_trials_per_task = num_trials_per_task
        self.max_steps = max_steps
        self.episode_data_dir = Path(episode_data_dir) if episode_data_dir else None
        self.video_dir = Path(video_dir) if video_dir else None
        self.summary_path = Path(summary_path) if summary_path else None
        self.episode_runner_kwargs = episode_runner_kwargs or {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def run(self) -> dict:
        """Run the full benchmark. Returns summary statistics dict."""
        import libero.libero.benchmark as bm

        from policy_abstraction.envs.libero.utils import get_max_steps_for_suite
        from policy_abstraction.evaluation.episode_runner import EpisodeRunner

        suite = bm.get_benchmark_dict()[self.task_suite_name]()
        num_tasks = suite.get_num_tasks()
        max_steps = self.max_steps or get_max_steps_for_suite(self.task_suite_name)

        all_results = {}
        total_success = 0
        total_trials = 0

        for task_id in tqdm.trange(num_tasks, desc="Tasks"):
            task = suite.get_task(task_id)
            task_name = task.name
            init_states = self.env_client.get_task_init_states(task_id)
            env_meta = self.env_client.make_env(task_id)
            task_description = env_meta.get("language_instruction", task.language)

            task_successes = 0
            for trial in range(self.num_trials_per_task):
                init_state = init_states[trial % len(init_states)]
                ep_data_path = None
                if self.episode_data_dir:
                    ep_data_path = str(
                        self.episode_data_dir
                        / f"task_{task_id:02d}_trial_{trial:02d}.vla"
                    )
                runner = EpisodeRunner(
                    env=self.env_client,
                    client=self.policy_client,
                    task_description=task_description,
                    initial_state=init_state,
                    max_steps=max_steps,
                    episode_data_path=ep_data_path,
                    **self.episode_runner_kwargs,
                )
                success, num_inferences = runner.run()
                if success:
                    task_successes += 1
                total_success += int(success)
                total_trials += 1

                if self.video_dir:
                    self.video_dir.mkdir(parents=True, exist_ok=True)
                    video_path = (
                        self.video_dir / f"task_{task_id:02d}_trial_{trial:02d}.mp4"
                    )
                    runner.save_video(
                        str(video_path), episode_idx=trial, success=success
                    )

            all_results[task_name] = {
                "success_rate": task_successes / self.num_trials_per_task,
                "successes": task_successes,
                "trials": self.num_trials_per_task,
            }
            self._logger.info(
                f"Task {task_id} ({task_name}): "
                f"{task_successes}/{self.num_trials_per_task} = "
                f"{task_successes / self.num_trials_per_task:.1%}"
            )

        summary = {
            "task_suite": self.task_suite_name,
            "overall_success_rate": total_success / max(total_trials, 1),
            "total_successes": total_success,
            "total_trials": total_trials,
            "per_task": all_results,
        }

        if self.summary_path:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            self._logger.info(f"Summary written to {self.summary_path}")

        return summary
