"""The demo launcher's argv, and the demo entry point's guards, without hardware.

The launcher tests run the real shell script against a stub interpreter, so
they assert the argv an operator's flags actually produce without starting a
server or reaching the arm. tyro accepts each flag only on its own side of
the subcommands, so *where* a flag lands in argv decides whether it is
honoured or rejected as unrecognised.

The entry-point tests import ``scripts/demo_ur5.py`` directly and check the
guard that runs before the GUI, the servers and the checkpoint load.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from policy_abstraction import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import demo_ur5  # noqa: E402

LAUNCHER = PROJECT_ROOT / "scripts" / "run_ur5_demo.sh"


def _stub(tmp_path):
    stub = tmp_path / "stub_python.sh"
    stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
    stub.chmod(0o755)
    return stub


def _demos(tmp_path):
    demos = tmp_path / "demos"
    demos.mkdir(exist_ok=True)
    return demos


def _run(tmp_path, *flags, env=None, simulate=True):
    """Run the launcher with a stub interpreter; return the finished process."""
    environ = {
        **os.environ,
        "PYTHON": str(_stub(tmp_path)),
        # Set so the launcher does not shell out to resolve it.
        "ROBOT_IP": "1.2.3.4",
        "RETRIEVAL_DIR": str(_demos(tmp_path)),
        **(env or {}),
    }
    # --simulate keeps the dashboard probe from dialling that address.
    return subprocess.run(
        ["bash", str(LAUNCHER), *(["--simulate"] if simulate else []), *flags],
        capture_output=True,
        text=True,
        env=environ,
        cwd=PROJECT_ROOT,
        timeout=60,
    )


def _argv(tmp_path, *flags, env=None, simulate=True) -> list[str]:
    """The argv the launcher hands the interpreter."""
    proc = _run(tmp_path, *flags, env=env, simulate=simulate)
    assert proc.returncode == 0, proc.stderr
    # First line is the echoed command; the rest is the stub's argv.
    return proc.stdout.splitlines()[1:]


class TestLauncherArgv:
    def test_guided_is_the_default(self, tmp_path):
        argv = _argv(tmp_path, "--no-viz")
        assert argv[0].endswith("scripts/demo_ur5.py")
        assert argv[argv.index("--perception.mode") + 1] == "REAL_PERCEPTION"
        at = argv.index("--retrieval.dataset-dirs")
        assert argv[at + 1] == str(_demos(tmp_path))
        assert at < argv.index("env:ur5"), "a root-level flag precedes the subcommands"
        assert "--no-guide" not in argv
        assert "--policy.rtc" in argv
        assert "--env.no-home-on-reset" in argv
        assert "--env.simulate-hardware" in argv

    def test_the_default_run_serves_the_viewer_from_the_demo_process(self, tmp_path):
        """No viewer binary, no tunnel: the 3-D scene is a panel of the window."""
        argv = _argv(tmp_path)
        assert "--viz.serve" in argv
        assert argv[argv.index("--viz.serve-grpc-port") + 1] == "9876"
        assert argv[argv.index("--viz.serve-web-port") + 1] == "9090"
        assert argv[argv.index("--viz.layout") + 1] == "scene"
        assert "--viz.connect" not in argv and "--viz.require-viewer" not in argv

    def test_external_viewer_falls_back_to_serving_when_nothing_listens(self, tmp_path):
        """Nothing on VIEWER_ADDR must never cost a run."""
        proc = _run(tmp_path, "--external-viewer", env={"VIEWER_ADDR": "127.0.0.1:1"})
        assert proc.returncode == 0, proc.stderr
        argv = proc.stdout.splitlines()[1:]
        assert "--viz.serve" in argv
        assert "in-window viewer" in proc.stderr

    def test_no_viz_disables_the_viewer(self, tmp_path):
        argv = _argv(tmp_path, "--no-viz")
        assert "--viz.no-enable" in argv
        assert "--viz.enable" not in argv
        assert "--viz.layout" not in argv

    def test_spawn_viewer_uses_the_scene_layout(self, tmp_path):
        argv = _argv(tmp_path, "--spawn-viewer")
        assert argv[argv.index("--viz.connect") + 1] == "None"
        assert "--viz.no-serve" in argv and "--viz.serve" not in argv
        assert argv[argv.index("--viz.layout") + 1] == "scene"
        assert argv[argv.index("--viz.robot-description") + 1] == "ur5e_mj_description"
        assert "--viz.require-viewer" not in argv
        # Root-level, so before the subcommands.
        assert argv.index("--viz.layout") < argv.index("env:ur5")

    def test_full_viewer_installs_the_analysis_layout(self, tmp_path):
        argv = _argv(tmp_path, "--full-viewer", "--spawn-viewer")
        assert argv[argv.index("--viz.layout") + 1] == "full"

    def test_unguided_drops_the_pipeline(self, tmp_path):
        argv = _argv(tmp_path, "--unguided", "--no-viz", env={"RETRIEVAL_DIR": ""})
        assert "--no-guide" in argv
        assert "--perception.mode" not in argv
        assert "--retrieval.dataset-dirs" not in argv

    def test_start_guided_sets_the_switch(self, tmp_path):
        argv = _argv(tmp_path, "--start-guided", "--no-viz")
        assert "--demo.guided" in argv
        assert argv.index("--demo.guided") < argv.index("env:ur5")

    def test_the_switch_is_off_by_default(self, tmp_path):
        argv = _argv(tmp_path, "--no-viz")
        assert "--demo.guided" not in argv

    def test_name_and_max_steps(self, tmp_path):
        argv = _argv(tmp_path, "--no-viz", "--name", "day2", "--max-steps", "900")
        assert argv[argv.index("--demo.name") + 1] == "day2"
        assert argv[argv.index("--demo.max-steps") + 1] == "900"

    def test_extra_flags_are_routed_by_prefix(self, tmp_path):
        argv = _argv(
            tmp_path,
            "--no-viz",
            "--",
            "--demo.refresh-hz",
            "10",
            "--env.max-steps",
            "900",
            "--policy.gpus",
            "1",
        )
        env_at, policy_at = argv.index("env:ur5"), argv.index("policy:lerobot")

        root_at = argv.index("--demo.refresh-hz")
        assert root_at < env_at
        assert argv[root_at + 1] == "10"

        assert env_at < argv.index("--env.max-steps") < policy_at
        assert policy_at < argv.index("--policy.gpus")

    def test_an_override_wins_over_the_scripts_own_value(self, tmp_path):
        """Last occurrence wins in argparse, so an override has to be
        appended after the launcher's own copy of the same flag."""
        argv = _argv(tmp_path, "--no-viz", "--", "--policy.gpus", "1")
        gpus = [i for i, tok in enumerate(argv) if tok == "--policy.gpus"]
        assert len(gpus) == 2
        assert argv[gpus[-1] + 1] == "1"

    def test_guided_without_a_retrieval_dir_is_refused_before_the_servers(
        self, tmp_path
    ):
        proc = _run(tmp_path, "--guided", "--no-viz", env={"RETRIEVAL_DIR": ""})
        assert proc.returncode == 2
        assert "RETRIEVAL_DIR" in proc.stderr
        # The interpreter was never reached: nothing echoed past the banner.
        assert "demo_ur5.py" not in proc.stdout

    def test_a_retrieval_dir_that_does_not_exist_is_refused(self, tmp_path):
        proc = _run(tmp_path, "--no-viz", env={"RETRIEVAL_DIR": str(tmp_path / "nope")})
        assert proc.returncode == 2
        assert "not a directory" in proc.stderr

    def test_several_retrieval_dirs_are_all_passed(self, tmp_path):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        argv = _argv(tmp_path, "--no-viz", env={"RETRIEVAL_DIR": f"{a} {b}"})
        at = argv.index("--retrieval.dataset-dirs")
        assert argv[at + 1 : at + 3] == [str(a), str(b)]

    def test_fake_policy_skips_the_checkpoint_and_never_moves_the_arm(self, tmp_path):
        # Without --simulate, so the fake forces --dry-run. A loopback robot
        # address makes the dashboard probe fail instantly instead of waiting
        # on an unroutable one.
        argv = _argv(
            tmp_path,
            "--fake-policy",
            "circular",
            "--no-viz",
            env={"ROBOT_IP": "127.0.0.1"},
            simulate=False,
        )
        assert "--guidance.debug" in argv
        assert argv[argv.index("--guidance.fake-policy") + 1] == "circular"
        assert "--env.dry-run" in argv
        assert "--env.simulate-hardware" not in argv
        assert "--policy.model-path" not in argv
        assert "--policy.rtc" not in argv
        # The subcommand still picks the backend, and replan-steps sizes the chunk.
        policy_at = argv.index("policy:lerobot")
        assert argv[policy_at + 1] == "--policy.replan-steps"

    def test_fake_policy_under_simulate_stays_simulated(self, tmp_path):
        argv = _argv(tmp_path, "--fake-policy", "--no-viz")
        assert argv[argv.index("--guidance.fake-policy") + 1] == "noop"
        assert "--env.simulate-hardware" in argv
        assert "--env.dry-run" not in argv

    def test_no_compile_and_fresh_analysis_by_default(self, tmp_path):
        """A demo trades steady-state speed for a start-up that does not
        compile, and re-stages the scene between rollouts, so it must not
        replay a cached live-object match or a cached pair analysis."""
        argv = _argv(tmp_path, "--no-viz")
        assert argv[argv.index("--policy.compile-model") + 1] == "off"
        assert argv.index("--policy.compile-model") > argv.index("policy:lerobot")
        for flag in (
            "--demo-analysis.no-cache-live-objects",
            "--demo-analysis.no-cache-pairs",
        ):
            assert flag in argv
            assert argv.index(flag) < argv.index("env:ur5")

    def test_compile_and_caches_can_be_turned_back_on(self, tmp_path):
        argv = _argv(
            tmp_path,
            "--no-viz",
            env={
                "COMPILE_MODEL": "auto",
                "CACHE_LIVE_OBJECTS": "1",
                "CACHE_PAIRS": "1",
            },
        )
        assert argv[argv.index("--policy.compile-model") + 1] == "auto"
        assert "--demo-analysis.no-cache-live-objects" not in argv
        assert "--demo-analysis.no-cache-pairs" not in argv

    def test_training_positions_are_passed_when_the_file_exists(self, tmp_path):
        positions = tmp_path / "grasps.json"
        positions.write_text('{"grasps": [[0, 0, 0]]}')
        argv = _argv(tmp_path, "--no-viz", env={"TRAINING_POSITIONS": str(positions)})
        assert argv[argv.index("--demo.training-positions") + 1] == str(positions)
        assert argv.index("--demo.training-positions") < argv.index("env:ur5")

    def test_missing_training_positions_only_warn(self, tmp_path):
        proc = _run(
            tmp_path,
            "--no-viz",
            env={"TRAINING_POSITIONS": str(tmp_path / "none.json")},
        )
        assert proc.returncode == 0, proc.stderr
        assert "--demo.training-positions" not in proc.stdout
        assert "heat map" in proc.stderr

    def test_recording_is_off_by_default_and_record_turns_it_on(self, tmp_path):
        argv = _argv(tmp_path, "--no-viz")
        assert "--output.no-save-episodes" in argv
        assert argv.index("--output.no-save-episodes") < argv.index("env:ur5")
        argv = _argv(tmp_path, "--no-viz", env={"RECORD": "1"})
        assert "--output.no-save-episodes" not in argv

    def test_the_env_tail_carries_the_cell(self, tmp_path):
        argv = _argv(tmp_path, "--no-viz")
        env_at = argv.index("env:ur5")
        assert argv[argv.index("--env.robot-ip") + 1] == "1.2.3.4"
        assert argv[argv.index("--env.tasks") + 1] == "Put all cables in the bag."
        cams = argv.index("--env.cameras")
        assert argv[cams + 1].startswith("exterior_image_1_left=rs:262722073432:")
        assert argv[cams + 2].startswith("wrist_image_left=rs:115222071166:")
        assert env_at < cams < argv.index("policy:lerobot")

    def test_the_guided_argv_parses_into_a_guided_run(self, tmp_path):
        """The command, parsed by the real arg classes: guidance on, the
        retrieval source set, the demo flags understood."""
        import tyro

        argv = _argv(
            tmp_path,
            "--start-guided",
            "--no-viz",
            "--name",
            "d",
            "--",
            "--demo.refresh-hz",
            "10",
        )
        # argv[0] is the script path the stub was handed.
        args = tyro.cli(demo_ur5.Args, args=argv[1:])
        assert args.guide and args.guidance.enable
        assert args.retrieval.enable and args.demo_analysis.enable
        assert args.retrieval.dataset_dirs == [str(_demos(tmp_path))]
        assert args.perception.mode.name == "REAL_PERCEPTION"
        assert args.demo.guided
        assert args.demo.name == "d"
        assert args.demo.refresh_hz == 10
        assert not args.viz.enable
        assert args.env.simulate_hardware
        assert args.policy.rtc

    def test_the_spawn_viewer_argv_parses(self, tmp_path):
        import tyro

        argv = _argv(tmp_path, "--spawn-viewer", "--full-viewer")
        args = tyro.cli(demo_ur5.Args, args=argv[1:])
        assert args.viz.enable and args.viz.connect is None
        assert args.viz.layout == "full"

    def test_help_prints_the_hotkeys(self, tmp_path):
        proc = subprocess.run(
            ["bash", str(LAUNCHER), "--help"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=60,
        )
        assert proc.returncode == 0
        assert "Space" in proc.stdout
        assert "RETRIEVAL_DIR" in proc.stdout

    def test_an_unknown_option_is_refused(self, tmp_path):
        proc = _run(tmp_path, "--bogus")
        assert proc.returncode == 2
        assert "unknown option" in proc.stderr


class TestEntryPointGuards:
    def test_an_unguided_args_builds(self):
        args = demo_ur5.Args(guide=False)
        assert not args.guidance.enable
        assert args.demo.overlays == "auto"

    @staticmethod
    def _nothing_launches(monkeypatch):
        def _boom(*_a, **_k):
            raise AssertionError("the guard must fire before anything launches")

        monkeypatch.setattr(demo_ur5, "_preflight_gui", _boom)
        monkeypatch.setattr(demo_ur5, "start_backends", _boom)
        monkeypatch.setitem(
            sys.modules,
            "policy_abstraction.demo.gui",
            type(sys)("policy_abstraction.demo.gui"),
        )
        sys.modules["policy_abstraction.demo.gui"].run_demo_gui = _boom

    def test_a_policy_server_told_to_refuse_guidance_is_refused(self, monkeypatch):
        from policy_abstraction.evaluation.abstraction import PerceptionMode
        from policy_abstraction.evaluation.config import (
            LerobotPolicyArgs,
            PerceptionArgs,
            Ur5EnvArgs,
        )

        self._nothing_launches(monkeypatch)
        args = demo_ur5.Args(
            env=Ur5EnvArgs(),
            policy=LerobotPolicyArgs(guidance="off", model_path="x"),
            perception=PerceptionArgs(mode=PerceptionMode.REAL_PERCEPTION),
        )
        with pytest.raises(SystemExit, match="guidance"):
            demo_ur5.main(args)

    def test_an_unguided_launch_only_warns(self, monkeypatch):
        """An unguided demo still streams; the guard says the switch is inert
        and lets the run go on to the GUI preflight."""
        self._nothing_launches(monkeypatch)
        messages: list[str] = []
        handle = demo_ur5.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            demo_ur5.check_guidance_configuration(demo_ur5.Args(guide=False))
        finally:
            demo_ur5.logger.remove(handle)
        assert any("launched unguided" in m for m in messages)
        assert any("REAL_PERCEPTION" in m for m in messages)

    def test_a_guided_lerobot_launch_passes_the_guard(self):
        from policy_abstraction.evaluation.abstraction import PerceptionMode
        from policy_abstraction.evaluation.config import (
            LerobotPolicyArgs,
            PerceptionArgs,
            Ur5EnvArgs,
        )

        args = demo_ur5.Args(
            env=Ur5EnvArgs(),
            policy=LerobotPolicyArgs(model_path="x"),
            perception=PerceptionArgs(mode=PerceptionMode.REAL_PERCEPTION),
        )
        demo_ur5.check_guidance_configuration(args)  # must not raise

    def test_the_task_is_the_envs_prompt(self):
        from policy_abstraction.evaluation.config import Ur5EnvArgs

        args = demo_ur5.Args(guide=False, env=Ur5EnvArgs(tasks=["Do the thing."]))
        assert demo_ur5._task_for(args) == "Do the thing."
        assert demo_ur5._task_for(demo_ur5.Args(guide=False)) is None
