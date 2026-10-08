"""The collection script's guards, run without hardware.

The GUI preflight is the one worth pinning down. Qt does not raise
when its platform plugin is missing — it calls ``qFatal`` and the
process aborts — so the check has to run out-of-process, and it has
to run *before* the servers and the ~3 minute checkpoint load rather
than after them.

The launcher tests below run the real shell script against a stub
interpreter, so they assert the argv an operator's flags actually
produce without starting a server or reaching the arm.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from policy_abstraction import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))

import collect_experiment as ce  # noqa: E402

LAUNCHERS = {
    "cable_position": "run_cable_position_experiment.sh",
    "object_disambiguation": "run_object_disambiguation_experiment.sh",
}
LAUNCHER = PROJECT_ROOT / "scripts" / "experiments" / LAUNCHERS["cable_position"]


class TestGuiPreflight:
    def test_passes_when_qt_can_start(self, monkeypatch):
        monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
        ce._preflight_gui()  # must not raise

    def test_a_missing_platform_plugin_names_the_fix(self, monkeypatch):
        """The real fail: Qt 6.5+ needs libxcb-cursor0."""
        qt_error = (
            "qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is "
            "needed to load the Qt xcb platform plugin.\n"
            'qt.qpa.plugin: Could not load the Qt platform plugin "xcb"'
        )
        monkeypatch.setattr(
            ce.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0] if a else [], 134, "", qt_error
            ),
        )
        with pytest.raises(SystemExit) as excinfo:
            ce._preflight_gui()
        message = str(excinfo.value)
        assert "libxcb-cursor0" in message
        # ... including the no-root route, since not every box
        # running this has sudo.
        assert "vendor_qt_xcb_cursor.sh" in message
        # Says nothing was loaded and the robot is untouched...
        assert "robot was not touched" in message
        # ...and offers the way to keep working.
        assert "--experiment.headless" in message
        # ...and keeps Qt's own diagnostic.
        assert "xcb" in message

    def test_no_display_is_diagnosed_separately(self, monkeypatch):
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.setattr(
            ce.subprocess,
            "run",
            lambda *a, **k: subprocess.CompletedProcess(
                a[0] if a else [], 1, "", "could not connect to display"
            ),
        )
        with pytest.raises(SystemExit, match="ssh -X"):
            ce._preflight_gui()

    def test_a_probe_that_will_not_run_does_not_block_the_session(self, monkeypatch):
        """The probe is a convenience; it must never be the thing
        that stops an operator from collecting data."""

        def _boom(*_a, **_k):
            raise OSError("no fork for you")

        monkeypatch.setattr(ce.subprocess, "run", _boom)
        ce._preflight_gui()  # warns, returns

    def test_a_hanging_probe_does_not_hang_the_session(self, monkeypatch):
        def _timeout(*_a, **_k):
            raise subprocess.TimeoutExpired(cmd="probe", timeout=1)

        monkeypatch.setattr(ce.subprocess, "run", _timeout)
        ce._preflight_gui()  # warns, returns


class TestArgGuards:
    def test_report_only_without_a_name_says_what_to_pass(self):
        args = ce.Args(guide=False)
        args.experiment.report_only = True
        with pytest.raises(SystemExit, match="experiment-name"):
            ce.main(args)

    def test_report_only_on_a_missing_experiment_names_the_path(self):
        args = ce.Args(guide=False)
        args.experiment.report_only = True
        args.experiment.experiment_name = "does_not_exist"
        with pytest.raises(SystemExit, match="No experiment at"):
            ce.main(args)

    def test_a_missing_config_says_where_the_examples_are(self):
        args = ce.Args(guide=False)
        with pytest.raises(SystemExit, match="experiment-config"):
            ce.main(args)


class TestEditedSettingsSurviveARelaunch:
    """An existing experiment resumes under its own stored settings.

    Settings edited in the GUI are written to the experiment, not the YAML, and
    the YAML may since have been edited for the other arm — or not be passed
    at all. The rule is that the name decides: what exists on disk under it is
    what runs, and the YAML only ever creates. These pin that down, because
    getting it wrong loses an afternoon's reshaping of the grid.
    """

    def _config(self, rows=2, cols=2, **kwargs):
        from policy_abstraction.experiments.config import ExperimentConfig
        from policy_abstraction.experiments.factors import GridPositionFactor
        from policy_abstraction.experiments.grid import TableGrid

        return ExperimentConfig(
            name="cable_in_box",
            task="Put the cable in the box.",
            factors=[
                GridPositionFactor(
                    "cable_position",
                    TableGrid(
                        origin=(0.3, -0.2, 0.0),
                        u_axis=(0, 1, 0),
                        v_axis=(1, 0, 0),
                        rows=rows,
                        cols=cols,
                        name="cable",
                    ),
                )
            ],
            **kwargs,
        )

    def _store(self, tmp_path, source, stored=None, snapshot=True):
        """An experiment collected under *stored*, launched from *source*."""
        from policy_abstraction.experiments.store import ExperimentStore

        store = ExperimentStore.open(tmp_path / "exp", stored or source)
        if snapshot:
            meta = store.load_metadata()
            meta[ce.SOURCE_SNAPSHOT] = source.to_dict()
            store.save_metadata(meta)
        return store

    def test_a_brand_new_experiment_uses_the_yaml(self, tmp_path):
        from policy_abstraction.experiments.store import ExperimentStore

        source = self._config()
        config, from_experiment = ce._effective_config(
            ExperimentStore(tmp_path / "nothing_here"), source
        )
        assert config is source
        assert not from_experiment

    def test_an_untouched_yaml_lets_the_gui_edit_stand(self, tmp_path):
        source = self._config()
        store = self._store(tmp_path, source, stored=self._config(rows=4, cols=3))
        config, from_experiment = ce._effective_config(store, source)
        assert from_experiment
        assert (config.factors[0].grid.rows, config.factors[0].grid.cols) == (4, 3)
        assert len(config.build_plan()) == 12

    def test_the_stored_settings_win_over_an_edited_yaml(self, tmp_path):
        """The name names what is on disk; the file only creates."""
        store = self._store(
            tmp_path, self._config(), stored=self._config(rows=4, cols=3)
        )
        edited_by_hand = self._config(rows=3, cols=3)
        messages: list[str] = []
        handle = ce.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            config, from_experiment = ce._effective_config(store, edited_by_hand)
        finally:
            ce.logger.remove(handle)
        assert from_experiment
        assert (config.factors[0].grid.rows, config.factors[0].grid.cols) == (4, 3)
        text = "\n".join(messages)
        assert "factors" in text and "overwrite" in text

    def test_an_experiment_from_before_the_snapshot_existed_still_resumes(
        self, tmp_path
    ):
        source = self._config()
        store = self._store(
            tmp_path, source, stored=self._config(rows=4, cols=3), snapshot=False
        )
        config, from_experiment = ce._effective_config(store, source)
        assert from_experiment
        assert config.factors[0].grid.rows == 4

    def test_no_yaml_at_all_resumes_an_existing_experiment(self, tmp_path):
        stored = self._config(rows=4, cols=3, timeout_s=45.0)
        store = self._store(tmp_path, stored, snapshot=False)
        config, from_experiment = ce._effective_config(store, None)
        assert from_experiment
        assert config.to_dict() == stored.to_dict()

    def test_no_yaml_and_no_experiment_is_nothing_to_run(self, tmp_path):
        from policy_abstraction.experiments.store import ExperimentStore

        config, from_experiment = ce._effective_config(
            ExperimentStore(tmp_path / "nothing_here"), None
        )
        assert config is None
        assert not from_experiment

    def test_identical_settings_resume_quietly(self, tmp_path):
        source = self._config()
        store = self._store(tmp_path, source)
        messages: list[str] = []
        handle = ce.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            config, from_experiment = ce._effective_config(store, source)
        finally:
            ce.logger.remove(handle)
        assert from_experiment
        assert config.to_dict() == source.to_dict()
        assert not messages

    def test_a_non_plan_edit_survives_too(self, tmp_path):
        """Timeouts and prompts are settings the operator edited on purpose."""
        source = self._config(timeout_s=90.0)
        store = self._store(tmp_path, source, stored=self._config(timeout_s=45.0))
        config, from_experiment = ce._effective_config(store, source)
        assert from_experiment
        assert config.timeout_s == 45.0

    def test_an_unreadable_stored_config_falls_back_to_the_yaml(self, tmp_path):
        source = self._config()
        store = self._store(tmp_path, source, stored=self._config(rows=4))
        store.config_path.write_text("[not a mapping]", encoding="utf-8")
        config, from_experiment = ce._effective_config(store, source)
        assert config is source
        assert not from_experiment

    def test_an_unreadable_stored_config_with_no_yaml_is_a_usage_error(self, tmp_path):
        store = self._store(tmp_path, self._config())
        store.config_path.write_text("[not a mapping]", encoding="utf-8")
        with pytest.raises(SystemExit, match="experiment-config"):
            ce._effective_config(store, None)

    def test_a_rename_leaves_a_forwarding_address(self, tmp_path, caplog):
        from policy_abstraction.experiments.store import ExperimentStore

        store = ExperimentStore.open(tmp_path / "exp", self._config())
        meta = store.load_metadata()
        meta["superseded_by"] = "cable_position_baseline_v2"
        store.save_metadata(meta)

        messages = []
        handle = ce.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            ce._announce_successor(store)
        finally:
            ce.logger.remove(handle)
        assert any("cable_position_baseline_v2" in m for m in messages)
        assert any("--name" in m for m in messages)

    def test_nothing_is_said_when_there_is_no_successor(self, tmp_path):
        from policy_abstraction.experiments.store import ExperimentStore

        store = ExperimentStore.open(tmp_path / "exp", self._config())
        messages = []
        handle = ce.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            ce._announce_successor(store)
        finally:
            ce.logger.remove(handle)
        assert not messages


@pytest.mark.parametrize("launcher", sorted(LAUNCHERS.values()))
class TestLauncherArgv:
    """What each experiment launcher hands to tyro.

    tyro accepts each flag only on its own side of the subcommands, so
    *where* a flag lands in argv decides whether it is honoured or
    rejected as unrecognised. Nothing else checks that, and the failure
    mode is a flag the operator typed being refused — or worse, one the
    parser accepts in the wrong scope.

    Run against every launcher, because they are supposed to be the same
    script: the cell, the pre-flight and this routing live once in
    ``_ur5_experiment.sh``, and a launcher that grew its own copy would
    show up here as two behaviours where there should be one.
    """

    @staticmethod
    def _run(tmp_path, launcher, *flags, env=None):
        """Run the launcher with a stub interpreter; return the finished process."""
        stub = tmp_path / "stub_python.sh"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
        stub.chmod(0o755)

        environ = {
            **os.environ,
            "PYTHON": str(stub),
            # Set so the launcher does not shell out to resolve it.
            "ROBOT_IP": "1.2.3.4",
            **(env or {}),
        }
        # --simulate keeps the dashboard probe from dialling that address.
        return subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "scripts" / "experiments" / launcher),
                "--simulate",
                *flags,
            ],
            capture_output=True,
            text=True,
            env=environ,
            cwd=PROJECT_ROOT,
            timeout=60,
        )

    @classmethod
    def _argv(cls, tmp_path, launcher, *flags, env=None) -> list[str]:
        """The argv the launcher hands the interpreter."""
        proc = cls._run(tmp_path, launcher, *flags, env=env)
        assert proc.returncode == 0, proc.stderr
        # First line is the echoed command; the rest is the stub's argv.
        return proc.stdout.splitlines()[1:]

    def test_overwrite_lands_before_the_subcommands(self, tmp_path, launcher):
        argv = self._argv(tmp_path, launcher, "--baseline", "--overwrite")
        assert argv.index("--experiment.overwrite") < argv.index("env:ur5")

    def test_extra_flags_are_routed_by_prefix(self, tmp_path, launcher):
        """The bug an operator hits: everything after ``--`` used to be
        appended past both subcommands, where a root-level flag is
        unrecognised."""
        argv = self._argv(
            tmp_path,
            launcher,
            "--baseline",
            "--",
            "--experiment.refresh-hz",
            "10",
            "--env.max-steps",
            "900",
            "--policy.gpus",
            "1",
        )
        env_at, policy_at = argv.index("env:ur5"), argv.index("policy:lerobot")

        # Root-level before both subcommands...
        root_at = argv.index("--experiment.refresh-hz")
        assert root_at < env_at
        # ...and its value rides with it.
        assert argv[root_at + 1] == "10"

        # ...each subcommand's own flags after that subcommand.
        assert env_at < argv.index("--env.max-steps") < policy_at
        assert policy_at < argv.index("--policy.gpus")

    def test_an_override_wins_over_the_scripts_own_value(self, tmp_path, launcher):
        """Last occurrence wins in argparse, so an override has to be
        appended after the launcher's own copy of the same flag."""
        argv = self._argv(tmp_path, launcher, "--baseline", "--", "--policy.gpus", "1")
        gpus = [i for i, tok in enumerate(argv) if tok == "--policy.gpus"]
        assert len(gpus) == 2
        assert argv[gpus[-1] + 1] == "1"

    def test_overwrite_is_refused_where_it_would_do_nothing(self, tmp_path, launcher):
        stub = tmp_path / "stub_python.sh"
        stub.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@"\n')
        stub.chmod(0o755)
        proc = subprocess.run(
            [
                "bash",
                str(PROJECT_ROOT / "scripts" / "experiments" / launcher),
                "--report",
                "--overwrite",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHON": str(stub), "ROBOT_IP": "1.2.3.4"},
            cwd=PROJECT_ROOT,
            timeout=60,
        )
        assert proc.returncode == 2
        assert "--overwrite applies to collection" in proc.stderr

    def test_the_calibration_default_declares_its_base_frame(self, tmp_path, launcher):
        """The server refuses a file that does not, after the checkpoint loads.

        Both launchers must therefore default to the checked-in calibration
        rather than to a raw hand-eye export. This is exactly the drift that
        broke the experiment launchers once: ``run_ur5_rollout.sh`` was moved
        onto the declared file and these were left pointing at Kalib's output,
        so every run died three minutes in with the servers already up.
        """
        argv = self._argv(tmp_path, launcher, "--baseline")
        assert "--env.calibration-path" in argv
        path = PROJECT_ROOT / argv[argv.index("--env.calibration-path") + 1]
        assert path.exists(), f"{path} is the default and must be in the repo"
        assert '"base_frame"' in path.read_text(), (
            "the default calibration must declare its base frame, or the env "
            "server refuses it"
        )
        # Nothing overrides the file's own declaration unless asked to.
        assert "--env.calibration-base-frame" not in argv

    def test_the_base_frame_can_be_overridden_for_a_file_that_lacks_one(
        self, tmp_path, launcher
    ):
        """The escape hatch for a calibration you cannot edit."""
        raw = tmp_path / "T_cam_base.json"
        raw.write_text('{"T_cam_base": [[1, 0, 0, 0]], "convention": "X_cam"}')
        argv = self._argv(
            tmp_path,
            launcher,
            "--baseline",
            env={"CALIBRATION": str(raw), "CALIBRATION_BASE_FRAME": "base_link"},
        )
        assert argv[argv.index("--env.calibration-base-frame") + 1] == "base_link"

    def test_a_calibration_with_no_base_frame_is_refused_before_the_servers(
        self, tmp_path, launcher
    ):
        """Up front, where the fix costs seconds instead of a checkpoint load."""
        raw = tmp_path / "T_cam_base.json"
        raw.write_text('{"T_cam_base": [[1, 0, 0, 0]], "convention": "X_cam"}')
        proc = self._run(
            tmp_path, launcher, "--baseline", env={"CALIBRATION": str(raw)}
        )
        assert proc.returncode != 0
        assert "does not declare a base frame" in proc.stderr
        assert "CALIBRATION_BASE_FRAME" in proc.stderr

    def test_each_launcher_names_its_own_experiment(self, tmp_path, launcher):
        """The one thing that legitimately differs between the two."""
        argv = self._argv(tmp_path, launcher, "--baseline")
        config = argv[argv.index("--experiment.experiment-config") + 1]
        name = argv[argv.index("--experiment.experiment-name") + 1]
        stem = launcher.removeprefix("run_").removesuffix("_experiment.sh")
        assert config.startswith("configs/experiments/") and config.endswith(".yaml")
        assert name == f"{stem}_baseline"


EXPERIMENT_CONFIGS = [
    "configs/experiments/cable_in_box_position.yaml",
    "configs/experiments/object_disambiguation.yaml",
]


class _StubSession:
    """Only what ``_build_session`` touches."""

    def __init__(self, *_args, **_kwargs) -> None:
        self.prepared = 0

    def prepare_env(self) -> None:
        self.prepared += 1

    def shutdown(self, timeout: float = 30.0) -> None:
        pass


class _StubGui:
    """Stands in for ``policy_abstraction.experiments.gui`` at its import."""

    def __init__(self, run_gui) -> None:
        self.run_gui = run_gui


class TestStartUpScreenIsShared:
    """The window comes up before the servers, for *every* experiment.

    There is one collector and one ``run_gui`` call in it, so this is true by
    construction — which is exactly the kind of thing that quietly stops being
    true. The property worth pinning is not "the code is shared" but the
    behaviour an operator sees: nothing expensive has started at the moment the
    window is asked for. Both shipped experiments run through it here, so an
    experiment that grew a start-up path of its own would fail this.
    """

    @staticmethod
    def _run(monkeypatch, tmp_path, config_path: str) -> dict:
        """Drive ``main`` with the servers and the window stubbed out."""
        seen: dict = {"backends_started": 0, "session_arg": None, "started_when": None}

        class _StubBackends:
            def shutdown(self):
                pass

        def fake_start_backends(*_args, **_kwargs):
            seen["backends_started"] += 1
            return _StubBackends()

        def fake_run_gui(session, _config, **_kwargs):
            # What run_gui is handed, and what had already happened by then.
            seen["session_arg"] = session
            seen["started_when"] = seen["backends_started"]
            return 0

        monkeypatch.setattr(ce, "_preflight_gui", lambda: None)
        monkeypatch.setattr(ce, "start_backends", fake_start_backends)
        monkeypatch.setattr(ce, "ExperimentSession", _StubSession)
        monkeypatch.setitem(
            sys.modules,
            "policy_abstraction.experiments.gui",
            _StubGui(fake_run_gui),
        )
        monkeypatch.chdir(PROJECT_ROOT)

        args = ce.Args(guide=False)
        args.experiment.experiment_config = config_path
        args.experiment.experiment_name = "startup_probe"
        args.output.run = str(tmp_path / "run")
        assert ce.main(args) == 0
        return seen

    @pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS)
    def test_the_window_is_asked_for_before_any_server_starts(
        self, monkeypatch, tmp_path, config_path
    ):
        seen = self._run(monkeypatch, tmp_path, config_path)
        assert seen["started_when"] == 0, (
            "a server was started before the window was opened; the operator "
            "would be back to watching a still terminal"
        )

    @pytest.mark.parametrize("config_path", EXPERIMENT_CONFIGS)
    def test_run_gui_is_handed_a_factory_not_a_session(
        self, monkeypatch, tmp_path, config_path
    ):
        seen = self._run(monkeypatch, tmp_path, config_path)
        factory = seen["session_arg"]
        assert callable(factory), "run_gui got a built session; the window would wait"
        # Calling it is what starts the servers and homes the arm — behind the
        # window, which is the whole point.
        session = factory()
        assert seen["backends_started"] == 1
        assert session.prepared == 1


@pytest.mark.parametrize("launcher", sorted(LAUNCHERS.values()))
class TestGuidedLauncher:
    """What ``--guided`` has to hand collect_experiment.py for guidance to
    actually steer: the real perception stack, a retrieval source, and the
    guidance chain left switched on. A guided arm missing any of these
    collects a baseline and files it under the wrong label.
    """

    _run = staticmethod(TestLauncherArgv._run)

    @classmethod
    def _argv(cls, tmp_path, launcher, *flags, env=None) -> list[str]:
        proc = cls._run(tmp_path, launcher, *flags, env=env)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.splitlines()[1:]

    def test_guided_without_a_retrieval_dir_is_refused_before_the_servers(
        self, tmp_path, launcher
    ):
        proc = self._run(tmp_path, launcher, "--guided", env={"RETRIEVAL_DIR": ""})
        assert proc.returncode != 0
        assert "RETRIEVAL_DIR" in proc.stderr
        # The interpreter was never reached: nothing echoed past the banner.
        assert "collect_experiment.py" not in proc.stdout

    def test_a_retrieval_dir_that_does_not_exist_is_refused(self, tmp_path, launcher):
        proc = self._run(
            tmp_path,
            launcher,
            "--guided",
            env={"RETRIEVAL_DIR": str(tmp_path / "nope")},
        )
        assert proc.returncode != 0
        assert "not a directory" in proc.stderr

    def test_guided_passes_retrieval_and_real_perception(self, tmp_path, launcher):
        demos = tmp_path / "demos"
        demos.mkdir()
        argv = self._argv(
            tmp_path, launcher, "--guided", env={"RETRIEVAL_DIR": str(demos)}
        )
        env_at = argv.index("env:ur5")
        at = argv.index("--retrieval.dataset-dirs")
        assert argv[at + 1] == str(demos)
        assert at < env_at, "a root-level flag must precede the subcommands"
        assert argv[argv.index("--perception.mode") + 1] == "REAL_PERCEPTION"
        assert argv[argv.index("--experiment.policy-setting") + 1] == "guided"
        assert "--no-guide" not in argv
        assert "--policy.rtc" in argv

    def test_several_retrieval_dirs_are_all_passed(self, tmp_path, launcher):
        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        argv = self._argv(
            tmp_path, launcher, "--guided", env={"RETRIEVAL_DIR": f"{a} {b}"}
        )
        at = argv.index("--retrieval.dataset-dirs")
        assert argv[at + 1 : at + 3] == [str(a), str(b)]

    def test_the_baseline_does_not_need_one(self, tmp_path, launcher):
        argv = self._argv(tmp_path, launcher, "--baseline", env={"RETRIEVAL_DIR": ""})
        assert "--no-guide" in argv
        assert "--retrieval.dataset-dirs" not in argv

    def test_the_guided_argv_parses_into_a_guided_run(self, tmp_path, launcher):
        """The command, parsed by the real arg classes: guidance on, the
        retrieval source set, and the loss reading joint targets."""
        import tyro

        from policy_abstraction.evaluation import backends

        demos = tmp_path / "demos"
        demos.mkdir()
        argv = self._argv(
            tmp_path, launcher, "--guided", env={"RETRIEVAL_DIR": str(demos)}
        )
        # argv[0] is the script path the stub was handed.
        args = tyro.cli(ce.Args, args=argv[1:])
        assert args.guide and args.guidance.enable
        assert args.retrieval.enable and args.demo_analysis.enable
        assert args.retrieval.dataset_dirs == [str(demos)]
        assert args.perception.mode.name == "REAL_PERCEPTION"
        assert args.experiment.policy_setting == "guided"
        assert (
            args.policy.rtc
            and args.policy.rtc_execution_horizon >= args.policy.replan_steps
        )
        assert backends.guidance_action_space_for(args) == "ur5e-joints"
        assert backends.guidance_gripper_geometry_for(args) == "robotiq_2f140"
        assert args.env.tcp_offset is not None
        ce.check_policy_setting(args)  # consistent: must not raise

    def test_the_baseline_argv_parses_into_an_unguided_run(self, tmp_path, launcher):
        import tyro

        argv = self._argv(tmp_path, launcher, "--baseline")
        args = tyro.cli(ce.Args, args=argv[1:])
        assert not args.guidance.enable
        assert args.experiment.policy_setting == "baseline"
        ce.check_policy_setting(args)


class TestPolicySettingGuard:
    """A label the flags contradict is a usage error, before any server."""

    @staticmethod
    def _args(setting: str, tmp_path, **overrides):
        from policy_abstraction.evaluation.config import (
            LerobotPolicyArgs,
            PerceptionArgs,
            Ur5EnvArgs,
        )

        demos = tmp_path / "demos"
        demos.mkdir(exist_ok=True)
        kwargs = dict(
            policy=LerobotPolicyArgs(
                model_path="/ckpt", rtc=True, rtc_execution_horizon=25, replan_steps=25
            ),
            env=Ur5EnvArgs(),
            # env:ur5 itself refuses any other mode while guidance is on.
            perception=PerceptionArgs(mode=ce.PerceptionMode.REAL_PERCEPTION),
            experiment=ce.ExperimentArgs(policy_setting=setting),
        )
        kwargs.update(overrides)
        args = ce.Args(**kwargs)
        if setting == "guided":
            args.retrieval.dataset_dirs = [str(demos)]
        return args, None

    def test_a_consistent_guided_run_passes(self, tmp_path):
        args, _ = self._args("guided", tmp_path)
        ce.check_policy_setting(args)

    def test_guided_with_no_guide_is_refused(self, tmp_path):
        args, _ = self._args("guided", tmp_path, guide=False)
        with pytest.raises(SystemExit, match="--guide"):
            ce.check_policy_setting(args)

    def test_guided_with_retrieval_off_is_refused(self, tmp_path):
        args, _ = self._args("guided", tmp_path)
        args.retrieval.enable = False
        with pytest.raises(SystemExit, match="retrieval.enable"):
            ce.check_policy_setting(args)

    def test_guided_with_demo_analysis_off_is_refused(self, tmp_path):
        args, _ = self._args("guided", tmp_path)
        args.demo_analysis.enable = False
        with pytest.raises(SystemExit, match="demo-analysis.enable"):
            ce.check_policy_setting(args)

    def test_guided_on_the_arm_needs_an_existing_retrieval_dir(self, tmp_path):
        args, _ = self._args("guided", tmp_path)
        args.retrieval.dataset_dirs = None
        with pytest.raises(SystemExit, match="RETRIEVAL_DIR"):
            ce.check_policy_setting(args)
        args.retrieval.dataset_dirs = [str(tmp_path / "missing")]
        with pytest.raises(SystemExit, match="do not exist"):
            ce.check_policy_setting(args)

    def test_guided_on_the_arm_needs_real_perception(self, tmp_path):
        args, _ = self._args("guided", tmp_path)
        args.perception.mode = ce.PerceptionMode.GT_SEG_RECONSTRUCT
        with pytest.raises(SystemExit, match="REAL_PERCEPTION"):
            ce.check_policy_setting(args)

    def test_a_baseline_with_guidance_on_is_refused(self, tmp_path):
        args, _ = self._args("baseline", tmp_path)
        with pytest.raises(SystemExit, match="baseline, but guidance is enabled"):
            ce.check_policy_setting(args)

    def test_a_baseline_with_no_guide_passes(self, tmp_path):
        args, _ = self._args("baseline", tmp_path, guide=False)
        ce.check_policy_setting(args)

    def test_an_unguided_perception_baseline_passes(self, tmp_path):
        # --guidance.no-enable: perception and alignment run, the gradient
        # does not. An honest baseline on the same perception path.
        args, _ = self._args("baseline", tmp_path)
        args.guidance.enable = False
        ce.check_policy_setting(args)

    def test_other_labels_and_calibration_are_left_alone(self, tmp_path):
        args, _ = self._args("t", tmp_path, guide=False)
        ce.check_policy_setting(args)
        args, _ = self._args("guided", tmp_path, guide=False)
        args.experiment.calibrate = True
        ce.check_policy_setting(args)
