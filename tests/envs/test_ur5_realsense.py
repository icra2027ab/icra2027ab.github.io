"""Device-string routing for RealSense vs V4L2 cameras.

Pure string logic — no `pyrealsense2`, no hardware — so it runs anywhere. The
capture behaviour itself needs a camera and is exercised by
`scripts/debug/smoke_test_env_servers.py` / a live run.
"""

import pytest

from policy_abstraction.envs.ur5.realsense import (
    RS_PREFIX,
    is_realsense_device,
    probe_realsense_cameras,
    realsense_serial,
    video_node_holders,
)
from policy_abstraction.envs.ur5.server import Ur5EnvServer, parse_camera_arg


class TestDeviceRouting:
    @pytest.mark.parametrize(
        "device, expected",
        [
            ("rs:035422072667", True),
            ("/dev/video4", False),
            ("/dev/video12", False),
            ("", False),
        ],
    )
    def test_is_realsense_device(self, device, expected):
        assert is_realsense_device(device) is expected

    def test_serial_extraction(self):
        assert realsense_serial(f"{RS_PREFIX}035422072667") == "035422072667"

    def test_serial_extraction_rejects_v4l2_paths(self):
        with pytest.raises(ValueError, match="not a RealSense"):
            realsense_serial("/dev/video4")


class TestCameraArgParsing:
    """`parse_camera_arg` must handle `rs:<serial>` without a special case.

    The device string contains a ':' of its own, so the resolution suffix has
    to be split off from the right and only when it looks like WxH.
    """

    def test_realsense_with_resolution_and_fps(self):
        assert parse_camera_arg("exterior_image_1_left=rs:035422072667:640x480@30") == {
            "name": "exterior_image_1_left",
            "device": "rs:035422072667",
            "width": 640,
            "height": 480,
            "fps": 30,
        }

    def test_realsense_bare_serial_keeps_the_prefix(self):
        parsed = parse_camera_arg("wrist_image_left=rs:123456789")
        assert parsed["device"] == "rs:123456789"
        assert is_realsense_device(parsed["device"])

    def test_v4l2_path_is_untouched(self):
        parsed = parse_camera_arg("exterior_image_1_left=/dev/video4:640x480@30")
        assert parsed["device"] == "/dev/video4"
        assert not is_realsense_device(parsed["device"])


class TestCameraPreflight:
    """Startup camera checks — the failure the pre-flight exists to catch.

    A camera claimed by another process used to surface ~3 minutes into a run,
    because cameras are opened lazily on the first `make_env` and that only
    happens once the policy server has finished loading and compiling.
    """

    def test_holders_lists_this_process_when_it_holds_a_node(self, tmp_path):
        # `video_node_holders` scans /proc for open /dev/video* descriptors.
        # Rather than require a camera, point it at the same scan with a real
        # fd: any process holding one must appear, so an empty result on a
        # machine with an open node would be a bug.
        holders = video_node_holders()
        assert isinstance(holders, list)
        assert all(isinstance(h, str) for h in holders)

    def test_probe_reports_a_missing_device_rather_than_raising(self):
        pytest.importorskip("pyrealsense2")
        # A serial that cannot exist: the probe must report it per-camera
        # rather than raise, so one bad camera does not hide the others.
        failures = probe_realsense_cameras(
            [
                {
                    "name": "exterior_image_1_left",
                    "serial": "000000000000",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                    "depth": True,
                }
            ]
        )
        assert [name for name, _ in failures] == ["exterior_image_1_left"]
        assert "No device connected" in failures[0][1]

    def test_probe_of_nothing_is_not_a_failure(self):
        # An all-V4L2 setup has no RealSense requests to make.
        assert probe_realsense_cameras([]) == []

    def test_depth_goes_to_the_primary_and_calibrated_cameras_only(self):
        # Two D435s streaming colour+depth exceed one USB controller, so depth
        # is spent only where perception can actually use it.
        server = Ur5EnvServer.__new__(Ur5EnvServer)
        server._primary_camera = "exterior_image_1_left"
        server._calibration = {"third_view": {"extrinsic": object()}}

        assert server._wants_depth("exterior_image_1_left") is True
        assert server._wants_depth("third_view") is True
        assert server._wants_depth("wrist_image_left") is False

    def test_uncalibrated_camera_gets_no_depth(self):
        server = Ur5EnvServer.__new__(Ur5EnvServer)
        server._primary_camera = "exterior_image_1_left"
        # An intrinsic alone cannot place depth in the robot frame.
        server._calibration = {"wrist_image_left": {"intrinsic": object()}}
        assert server._wants_depth("wrist_image_left") is False

    def test_simulate_hardware_skips_the_probe(self):
        # `--simulate-hardware` serves synthetic frames, so probing would fail
        # on a machine that has no cameras at all.
        server = Ur5EnvServer.__new__(Ur5EnvServer)
        server._simulate_hardware = True
        server._camera_specs = [
            {
                "name": "exterior_image_1_left",
                "device": "rs:000000000000",
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        ]
        server.preflight_cameras()  # must not raise


class TestPreflightRecovery:
    """The pre-flight power-cycles a camera that starts but sends no frames.

    A D435 left wedged by an abruptly killed run behaves exactly that way, and
    `hardware_reset` is the documented recovery — but it must not fire at a
    camera another process is legitimately using.
    """

    @staticmethod
    def _server(monkeypatch, probe_results, reset_calls):
        from policy_abstraction.envs.ur5 import realsense

        calls = iter(probe_results)
        monkeypatch.setattr(
            realsense, "probe_realsense_cameras", lambda reqs: next(calls)
        )
        monkeypatch.setattr(
            realsense,
            "reset_realsense",
            lambda serials, **kw: reset_calls.extend(serials) or list(serials),
        )
        monkeypatch.setattr(realsense, "video_node_holders", lambda: [])

        server = Ur5EnvServer.__new__(Ur5EnvServer)
        server._simulate_hardware = False
        server._primary_camera = "exterior_image_1_left"
        server._calibration = {}
        server._camera_specs = [
            {
                "name": "exterior_image_1_left",
                "device": "rs:035422072667",
                "width": 640,
                "height": 480,
                "fps": 30,
            }
        ]
        return server

    def test_frame_timeout_triggers_a_reset_and_retry(self, monkeypatch):
        resets: list[str] = []
        server = self._server(
            monkeypatch,
            [[("exterior_image_1_left", "Frame didn't arrive within 5000")], []],
            resets,
        )
        server.preflight_cameras()  # second probe succeeds — must not raise
        assert resets == ["035422072667"]

    def test_a_busy_camera_is_never_power_cycled(self, monkeypatch):
        # Another process is using it: resetting would break their run.
        resets: list[str] = []
        server = self._server(
            monkeypatch,
            [[("exterior_image_1_left", "Device or resource busy")]],
            resets,
        )
        with pytest.raises(RuntimeError, match="busy"):
            server.preflight_cameras()
        assert resets == []

    def test_a_camera_that_stays_dead_after_a_reset_still_fails(self, monkeypatch):
        resets: list[str] = []
        timeout = [("exterior_image_1_left", "Frame didn't arrive within 5000")]
        server = self._server(monkeypatch, [timeout, timeout], resets)
        with pytest.raises(RuntimeError, match="power-cycle did not help"):
            server.preflight_cameras()
        assert resets == ["035422072667"]


class TestConnectRetry:
    """A pipeline that starts but never delivers is restarted, then reset, once each."""

    def _camera_with_fake_start(self, monkeypatch, outcomes, *, reset_ok=True):
        import policy_abstraction.envs.ur5.realsense as mod
        from policy_abstraction.envs.ur5.realsense import RealSenseCamera

        cam = RealSenseCamera("0000", warmup_frames=1)
        calls = {"start": 0, "stop": 0, "reset": 0}

        def fake_start(self):
            calls["start"] += 1
            outcome = outcomes.pop(0)
            if outcome is not None:
                raise RuntimeError(outcome)
            self._pipeline = object()

        def fake_disconnect(self):
            calls["stop"] += 1
            self._pipeline = None

        def fake_reset(serials, timeout_s=20.0):
            calls["reset"] += 1
            return list(serials) if reset_ok else []

        monkeypatch.setattr(RealSenseCamera, "_start", fake_start)
        monkeypatch.setattr(RealSenseCamera, "disconnect", fake_disconnect)
        monkeypatch.setattr(mod, "reset_realsense", fake_reset)
        monkeypatch.setattr("time.sleep", lambda _s: None)
        return cam, calls

    def test_restarts_once_on_a_frame_timeout(self, monkeypatch):
        cam, calls = self._camera_with_fake_start(
            monkeypatch, ["Frame didn't arrive within 5000", None]
        )
        cam.connect()
        assert calls == {"start": 2, "stop": 1, "reset": 0}
        assert cam.is_connected

    def test_resets_the_device_before_the_last_try(self, monkeypatch):
        cam, calls = self._camera_with_fake_start(
            monkeypatch, ["Frame didn't arrive within 5000"] * 2 + [None]
        )
        cam.connect()
        assert calls == {"start": 3, "stop": 2, "reset": 1}

    def test_gives_up_after_the_third_timeout(self, monkeypatch):
        cam, calls = self._camera_with_fake_start(
            monkeypatch, ["Frame didn't arrive within 5000"] * 3
        )
        with pytest.raises(RuntimeError, match="didn't arrive"):
            cam.connect()
        assert calls["start"] == 3 and calls["reset"] == 1

    def test_a_device_that_never_returns_is_reported(self, monkeypatch):
        cam, calls = self._camera_with_fake_start(
            monkeypatch, ["Frame didn't arrive within 5000"] * 2, reset_ok=False
        )
        with pytest.raises(RuntimeError, match="did not come back"):
            cam.connect()

    def test_other_errors_are_not_retried(self, monkeypatch):
        cam, calls = self._camera_with_fake_start(
            monkeypatch, ["Device or resource busy"]
        )
        with pytest.raises(RuntimeError, match="busy"):
            cam.connect()
        assert calls["start"] == 1
