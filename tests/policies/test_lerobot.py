"""Unit tests for the LeRobot policy backend.

Covers the two halves of the adapter that can be exercised without a GPU or a
lerobot install:

* client-side (``policy_abstraction.policies.lerobot``) — view extraction from
  a PA element, action-chunk reshaping/gripper remap, guidance refusal;
* server-side (``scripts/servers/serve_lerobot.py``) — camera-map resolution,
  state adaptation, and the msgpack ndarray codec both ends share.

The server module keeps every ``lerobot`` import inside ``LerobotRuntime``
precisely so this file can load it from the project venv.
"""

from __future__ import annotations

import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.policies.lerobot import (
    LerobotPolicyClient,
    extract_views,
)


def _load_server_module():
    """Import ``scripts/servers/serve_lerobot.py`` by path (not a package)."""
    path = PROJECT_ROOT / "scripts" / "servers" / "serve_lerobot.py"
    spec = importlib.util.spec_from_file_location("serve_lerobot_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


serve_lerobot = _load_server_module()


def _feature(kind: str):
    """Stand-in for lerobot's ``PolicyFeature`` (its ``type`` is a str Enum)."""
    return SimpleNamespace(type=kind, shape=(3, 224, 224))


def _element(**overrides) -> dict:
    element = {
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image": np.ones((8, 8, 3), dtype=np.uint8),
        "observation/state": np.arange(8, dtype=np.float32),
        "prompt": "pick up the mug",
        "__rng_seed__": np.array(7, dtype=np.int64),
    }
    element.update(overrides)
    return element


# ---------------------------------------------------------------------------
# Client: element → payload
# ---------------------------------------------------------------------------


class TestExtractViews:
    def test_returns_cameras_in_canonical_order(self):
        element = _element(
            **{"observation/secondary_image": np.zeros((8, 8, 3), dtype=np.uint8)}
        )
        # Insert the wrist view last to prove ordering comes from the canonical
        # list, not from the element's own key order.
        element["observation/wrist_image"] = element.pop("observation/wrist_image")
        assert list(extract_views(element)) == [
            "image",
            "wrist_image",
            "secondary_image",
        ]

    def test_skips_state_and_scalar_keys(self):
        views = extract_views(_element())
        assert set(views) == {"image", "wrist_image"}

    def test_unknown_views_sort_after_canonical_ones(self):
        element = _element(
            **{"observation/zed_image": np.zeros((8, 8, 3), dtype=np.uint8)}
        )
        element["observation/aux_image"] = np.zeros((8, 8, 3), dtype=np.uint8)
        assert list(extract_views(element))[2:] == ["aux_image", "zed_image"]

    def test_accepts_unprefixed_camera_keys(self):
        """The UR5e spec names policy image keys after the raw LeRobot
        cameras, with no ``observation/`` prefix."""
        element = {
            "exterior_image_1_left": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_image_left": np.ones((8, 8, 3), dtype=np.uint8),
            "observation/state": np.arange(7, dtype=np.float32),
            "prompt": "put all cables in the bag",
            "__rng_seed__": np.array(7, dtype=np.int64),
        }
        assert list(extract_views(element)) == [
            "exterior_image_1_left",
            "wrist_image_left",
        ]

    def test_rejects_non_uint8_images(self):
        element = _element(
            **{"observation/image": np.zeros((8, 8, 3), dtype=np.float32)}
        )
        with pytest.raises(ValueError, match="uint8"):
            extract_views(element)

    def test_raises_when_no_camera_present(self):
        with pytest.raises(KeyError, match="no camera images"):
            extract_views({"observation/state": np.zeros(8, dtype=np.float32)})


class TestClientTranslation:
    """Exercise the pure translation methods without touching ZMQ."""

    @staticmethod
    def _client(**kwargs) -> LerobotPolicyClient:
        # __init__ opens a socket and blocks on the server; the translation
        # helpers only need the fields it would have set.
        client = LerobotPolicyClient.__new__(LerobotPolicyClient)
        from policy_abstraction.policies.groot import resolve_gripper_remap

        client._gripper_remap = resolve_gripper_remap(
            kwargs.get("gripper_remap", "passthrough")
        )
        client._replan_steps = kwargs.get("replan_steps", 5)
        return client

    def test_build_payload_shape(self):
        payload = self._client()._build_payload(_element())
        assert set(payload) == {"images", "state", "task", "consumed_steps"}
        assert set(payload["images"]) == {"image", "wrist_image"}
        assert payload["state"].dtype == np.float32
        assert payload["task"] == "pick up the mug"
        assert payload["consumed_steps"] == 5

    def test_build_payload_requires_state_and_prompt(self):
        element = _element()
        del element["observation/state"]
        with pytest.raises(KeyError, match="observation/state"):
            self._client()._build_payload(element)

    def test_shape_action_chunk_adds_chunk_axis(self):
        actions = np.arange(3 * 7, dtype=np.float32).reshape(3, 7)
        out = self._client()._shape_action_chunk(actions)
        assert out.shape == (1, 3, 7)
        assert out.dtype == np.float32
        np.testing.assert_allclose(out[0], actions)

    def test_shape_action_chunk_applies_gripper_remap(self):
        actions = np.zeros((2, 7), dtype=np.float32)
        actions[:, 6] = [0.0, 1.0]  # RLDS: 0=close, 1=open
        out = self._client(gripper_remap="libero_robosuite")._shape_action_chunk(
            actions
        )
        # sign(1 - 2x): 0 → +1 (close), 1 → -1 (open)
        np.testing.assert_allclose(out[0, :, 6], [1.0, -1.0])

    def test_shape_action_chunk_does_not_mutate_input(self):
        actions = np.zeros((2, 7), dtype=np.float32)
        self._client(gripper_remap="libero_robosuite")._shape_action_chunk(actions)
        np.testing.assert_allclose(actions[:, 6], [0.0, 0.0])

    def test_shape_action_chunk_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\(T, action_dim\)"):
            self._client()._shape_action_chunk(np.zeros((1, 3, 7), dtype=np.float32))

    def test_build_payload_strips_guidance_keys(self):
        """``__gp_*__`` params travel in ``options``, not in the observation —
        leaving them in would make the server reject the frame."""
        from policy_steering.params import GP_SENTINEL_KEY, extract_guidance_params

        element = _element()
        element[GP_SENTINEL_KEY] = np.zeros((2, 4), dtype=np.float32)
        guidance_params, cleaned = extract_guidance_params(element)
        assert guidance_params is not None
        assert GP_SENTINEL_KEY in guidance_params
        payload = self._client()._build_payload(cleaned)
        assert not any(k.startswith("__gp_") for k in payload)


# ---------------------------------------------------------------------------
# Server: checkpoint introspection and observation mapping
# ---------------------------------------------------------------------------


class TestCameraFeatureKeys:
    def test_keeps_visual_features_in_order_and_drops_placeholders(self):
        features = {
            "observation.images.image": _feature("VISUAL"),
            "observation.state": _feature("STATE"),
            "observation.images.image2": _feature("VISUAL"),
            "observation.images.empty_camera_0": _feature("VISUAL"),
        }
        assert serve_lerobot.camera_feature_keys(features) == [
            "observation.images.image",
            "observation.images.image2",
        ]


class TestResolveCameraMap:
    UR5E = [
        "observation.images.exterior_image_1_left",
        "observation.images.wrist_image_left",
    ]
    LIBERO = ["observation.images.image", "observation.images.image2"]

    def test_auto_map_binds_wrist_by_name(self):
        mapping, unmapped, unused = serve_lerobot.resolve_camera_map(
            ["image", "wrist_image"], self.UR5E
        )
        assert mapping == {
            "image": "observation.images.exterior_image_1_left",
            "wrist_image": "observation.images.wrist_image_left",
        }
        assert not unmapped and not unused

    def test_wrist_view_does_not_steal_the_third_person_slot(self):
        """A wrist-first feature order must not push the wrist view onto the
        exterior camera; wrist binding happens before the generic pass."""
        mapping, _, _ = serve_lerobot.resolve_camera_map(
            ["image", "wrist_image"], list(reversed(self.UR5E))
        )
        assert mapping["wrist_image"] == "observation.images.wrist_image_left"
        assert mapping["image"] == "observation.images.exterior_image_1_left"

    def test_auto_map_falls_back_to_declaration_order(self):
        """LIBERO checkpoints name both cameras generically; the primary view
        must take the first-declared feature."""
        mapping, unmapped, unused = serve_lerobot.resolve_camera_map(
            ["image", "wrist_image"], self.LIBERO
        )
        assert mapping == {
            "image": "observation.images.image",
            "wrist_image": "observation.images.image2",
        }
        assert not unmapped and not unused

    def test_extra_view_is_reported_as_unmapped(self):
        mapping, unmapped, unused = serve_lerobot.resolve_camera_map(
            ["image", "wrist_image", "secondary_image"], self.UR5E
        )
        assert unmapped == ["secondary_image"]
        assert len(mapping) == 2
        assert not unused

    def test_spare_checkpoint_camera_is_reported_as_unused(self):
        mapping, unmapped, unused = serve_lerobot.resolve_camera_map(
            ["image"], self.UR5E
        )
        assert mapping == {"image": "observation.images.exterior_image_1_left"}
        assert unused == ["observation.images.wrist_image_left"]
        assert not unmapped

    def test_lone_third_person_view_avoids_a_wrist_camera(self):
        """Even when the wrist feature is declared first, a non-wrist view must
        not be fed the wrist camera while an exterior one is free."""
        mapping, _, unused = serve_lerobot.resolve_camera_map(
            ["image"], list(reversed(self.UR5E))
        )
        assert mapping == {"image": "observation.images.exterior_image_1_left"}
        assert unused == ["observation.images.wrist_image_left"]

    def test_explicit_map_is_used_verbatim(self):
        mapping, unmapped, unused = serve_lerobot.resolve_camera_map(
            ["image", "wrist_image"],
            self.UR5E,
            explicit=(
                "image=observation.images.wrist_image_left,"
                "wrist_image=observation.images.exterior_image_1_left"
            ),
        )
        assert mapping["image"] == "observation.images.wrist_image_left"
        assert mapping["wrist_image"] == "observation.images.exterior_image_1_left"
        assert not unmapped and not unused

    def test_explicit_map_rejects_unknown_feature(self):
        with pytest.raises(ValueError, match="not a camera feature"):
            serve_lerobot.resolve_camera_map(
                ["image"], self.UR5E, explicit="image=observation.images.nope"
            )

    def test_explicit_map_rejects_malformed_entry(self):
        with pytest.raises(ValueError, match="<view>=<feature_key>"):
            serve_lerobot.resolve_camera_map(["image"], self.UR5E, explicit="image")

    def test_explicit_map_rejects_two_views_on_one_camera(self):
        """The dangerous typo: both views aimed at the same feature would let
        one image overwrite the other in the frame dict and leave the other
        camera padded as empty — a silently wrong rollout, not a crash."""
        with pytest.raises(ValueError, match="Each camera feature"):
            serve_lerobot.resolve_camera_map(
                ["image", "wrist_image"],
                self.UR5E,
                explicit=(
                    "image=observation.images.wrist_image_left,"
                    "wrist_image=observation.images.wrist_image_left"
                ),
            )

    def test_explicit_map_rejects_repeated_view(self):
        with pytest.raises(ValueError, match="twice"):
            serve_lerobot.resolve_camera_map(
                ["image"],
                self.UR5E,
                explicit=(
                    "image=observation.images.exterior_image_1_left,"
                    "image=observation.images.wrist_image_left"
                ),
            )


class TestAdaptState:
    def test_identity_passes_matching_dim(self):
        state = np.arange(7, dtype=np.float32)
        np.testing.assert_allclose(
            serve_lerobot.adapt_state(state, 7, "identity"), state
        )

    def test_identity_rejects_mismatch(self):
        with pytest.raises(ValueError, match="State dim mismatch"):
            serve_lerobot.adapt_state(np.zeros(8, dtype=np.float32), 7, "identity")

    def test_pad_or_truncate_pads_with_zeros(self):
        out = serve_lerobot.adapt_state(
            np.ones(3, dtype=np.float32), 7, "pad_or_truncate"
        )
        np.testing.assert_allclose(out, [1, 1, 1, 0, 0, 0, 0])

    def test_pad_or_truncate_clips(self):
        out = serve_lerobot.adapt_state(
            np.arange(8, dtype=np.float32), 7, "pad_or_truncate"
        )
        np.testing.assert_allclose(out, np.arange(7))

    def test_unknown_adapter_raises(self):
        with pytest.raises(ValueError, match="Unknown state adapter"):
            serve_lerobot.adapt_state(np.zeros(3, dtype=np.float32), 7, "magic")


class TestRtcKwargs:
    """`LerobotRuntime._rtc_kwargs` decides what the previous chunk hands to
    the next one. It needs no lerobot import, so it is exercised directly on a
    bare instance with a stub RTC config.
    """

    CHUNK = 50
    ACTION_DIM = 7

    @staticmethod
    def _runtime(rtc: bool, execution_horizon: int = 10):
        torch = pytest.importorskip("torch")
        runtime = serve_lerobot.LerobotRuntime.__new__(serve_lerobot.LerobotRuntime)
        # ``rtc_config`` is also switched on to expose the guidance hook, so
        # the leftover is gated on ``rtc_requested`` — what the user asked for.
        runtime.rtc_config = SimpleNamespace(
            enabled=True, execution_horizon=execution_horizon
        )
        runtime.rtc_requested = rtc
        runtime._prev_chunk = None
        runtime._torch = torch
        return runtime

    def _chunk(self, runtime):
        return runtime._torch.arange(
            self.CHUNK * self.ACTION_DIM, dtype=runtime._torch.float32
        ).reshape(1, self.CHUNK, self.ACTION_DIM)

    def test_no_kwargs_when_rtc_not_requested(self):
        """Guidance alone enables the lerobot RTCConfig; that must not smuggle
        a previous chunk into the next one behind the user's back."""
        runtime = self._runtime(rtc=False)
        runtime._prev_chunk = self._chunk(runtime)
        assert runtime._rtc_kwargs(5) == {}

    def test_first_call_has_no_previous_chunk(self):
        kwargs = self._runtime(rtc=True)._rtc_kwargs(5)
        assert kwargs["prev_chunk_left_over"] is None
        assert kwargs["inference_delay"] == 0
        assert kwargs["execution_horizon"] == 10

    def test_leftover_is_the_tail_past_consumed_steps(self):
        runtime = self._runtime(rtc=True)
        prev = self._chunk(runtime)
        runtime._prev_chunk = prev
        leftover = runtime._rtc_kwargs(5)["prev_chunk_left_over"]
        assert leftover.shape == (1, self.CHUNK - 5, self.ACTION_DIM)
        # Must be the *unexecuted* tail: PA already ran the first 5 steps.
        assert runtime._torch.equal(leftover, prev[:, 5:])

    def test_fully_consumed_chunk_yields_no_leftover(self):
        """Nothing is left to blend into, so RTC must fall back to the
        unguided path rather than passing an empty tensor."""
        runtime = self._runtime(rtc=True)
        runtime._prev_chunk = self._chunk(runtime)
        assert runtime._rtc_kwargs(self.CHUNK)["prev_chunk_left_over"] is None

    def test_over_consumption_is_clamped_not_negative_indexed(self):
        runtime = self._runtime(rtc=True)
        runtime._prev_chunk = self._chunk(runtime)
        assert runtime._rtc_kwargs(self.CHUNK + 20)["prev_chunk_left_over"] is None

    def test_reset_clears_the_previous_chunk(self):
        runtime = self._runtime(rtc=True)
        runtime._prev_chunk = self._chunk(runtime)
        # reset() also pokes the processor pipelines; stub them out.
        runtime.preprocess = SimpleNamespace(reset=lambda: None)
        runtime.postprocess = SimpleNamespace(reset=lambda: None)
        runtime.reset()
        assert runtime._prev_chunk is None
        assert runtime._rtc_kwargs(5)["prev_chunk_left_over"] is None


class TestSplitGuidanceOptions:
    def test_absent_sentinel_leaves_options_untouched(self):
        opts = {"something": 1}
        guidance, cleaned = serve_lerobot.split_guidance_options(opts)
        assert guidance is None
        assert cleaned is opts

    def test_none_options_is_unguided(self):
        assert serve_lerobot.split_guidance_options(None) == (None, None)

    def test_splits_gp_keys_from_the_rest(self):
        from policy_steering.params import GP_SENTINEL_KEY

        opts = {
            GP_SENTINEL_KEY: np.zeros((2, 4), dtype=np.float32),
            "__gp_line_search__": 1,
            "unrelated": "keep",
        }
        guidance, cleaned = serve_lerobot.split_guidance_options(opts)
        assert set(guidance) == {GP_SENTINEL_KEY, "__gp_line_search__"}
        # Bool keys must come back as real bools; a numpy scalar here would
        # break the `if use_line_search` branch inside the kernel.
        assert guidance["__gp_line_search__"] is True
        assert cleaned == {"unrelated": "keep"}


class TestGuidanceKernel:
    """The kernel is deliberately lerobot-free so it can be built and run
    against a real gradient here, without a checkpoint or a GPU.
    """

    HORIZON = 8
    ACTION_DIM = 32  # lerobot denoises in the padded action space

    @staticmethod
    def _params(horizon: int, *, line_search: bool = False) -> dict:
        """A single centroid pair pulling the end-effector toward a waypoint."""
        return {
            "__gp_ee_pos_0__": np.zeros(3, dtype=np.float32),
            "__gp_ee_ori_0__": np.zeros(3, dtype=np.float32),
            "__gp_dt__": np.float32(0.02),
            "__gp_expected_dists__": np.zeros((1, horizon), dtype=np.float32),
            "__gp_expected_stds__": np.ones((1, horizon), dtype=np.float32),
            "__gp_dist_types__": np.zeros(1, dtype=np.int64),
            "__gp_waypoints__": np.array([[0.2, 0.0, 0.0]], dtype=np.float32),
            "__gp_has_axis__": np.zeros(1, dtype=np.int64),
            "__gp_axes__": np.zeros((1, 3), dtype=np.float32),
            "__gp_has_plane__": np.zeros(1, dtype=np.int64),
            "__gp_plane_normals__": np.zeros((1, 3), dtype=np.float32),
            "__gp_surface_counts__": np.zeros(1, dtype=np.int64),
            "__gp_surface_pts__": np.zeros((1, 1, 3), dtype=np.float32),
            "__gp_guidance_weights__": np.ones(1, dtype=np.float32),
            "__gp_source_type__": np.zeros(1, dtype=np.uint8),
            "__gp_guidance_factor__": np.float32(1.0),
            "__gp_line_search__": line_search,
            "__gp_normalize__": True,
        }

    def _kernel(self, **kwargs):
        pytest.importorskip("torch")
        return serve_lerobot.GuidanceKernel(
            self._params(self.HORIZON, **kwargs),
            device="cpu",
            dtype=__import__("torch").float32,
            action_horizon=self.HORIZON,
            num_steps=10,
        )

    def test_dt_matches_lerobots_negative_convention(self):
        """lerobot integrates with ``dt = -1/num_steps`` (t: 1 → 0), the same
        sign openpi uses — the shared kernel keys its step sign off this."""
        assert self._kernel().dt == pytest.approx(-0.1)

    def test_expected_dists_are_truncated_to_the_action_horizon(self):
        torch = pytest.importorskip("torch")
        params = self._params(self.HORIZON * 3)
        kernel = serve_lerobot.GuidanceKernel(
            params,
            device="cpu",
            dtype=torch.float32,
            action_horizon=self.HORIZON,
            num_steps=10,
        )
        assert kernel.params["__gp_expected_dists__"].shape == (1, self.HORIZON)
        # The caller's dict must not be mutated in place.
        assert params["__gp_expected_dists__"].shape == (1, self.HORIZON * 3)

    def test_apply_changes_the_velocity(self):
        torch = pytest.importorskip("torch")
        kernel = self._kernel()
        x_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        v_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        out = kernel.apply(x_t, v_t, step_idx=0)
        assert out.shape == v_t.shape
        assert not torch.allclose(out, v_t), "guidance produced no correction"

    def test_correction_pushes_the_end_effector_toward_the_waypoint(self):
        """The waypoint sits at +x, so the position channel of the corrected
        velocity must move the integrated trajectory that way."""
        torch = pytest.importorskip("torch")
        kernel = self._kernel()
        x_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        v_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        out = kernel.apply(x_t, v_t, step_idx=0)
        # x_{t+dt} = x_t + dt * v_t with dt < 0, so a positive resulting
        # x-velocity requires a negative correction in v_t.
        next_x = x_t + kernel.dt * out
        assert next_x[0, :, 0].sum().item() > 0.0

    def test_padded_action_dims_receive_no_correction(self):
        """Dims 7..31 are lerobot's zero padding; the loss never reads them, so
        writing into them would inject pure noise into the chunk."""
        torch = pytest.importorskip("torch")
        kernel = self._kernel()
        x_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        v_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        out = kernel.apply(x_t, v_t, step_idx=0)
        assert torch.allclose(out[..., 7:], torch.zeros_like(out[..., 7:]))

    def test_debug_guidance_zeroes_the_network_velocity(self):
        """``--guidance.debug`` isolates the guidance field by discarding the
        model's prediction, matching pi0grounded's ``debug_guidance`` branch."""
        torch = pytest.importorskip("torch")
        params = self._params(self.HORIZON)
        params["__gp_debug_guidance__"] = True
        kernel = serve_lerobot.GuidanceKernel(
            params,
            device="cpu",
            dtype=torch.float32,
            action_horizon=self.HORIZON,
            num_steps=10,
        )
        x_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        huge_v = torch.full((1, self.HORIZON, self.ACTION_DIM), 1e3)
        out = kernel.apply(x_t, huge_v, step_idx=0)
        # The 1e3 prediction must be gone, leaving only the guidance term.
        assert out.abs().max().item() < 1e3

    def test_guidance_factor_scales_the_correction(self):
        torch = pytest.importorskip("torch")
        weak = self._kernel()
        strong_params = self._params(self.HORIZON)
        strong_params["__gp_guidance_factor__"] = np.float32(4.0)
        strong = serve_lerobot.GuidanceKernel(
            strong_params,
            device="cpu",
            dtype=torch.float32,
            action_horizon=self.HORIZON,
            num_steps=10,
        )
        x_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        v_t = torch.zeros(1, self.HORIZON, self.ACTION_DIM)
        w = weak.apply(x_t, v_t, 0).abs().sum().item()
        s = strong.apply(x_t, v_t, 0).abs().sum().item()
        assert s == pytest.approx(4.0 * w, rel=1e-3)


class TestRtcCapablePolicyTypes:
    def test_list_matches_the_policies_that_read_the_rtc_kwargs(self):
        """Derived by grepping lerobot for ``prev_chunk_left_over``. Policies
        outside this set (notably ``pi0_fast``) declare ``**kwargs`` with a
        different TypedDict and drop the RTC arguments without erroring, so a
        stale list here would let ``--rtc`` report success while doing nothing.
        Re-run that grep when bumping the lerobot pin.
        """
        assert serve_lerobot.RTC_CAPABLE_POLICY_TYPES == frozenset(
            {"pi0", "pi05", "smolvla"}
        )
        assert "pi0_fast" not in serve_lerobot.RTC_CAPABLE_POLICY_TYPES


class TestEpisodeResetOptIn:
    """``EpisodeRunner`` resets only clients that advertise
    ``resets_per_episode``. Every policy client in the repo defines ``reset``,
    so duck-typing the hook would silently change behaviour for backends that
    never asked for it — including GR00T, whose ``reset`` sends an endpoint its
    server may not implement.
    """

    def test_lerobot_client_opts_in(self):
        assert LerobotPolicyClient.resets_per_episode is True

    @pytest.mark.parametrize(
        "module_path, class_name",
        [
            ("policy_abstraction.policies.groot", "GrootPolicyClient"),
            ("policy_abstraction.policies.cosmos", "CosmosPolicyClient"),
            ("policy_abstraction.policies.dummy", "DummyPolicy"),
            ("policy_abstraction.policies.dummy", "CircularPolicy"),
            ("policy_abstraction.policies.openpi", "PatientWebsocketClient"),
        ],
    )
    def test_existing_backends_stay_opted_out(self, module_path, class_name):
        import importlib

        cls = getattr(importlib.import_module(module_path), class_name)
        assert getattr(cls, "resets_per_episode", False) is False


class TestWireCodec:
    """The client and server each carry their own copy of the codec; a shape
    or dtype drift between them would corrupt every observation."""

    def test_round_trips_arrays_between_client_and_server_codecs(self):
        import msgpack

        from policy_abstraction.policies import lerobot as client_mod

        arrays: dict[str, np.ndarray] = {
            "img": np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
            "state": np.linspace(0, 1, 7, dtype=np.float32),
        }
        payload: dict = {**arrays, "task": "wipe the table"}
        packed = msgpack.packb(payload, default=client_mod._encode_custom)
        decoded = msgpack.unpackb(packed, object_hook=serve_lerobot._decode_custom)
        assert decoded["task"] == "wipe the table"
        for key, original in arrays.items():
            assert decoded[key].dtype == original.dtype
            assert decoded[key].shape == original.shape
            np.testing.assert_array_equal(decoded[key], original)

        # ...and back the other way.
        repacked = msgpack.packb(decoded, default=serve_lerobot._encode_custom)
        again = msgpack.unpackb(repacked, object_hook=client_mod._decode_custom)
        np.testing.assert_array_equal(again["img"], arrays["img"])


# ---------------------------------------------------------------------------
# Spec -> element -> client, for the specs that run on this backend
# ---------------------------------------------------------------------------


class TestEnvSpecsProduceUsableElements:
    """The element an ``EnvSpec`` produces must be one this client accepts.

    Nothing else ties the two ends together: ``policy_input_keys.images``
    names the element keys, and ``extract_views`` decides which of them are
    cameras by the ``observation/`` prefix. A spec that omits the prefix
    type-checks, round-trips through the wire format and passes every spec
    test -- then fails on the first inference of a real rollout with "found
    no camera images". So assert the contract across the specs rather than
    inside either half of it.
    """

    @staticmethod
    def _element(spec, obs, images):
        """Build the element exactly as the runner does mid-rollout."""
        from policy_abstraction.evaluation.episode_runner import EpisodeRunner

        stub = SimpleNamespace(
            env_spec=spec,
            task_description="put the cable in the box",
            _episode_rng=np.random.default_rng(0),
        )
        return EpisodeRunner._build_policy_input(stub, obs, images)

    def test_ur5_spec_views_are_the_raw_camera_keys(self):
        from policy_abstraction.envs.ur5.config import (
            DEFAULT_EXTERIOR_CAMERA,
            DEFAULT_WRIST_CAMERA,
            UR5E_HOME_JOINT_POSITIONS,
            ur5_env_spec,
        )

        spec = ur5_env_spec()
        obs = {
            "joint_pos": np.asarray(UR5E_HOME_JOINT_POSITIONS, dtype=np.float64),
            "gripper_qpos": np.array([0.75]),
        }
        images = {c.name: np.zeros((8, 8, 3), dtype=np.uint8) for c in spec.cameras}

        views = extract_views(self._element(spec, obs, images))

        # The server pairs these with the checkpoint's
        # ``observation.images.*`` features by name, so they stay raw.
        assert list(views) == [DEFAULT_EXTERIOR_CAMERA, DEFAULT_WRIST_CAMERA]

    def test_libero_spec_views_are_the_canonical_names(self):
        from policy_abstraction.envs.libero.config import libero_env_spec

        spec = libero_env_spec()
        obs = {
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }
        images = {c.name: np.zeros((8, 8, 3), dtype=np.uint8) for c in spec.cameras}

        views = extract_views(self._element(spec, obs, images))

        assert list(views) == ["image", "wrist_image"]


class TestStartLerobotServerFlags:
    """The launcher must tell the server how guidance should read the chunk."""

    @staticmethod
    def _launch(monkeypatch, tmp_path, **kwargs) -> list[str]:
        from policy_abstraction.policies import lerobot as mod

        repo = tmp_path / "sdk"
        (repo / ".venv" / "bin").mkdir(parents=True)
        (repo / ".venv" / "bin" / "python").write_text("")
        ckpt = tmp_path / "ckpt"
        ckpt.mkdir()
        seen: dict = {}

        def _popen(cmd, **_):
            seen["cmd"] = [str(c) for c in cmd]
            return SimpleNamespace(pid=4242)

        monkeypatch.setattr(mod.subprocess, "Popen", _popen)
        monkeypatch.setattr(mod, "assert_port_free", lambda *a, **k: None)
        mod.start_lerobot_server(
            port=5599, model_path=str(ckpt), repo_path=str(repo), **kwargs
        )
        return seen["cmd"]

    def test_the_default_keeps_the_simulators_integration(self, monkeypatch, tmp_path):
        cmd = self._launch(monkeypatch, tmp_path)
        assert cmd[cmd.index("--action-space") + 1] == "cartesian-delta"
        assert "--tcp-offset" not in cmd
        assert "--gripper-geometry" not in cmd

    def test_the_ur5_kinematics_carry_their_offset_and_gripper(
        self, monkeypatch, tmp_path
    ):
        tcp = [0.00052, 0.00109, 0.22347, 0.0, 0.0, 0.0]
        cmd = self._launch(
            monkeypatch,
            tmp_path,
            guidance_action_space="ur5e-joints",
            tcp_offset=tcp,
            gripper_geometry="robotiq_2f140",
        )
        assert cmd[cmd.index("--action-space") + 1] == "ur5e-joints"
        i = cmd.index("--tcp-offset")
        assert [float(v) for v in cmd[i + 1 : i + 7]] == tcp
        assert cmd[cmd.index("--gripper-geometry") + 1] == "robotiq_2f140"

    def test_a_malformed_tcp_offset_is_refused_before_launch(
        self, monkeypatch, tmp_path
    ):
        with pytest.raises(ValueError, match="tcp_offset"):
            self._launch(monkeypatch, tmp_path, tcp_offset=[0.0, 0.0, 0.2])
