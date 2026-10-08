"""Tiptop-style detect.gemini -> sam2(bbox) wiring, with stubbed clients.

No network / no Gemini. The box proposer is a ``StubBoxDetector`` and SAM2 is
a ``StubSam2`` that returns one filled-rectangle mask per box.
"""

from __future__ import annotations

import numpy as np
from _stubs import StubBoxDetector, StubSam2
from open_perception import OpenPerception, PerceptionConfig
from open_perception.stages.detect_gemini import (
    GeminiDetectStage,
)
from open_perception.stages.sam2 import Sam2SegmentStage

# --- GeminiBoxDetector parsing/conversion (no genai) ------------------------


def test_gemini_box_detector_converts_0_1000_to_pixel_xyxy() -> None:
    from open_perception.models.gemini import GeminiBoxDetector

    det = GeminiBoxDetector()
    # box_2d is [ymin, xmin, ymax, xmax] in 0-1000; image is 200x100 (HxW).
    dets = [{"box_2d": [100, 250, 500, 750], "label": "red mug"}]
    boxes = det._to_pixel_xyxy(dets, H=200, W=100)
    assert set(boxes) == {"red_mug"}  # label sanitized
    # xyxy pixel: [(250/1000)*100, (100/1000)*200, (750/1000)*100, (500/1000)*200]
    np.testing.assert_allclose(boxes["red_mug"][0], [25.0, 20.0, 75.0, 100.0])


def test_gemini_box_detector_stacks_duplicate_labels() -> None:
    from open_perception.models.gemini import GeminiBoxDetector

    det = GeminiBoxDetector()
    dets = [
        {"box_2d": [0, 0, 100, 100], "label": "block"},
        {"box_2d": [500, 500, 600, 600], "label": "block"},
    ]
    boxes = det._to_pixel_xyxy(dets, H=1000, W=1000)
    assert boxes["block"].shape == (2, 4)


def test_gemini_box_detector_parses_fenced_json() -> None:
    from open_perception.models.gemini import GeminiBoxDetector

    det = GeminiBoxDetector()
    text = '```json\n{"bboxes": [{"box_2d": [0,0,10,10], "label": "x"}]}\n```'
    assert det._parse_json(text) == [{"box_2d": [0, 0, 10, 10], "label": "x"}]
    assert det._parse_json("not json") == []
    assert det._parse_json(None) == []


# --- detect.gemini stage ----------------------------------------------------


def test_detect_gemini_open_vocab_sets_queries(default_context) -> None:
    boxes = {"red_box": np.array([[4, 4, 12, 12]], dtype=np.float32)}
    stage = GeminiDetectStage(
        detector_getter=lambda: StubBoxDetector(boxes),
        stage="open_vocab",
        task="grab it",
    )
    ctx = stage(default_context)
    assert ctx.box_prompts is not None and set(ctx.box_prompts) == {"red_box"}
    # open-vocab: proposed labels replace the queries.
    assert ctx.queries == ["red_box"]


def test_detect_gemini_grounded_forwards_queries(default_context) -> None:
    detector = StubBoxDetector({"mug": np.array([[4, 4, 12, 12]], dtype=np.float32)})
    stage = GeminiDetectStage(detector_getter=lambda: detector, stage="grounded")
    ctx = stage(default_context)
    # grounded: original queries preserved; the passed queries reach the client.
    assert ctx.queries == ["mug", "plate"]
    assert detector.calls == [(["mug", "plate"], None)]


def test_detect_gemini_no_client_is_noop(default_context) -> None:
    ctx = GeminiDetectStage(detector_getter=lambda: None)(default_context)
    assert ctx.box_prompts is None


def test_detect_gemini_grounded_rekeys_sanitized_labels(default_context) -> None:
    # Queries use spaces (the live client's canonical form); the detector
    # sanitizes labels to underscores. Grounded mode must re-key the prompts
    # back to the verbatim queries so segmenter lookups match ctx.queries.
    default_context.queries = ["alphabet soup", "cream cheese"]
    detector = StubBoxDetector(
        {"alphabet_soup": np.array([[4, 4, 12, 12]], dtype=np.float32)}
    )
    stage = GeminiDetectStage(detector_getter=lambda: detector, stage="grounded")
    ctx = stage(default_context)
    assert ctx.box_prompts is not None and set(ctx.box_prompts) == {"alphabet soup"}


# --- sam2 bbox mode consumes ctx.box_prompts --------------------------------


def _ctx_with_box_prompts(default_context, boxes):
    default_context.queries = list(boxes.keys())
    default_context.box_prompts = boxes
    return default_context


def test_sam2_bbox_consumes_box_prompts(default_context) -> None:
    boxes = {"red_box": np.array([[4, 4, 12, 12]], dtype=np.float32)}
    ctx = _ctx_with_box_prompts(default_context, boxes)
    stage = Sam2SegmentStage(sam2_getter=lambda: StubSam2(), mode="bbox")
    out = stage(ctx)
    assert len(out.detections["red_box"]) == 1
    assert out.num_instances == 1


def test_sam2_static_prompts_take_precedence(default_context) -> None:
    # box_prompts present, but a static prompts dict should win.
    default_context.queries = ["red_box"]
    default_context.box_prompts = {"red_box": np.array([[0, 0, 1, 1]], np.float32)}
    static = {"red_box": [(4, 4, 12, 12), (20, 20, 28, 28)]}
    stage = Sam2SegmentStage(
        sam2_getter=lambda: StubSam2(), mode="bbox", prompts=static
    )
    out = stage(default_context)
    # Two static boxes -> two masks, proving box_prompts was ignored.
    assert len(out.detections["red_box"]) == 2


# --- point proposals: parsing, stage, sam2 point mode ------------------------


def test_gemini_point_detector_converts_0_1000_to_pixel_xy() -> None:
    from open_perception.models.gemini import GeminiBoxDetector

    det = GeminiBoxDetector()
    # point is [y, x] in 0-1000; image is 200x100 (HxW).
    dets = [
        {"point": [100, 250], "label": "red mug"},
        {"point": [500, 750], "label": "red mug"},
    ]
    pts = det._to_pixel_xy(dets, H=200, W=100)
    assert set(pts) == {"red_mug"}  # label sanitized
    assert pts["red_mug"].shape == (2, 2)  # duplicates stack, one row each
    # xy pixel: [(250/1000)*100, (100/1000)*200]
    np.testing.assert_allclose(pts["red_mug"][0], [25.0, 20.0])


def test_gemini_point_detector_parses_points_key() -> None:
    from open_perception.models.gemini import GeminiBoxDetector

    det = GeminiBoxDetector()
    text = '```json\n{"points": [{"point": [10, 20], "label": "x"}]}\n```'
    assert det._parse_json(text, key="points") == [{"point": [10, 20], "label": "x"}]


def test_detect_gemini_point_proposal_sets_point_prompts(default_context) -> None:
    points = {"mug": np.array([[8.0, 8.0], [24.0, 24.0]], dtype=np.float32)}
    stage = GeminiDetectStage(
        detector_getter=lambda: StubBoxDetector(points=points),
        stage="grounded",
        proposal="point",
    )
    ctx = stage(default_context)
    assert ctx.box_prompts is None
    assert ctx.point_prompts is not None and set(ctx.point_prompts) == {"mug"}
    assert ctx.point_prompts["mug"].shape == (2, 2)


def test_sam2_point_consumes_point_prompts(default_context) -> None:
    default_context.queries = ["mug"]
    default_context.point_prompts = {
        "mug": np.array([[8.0, 8.0], [24.0, 24.0]], dtype=np.float32)
    }
    stage = Sam2SegmentStage(sam2_getter=lambda: StubSam2(), mode="point")
    out = stage(default_context)
    # Two proposed points -> two instance masks for the same query.
    assert len(out.detections["mug"]) == 2
    assert out.num_instances == 2
    assert out.detections["mug"][0].mask[8, 8]
    assert out.detections["mug"][1].mask[24, 24]


# --- sam3 geometric (tracker-head) modes -------------------------------------


def test_sam3_bbox_mode_consumes_box_prompts(default_context) -> None:
    from _stubs import StubSam3Geometric
    from open_perception.stages.sam3 import Sam3SegmentStage

    default_context.queries = ["mug"]
    default_context.box_prompts = {"mug": np.array([[4, 4, 12, 12]], dtype=np.float32)}
    stage = Sam3SegmentStage(
        sam3_getter=lambda: StubSam3Geometric(),
        prompt_templates=["<object_names>"],
        score_threshold=0.05,
        mask_threshold=0.4,
        mode="bbox",
    )
    out = stage(default_context)
    assert len(out.detections["mug"]) == 1
    assert out.detections["mug"][0].mask[8, 8]


def test_sam3_point_mode_consumes_point_prompts(default_context) -> None:
    from _stubs import StubSam3Geometric
    from open_perception.stages.sam3 import Sam3SegmentStage

    default_context.queries = ["mug"]
    default_context.point_prompts = {
        "mug": np.array([[8.0, 8.0], [24.0, 24.0]], dtype=np.float32)
    }
    stage = Sam3SegmentStage(
        sam3_getter=lambda: StubSam3Geometric(),
        prompt_templates=["<object_names>"],
        score_threshold=0.05,
        mask_threshold=0.4,
        mode="point",
    )
    out = stage(default_context)
    assert len(out.detections["mug"]) == 2
    assert out.num_instances == 2


# --- OpenRouter backend: shares prompts/parsing, swaps transport ------------


class _FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = type("M", (), {"content": self._content})()
        choice = type("C", (), {"message": msg})()
        return type("R", (), {"choices": [choice]})()


class _FakeOpenAI:
    def __init__(self, content: str) -> None:
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions(content)})()


def test_openrouter_detector_parses_boxes_and_sends_image() -> None:
    from open_perception.models.openrouter import OpenRouterBoxDetector

    det = OpenRouterBoxDetector(model_id="google/gemini-3-flash-preview")
    fake = _FakeOpenAI('{"bboxes": [{"box_2d": [100, 250, 500, 750], "label": "mug"}]}')
    det._client = fake  # inject; skips make_openai_client / network

    rgb = np.zeros((200, 100, 3), dtype=np.uint8)
    res = det.detect(rgb, queries=["mug"], task=None, proposal="box")
    assert set(res.boxes) == {"mug"}
    np.testing.assert_allclose(res.boxes["mug"][0], [25.0, 20.0, 75.0, 100.0])

    # The request carried the prompt + a base64 image_url content block.
    call = fake.chat.completions.calls[0]
    assert call["model"] == "google/gemini-3-flash-preview"
    content = call["messages"][0]["content"]
    assert any(c["type"] == "text" for c in content)
    img = next(c for c in content if c["type"] == "image_url")
    assert img["image_url"]["url"].startswith("data:image/png;base64,")


def test_openrouter_detector_parses_points_with_prose_prefix() -> None:
    from open_perception.models.openrouter import OpenRouterBoxDetector

    det = OpenRouterBoxDetector()
    # Reasoning-style models may prepend prose before the JSON object.
    det._client = _FakeOpenAI(
        'Here are the points:\n{"points": [{"point": [100, 250], "label": "mug"}]}'
    )
    res = det.detect(np.zeros((200, 100, 3), np.uint8), ["mug"], None, proposal="point")
    assert set(res.points) == {"mug"}
    np.testing.assert_allclose(res.points["mug"][0], [25.0, 20.0])


# --- end-to-end via OpenPerception(box_detector=...) ------------------------


def test_end_to_end_open_vocab(synthetic_rgb) -> None:
    boxes = {
        "red_block": np.array([[4, 4, 12, 12]], dtype=np.float32),
        "blue_block": np.array([[20, 20, 28, 28]], dtype=np.float32),
    }
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["detect.gemini", "sam2"])
    cfg.update("sam2", mode="bbox")
    perc = OpenPerception(cfg, box_detector=StubBoxDetector(boxes))
    perc._sam2._instance = StubSam2()  # inject stub, skip real SAM2 load

    # Empty queries: open-vocab proposal supplies the labels.
    out = perc.detect([], synthetic_rgb)
    assert set(out.detections) == {"red_block", "blue_block"}
    assert all(len(v) == 1 for v in out.detections.values())


def test_end_to_end_point_proposal(synthetic_rgb) -> None:
    points = {
        "red_block": np.array([[8.0, 8.0]], dtype=np.float32),
        "blue_block": np.array([[24.0, 24.0]], dtype=np.float32),
    }
    cfg = PerceptionConfig(eager_load=False)
    cfg.order(["detect.gemini", "sam2"])
    cfg.update("detect.gemini", proposal="point")
    cfg.update("sam2", mode="point")
    perc = OpenPerception(cfg, box_detector=StubBoxDetector(points=points))
    perc._sam2._instance = StubSam2()  # inject stub, skip real SAM2 load

    out = perc.detect([], synthetic_rgb)
    assert set(out.detections) == {"red_block", "blue_block"}
    assert all(len(v) == 1 for v in out.detections.values())
