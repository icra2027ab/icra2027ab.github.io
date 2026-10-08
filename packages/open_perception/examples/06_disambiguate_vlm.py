"""Disambiguate similar instances with a VLM reranker (OpenAI / OpenRouter).

Same three-mug scene as 04/05. Instead of CLIP, a minimal OpenAI-backed
``VlmRerankClient`` scores each candidate crop against each query.

Requires the ``vlm`` extra and an API key. Set ``OPENROUTER_API_KEY`` to route
through OpenRouter (https://openrouter.ai/api/v1) — pass an OpenRouter model id
like ``openai/gpt-4o-mini`` — or set ``OPENAI_API_KEY`` to use OpenAI directly:

    uv sync --extra vlm --extra examples
    OPENAI_API_KEY=... uv run python examples/06_disambiguate_vlm.py
    # or
    OPENROUTER_API_KEY=... uv run python examples/06_disambiguate_vlm.py
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from open_perception import (
    InferenceOptions,
    OpenPerception,
    PerceptionConfig,
    overlay_detections,
)

ASSETS = Path(__file__).resolve().parent.parent / "assets"
QUERIES = ["white mug", "yellow and white mug", "red mug"]


class OpenAIVlmRerankClient:
    """Implements the ``VlmRerankClient`` protocol via the OpenAI Chat API."""

    def __init__(self, model: str = "gpt-4o-mini", detail: str = "low") -> None:
        from open_perception.utils.openai_client import make_openai_client

        # Honors OPENROUTER_API_KEY (routed to OpenRouter) or OPENAI_API_KEY.
        self._client = make_openai_client()
        self.model = model
        self.detail = detail

    @staticmethod
    def _encode(img: np.ndarray) -> str:
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img.astype(np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def score_candidates(
        self,
        crops: list[np.ndarray],
        queries: list[str],
        reference_images: dict[str, list[np.ndarray]] | None = None,
    ) -> np.ndarray:
        queries_list = "\n".join(f"  Q{i}: {q}" for i, q in enumerate(queries))
        content: list[dict] = [
            {
                "type": "text",
                "text": (
                    f"Given {len(crops)} candidate image crops (indexed 0..{len(crops) - 1}) "
                    f"and {len(queries)} object descriptions:\n{queries_list}\n\n"
                    "Score each (crop, query) pair in [0, 1]. Return JSON only: "
                    '{"scores": [[s_c0_q0, s_c0_q1, ...], [s_c1_q0, ...], ...]}.'
                ),
            }
        ]
        for i, crop in enumerate(crops):
            content.append({"type": "text", "text": f"Crop {i}:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self._encode(crop)}",
                        "detail": self.detail,
                    },
                }
            )

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
        )
        scores = json.loads(resp.choices[0].message.content)["scores"]
        return np.asarray(scores, dtype=np.float32)


def main() -> None:
    from open_perception.utils.openai_client import vlm_api_key_present

    if not vlm_api_key_present():
        raise SystemExit(
            "set OPENROUTER_API_KEY or OPENAI_API_KEY to run the VLM rerank example"
        )

    rgb = iio.imread(ASSETS / "demo" / "frame_00.png")

    cfg = PerceptionConfig()
    cfg.order(["sam3", "filter.overlap", "filter.size", "disambiguate.vlm", "assign"])
    perc = OpenPerception(cfg, vlm_client=OpenAIVlmRerankClient())

    out = perc.detect(QUERIES, rgb, options=InferenceOptions(match_threshold=0.2))
    for q, dets in out.detections.items():
        print(f"{q}: scores={[round(d.score, 3) for d in dets]}")

    out_path = Path("tmp/06/disambiguate_vlm.png")
    out_path.parent.mkdir(exist_ok=True)
    iio.imwrite(
        out_path, overlay_detections(rgb, out, show_label=True, show_score=True)
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
