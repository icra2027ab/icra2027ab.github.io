"""Visualisation utilities for demo retrieval results."""

from __future__ import annotations

import io
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from policy_abstraction.retrieval.demo_source import DemoRef, build_source


def load_first_frame(result: dict) -> np.ndarray | None:
    """Load the first frame for a retrieval result via its source backend.

    The result dict must carry ``source_kind`` + ``source_path`` + ``demo_key``
    (the schema produced by :class:`EmbeddingRetriever`). Frames come back
    upright; the source adapter handles any per-format flips.
    """
    try:
        source = build_source(result["source_kind"], result["source_path"])
        return source.load_first_frame(
            DemoRef(result["source_path"], result["demo_key"])
        )
    except Exception:
        return None


def render_retrieval_figure(
    task_description: str,
    query_frame: np.ndarray,
    results: list[dict],
) -> np.ndarray:
    """Render a retrieval grid as a uint8 RGB numpy array.

    Layout: query panel (navy border) followed by one panel per retrieved demo.

    Parameters
    ----------
    task_description:
        Task string shown in the query panel title.
    query_frame:
        uint8 RGB ``(H, W, 3)`` image of the query initial frame.
    results:
        List of retrieval result dicts as returned by
        :meth:`EmbeddingRetriever.retrieve`.  Each dict must contain
        ``"source_kind"``, ``"source_path"``, ``"demo_key"``, ``"task"``,
        ``"score"``, and optionally ``"visual_score"`` / ``"text_score"``.
    """

    def _wrap(text: str, width: int = 28) -> str:
        return "\n".join(textwrap.wrap(text, width=width))

    k = len(results)
    fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 4.5))
    if k == 0:
        axes = [axes]
    fig.suptitle("Demo Retrieval — Query vs. Top Retrieved", fontsize=12, y=1.01)

    # Query panel
    axes[0].imshow(query_frame)
    axes[0].set_title("QUERY\n" + _wrap(task_description), fontsize=8, color="navy")
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor("navy")
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Retrieved panels
    for i, result in enumerate(results):
        ax = axes[i + 1]
        frame = load_first_frame(result)
        if frame is not None:
            ax.imshow(frame)
        else:
            ax.text(
                0.5,
                0.5,
                "[image unavailable]",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
            )
        score_str = (
            f"score={result['score']:.3f}"
            f"  (vis={result.get('visual_score', 0):.2f}"
            f" txt={result.get('text_score', 0):.2f})"
        )
        ax.set_title(
            f"#{i + 1}  [{result['demo_key']}]\n"
            + _wrap(result["task"])
            + f"\n{score_str}",
            fontsize=7,
        )
        ax.axis("off")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    from PIL import Image

    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
