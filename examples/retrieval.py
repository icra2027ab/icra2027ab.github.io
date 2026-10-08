"""Showcase demo retrieval: given a query demo, display the top-5 most similar.

The script loads a query demo from one of the LIBERO datasets, builds a
retrieval index over all three encoded datasets, and renders a matplotlib
figure with the reference demo alongside the five retrieved demos.

Each panel shows the first frame (agentview, upright) with the task
description and similarity score.

Usage
-----
    python scripts/example_retrieval.py \
        --query-file datasets/libero_object_new/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
        --demo-key demo_0 \
        --output outputs/retrieval_example.png
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_description_from_path(hdf5_path: Path) -> str:
    stem = hdf5_path.stem
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return stem.replace("_", " ")


def _load_query(hdf5_path: Path, demo_key: str) -> dict:
    """Load frames and task for a query demo."""
    task = _task_description_from_path(hdf5_path)
    with h5py.File(hdf5_path, "r") as f:
        raw = f["data"][demo_key]["obs"]["agentview_rgb"][:]  # (T, H, W, 3)
    # LIBERO images are upside-down; flip to upright
    frames = list(raw[:, ::-1, ::-1, :])
    return {"frames": frames, "task": task}


def _load_first_frame(hdf5_path: str, demo_key: str) -> np.ndarray:
    """Load and flip the first agentview frame from an HDF5 demo."""
    with h5py.File(hdf5_path, "r") as f:
        raw = f["data"][demo_key]["obs"]["agentview_rgb"][0]  # (H, W, 3)
    return raw[::-1, ::-1, :]


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def visualise_results(
    query: dict,
    query_hdf5: Path,
    query_demo_key: str,
    results: list[dict],
    output_path: Path,
) -> None:
    """Render and save a 1+k panel figure."""
    k = len(results)
    fig, axes = plt.subplots(1, k + 1, figsize=(4 * (k + 1), 4.5))
    fig.suptitle("Demo Retrieval — Reference vs. Top-5 Retrieved", fontsize=13, y=1.01)

    # --- Query panel -------------------------------------------------------
    query_frame = query["frames"][0]
    axes[0].imshow(query_frame)
    axes[0].set_title(
        f"QUERY  [{query_demo_key}]\n" + _wrap(query["task"], width=28),
        fontsize=8,
        color="navy",
    )
    axes[0].axis("off")
    # highlight the query panel with a coloured border
    for spine in axes[0].spines.values():
        spine.set_edgecolor("navy")
        spine.set_linewidth(3)
        spine.set_visible(True)

    # --- Retrieved panels ---------------------------------------------------
    for i, result in enumerate(results):
        ax = axes[i + 1]
        try:
            frame = _load_first_frame(result["hdf5_path"], result["demo_key"])
            ax.imshow(frame)
        except Exception as exc:
            ax.text(
                0.5,
                0.5,
                f"[image error]\n{exc}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=7,
            )

        score_str = f"score={result['score']:.3f}  (vis={result['visual_score']:.2f} txt={result['text_score']:.2f})"
        title = (
            f"#{i + 1}  [{result['demo_key']}]\n"
            + _wrap(result["task"], width=28)
            + f"\n{score_str}"
        )
        ax.set_title(title, fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to: {output_path}")
    plt.show()


def _wrap(text: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    default_query = (
        project_root
        / "datasets/libero_object_new"
        / "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-file",
        type=Path,
        default=default_query,
        help="Path to the query .hdf5 file.",
    )
    parser.add_argument(
        "--demo-key",
        default="demo_0",
        help="Demo key within the query file (default: demo_0).",
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=None,
        help="Dataset directories to index. Defaults to the three LIBERO splits.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to retrieve (default: 5).",
    )
    parser.add_argument(
        "--visual-weight",
        type=float,
        default=0.5,
        help="Weight for visual similarity (default: 0.5).",
    )
    parser.add_argument(
        "--text-weight",
        type=float,
        default=0.5,
        help="Weight for text similarity (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "retrieval_example.png",
        help="Output path for the figure.",
    )
    parser.add_argument(
        "--dino-model",
        default="facebook/dinov2-base",
    )
    parser.add_argument(
        "--clip-model",
        default="openai/clip-vit-base-patch16",
    )
    args = parser.parse_args()

    # --- Dataset dirs -------------------------------------------------------
    if args.dataset_dirs is not None:
        dataset_dirs = [Path(d) for d in args.dataset_dirs]
    else:
        dataset_dirs = [
            project_root / "datasets" / "libero_spatial_new",
            project_root / "datasets" / "libero_object_new",
            project_root / "datasets" / "libero_goal_new",
        ]

    # --- Check embeddings exist --------------------------------------------
    missing = [
        d for d in dataset_dirs if d.exists() and not list(d.glob("*_embeddings.npz"))
    ]
    if missing:
        print(
            "Warning: the following dataset directories have no *_embeddings.npz files:"
        )
        for d in missing:
            print(f"  {d}")
        print("Run  python scripts/encode_datasets.py  first to generate embeddings.\n")

    # --- Build index -------------------------------------------------------
    from policy_abstraction.retrieval import (
        DatasetIndex,
        DemoEncoder,
        EmbeddingRetriever,
    )

    print("Building retrieval index…")
    index = DatasetIndex(list(dataset_dirs))
    if index.size == 0:
        print("Index is empty. Exiting.")
        return

    print(f"Loading encoder (DINOv2={args.dino_model}, CLIP={args.clip_model})…")
    encoder = DemoEncoder(dino_model=args.dino_model, clip_model=args.clip_model)

    retriever = EmbeddingRetriever(
        index=index,
        encoder=encoder,
        visual_weight=args.visual_weight,
        text_weight=args.text_weight,
    )

    # --- Load query --------------------------------------------------------
    if not args.query_file.exists():
        print(f"Error: query file not found: {args.query_file}")
        return

    print(f"\nLoading query: {args.query_file.name}  [{args.demo_key}]")
    query = _load_query(args.query_file, args.demo_key)
    print(f"Task: {query['task']}")
    print(f"Frames: {len(query['frames'])}")

    # --- Retrieve ----------------------------------------------------------
    print(f"\nRetrieving top-{args.k} demos…")
    results = retriever.retrieve(query, k=args.k)

    print(f"\nTop-{args.k} results:")
    for i, r in enumerate(results):
        print(
            f"  #{i + 1}  score={r['score']:.4f}"
            f"  (vis={r['visual_score']:.3f}, txt={r['text_score']:.3f})"
            f"  {Path(r['hdf5_path']).stem}/{r['demo_key']}"
            f"  task={r['task']!r}"
        )

    # --- Visualise ---------------------------------------------------------
    visualise_results(
        query=query,
        query_hdf5=args.query_file,
        query_demo_key=args.demo_key,
        results=results,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
