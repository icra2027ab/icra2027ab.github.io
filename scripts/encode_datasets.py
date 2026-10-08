"""Encode demo datasets with DINOv2 (visual) + CLIP (text) embeddings.

Supports three formats via the :class:`DemoSource` abstraction:

* ``libero`` — per-task ``*.hdf5`` files.
* ``calvin`` — NPZ-per-frame split directories with sparse language
  annotations.
* ``robocasa`` — LeRobot per-task directories (Parquet + MP4).
* ``vla`` — a directory of ``.vla`` rollouts recorded by ``evaluate.py``.

For each discovered source we produce one ``<name>_embeddings.npz`` cache
sitting next to the source data. The cache schema is:

* ``visual_embeddings`` ``(N, 768) float32`` — one per demo, L2-normalised.
* ``text_embeddings`` ``(N, 512) float32`` — one per demo, L2-normalised.
* ``demo_keys`` ``(N,) str`` — the per-demo identifier within the source.
* ``source_kind`` scalar str — ``"libero"`` / ``"calvin"`` / ``"robocasa"`` /
  ``"vla"``.
* ``source_path`` scalar str — absolute path identifying the source.
* ``task`` ``(N,) str`` — per-demo language string used as the CLIP query.
* ``dino_model`` / ``clip_model`` scalar str — model IDs the cache was built with.

Usage
-----
    # Single source (auto-detect kind)
    python scripts/encode_datasets.py \\
        --source-path /scratch/.../calvin/calvin_debug_dataset/training

    # A folder of .vla rollouts (one demo per file); the cache lands
    # inside the folder, so point --retrieval.dataset-dirs at the same path.
    python scripts/encode_datasets.py \\
        --source-path data/ur5_baseline/ur5/<task>/episodes

    # Explicit kind
    python scripts/encode_datasets.py \\
        --source-kind robocasa \\
        --source-path /scratch/.../robocasa/.../atomic/CloseDrawer

    # Recursively discover sub-sources under a root (useful for the
    # robocasa atomic split, where each task is its own source).
    python scripts/encode_datasets.py \\
        --source-kind robocasa --discover-children \\
        --source-path /scratch/.../robocasa/v1.0/pretrain/atomic

    # Legacy LIBERO multi-dir mode still works.
    python scripts/encode_datasets.py \\
        --datasets-root datasets \\
        --dataset-dirs libero_spatial_new libero_object_new
"""

from __future__ import annotations

import argparse
from pathlib import Path

from policy_abstraction.retrieval.demo_source import DemoSource, build_source
from policy_abstraction.retrieval.encode import (
    SOURCE_KINDS,
    cache_path_for,
    discover_sources,
    encode_source,
)
from policy_abstraction.utils.seeding import seed_everything

_DEFAULT_DATASET_DIRS = [
    "datasets/libero_spatial_new",
    "datasets/libero_object_new",
    "datasets/libero_goal_new",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-kind",
        choices=list(SOURCE_KINDS),
        default=None,
        help="Format of the source data. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--source-path",
        type=Path,
        default=None,
        help="Path to the source (file or directory). Mutually exclusive with --datasets-root.",
    )
    parser.add_argument(
        "--discover-children",
        action="store_true",
        help="Treat --source-path as a parent directory; encode each immediate subdir as its own source.",
    )
    # Legacy libero-only multi-dir mode.
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="(Legacy LIBERO mode) Root directory containing dataset subdirectories.",
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        default=None,
        help="(Legacy LIBERO mode) Subdirectories of --datasets-root to encode.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch16")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    seed_everything(args.seed)

    if (
        args.source_path is None
        and args.datasets_root is None
        and args.dataset_dirs is None
    ):
        # Default: legacy libero scan from project root.
        project_root = Path(__file__).resolve().parent.parent
        args.datasets_root = project_root

    sources: list[DemoSource] = []
    if args.source_path is not None:
        sources.extend(
            discover_sources(args.source_kind, args.source_path, args.discover_children)
        )
    if args.datasets_root is not None or args.dataset_dirs is not None:
        root = args.datasets_root or Path(__file__).resolve().parent.parent
        for rel in args.dataset_dirs or _DEFAULT_DATASET_DIRS:
            d = root / rel
            if not d.exists():
                print(f"[skip] Directory not found: {d}")
                continue
            for hdf5_path in sorted(d.glob("*.hdf5")):
                sources.append(build_source("libero", hdf5_path))

    if not sources:
        print("No sources to encode.")
        return

    from policy_abstraction.retrieval.encoder import DemoEncoder

    print(f"Loading models (DINOv2={args.dino_model}, CLIP={args.clip_model})…")
    encoder = DemoEncoder(dino_model=args.dino_model, clip_model=args.clip_model)
    print(f"Models loaded on device: {encoder.device}\n")

    for source in sources:
        out_path = cache_path_for(source)
        if not args.force and out_path.exists():
            print(f"[skip] {out_path}  (already encoded; pass --force to overwrite)")
            continue
        print(f"=== {source.source_kind}: {source.source_path} ===")
        encode_source(source, encoder, out_path, batch_size=args.batch_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
