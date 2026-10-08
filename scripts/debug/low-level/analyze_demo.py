"""Save videos of camera views from a LIBERO dataset demo file.

When ``--annotate-fingers`` is passed (the default), the video also overlays
the projected 3-D positions of both Panda finger tips on every frame:
  • left  finger → green dot
  • right finger → red   dot

Pass ``--rerun`` to visualise trajectories and camera feeds live in Rerun
instead of (or in addition to) saving a video.
"""

import argparse
import json
from pathlib import Path

import h5py

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.analysis.demo_analyzer import DemoAnalyzer
from policy_abstraction.utils.seeding import seed_everything
from policy_abstraction.visualization.annotation import (
    LIBERO_DEPTH_FAR,
    LIBERO_DEPTH_NEAR,
    save_demo_video,
    visualize_in_rerun,
)


def _task_description_from_path(task_file: Path) -> str:
    """Derive a human-readable task string from the HDF5 filename."""
    stem = task_file.stem
    if stem.endswith("_demo"):
        stem = stem[:-5]
    return stem.replace("_", " ")


def process_file(
    task_file: Path,
    demo_index: int | None,
    output_root: Path,
    fps: int,
    annotate_fingers: bool = True,
    use_rerun: bool = False,
    vlm_model: str = "gpt-5.4",
    cache_dir: Path | None = None,
) -> None:
    task_name = task_file.stem
    task_description = _task_description_from_path(task_file)
    video_dir = output_root / task_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    analyzer = DemoAnalyzer(vlm_model=vlm_model, cache_dir=cache_dir)

    with h5py.File(task_file, "r") as f:
        # Read near/far stored by the regeneration script (fall back to defaults).
        depth_near = float(f["data"].attrs.get("depth_near", LIBERO_DEPTH_NEAR))
        depth_far = float(f["data"].attrs.get("depth_far", LIBERO_DEPTH_FAR))
        # Read segmentation ID → name mapping if available.
        _seg_map_json = f["data"].attrs.get("segmentation_name_mapping", None)
        seg_name_map: dict[str, str] = (
            json.loads(_seg_map_json) if _seg_map_json else {}
        )
        demos = f["data"]
        keys = (
            [f"demo_{demo_index}"]
            if demo_index is not None
            else sorted(demos.keys(), key=lambda k: int(k.split("_")[1]))
        )
        for key in keys:
            demo = demos[key]

            print(f"[{key}] Analysing demo…")
            cache_key = f"{task_name}/{key}" if cache_dir is not None else None
            result = analyzer.analyze(
                demo, task_description, seg_name_map, cache_key=cache_key
            )

            id_to_name = (
                {int(k): v for k, v in seg_name_map.items()} if seg_name_map else {}
            )
            if id_to_name:
                id_name_pairs = [
                    f"{sid} ({id_to_name.get(sid, '?')})"
                    for sid in sorted(result.interacting_ids)
                ]
                print(f"[{key}] Interacting instance IDs: {', '.join(id_name_pairs)}")
            else:
                print(
                    f"[{key}] Interacting instance IDs: {sorted(result.interacting_ids)}"
                )

            print(
                f"[{key}] Distance pairs: {[(p.from_entity, p.to_entity, p.label) for p in result.distance_pairs]}"
            )

            # Derive VLM-selected segmentation IDs from distance pairs.
            vlm_selected_ids: set[int] = set()
            for pair in result.distance_pairs:
                if isinstance(pair.from_entity, int):
                    vlm_selected_ids.add(pair.from_entity)
                if isinstance(pair.to_entity, int):
                    vlm_selected_ids.add(pair.to_entity)

            out_path = video_dir / f"{key}.mp4"
            save_demo_video(
                demo,
                out_path,
                fps=fps,
                annotate_fingers=annotate_fingers,
                interacting_ids=result.interacting_ids,
                vlm_selected_ids=vlm_selected_ids or None,
                distance_pairs=result.distance_pairs or None,
                seg_name_map=seg_name_map or None,
                depth_near=depth_near,
                depth_far=depth_far,
                dino_projector=None,
            )
            if use_rerun:
                visualize_in_rerun(
                    demo,
                    key,
                    interacting_ids=result.interacting_ids,
                    vlm_selected_ids=vlm_selected_ids or None,
                    seg_annotated_image=result.seg_index_image,
                    distance_pairs=result.distance_pairs or None,
                    depth_near=depth_near,
                    depth_far=depth_far,
                )
            print(f"Saved {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sample_dataset = str(
        PROJECT_ROOT
        / "datasets"
        / "libero_object_new"
        / "pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5"
    )

    parser.add_argument(
        "--task_file",
        type=Path,
        default=sample_dataset,
        help="Path to the .hdf5 demo file.",
    )
    parser.add_argument(
        "--demo-index",
        type=int,
        default=3,
        help="Demo index to process (default: all).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Root output directory."
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--no-annotate-fingers",
        dest="annotate_fingers",
        action="store_false",
        help="Disable finger-tip projection overlay.",
    )
    parser.add_argument(
        "--rerun",
        dest="use_rerun",
        action="store_true",
        help="Visualize trajectories and camera views in Rerun instead of saving a video.",
    )
    parser.add_argument(
        "--vlm-model",
        default="gpt-5.4",
        help="OpenAI model for VLM pair selection.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Directory for caching VLM distance-pair results per demo. "
        "If set, results are saved as editable JSON files and reused on subsequent runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for python/numpy/torch/jax RNGs.",
    )
    parser.set_defaults(annotate_fingers=True, use_rerun=True)
    args = parser.parse_args()
    seed_everything(args.seed)

    if args.output_dir is None:
        args.output_dir = args.task_file.parent

    process_file(
        args.task_file,
        args.demo_index,
        args.output_dir,
        args.fps,
        args.annotate_fingers,
        args.use_rerun,
        vlm_model=args.vlm_model,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
