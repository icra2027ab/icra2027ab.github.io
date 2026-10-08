# ruff: noqa: E402
"""Replay LeRobot RoboCasa demos in a RoboCasa sim and dump a LIBERO HDF5.

This script bridges robocasa's LeRobot dataset distribution (per-task parquet
+ per-camera MP4 + per-episode ``extras/`` init-state bundle) to the LIBERO
HDF5 schema that policy_abstraction's :class:`DemoAnalyzer` and downstream
pipeline already consume (segmentation-based VLM pair selection, point-cloud
guidance).

Run from the **robocasa venv** (``third_party/robocasa/.venv``), not the
main PA venv -- robosuite/MuJoCo are only installed there.  Mirrors how
``scripts/servers/serve_robocasa_env.py`` is launched.

    cd third_party/robocasa
    uv run python ../../scripts/setup/collect_robocasa_demos.py \\
        --lerobot-root datasets/robocasa_orig/lerobot \\
        --task-name CloseDrawer \\
        --out-dir datasets/robocasa \\
        --num-episodes 3

Init-state restoration
----------------------
Robocasa's LeRobot tars include a sibling ``extras/episode_NNNNNN/`` folder
per episode containing ``model.xml.gz`` (MuJoCo scene XML), ``ep_meta.json``
(kitchen layout / object cfgs), and ``states.npz`` (per-step flattened sim
state).  We feed all three to robocasa's standard ``reset_to`` primitive
(see third_party/robocasa/.../playback_dataset.py:165) so the replay starts
in the exact recorded scene.  This makes the replayed trajectory
*deterministically equal* to the original, modulo floating-point determinism
in MuJoCo.

When ``--no-use-init-states`` is passed we fall back to a plain
``env.reset()`` seeded with ``--seed``; the kitchen layout then diverges
from the recording (suitable for retrieval/seg analysis only).

Other notes
-----------
* The instance-segmentation mask is rendered per-step via
  ``env.sim.render(..., segmentation=True)`` because ``Kitchen.__init__`` does
  not accept ``camera_segmentations`` (see envs/robocasa/server.py:541).
  Only the instance-id channel is stored.
* Action width: the parquet ``action`` column is the recorded 12-D PandaOmron
  action ``[eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]``.
  We replay the full 12-D action through ``env.step``, but save only the
  arm-only 7-D action (eef_pos + eef_rot + gripper) into the HDF5 to match
  LIBERO's schema and pi0's I/O contract.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Make src/ importable when launched from inside third_party/robocasa.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
for _p in (_SRC_ROOT, _REPO_ROOT):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import h5py
import numpy as np
import pyarrow.parquet as pq
import tyro
from loguru import logger

from policy_abstraction.envs.robosuite_geom_utils import (
    get_camera_extrinsic_matrix,
    get_camera_intrinsic_matrix,
)
from policy_abstraction.utils.segmentation import ROBOT_SEGMENT_KEYWORDS
from policy_abstraction.utils.transforms import quat2axisangle

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    """CLI arguments for :mod:`collect_robocasa_demos`."""

    lerobot_root: str
    """Path to the ``lerobot/`` dir, its date-stamped parent, or the task root.

    Accepts e.g.
    ``$BASE_DATASETS_DIR/RoboCasa-Cosmos-Policy/atomic/OpenSingleDoor`` or
    ``.../OpenSingleDoor/20240501/lerobot``.  We probe for the
    ``meta/tasks.jsonl`` marker exactly like
    :func:`policy_abstraction.retrieval.sources.robocasa_lerobot._resolve_lerobot_root`.
    """

    task_name: str
    """RoboCasa env-class name (e.g. ``OpenSingleDoor``, ``PnPCounterToCab``).

    Episodes are filtered by ``task_index`` -> ``meta/tasks.jsonl`` task-name
    match.  If the LeRobot dataset only contains one task we accept any
    ``task_index`` so this is essentially a label.
    """

    out_dir: str
    """Directory to write ``<task_name>_demo.hdf5`` into.

    Created if missing.
    """

    num_episodes: int = 5
    """Number of episodes to replay (capped to the number of available parquet files)."""

    camera_height: int = 256
    """Render height for all cameras (matches LIBERO/RoboCasa default 256x256)."""

    camera_width: int = 256
    """Render width."""

    robots: str = "PandaOmron"
    """Robosuite robot to instantiate.  Should match the recording robot."""

    seed: int = 0
    """Deterministic seed for env construction + reset."""

    layout_and_style_ids: Optional[list[tuple[int, int]]] = None
    """Optional explicit ``(layout_id, style_id)`` pairs to pass to robosuite.

    When ``None`` we let robocasa's defaults decide.  RoboCasa's leaderboard
    "target" split uses ``list(zip(range(1, 11), range(1, 11)))``.
    """

    overwrite: bool = False
    """If True, overwrite an existing ``<task_name>_demo.hdf5`` instead of erroring."""

    use_init_states: bool = True
    """If True (default) and the LeRobot dataset has a sibling ``extras/`` dir,
    restore each episode from its recorded MuJoCo state via
    ``reset_to({"model": <xml>, "ep_meta": <json>, "states": <flat>})`` so the
    replay matches the original recording exactly.  Falls back to
    ``env.reset()`` when ``extras/`` is missing or the file set is incomplete."""


# ---------------------------------------------------------------------------
# LeRobot path resolution
# ---------------------------------------------------------------------------


def _resolve_lerobot_root(path: str) -> Path:
    """Mirror :func:`robocasa_lerobot._resolve_lerobot_root`.

    Accepts the ``lerobot/`` dir, its date-stamped parent, or the task root.
    """
    p = Path(path).resolve()
    candidates: list[Path] = [p, p / "lerobot"]
    if p.is_dir():
        for child in sorted(p.iterdir()):
            if child.is_dir():
                candidates.append(child / "lerobot")
                candidates.append(child)
    for candidate in candidates:
        if (candidate / "meta" / "tasks.jsonl").exists():
            return candidate
    raise FileNotFoundError(f"could not find lerobot/meta/tasks.jsonl under {p}")


def _load_tasks_jsonl(lerobot_root: Path) -> dict[int, str]:
    """Parse ``meta/tasks.jsonl`` -> ``{task_index: task_name}``."""
    out: dict[int, str] = {}
    with (lerobot_root / "meta" / "tasks.jsonl").open() as f:
        for line in f:
            rec = json.loads(line)
            out[int(rec["task_index"])] = str(rec["task"])
    return out


def _select_episodes(
    lerobot_root: Path,
    task_name: str,
    num_episodes: int,
) -> list[Path]:
    """Return up to *num_episodes* parquet paths whose ``task_index`` matches *task_name*.

    Matching rule: case-insensitive substring match against the
    ``meta/tasks.jsonl`` entry for the parquet's ``task_index``.  If the
    dataset only has one task we accept everything (the substring still has
    to match, but that should be obvious).
    """
    tasks = _load_tasks_jsonl(lerobot_root)
    # Match task_name against the meta entries via several heuristics so we
    # tolerate the natural-language variants robocasa records (e.g.
    # task_name="CloseDrawer" should match "Close the right drawer." and
    # "Close the left drawer.").  We accept any entry whose normalized form
    # shares all the camel-case tokens with the requested task.
    target_lower = task_name.lower()
    # Camel-case → space-separated tokens (e.g. "CloseDrawer" → ["close", "drawer"]).
    import re as _re

    tokens = [t.lower() for t in _re.findall(r"[A-Z][a-z]*|[a-z]+|\d+", task_name) if t]

    def _normalize(name: str) -> str:
        return _re.sub(r"[^a-z0-9]+", " ", name.lower())

    matching_indices: set[int] = set()
    for idx, name in tasks.items():
        norm = _normalize(name)
        if target_lower in norm.replace(" ", ""):
            matching_indices.add(idx)
            continue
        if tokens and all(tok in norm.split() for tok in tokens):
            matching_indices.add(idx)
    if not matching_indices:
        raise ValueError(
            f"No task_index in {lerobot_root / 'meta' / 'tasks.jsonl'} matches "
            f"task_name={task_name!r}. Known tasks: "
            f"{sorted(set(tasks.values()))}"
        )
    logger.info(
        f"Matched task_name={task_name!r} to task_indices={sorted(matching_indices)} "
        f"({[tasks[i] for i in sorted(matching_indices)]})"
    )

    data_dir = lerobot_root / "data" / "chunk-000"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Missing parquet chunk dir: {data_dir}.  Is this a valid LeRobot dataset?"
        )

    parquet_files = sorted(data_dir.glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No episode_*.parquet under {data_dir}")

    selected: list[Path] = []
    for pq_path in parquet_files:
        # Cheap: only read task_index column, only first row.
        try:
            table = pq.read_table(str(pq_path), columns=["task_index"])
        except Exception as exc:
            logger.warning(f"Failed to read {pq_path.name}: {exc}; skipping")
            continue
        if len(table) == 0:
            continue
        idx = int(table["task_index"][0].as_py())
        if idx in matching_indices:
            selected.append(pq_path)
        if len(selected) >= num_episodes:
            break
    if not selected:
        raise ValueError(
            f"No parquet files matched task_name={task_name!r} "
            f"(task_indices={sorted(matching_indices)})."
        )
    return selected


def _extras_dir_for(lerobot_root: Path, episode_id: int) -> Optional[Path]:
    """Return ``lerobot/extras/episode_NNNNNN`` if present, else ``None``."""
    candidate = lerobot_root / "extras" / f"episode_{episode_id:06d}"
    return candidate if candidate.is_dir() else None


def _load_init_state(extras_dir: Path) -> Optional[dict]:
    """Load ``{model, ep_meta, states[0]}`` from a LeRobot ``extras/`` folder.

    Robocasa's tar layout (see download_datasets.py + playback_dataset.py):
      extras/episode_NNNNNN/
        model.xml.gz   — gzipped MuJoCo scene XML
        ep_meta.json   — kitchen layout / object cfgs (JSON)
        states.npz     — np.savez(states=<flat-state per step>)

    Returns ``None`` if any file is missing, so the caller can fall back to a
    plain ``env.reset()``.
    """
    import gzip

    model_path = extras_dir / "model.xml.gz"
    ep_meta_path = extras_dir / "ep_meta.json"
    states_path = extras_dir / "states.npz"
    if not (model_path.exists() and ep_meta_path.exists() and states_path.exists()):
        logger.warning(
            f"Incomplete extras/ at {extras_dir} -- "
            f"have {[p.name for p in extras_dir.iterdir()]}"
        )
        return None
    with gzip.open(model_path, "rt") as f:
        model_xml = f.read()
    with ep_meta_path.open() as f:
        ep_meta_str = f.read()
    states = np.load(states_path)["states"]
    return {
        "model": model_xml,
        "ep_meta": ep_meta_str,  # robocasa's reset_to calls json.loads internally
        "states": states[0],
        "states_full": states,  # caller may use for per-step divergence check
    }


def _reset_to(env, init_state: dict) -> None:
    """Mirror :func:`robocasa.scripts.dataset_scripts.playback_dataset.reset_to`.

    We inline it (rather than importing) so this script remains importable in
    contexts where the playback_dataset module hasn't been loaded yet (it
    imports lerobot_utils which has its own heavy deps).
    """
    import json as _json

    import robosuite

    if "model" in init_state:
        ep_meta = (
            _json.loads(init_state["ep_meta"])
            if init_state.get("ep_meta") is not None
            else {}
        )
        if hasattr(env, "set_attrs_from_ep_meta"):
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):
            env.set_ep_meta(ep_meta)
        env.reset()
        robosuite_minor = int(robosuite.__version__.split(".")[1])
        if robosuite_minor <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(init_state["model"])
        else:
            xml = env.edit_model_xml(init_state["model"])
        env.reset_from_xml_string(xml)
        env.sim.reset()
    if "states" in init_state:
        env.sim.set_state_from_flattened(init_state["states"])
        env.sim.forward()
    if hasattr(env, "update_sites"):
        env.update_sites()
    if hasattr(env, "update_state"):
        env.update_state()


def _read_actions(parquet_path: Path) -> np.ndarray:
    """Return ``[T, A]`` action array stacked from a LeRobot parquet.

    Each row's ``action`` field is itself a list/array of floats (the recorded
    12-D PandaOmron action), so we ``np.asarray`` each cell and stack.
    """
    table = pq.read_table(str(parquet_path), columns=["action"])
    col = table["action"].to_pylist()
    if not col:
        raise ValueError(f"Empty action column in {parquet_path}")
    arr = np.asarray([np.asarray(row, dtype=np.float64) for row in col])
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2-D action array from {parquet_path}, got shape {arr.shape}"
        )
    return arr


# ---------------------------------------------------------------------------
# RoboCasa env factory (mirrors envs/robocasa/server.py:_make_env_by_name)
# ---------------------------------------------------------------------------


def _make_env(
    env_name: str,
    camera_height: int,
    camera_width: int,
    robots: str,
    seed: int,
    layout_and_style_ids: Optional[list[tuple[int, int]]],
):
    """Build a RoboCasa env with depth + offscreen rendering enabled.

    Mirrors :meth:`RobocasaEnvServer._make_env_by_name`
    (envs/robocasa/server.py:483-558).  We do NOT pass ``camera_segmentations``
    because ``Kitchen.__init__`` does not accept that kwarg -- segmentation is
    rendered explicitly per-step via ``env.sim.render(..., segmentation=True)``.
    """
    import robocasa  # noqa: F401  -- side effect: registers Kitchen envs
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    controller_config = load_composite_controller_config(controller=None, robot=robots)

    # robosuite expects ``layout_and_style_ids`` as a list of (layout, style)
    # tuples; pass through verbatim when given.
    env_kwargs: dict = dict(
        env_name=env_name,
        robots=robots,
        controller_configs=controller_config,
        camera_names=[
            "robot0_agentview_left",
            "robot0_eye_in_hand",
        ],
        camera_widths=camera_width,
        camera_heights=camera_height,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=True,
        seed=seed,
        generative_textures=None,
        randomize_cameras=False,
        translucent_robot=False,
    )
    if layout_and_style_ids is not None:
        env_kwargs["layout_and_style_ids"] = [
            tuple(pair) for pair in layout_and_style_ids
        ]

    env = robosuite.make(**env_kwargs)
    logger.info(
        f"Created robocasa env {env_name!r} "
        f"(res={camera_height}x{camera_width}, seed={seed})"
    )
    return env


# ---------------------------------------------------------------------------
# Per-step helpers
# ---------------------------------------------------------------------------


# Cameras stored in the LIBERO HDF5.  Keys are the *output prefix* used in the
# HDF5 obs group; values are the robosuite camera names used to render +
# read camera params.
_AGENTVIEW_CAM = "robot0_agentview_left"
_EIH_CAM = "robot0_eye_in_hand"


def _render_segmentation(
    env, camera_name: str, camera_height: int, camera_width: int
) -> np.ndarray:
    """Render an instance-segmentation mask via ``env.sim.render``.

    Returns ``(H, W, 1)`` uint8 with only the instance-id channel kept
    (channel 0 of MuJoCo's ``(H, W, 2)`` segmentation buffer).  Mask IDs
    saturate above 255 in MuJoCo, so we clip to uint8 -- the LIBERO schema
    uses uint8 for segmentation as well.

    MuJoCo's offscreen render is OpenGL-style (origin at bottom-left), i.e.
    flipped vs. the natural image orientation.  The LIBERO pipeline expects
    on-disk frames to be stored in that OpenGL orientation and re-flips them
    with ``[::-1, ::-1]`` when reading (see
    :func:`policy_abstraction.utils.segmentation.accumulate_interacting_ids`).
    ``env.sim.render`` already returns frames in that on-disk orientation, so
    we pass them through verbatim.
    """
    seg = env.sim.render(
        camera_name=camera_name,
        width=camera_width,
        height=camera_height,
        segmentation=True,
    )
    # robosuite/mujoco return (H, W, 2) int32 -- channel 0 is instance id,
    # channel 1 is geom-type id.  Keep only the instance id.
    if seg.ndim != 3 or seg.shape[-1] < 1:
        raise RuntimeError(
            f"Unexpected segmentation shape from env.sim.render: {seg.shape}"
        )
    inst = seg[..., 0].astype(np.int32)
    # uint8 is the schema dtype LIBERO uses.  Mask IDs >255 would wrap; clip
    # to 255 instead so the visualisation overlay stays sensible.  In RoboCasa
    # kitchen scenes typically <100 visible instances, so this is safe.
    if inst.max() > 255:
        logger.warning(
            f"Segmentation has IDs >255 (max={inst.max()}); clipping to uint8 may "
            f"collide instance IDs in cam={camera_name!r}"
        )
    inst_u8 = np.clip(inst, 0, 255).astype(np.uint8)
    return inst_u8[..., np.newaxis]  # (H, W, 1)


def _build_seg_name_mapping(env) -> dict[str, str]:
    """Build a ``{seg_id_str: object_name}`` mapping for non-robot geoms.

    The LIBERO schema stores this as a JSON-encoded HDF5 attribute on
    ``/data``.  By *omitting* robot/gripper geoms from this map we let
    :func:`get_robot_segment_ids` route them via its "unmapped -> robot" branch
    (segmentation.py:71-73), which is exactly the convention LIBERO uses.

    We use ``env.model.instances_to_ids`` to enumerate instances and
    ``env.segmentation_id_mapping`` (raw_seg_id -> instance name) to recover
    the pixel-value mapping.  Pixel value = raw_seg_id + 1 (robosuite
    convention; see envs/robocasa/server.py:627-640).
    """
    seg_mapping = getattr(env, "segmentation_id_mapping", None) or {}
    out: dict[str, str] = {}
    for raw_seg_id, instance_name in seg_mapping.items():
        if not isinstance(instance_name, str):
            instance_name = str(instance_name)
        # Drop anything that looks like a robot/gripper part.
        if any(kw in instance_name for kw in ROBOT_SEGMENT_KEYWORDS):
            continue
        mask_id = int(raw_seg_id) + 1
        out[str(mask_id)] = instance_name
    return out


def _strip_action_to_arm_only(action: np.ndarray) -> np.ndarray:
    """Slice the recorded 12-D PandaOmron action down to 7-D arm-only.

    Layout: ``[eef_pos(3), eef_rot(3), gripper(1), base_motion(4), control_mode(1)]``
    -> ``[eef_pos(3), eef_rot(3), gripper(1)]``.  If the input is already 7-D
    we pass it through unchanged.
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape[0] >= 7:
        return a[:7].astype(np.float32)
    out = np.zeros(7, dtype=np.float32)
    out[: a.shape[0]] = a
    return out


def _pad_action_for_env(action: np.ndarray, env) -> np.ndarray:
    """Pad/trim the recorded action to ``env.action_dim`` for ``env.step``.

    For PandaOmron (12-D) we pass the original 12-D action through; for arms
    whose native dim equals 7-D we slice the arm-only portion.  Matches
    :meth:`RobocasaEnvServer._pad_action` (envs/robocasa/server.py:597-611).
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    target = int(env.action_dim)
    if a.shape[0] == target:
        return a
    if a.shape[0] > target:
        return a[:target]
    out = np.zeros(target, dtype=np.float64)
    out[: a.shape[0]] = a
    return out


# ---------------------------------------------------------------------------
# Episode replay
# ---------------------------------------------------------------------------


def _replay_episode(
    env,
    actions_12d: np.ndarray,
    camera_height: int,
    camera_width: int,
    *,
    init_state: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Step *env* through *actions_12d* and accumulate the LIBERO obs payload.

    When ``init_state`` is provided, we restore the recorded MuJoCo state via
    :func:`_reset_to` so the replay matches the original recording exactly.
    Otherwise we fall back to a plain ``env.reset()`` and the kitchen layout
    will differ from the recording.

    Returns a dict of stacked arrays keyed by the on-disk HDF5 dataset name.
    """
    # Per-step buffers
    agentview_rgb: list[np.ndarray] = []
    agentview_depth: list[np.ndarray] = []
    agentview_seg: list[np.ndarray] = []
    agentview_K: list[np.ndarray] = []
    agentview_E: list[np.ndarray] = []

    eih_rgb: list[np.ndarray] = []
    eih_depth: list[np.ndarray] = []
    eih_K: list[np.ndarray] = []
    eih_E: list[np.ndarray] = []

    ee_pos: list[np.ndarray] = []
    ee_ori: list[np.ndarray] = []
    gripper_states: list[np.ndarray] = []
    saved_actions: list[np.ndarray] = []

    if init_state is not None:
        _reset_to(env, init_state)
        # _reset_to does not return obs; pull it explicitly.
        obs = (
            env._get_observations()
            if hasattr(env, "_get_observations")
            else env.reset()
        )
    else:
        obs = env.reset()

    T = actions_12d.shape[0]
    for t in range(T):
        # ---- Record obs at *current* state (i.e. *before* applying action t)
        # so that obs[t] and actions[t] are an aligned (state, action) pair --
        # matches LIBERO's data/regenerator.py convention.
        agentview_rgb.append(np.asarray(obs[f"{_AGENTVIEW_CAM}_image"], dtype=np.uint8))
        agentview_depth.append(
            np.asarray(obs[f"{_AGENTVIEW_CAM}_depth"], dtype=np.float32).squeeze(-1)
            if np.asarray(obs[f"{_AGENTVIEW_CAM}_depth"]).ndim == 3
            else np.asarray(obs[f"{_AGENTVIEW_CAM}_depth"], dtype=np.float32)
        )
        agentview_seg.append(
            _render_segmentation(env, _AGENTVIEW_CAM, camera_height, camera_width)
        )
        agentview_K.append(
            get_camera_intrinsic_matrix(
                env.sim, _AGENTVIEW_CAM, camera_height, camera_width
            ).astype(np.float32)
        )
        agentview_E.append(
            get_camera_extrinsic_matrix(env.sim, _AGENTVIEW_CAM).astype(np.float32)
        )

        eih_rgb.append(np.asarray(obs[f"{_EIH_CAM}_image"], dtype=np.uint8))
        eih_d = np.asarray(obs[f"{_EIH_CAM}_depth"], dtype=np.float32)
        if eih_d.ndim == 3:
            eih_d = eih_d.squeeze(-1)
        eih_depth.append(eih_d)
        eih_K.append(
            get_camera_intrinsic_matrix(
                env.sim, _EIH_CAM, camera_height, camera_width
            ).astype(np.float32)
        )
        eih_E.append(get_camera_extrinsic_matrix(env.sim, _EIH_CAM).astype(np.float32))

        # EE pose: robosuite stores quat as xyzw.
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(3)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
        ee_pos.append(eef_pos)
        ee_ori.append(quat2axisangle(eef_quat.copy()).astype(np.float64))

        # Raw 2-D gripper qpos.
        gq = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
        if gq.shape[0] < 2:
            gq = np.pad(gq, (0, 2 - gq.shape[0]))
        gripper_states.append(gq[:2])

        saved_actions.append(_strip_action_to_arm_only(actions_12d[t]))

        # ---- Step
        padded = _pad_action_for_env(actions_12d[t], env)
        obs, _reward, _done, _info = env.step(padded)

    return {
        "agentview_rgb": np.stack(agentview_rgb, axis=0).astype(np.uint8),
        "agentview_depth": np.stack(agentview_depth, axis=0).astype(np.float32),
        "agentview_segmentation": np.stack(agentview_seg, axis=0).astype(np.uint8),
        "agentview_intrinsic": np.stack(agentview_K, axis=0).astype(np.float32),
        "agentview_extrinsic": np.stack(agentview_E, axis=0).astype(np.float32),
        "robot0_eye_in_hand_image": np.stack(eih_rgb, axis=0).astype(np.uint8),
        "robot0_eye_in_hand_depth": np.stack(eih_depth, axis=0).astype(np.float32),
        "robot0_eye_in_hand_intrinsic": np.stack(eih_K, axis=0).astype(np.float32),
        "robot0_eye_in_hand_extrinsic": np.stack(eih_E, axis=0).astype(np.float32),
        "ee_pos": np.stack(ee_pos, axis=0).astype(np.float64),
        "ee_ori": np.stack(ee_ori, axis=0).astype(np.float64),
        "gripper_states": np.stack(gripper_states, axis=0).astype(np.float32),
        "actions": np.stack(saved_actions, axis=0).astype(np.float32),
    }


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------


_OBS_KEYS_IN_OBS_GROUP = (
    "agentview_rgb",
    "agentview_depth",
    "agentview_segmentation",
    "agentview_intrinsic",
    "agentview_extrinsic",
    "robot0_eye_in_hand_image",
    "robot0_eye_in_hand_depth",
    "robot0_eye_in_hand_intrinsic",
    "robot0_eye_in_hand_extrinsic",
    "ee_pos",
    "ee_ori",
    "gripper_states",
)


def _write_demo(
    grp: h5py.Group,
    demo_idx: int,
    payload: dict[str, np.ndarray],
    *,
    episode_id: int,
    parquet_name: str,
) -> None:
    """Write one ``demo_<idx>`` group with the LIBERO obs layout.

    ``episode_id`` / ``parquet_name`` are stashed as per-demo attrs so
    ``scripts/debug/verify_robocasa_demos.py`` can locate the reference LeRobot
    MP4 this demo was replayed from (the file index is shuffled by
    ``--num-episodes`` filtering, so ``demo_<i>`` is NOT the same as
    ``episode_<i>``).
    """
    demo_grp = grp.create_group(f"demo_{demo_idx}")
    demo_grp.attrs["num_samples"] = int(payload["actions"].shape[0])
    demo_grp.attrs["episode_id"] = int(episode_id)
    demo_grp.attrs["parquet_name"] = parquet_name

    obs_grp = demo_grp.create_group("obs")
    for key in _OBS_KEYS_IN_OBS_GROUP:
        if key not in payload:
            raise KeyError(f"Missing payload key: {key}")
        obs_grp.create_dataset(key, data=payload[key])

    demo_grp.create_dataset("actions", data=payload["actions"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(args: Args) -> None:
    logger.info(
        f"collect_robocasa_demos: task={args.task_name} "
        f"lerobot_root={args.lerobot_root} out_dir={args.out_dir}"
    )

    lerobot_root = _resolve_lerobot_root(args.lerobot_root)
    logger.info(f"Resolved LeRobot root -> {lerobot_root}")

    parquet_paths = _select_episodes(lerobot_root, args.task_name, args.num_episodes)
    logger.info(
        f"Selected {len(parquet_paths)} episode(s): {[p.name for p in parquet_paths]}"
    )

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.task_name}_demo.hdf5"
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {out_path} (pass --overwrite to replace)"
        )
    if out_path.exists():
        out_path.unlink()

    env = _make_env(
        env_name=args.task_name,
        camera_height=args.camera_height,
        camera_width=args.camera_width,
        robots=args.robots,
        seed=args.seed,
        layout_and_style_ids=args.layout_and_style_ids,
    )

    total_frames = 0
    seg_name_mapping: dict[str, str] = {}
    try:
        with h5py.File(out_path, "w") as h5:
            data_grp = h5.create_group("data")

            for i, parquet_path in enumerate(parquet_paths):
                logger.info(
                    f"[{i + 1}/{len(parquet_paths)}] Replaying {parquet_path.name}"
                )
                try:
                    actions_12d = _read_actions(parquet_path)
                except Exception as exc:
                    logger.warning(f"Failed to read actions from {parquet_path}: {exc}")
                    continue
                # parquet filenames are ``episode_NNNNNN.parquet``; parse the
                # zero-padded ID so we can locate matching ``extras/`` and so
                # verify_robocasa_demos.py can locate the matching MP4.
                try:
                    episode_id = int(parquet_path.stem.split("_", 1)[1])
                except (IndexError, ValueError):
                    episode_id = i

                init_state: Optional[dict] = None
                if args.use_init_states:
                    extras = _extras_dir_for(lerobot_root, episode_id)
                    if extras is not None:
                        init_state = _load_init_state(extras)
                    if init_state is None:
                        logger.warning(
                            f"  episode {episode_id}: no usable init state under "
                            f"{lerobot_root / 'extras'}; falling back to env.reset() "
                            "(kitchen layout will not match recording)"
                        )

                divergence_note = (
                    "exact replay (init state restored)"
                    if init_state is not None
                    else "open-loop replay -- kitchen layout differs from recording"
                )
                logger.info(
                    f"  episode actions shape={actions_12d.shape} "
                    f"(replaying {actions_12d.shape[0]} steps; {divergence_note})"
                )

                try:
                    payload = _replay_episode(
                        env,
                        actions_12d=actions_12d,
                        camera_height=args.camera_height,
                        camera_width=args.camera_width,
                        init_state=init_state,
                    )
                except Exception as exc:
                    logger.exception(f"Replay failed for {parquet_path.name}: {exc}")
                    continue

                _write_demo(
                    data_grp,
                    i,
                    payload,
                    episode_id=episode_id,
                    parquet_name=parquet_path.name,
                )
                total_frames += int(payload["actions"].shape[0])

                if not seg_name_mapping:
                    # Capture the mapping once from the first successful replay.
                    seg_name_mapping = _build_seg_name_mapping(env)

            # Stash the seg name map JSON on /data so DemoAnalyzer can find it
            # (analysis/demo_analyzer.py:400-410 reads ``segmentation_name_mapping``).
            data_grp.attrs["segmentation_name_mapping"] = json.dumps(seg_name_mapping)
    finally:
        try:
            env.close()
        except Exception:
            pass

    logger.info(
        f"Done.  Wrote {len(parquet_paths)} episode(s) "
        f"({total_frames} total frames) -> {out_path}"
    )
    logger.info(
        f"segmentation_name_mapping has {len(seg_name_mapping)} non-robot entries; "
        f"robot/gripper geoms are intentionally omitted so "
        f"get_robot_segment_ids() routes them via the unmapped -> robot branch "
        f"(utils/segmentation.py:71-73)."
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
