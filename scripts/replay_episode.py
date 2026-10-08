"""Replay a robodm ``.vla`` episode in the Rerun viewer.

Usage
-----
    python scripts/replay_episode.py path/to/episode.vla [options]

The script loads the episode with :class:`VLALoader`, then streams every
timestep into a :class:`RerunVisualizer` (Rerun viewer).

Examples
--------
Replay with the Rerun viewer spawned automatically::

    python scripts/replay_episode.py \\
        data/libero/episodes/task_00_episode_001.vla

Connect to a running ``rerun`` gRPC server instead of spawning::

    python scripts/replay_episode.py episode.vla --connect 127.0.0.1:9876

Disable camera images (faster for large episodes)::

    python scripts/replay_episode.py episode.vla --no-images

Show what the policy would have done *without* guidance next to what it did.
Every replan step's recorded observation is sent to a LeRobot policy server
with no guidance params, and the resulting chunk is drawn in magenta in the
3-D scene and on the camera overlays (the executed chunk stays yellow /
gripper-coloured). The server is launched from the experiment's own
``run_metadata.json`` unless ``--policy-server`` points at a running one::

    python scripts/replay_episode.py episode.vla --unguided
    python scripts/replay_episode.py episode.vla --unguided \
        --policy-server 127.0.0.1:8009 --unguided-cache tmp/unguided.npz

Add objects the rollout did not track (the distractors of a disambiguation
run) as point clouds. They are segmented once, on the first frame with depth,
by the same perception stack the rollout used, and stay put in the scene::

    python scripts/replay_episode.py episode.vla --extra-objects cable spoon
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

from policy_abstraction.data.loader import VLALoader
from policy_abstraction.utils.seeding import seed_everything
from policy_abstraction.visualization.rerun import RerunVisualizer, viewer_url


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a robodm .vla episode file in Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--episode",
        type=str,
        default="data/libero_object/episodes/task_00_pick_the_alphabet_soup_and_place_it_in_the_basket_episode_000.vla",
        help="Path to the .vla episode file.",
    )
    parser.add_argument(
        "--connect",
        metavar="HOST:PORT",
        default="0.0.0.0:9876",
        help=(
            "Connect to a running rerun gRPC server instead of spawning a new viewer. "
            "Example: 0.0.0.0:9876"
        ),
    )
    parser.add_argument(
        "--app-id",
        default="libero_eval",
        help="Rerun application ID shown in the viewer title bar.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip logging camera images (useful for faster replay of large episodes).",
    )
    parser.add_argument(
        "--no-action-chunks",
        action="store_true",
        help="Skip logging action-chunk tensors.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help=(
            "Target replay frame-rate in Hz.  When set the script will sleep "
            "between frames to approximate the requested rate.  Default: as "
            "fast as Rerun can ingest."
        ),
    )
    parser.add_argument(
        "--robot-model",
        metavar="NAME_OR_PATH",
        default=None,
        help=(
            "MuJoCo Menagerie robot name (e.g. 'panda') resolved from "
            "~/.mujoco_menagerie/<name>/<name>.xml, or an absolute path to "
            "an MJCF .xml file.  When set the robot model is displayed in "
            "Rerun with its base placed at the pose stored in the episode."
        ),
    )
    parser.add_argument(
        "--gripper-model",
        metavar="NAME",
        default="auto",
        help=(
            "MuJoCo Menagerie directory name of a gripper to bolt onto the "
            "arm's tool flange, e.g. 'robotiq_2f140'.  Menagerie publishes its "
            "arms bare, so a UR5e renders as an empty flange without this.  "
            "'auto' (default) picks one from the episode's own gripper "
            "geometry when a model for it is available; 'none' disables it."
        ),
    )
    parser.add_argument(
        "--save-rrd",
        metavar="PATH",
        default=None,
        help=(
            "Write the recording to a .rrd file instead of streaming it to a "
            "viewer.  Use this on a machine with no display; open the file "
            "later with `rerun <PATH>`."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (useful for a quick check).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for python/numpy/torch/jax RNGs.",
    )
    unguided = parser.add_argument_group(
        "unguided chunk",
        "Recompute the policy's unguided action chunk at every replan step "
        "from the recorded observation and show it next to the executed "
        "(guided) one. Needs a LeRobot policy server; one is launched from "
        "the experiment's run_metadata.json unless --policy-server is given.",
    )
    unguided.add_argument(
        "--unguided",
        action="store_true",
        help="Recompute and display the unguided action chunk.",
    )
    unguided.add_argument(
        "--policy-server",
        metavar="HOST:PORT",
        default=None,
        help=(
            "Use this running LeRobot policy server instead of launching one. "
            "It must serve the episode's checkpoint."
        ),
    )
    unguided.add_argument(
        "--policy-model-path",
        default=None,
        help=(
            "Checkpoint for the launched server. Default: the model_path in "
            "the experiment's run_metadata.json / metadata.json."
        ),
    )
    unguided.add_argument(
        "--policy-repo-path",
        default=None,
        help=(
            "Directory holding the lerobot venv for the launched server. "
            "Default: the experiment's repo_path, else third_party/physical-ai-sdk."
        ),
    )
    unguided.add_argument(
        "--policy-port",
        type=int,
        default=8019,
        help="ZMQ port for the launched server.",
    )
    unguided.add_argument(
        "--policy-gpus",
        default="0",
        help="CUDA_VISIBLE_DEVICES for the launched server.",
    )
    unguided.add_argument(
        "--unguided-cache",
        metavar="PATH",
        default=None,
        help=(
            "An .npz of recomputed chunks. Loaded instead of querying the "
            "policy when it exists, written after the replay otherwise."
        ),
    )
    extra = parser.add_argument_group(
        "extra objects",
        "Reconstruct objects the rollout did not track from the first frame's "
        "RGB-D and show them as static point clouds. Runs the perception "
        "stack (SAM on the GPU, the box detector through its API key).",
    )
    extra.add_argument(
        "--extra-objects",
        nargs="+",
        metavar="NAME",
        default=None,
        help="Object names to detect, e.g. 'cable spoon'.",
    )
    extra.add_argument(
        "--detector-backend",
        choices=("openrouter", "gemini"),
        default="openrouter",
        help="Box detector backend for --extra-objects.",
    )
    extra.add_argument(
        "--detector-model",
        default=None,
        help="Box detector model for --extra-objects (backend default when unset).",
    )
    extra.add_argument(
        "--extra-objects-cache",
        metavar="PATH",
        default=None,
        help=(
            "An .npz of the reconstructed clouds. Loaded instead of running "
            "perception when it exists, written otherwise."
        ),
    )
    return parser.parse_args(argv)


def _reconstruct_extra_objects(
    episode_path: str | Path,
    names: list[str],
    *,
    detector_backend: str = "openrouter",
    detector_model: str | None = None,
    cache_path: str | Path | None = None,
    erode_px: int = 3,
    min_points: int = 20,
) -> dict[str, np.ndarray]:
    """``{name: [M, 7]}`` point clouds of *names* from the episode's first frame.

    Same path as the staged-position measurement: the first frame with depth
    and a calibrated camera, the rollout's perception client for the masks,
    and the abstractor's own back-projection (with its border erosion) for
    the geometry. A name the detector finds several of keeps the instance
    with the most depth returns.
    """
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.is_file():
        with np.load(cache) as data:
            clouds = {k: np.asarray(data[k], dtype=np.float32) for k in data.files}
        missing = [n for n in names if n not in clouds]
        if not missing:
            logging.info(f"Loaded {sorted(clouds)} point clouds from {cache}")
            return {n: clouds[n] for n in names}
        logging.info(f"Cache {cache} lacks {missing}; running perception.")

    from policy_abstraction.experiments.measure import (
        first_frame,
        open_perception_measurer,
    )
    from policy_abstraction.utils.segmentation import (
        build_segmented_pointcloud,
        erode_segmentation,
    )

    frame = first_frame(Path(episode_path))
    logging.info(
        f"Reconstructing {names} from frame {frame.timestep} of {frame.camera}"
    )
    perceive = open_perception_measurer(
        detector_backend=detector_backend,
        detector_model=detector_model,
        pixels_mirrored=frame.pixels_mirrored,
    )
    masks = perceive(frame.rgb, frame.depth_m, frame.intrinsic, frame.extrinsic, names)

    seg = np.zeros(frame.depth_m.shape[:2], dtype=np.int32)
    by_name: dict[str, list[int]] = {}
    for inst_id, (name, mask) in masks.items():
        if mask is None or int(inst_id) <= 0:
            continue
        seg[np.asarray(mask, dtype=bool)] = int(inst_id)
        by_name.setdefault(name, []).append(int(inst_id))
    if erode_px > 0:
        seg = erode_segmentation(seg, erode_px)
    depth = frame.depth_m.copy()
    depth[~np.isfinite(depth)] = 0.0
    pcd = build_segmented_pointcloud(
        frame.rgb,
        depth,
        seg,
        frame.intrinsic,
        frame.extrinsic,
        depth_max=3.0,
        preprocessed=frame.pixels_mirrored,
    )
    ids = pcd[:, 6].astype(np.int32) if len(pcd) else np.zeros(0, dtype=np.int32)

    clouds: dict[str, np.ndarray] = {}
    for name in names:
        candidates = [
            pcd[ids == inst] for inst in by_name.get(name, []) if (ids == inst).any()
        ]
        candidates = [c for c in candidates if len(c) >= min_points]
        if not candidates:
            logging.warning(f"Perception did not find {name!r} on the first frame.")
            continue
        clouds[name] = max(candidates, key=len).astype(np.float32)
        logging.info(f"{name}: {len(clouds[name])} points")

    if cache is not None and clouds:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **clouds)
        logging.info(f"Saved point clouds to {cache}")
    return clouds


# Arms whose menagerie model ships without an end-effector, mapped to the
# gripper the cell actually has. Only consulted when the episode itself does
# not say (older recordings carry no env_spec).
_ARM_DEFAULT_GRIPPER = {"ur5e_mj_description": "robotiq_2f140"}


def _resolve_gripper_model(
    requested: str | None, env_spec, robot_description_name: str | None
) -> str | None:
    """Decide which gripper MJCF to graft onto the arm, if any.

    ``auto`` asks the episode first — its ``gripper.geometry_id`` names the
    hardware it was recorded on — and only falls back to a per-arm default for
    recordings that pre-date the env_spec metadata field.
    """
    from policy_abstraction.visualization.robot_assembly import gripper_mjcf_path

    if requested is None or requested.lower() in ("none", "off"):
        return None
    if requested.lower() != "auto":
        return requested

    candidate = None
    if env_spec is not None:
        candidate = getattr(env_spec.robot.gripper, "geometry_id", None)
    if not candidate:
        candidate = _ARM_DEFAULT_GRIPPER.get(robot_description_name or "")
    if not candidate:
        return None
    try:
        gripper_mjcf_path(candidate)
    except FileNotFoundError:
        logging.info(
            f"No MJCF for gripper {candidate!r}; rendering the arm without one."
        )
        return None
    return candidate


def _dynamics_model_for_episode(env_spec):
    """The model that turns the recorded action chunks into end-effector paths.

    Left to itself the visualiser integrates the first three action dims as
    Cartesian velocities. That is right for the simulators' delta actions and
    nonsense for the UR5e, whose chunks are absolute joint targets: summing
    radians onto the gripper position walks the drawn chunk off the table.
    Mirror ``evaluation.backends.dynamics_model_for`` from what the recording
    says about its own action space.
    """
    from policy_abstraction.dynamics.integration import IntegrationDynamicsModel

    layout = getattr(getattr(env_spec, "robot", None), "action_layout", None)
    if getattr(layout, "space", None) == "joint_absolute":
        from policy_abstraction.dynamics.ur5_kinematics import (
            Ur5KinematicsDynamicsModel,
        )

        tcp_offset = (getattr(env_spec, "metadata", None) or {}).get("tcp_offset")
        return Ur5KinematicsDynamicsModel(tcp_offset=tcp_offset)
    return IntegrationDynamicsModel()


# Magenta, matching RerunVisualizer._UNGUIDED_CHUNK_COLOR, so the same chunk
# is the same colour in the 3-D scene and on the camera frames.
UNGUIDED_OVERLAY_RGB = (255, 60, 220)


def _experiment_policy_args(episode_path: str | Path) -> dict:
    """The ``policy`` block of the experiment that recorded ``episode_path``.

    Experiments keep their episodes under ``<experiment>/episodes/`` next to a
    ``run_metadata.json`` (and an older ``metadata.json``) that records the
    exact ``evaluate.py`` arguments, checkpoint included. Returns ``{}`` when
    the episode is not laid out that way.
    """
    episode_path = Path(episode_path)
    for exp_dir in (episode_path.parent.parent, episode_path.parent):
        for name in ("run_metadata.json", "metadata.json"):
            candidate = exp_dir / name
            if not candidate.is_file():
                continue
            try:
                meta = json.loads(candidate.read_text())
            except (OSError, ValueError):
                continue
            block = meta.get("args", meta.get("eval_args", {})) or {}
            policy = block.get("policy")
            if isinstance(policy, dict):
                logging.info(f"Policy settings from {candidate}")
                return dict(policy)
    return {}


def _parse_host_port(text: str) -> tuple[str, int]:
    host, _, port = text.rpartition(":")
    return host or "127.0.0.1", int(port)


class UnguidedChunkSource:
    """Re-queries the policy, unguided, from a recorded observation.

    The policy input is rebuilt exactly as ``EpisodeRunner`` builds it — the
    same cameras resized with padding to the policy resolution, the same
    state layout from the recorded joints — but with no ``__gp_*__`` params,
    so the server takes its plain path. Two things are deliberately *not*
    reproduced: the sampled noise (the server draws its own) and Real-Time
    Chunking (the server runs without it, so the chunk is not inpainted with
    the previously executed tail). The result is therefore the policy's own
    proposal from that observation, a counterfactual sample rather than a
    bit-exact replay of the guided inference minus guidance.
    """

    def __init__(
        self,
        *,
        connect,
        env_spec,
        task_description: str,
        resize_size: int,
        seed: int,
        cache_path: str | Path | None = None,
    ) -> None:
        from policy_abstraction.evaluation.episode_runner import (
            build_policy_input,
            policy_preprocess_imgs,
        )

        self._build_policy_input = build_policy_input
        self._preprocess = policy_preprocess_imgs
        # ``connect()`` returns ``(client, server_process)``. It is only
        # called on the first chunk the cache cannot answer, so a fully
        # cached replay never touches the GPU.
        self._connect = connect
        self._client = None
        self._server = None
        self._env_spec = env_spec
        self._task = task_description
        self._resize = int(resize_size)
        self._rng = np.random.default_rng([int(seed), 0])
        self._cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[int, np.ndarray] = {}
        self._dirty = False
        if self._cache_path is not None and self._cache_path.is_file():
            with np.load(self._cache_path) as data:
                for t, chunk in zip(data["timesteps"], data["chunks"], strict=True):
                    self._cache[int(t)] = np.asarray(chunk, dtype=np.float32)
            logging.info(
                f"Loaded {len(self._cache)} unguided chunks from {self._cache_path}"
            )

    def chunk_at(
        self, t: int, obs: dict, robot_joints: np.ndarray
    ) -> np.ndarray | None:
        """The unguided ``[N, T, action_dim]`` chunk for timestep ``t``."""
        if t in self._cache:
            return self._cache[t]
        if self._client is None:
            self._client, self._server = self._connect()
        robot = self._env_spec.robot
        images = {
            cam.name: np.ascontiguousarray(
                cam.apply_image_transform(obs[cam.image_key])
            )
            for cam in self._env_spec.cameras
            if cam.image_key in obs
        }
        joints = np.asarray(robot_joints, dtype=np.float32)
        state_obs = {
            robot.joint_pos_key: joints[:6],
            robot.gripper_qpos_key: joints[6:],
        }
        element = self._build_policy_input(
            self._env_spec,
            state_obs,
            self._preprocess(images, self._resize),
            task_description=self._task,
            rng_seed=int(self._rng.integers(0, 2**31)),
        )
        t0 = time.monotonic()
        chunk = np.asarray(self._client.infer(element)["actions"], dtype=np.float32)
        logging.info(
            f"t={t}: unguided chunk {chunk.shape} in "
            f"{(time.monotonic() - t0) * 1000:.0f} ms"
        )
        self._cache[t] = chunk
        self._dirty = True
        return chunk

    def close(self) -> None:
        """Write the cache and stop the policy server, if one was started."""
        self.save()
        _stop_unguided_policy(self._client, self._server)
        self._client = self._server = None

    def save(self) -> None:
        if self._cache_path is None or not self._dirty or not self._cache:
            return
        ts = sorted(self._cache)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self._cache_path,
            timesteps=np.asarray(ts, dtype=np.int64),
            chunks=np.stack([self._cache[t] for t in ts]),
        )
        logging.info(f"Saved {len(ts)} unguided chunks to {self._cache_path}")


def _connect_unguided_policy(args, episode_path: str, env_spec, replan_steps: int):
    """A LeRobot client for the unguided recompute, plus the server it owns.

    Returns ``(client, server_process)``; the process is ``None`` when
    ``--policy-server`` named a running one. The launched server runs the
    episode's checkpoint with guidance and RTC both off, so it is the plain
    policy and nothing else.
    """
    from policy_abstraction.policies.lerobot import (
        LerobotPolicyClient,
        start_lerobot_server,
    )

    policy_args = _experiment_policy_args(episode_path)
    gripper_remap = policy_args.get("gripper_remap") or "passthrough"

    if args.policy_server:
        host, port = _parse_host_port(args.policy_server)
        logging.info(f"Using the running policy server at {host}:{port}")
        client = LerobotPolicyClient(
            host=host,
            port=port,
            gripper_remap=gripper_remap,
            replan_steps=replan_steps,
        )
        return client, None

    model_path = args.policy_model_path or policy_args.get("model_path")
    if not model_path:
        raise SystemExit(
            "--unguided needs a checkpoint: pass --policy-model-path, or "
            "--policy-server for a server that already runs it (no "
            "run_metadata.json with a policy.model_path was found next to "
            "the episode)."
        )
    repo_path = (
        args.policy_repo_path
        or policy_args.get("repo_path")
        or "third_party/physical-ai-sdk"
    )
    env_meta = getattr(env_spec, "metadata", None) or {}
    log_dir = Path("tmp") / "replay_policy_server"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info(
        f"Launching an unguided policy server on port {args.policy_port} "
        f"(checkpoint {model_path}, logs in {log_dir})"
    )
    process = start_lerobot_server(
        args.policy_port,
        model_path,
        gpus=args.policy_gpus,
        repo_path=repo_path,
        policy_type=policy_args.get("policy_type"),
        camera_map=policy_args.get("camera_map"),
        state_adapter=policy_args.get("state_adapter") or "identity",
        num_inference_steps=policy_args.get("num_inference_steps"),
        compile_model="off",
        guidance="off",
        rtc=False,
        robot_type=policy_args.get("robot_type") or "",
        tcp_offset=env_meta.get("tcp_offset"),
        gripper_geometry=getattr(env_spec.robot.gripper, "geometry_id", None),
        log_dir=log_dir,
    )
    try:
        client = LerobotPolicyClient(
            host="127.0.0.1",
            port=args.policy_port,
            server_process=process,
            gripper_remap=gripper_remap,
            replan_steps=replan_steps,
        )
    except Exception:
        process.terminate()
        raise
    return client, process


def _overlay_unguided(frame_bundle, obs: dict, env_spec, projector):
    """Draw the current unguided chunk, in magenta, on every camera overlay.

    The recorded overlays already carry the executed chunk in the gripper
    colours; the unguided one goes on top of them. A camera the projector has
    no matrices for (the wrist camera of the real cell) is left untouched. An
    episode with no overlays at all gets fresh ones drawn on its raw frames.
    """
    from policy_abstraction.visualization.frame_bundle import FrameBundle

    primary = next((c.name for c in env_spec.cameras if c.is_primary), None)
    if frame_bundle is None:
        images = {
            cam.name: np.ascontiguousarray(
                cam.apply_image_transform(obs[cam.image_key])
            )
            for cam in env_spec.cameras
            if cam.image_key in obs
        }
        frame_bundle = FrameBundle(
            images=images, primary_name=primary or "", chunk_overlays=dict(images)
        )
    for cam_name, img in list(frame_bundle.chunk_overlays.items()):
        if img is None:
            continue
        frame_bundle.chunk_overlays[cam_name] = projector.draw(
            np.ascontiguousarray(img), cam_name, color=UNGUIDED_OVERLAY_RGB
        )
    return frame_bundle


def _stop_unguided_policy(client, process) -> None:
    if client is not None:
        try:
            client.close()
        except Exception as exc:  # noqa: BLE001 — shutting down anyway
            logging.debug(f"Policy client close failed: {exc}")
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except Exception:  # noqa: BLE001
        process.kill()
    logging.info("Stopped the unguided policy server.")


def main(argv=None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    )

    episode_path = args.episode
    frame_period = 1.0 / args.fps if args.fps else None

    # ------------------------------------------------------------------
    # Load episode
    # ------------------------------------------------------------------
    with VLALoader(episode_path) as loader:
        task_desc = loader.task_description or Path(episode_path).stem
        logging.info(
            f"Loaded episode: {episode_path!r}  "
            f"({loader.length} steps, task={task_desc!r})"
        )
        logging.info(f"Features: {loader.feature_keys}")

        meta = loader.metadata

        # ------------------------------------------------------------------
        # Set up visualiser
        # ------------------------------------------------------------------
        # Robot model: CLI flag takes precedence, fall back to saved metadata.
        # Either source may name a file or a robot_descriptions entry, so the
        # two are told apart the same way for both: anything with a path
        # separator or an .xml suffix is a file, everything else is a package
        # name. Getting this wrong sends a name like "ur5e_mj_description" down
        # the file branch, where it is looked for under ~/.mujoco_menagerie and
        # quietly not found.
        requested = args.robot_model or meta.get("robot_model_name")
        robot_model_path = robot_description_name = None
        if requested:
            if "/" in requested or requested.endswith(".xml"):
                robot_model_path = requested
            else:
                robot_description_name = requested

        gripper_model_name = _resolve_gripper_model(
            args.gripper_model, loader.env_spec, robot_description_name
        )
        if gripper_model_name:
            logging.info(f"Gripper model: {gripper_model_name}")

        # Writing to a file and connecting to a viewer are mutually exclusive
        # sinks, so --save-rrd has to suppress both connecting and spawning.
        viz = RerunVisualizer(
            application_id=args.app_id,
            spawn=args.connect is None and not args.save_rrd,
            connect_addr=(
                None
                if args.save_rrd
                else (viewer_url(args.connect) if args.connect else None)
            ),
            save_path=args.save_rrd,
            log_images=not args.no_images,
            log_action_chunks=not args.no_action_chunks,
            log_predicted_states=False,  # no dynamics model during replay
            robot_model_path=robot_model_path,
            robot_description_name=robot_description_name,
            gripper_model_name=gripper_model_name,
        )
        # Wire the env's camera layout into the visualiser so per-camera
        # entity paths (live and reference) and the blueprint match the
        # rollout — supports any env (LIBERO, RoboCasa, …) with varying
        # camera counts. Falls back gracefully for episodes that pre-date
        # the env_spec metadata field.
        env_spec = loader.env_spec
        dynamics = _dynamics_model_for_episode(env_spec)
        control_hz = float(
            (getattr(env_spec, "metadata", None) or {}).get("control_hz") or 50.0
        )
        logging.info(f"Chunk states via {type(dynamics).__name__} at {control_hz:g} Hz")
        replan_steps = int(meta.get("replan_steps", 1) or 1)

        unguided_source = projector = None
        unguided_states = None
        if args.unguided:
            if env_spec is None:
                raise SystemExit(
                    "--unguided needs the episode's env_spec (old recording?)."
                )
            from policy_abstraction.visualization.chunk_projector import (
                ChunkProjector,
            )

            unguided_source = UnguidedChunkSource(
                connect=lambda: _connect_unguided_policy(
                    args, episode_path, env_spec, replan_steps
                ),
                env_spec=env_spec,
                task_description=task_desc,
                resize_size=int(meta.get("resize_size", 224) or 224),
                seed=args.seed,
                cache_path=args.unguided_cache,
            )
            # Chunk states from the dynamics model put the end-effector XYZ
            # first whatever the action space, so the slice is fixed here.
            projector = ChunkProjector(
                camera_height=int(meta.get("camera_height", 480) or 480),
                camera_width=int(meta.get("camera_width", 640) or 640),
                cameras=env_spec.cameras,
                eef_pos_slice=(0, 3),
                gripper_dim=None,
            )
        if env_spec is not None:
            viz.set_env_spec(env_spec)
            logging.info(
                f"Visualiser env_spec: cameras={[c.name for c in env_spec.cameras]}"
            )
        else:
            logging.warning(
                "No env_spec available (old recording without metadata). "
                "Wrist / ref-wrist panels may be empty."
            )
        # Show the backend that *produced* the recording as the primary
        # label so the viewer's Policy panel matches a live rollout. A
        # "(replay)" suffix keeps it obvious that this is not a live run.
        # Older episodes that pre-date the metadata field fall back to
        # "unknown".
        recorded_backend = meta.get("policy_backend") or "unknown"
        policy_label = f"{recorded_backend} (replay)"
        viz.setup(task_description=task_desc, policy_label=policy_label)

        # Set robot base pose from saved metadata so the model is placed
        # at the correct world-frame position from the very first frame.
        robot_base = loader.robot_base
        if robot_base["base_pos"] is not None and robot_base["base_ori"] is not None:
            viz.set_robot_base(robot_base["base_pos"], robot_base["base_ori"])
            logging.info(
                f"Robot base set: pos={robot_base['base_pos'].tolist()}, "
                f"ori={robot_base['base_ori'].tolist()}"
            )
        elif robot_model_path or robot_description_name:
            logging.warning(
                "Robot model requested but no base pose found in episode metadata "
                "(was the episode recorded with a recent version?)."
            )

        # Log episode-level retrieval info (query grid, ranked demos, VLM analysis)
        retrieval_data = loader.retrieval_data
        if retrieval_data:
            viz.log_retrieval_info(**retrieval_data)
            logging.info(f"Logged retrieval info: {list(retrieval_data.keys())}")

        # Log abstraction setup info from metadata (pair labels, ref traj,
        # equivalence mapping, constraints).
        if meta.get("abstraction_pair_labels"):
            pair_labels = meta["abstraction_pair_labels"]
            ref_traj_list = meta.get("abstraction_ref_traj")
            ref_traj = (
                np.asarray(ref_traj_list, dtype=np.float32) if ref_traj_list else None
            )
            setup_extra = loader.abstraction_setup_data
            viz.log_abstraction_setup(
                pair_labels=pair_labels,
                ref_traj=ref_traj,
                **setup_extra,
            )
            logging.info(
                f"Logged abstraction setup: {pair_labels} "
                f"(extras={list(setup_extra.keys())})"
            )

        if args.extra_objects:
            extra_clouds = _reconstruct_extra_objects(
                episode_path,
                list(args.extra_objects),
                detector_backend=args.detector_backend,
                detector_model=args.detector_model,
                cache_path=args.extra_objects_cache,
            )
            if extra_clouds:
                viz.log_static_object_pointclouds(extra_clouds)

        # Send blueprint after all static data is logged
        viz.send_blueprint()

        # ------------------------------------------------------------------
        # Replay loop
        # ------------------------------------------------------------------
        logging.info("Starting replay …")
        t0 = time.monotonic()

        replayed = 0
        for t, frame in loader.iter_frames():
            if args.max_frames is not None and replayed >= args.max_frames:
                logging.info(f"Stopping at --max-frames {args.max_frames}")
                break
            replayed += 1
            frame_start = time.monotonic()

            chunk_states = None
            if frame["action_chunks"] is not None and frame["robot_state"] is not None:
                chunk_states = dynamics.predict_states(
                    np.asarray(frame["action_chunks"]),
                    frame["robot_state"],
                    dt=1.0 / control_hz,
                )

            frame_bundle = frame.get("frame_bundle")
            unguided_states = None
            if unguided_source is not None and frame["robot_state"] is not None:
                if frame["action_chunks"] is not None:
                    # A replan step: the recorded observation is the one the
                    # guided inference saw, so ask the policy what it would
                    # have done with it unguided.
                    unguided_chunk = unguided_source.chunk_at(
                        t, frame["obs"], frame["robot_joints"]
                    )
                    if unguided_chunk is not None:
                        unguided_states = dynamics.predict_states(
                            unguided_chunk, frame["robot_state"], dt=1.0 / control_hz
                        )
                        projector.set_camera_params(loader.camera_params_at(t))
                        projector.update(unguided_states)
                frame_bundle = _overlay_unguided(
                    frame_bundle, frame["obs"], env_spec, projector
                )

            viz.update(
                timestep=t,
                obs=frame["obs"],
                robot_state=frame["robot_state"],
                robot_joints=frame["robot_joints"],
                action=frame["action"],
                action_chunks=frame["action_chunks"],
                action_chunk_states=chunk_states,
                unguided_chunk_states=unguided_states,
                reward=frame["reward"],
                done=frame["done"],
                abstraction_result=frame.get("abstraction_result"),
                frame_bundle=frame_bundle,
                guidance_info=frame.get("guidance_info"),
            )

            if frame_period is not None:
                elapsed = time.monotonic() - frame_start
                sleep_time = frame_period - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        total = time.monotonic() - t0
        logging.info(
            f"Replay complete: {replayed}/{loader.length} frames in "
            f"{total:.2f}s ({replayed / max(total, 1e-9):.1f} fps)"
        )

        if unguided_source is not None:
            unguided_source.close()
        viz.close()


if __name__ == "__main__":
    main()
