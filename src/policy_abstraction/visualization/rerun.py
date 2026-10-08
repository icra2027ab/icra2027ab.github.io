"""Visualizer components for the EpisodeRunner action loop.

A Visualizer consumes timestep data (observations, actions, robot state,
action chunks, …) and renders it for human inspection.

The default ``RerunVisualizer`` streams data to a `Rerun <https://rerun.io>`_
viewer in real time.  It optionally loads a robot URDF/MJCF and renders the
robot pose alongside the camera feeds, using either a
`mujoco_menagerie <https://github.com/google-deepmind/mujoco_menagerie>`_
asset path/name or a
`robot_descriptions <https://github.com/robot-descriptions/robot_descriptions.py>`_
package name.
"""

from __future__ import annotations

import base64
import colorsys
import hashlib
import io
import os
import pathlib
import sys
import time
from collections import deque
from typing import Any

import numpy as np
import rerun as rr
import rerun_loader_mjcf
from loguru import logger
from PIL import Image
from scipy.spatial.transform import Rotation as R

from policy_abstraction.visualization.base import BaseVisualizer
from policy_abstraction.visualization.constraint_panel import render_constraint_panel


class _VideoStreamLogger:
    """Per-entity H.264/H.265 video stream encoder for Rerun.

    Lazily initialises the encoder on the first frame so width/height are
    inferred from the input image. A FIFO of submitted timesteps is kept so
    encoded packets are emitted on the timeline at the step of the *original*
    frame (correct under encoder latency, since ``max_b_frames=0`` preserves
    submission order).
    """

    _FORMATS = {"h264": "h264", "h265": "hevc"}
    _ENCODERS = {"h264": "libx264", "h265": "libx265"}

    def __init__(
        self,
        entity: str,
        *,
        codec: str = "h264",
        fps: int = 30,
        timeline: str = "it",
    ) -> None:
        if codec not in self._FORMATS:
            raise ValueError(
                f"Unsupported video codec {codec!r}; expected one of {list(self._FORMATS)}"
            )
        self.entity = entity
        self.codec = codec
        self.fps = fps
        self.timeline = timeline
        self._container: Any = None
        self._stream: Any = None
        self._width: int | None = None
        self._height: int | None = None
        self._submitted_steps: deque[int] = deque()

    def _init_stream(self, width: int, height: int) -> None:
        import av

        codec_obj = rr.VideoCodec.H265 if self.codec == "h265" else rr.VideoCodec.H264
        self._container = av.open("/dev/null", "w", format=self._FORMATS[self.codec])
        stream = self._container.add_stream(self._ENCODERS[self.codec], rate=self.fps)
        assert isinstance(stream, av.video.stream.VideoStream)
        stream.width = width
        stream.height = height
        # B-frames produce out-of-order packets; rerun video streams don't
        # support them and we rely on FIFO order to map packets back to steps.
        stream.max_b_frames = 0
        self._stream = stream
        self._width = width
        self._height = height
        rr.log(self.entity, rr.VideoStream(codec=codec_obj), static=True)

    def log(self, img: np.ndarray, timestep: int) -> None:
        import av

        img = np.ascontiguousarray(img)
        h, w = img.shape[:2]
        # H.264/H.265 yuv420p require even dimensions; trim if needed.
        if h % 2 or w % 2:
            img = np.ascontiguousarray(img[: h - (h % 2), : w - (w % 2)])
            h, w = img.shape[:2]
        if self._stream is None:
            self._init_stream(w, h)
        elif self._width != w or self._height != h:
            logger.warning(
                f"[RerunVisualizer] Video stream {self.entity!r}: image size "
                f"({w}, {h}) differs from initialised ({self._width}, {self._height}); "
                "skipping frame."
            )
            return
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        self._submitted_steps.append(timestep)
        try:
            packets = list(self._stream.encode(frame))
        except Exception as exc:
            logger.warning(
                f"[RerunVisualizer] Encode failed for {self.entity!r}: {exc}"
            )
            self._submitted_steps.pop()
            return
        self._emit_packets(packets, current_timestep=timestep)

    def _emit_packets(self, packets, *, current_timestep: int) -> None:
        for packet in packets:
            if packet.pts is None:
                continue
            step = (
                self._submitted_steps.popleft()
                if self._submitted_steps
                else current_timestep
            )
            rr.set_time(self.timeline, sequence=step)
            rr.log(self.entity, rr.VideoStream.from_fields(sample=bytes(packet)))
        # Restore timeline cursor so subsequent logs land on the current step.
        rr.set_time(self.timeline, sequence=current_timestep)

    def close(self) -> None:
        if self._stream is None:
            return
        try:
            packets = list(self._stream.encode())
        except Exception as exc:
            logger.warning(
                f"[RerunVisualizer] Flush-encode failed for {self.entity!r}: {exc}"
            )
            packets = []
        for packet in packets:
            if packet.pts is None:
                continue
            step = self._submitted_steps.popleft() if self._submitted_steps else 0
            rr.set_time(self.timeline, sequence=step)
            rr.log(self.entity, rr.VideoStream.from_fields(sample=bytes(packet)))
        self._stream = None
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None


# ---------------------------------------------------------------------------
# Default implementation: rerun
# ---------------------------------------------------------------------------


def display_frame(
    name: str, tcp_pose: np.ndarray, color_alpha=1.0, axis_length: float = 0.1
):
    """Display the pose in Rerun as a coordinate frame."""
    # tcp_pose is typically [x, y, z, rx, ry, rz] or [x, y, z, qx, qy, qz, qw]

    position = tcp_pose[:3]
    ori = tcp_pose[3:]
    # Convert orientation to rotation matrix if needed
    if len(ori) == 3:
        # Assume ori is in axis-angle (rx, ry, rz)
        rotation = R.from_rotvec(ori)
    elif len(ori) == 4:
        # Assume ori is in quaternion (qx, qy, qz, qw)
        rotation = R.from_quat(ori)
    else:
        raise ValueError("Unsupported orientation format in tcp_pose.")
    # Get rotation matrix
    coord_frame = rotation.as_matrix()

    rr.log(
        f"{name}",
        rr.Transform3D(
            mat3x3=coord_frame,
            translation=position,
        ),
    )
    rr.log(
        f"{name}/axes",
        rr.Arrows3D(
            vectors=[[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]],
            colors=[
                [int(255 * color_alpha), 0, 0],
                [0, int(255 * color_alpha), 0],
                [0, 0, int(255 * color_alpha)],
            ],
        ),
    )


def _tcp_reachable(addr: str, timeout: float = 2.0) -> bool:
    """Whether something accepts TCP connections at a ``host:port`` address."""
    import socket

    host, _, port = addr.rpartition(":")
    if host in ("0.0.0.0", "", "[::]"):
        host = "127.0.0.1"
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def viewer_url(connect: str) -> str:
    """``host:port`` -> the gRPC proxy URL the Rerun SDK connects to.

    ``0.0.0.0`` is a *bind* address; as a connect target it is at best an
    accident that resolves to loopback. It is also the wrong thing to send
    over an SSH reverse tunnel, whose remote end binds ``127.0.0.1`` only
    (absent ``GatewayPorts``), so normalise it to loopback here rather than
    relying on the OS to be forgiving.
    """
    host, _, port = connect.rpartition(":")
    if host in ("0.0.0.0", "", "[::]", "::"):
        host = "127.0.0.1"
    return f"rerun+http://{host}:{port}/proxy"


def require_viewer(connect_addr: str) -> None:
    """Fail loudly when no viewer is listening at *connect_addr*.

    ``rr.connect_grpc`` does not raise, and its handshake is asynchronous, so
    logging into a viewer that was never started looks exactly like logging
    into one that works -- the run completes and the data is simply gone. A
    two-second TCP check up front is worth an entire rollout.
    """
    # Strip the "rerun+http://" scheme and the "/proxy" path back to host:port.
    addr = connect_addr.split("://", 1)[-1].split("/", 1)[0]
    if _tcp_reachable(addr):
        logger.info(f"[RerunVisualizer] Viewer reachable at {addr}")
        return
    port = addr.rpartition(":")[2] or "9876"
    raise RuntimeError(
        f"No Rerun viewer is listening at {addr}, so nothing would be "
        "recorded. Start one:\n"
        f"    rerun --port {port}\n"
        "(the `rerun` binary lives in this repo's .venv/bin, and is not "
        "normally on PATH)\n"
        "or pass --viz.connect None to spawn a viewer for the run, or "
        "--viz.no-enable to run without visualisation."
    )


class RerunVisualizer(BaseVisualizer):
    """Streams rollout data to the `Rerun <https://rerun.io>`_ viewer.

    Robot model loading
    -------------------
    Pass *exactly one* of the following constructor arguments to enable robot
    rendering:

    ``robot_model_path``
        Absolute path to a MuJoCo MJCF ``.xml`` file (e.g. from a local
        clone of *mujoco_menagerie*) **or** a *mujoco_menagerie* robot name
        such as ``"unitree_go2"`` (the asset is then resolved automatically
        from a local menagerie checkout at
        ``~/.mujoco_menagerie/<name>/<name>.xml``).

    ``robot_description_name``
        Name of a *robot_descriptions* package entry, e.g. ``"go2_mj_description"``.
        The URDF path is resolved via :pymod:`robot_descriptions`.

    Parameters
    ----------
    application_id:
        Rerun application ID (shown in the viewer title bar).
    spawn:
        If ``True`` (default), spawn a new Rerun viewer process automatically.
    connect_addr:
        If given, connect to a running ``rerun`` gRPC server instead of
        spawning one (e.g. ``"127.0.0.1:9876"``).
    save_path:
        If given, write the recording to this ``.rrd`` file instead of
        streaming it anywhere. Mutually exclusive with ``connect_addr`` and
        ``spawn`` — a stream has one sink — and the way to record on a machine
        with no display.
    robot_model_path:
        Path or mujoco_menagerie name for robot rendering (optional).
    robot_description_name:
        robot_descriptions package name for robot rendering (optional).
    gripper_model_name:
        Menagerie directory name of a gripper to bolt onto the arm's tool
        flange, e.g. ``"robotiq_2f140"`` (optional). Only meaningful together
        with ``robot_description_name``: menagerie's arms are published
        without end-effectors, so a UR5e otherwise renders as a bare flange.
        The gripper is posed from the observed aperture — see
        :mod:`policy_abstraction.visualization.robot_assembly`.
    log_images:
        Whether to log camera images.  Default ``True``.
    log_action_chunks:
        Whether to log the full action-chunk tensor.  Default ``True``.
    log_predicted_states:
        Whether to log the predicted state trajectory.  Default ``True``.
    log_action_chunk_states:
        Whether to log predicted states for each action chunk.  Default ``True``.
    use_video_stream:
        If ``True``, encode per-frame images as an H.264/H.265 video stream
        instead of logging individual JPEG frames.  Static images
        (retrieval grids, perception history, etc.) remain JPEG.  Default
        ``False``.
    video_codec:
        Codec for video streaming; ``"h264"`` (default) or ``"h265"``.
    video_fps:
        Frame rate hint for the video encoder.  Default ``30``.
    serve:
        Serve the recording from this process instead of connecting to or
        spawning a viewer: a gRPC server on ``serve_grpc_port`` and the web
        viewer on ``serve_web_port`` (both default to Rerun's usual ports).
        Nothing external is needed — not even the ``rerun`` binary — and
        :attr:`web_viewer_url` is a page any browser or embedded web view can
        show. This is how the demo window gets its 3-D panel. Wins over
        ``connect_addr`` and ``spawn``.
    layout:
        Which viewer layout :meth:`send_blueprint` installs: ``"full"`` (the
        analysis layout — 3-D scene, cameras, retrieval, charts) or
        ``"scene"`` (the 3-D scene alone, every panel collapsed — what a
        demo puts on the projector). Switchable at runtime with
        :meth:`set_layout`.
    """

    LAYOUTS = ("full", "scene")

    def __init__(
        self,
        application_id: str = "libero_eval",
        *,
        spawn: bool = False,
        connect_addr: str | None = None,
        save_path: str | None = None,
        robot_model_path: str | None = None,
        robot_description_name: str | None = None,
        gripper_model_name: str | None = None,
        gripper_mount_yaw_deg: float = 0.0,
        log_images: bool = True,
        log_action_chunks: bool = True,
        log_predicted_states: bool = True,
        log_action_chunk_states: bool = True,
        rgb_quality: int = 50,
        use_video_stream: bool = False,
        video_codec: str = "h264",
        video_fps: int = 30,
        layout: str = "full",
        serve: bool = False,
        serve_grpc_port: int | None = None,
        serve_web_port: int | None = None,
    ) -> None:
        if layout not in self.LAYOUTS:
            raise ValueError(f"layout must be one of {self.LAYOUTS}; got {layout!r}")
        self.layout = layout
        """Which blueprint :meth:`send_blueprint` sends — see :attr:`LAYOUTS`."""
        self._application_id = application_id
        self._run_id = str(int(time.time()))
        self._spawn = spawn
        self._connect_addr = connect_addr
        self._save_path = save_path
        self._serve = serve
        self._serve_grpc_port = serve_grpc_port
        self._serve_web_port = serve_web_port
        self._serve_url: str | None = None
        self.web_viewer_url: str | None = None
        """Where the served web viewer is, once :meth:`start` has run in
        ``serve`` mode: ``http://localhost:<web_port>/?url=<grpc url>``. A
        window embeds it; a browser opens it. ``None`` in every other mode."""
        self._boot_recording = None
        self._robot_model_path = robot_model_path
        self._robot_description_name = robot_description_name
        self._gripper_model_name = gripper_model_name
        self._gripper_mount_yaw_deg = float(gripper_mount_yaw_deg)
        self._log_images = log_images
        self._log_action_chunks = log_action_chunks
        self._log_predicted_states = log_predicted_states
        self._log_action_chunk_states = log_action_chunk_states
        self.rgb_quality = rgb_quality
        self._use_video_stream = use_video_stream
        self._video_codec = video_codec
        self._video_fps = video_fps
        self._video_streams: dict[str, _VideoStreamLogger] = {}

        self.first_time = True
        self._constraints: list = []  # SubtrajConstraint list, set during log_abstraction_setup

        self._mj = None  # mujoco module, imported lazily when needed
        self._mj_model = None  # mujoco.MjModel instance
        self._mj_data = None  # mujoco.MjData instance
        # Set when the arm model had a closed-chain gripper bolted on; carries
        # the solved aperture->qpos table and where the arm's own qpos ends.
        self._gripper_linkage = None
        self._arm_nq: int | None = None
        self._gripper_geometry = None
        self._aperture_clamped = False  # warn once, not every frame
        self.mjcf_logger: rerun_loader_mjcf.MJCFLogger | None = None
        self._urdf_path: str | None = None
        self._episode_count: int = 0  # tracks how many episodes have been set up
        self._base_pos: np.ndarray | None = None  # robot base position in world frame
        self._base_ori: np.ndarray | None = None  # robot base orientation (3x3 rot mat)
        # Root-body pose as authored in the MJCF, captured at load. It can
        # encode a frame convention (menagerie's UR5e yaws 180° to go from ROS
        # base_link to the controller's base), so the world base pose composes
        # onto it instead of replacing it.
        self._model_base_pose0: tuple[np.ndarray, np.ndarray] = (
            np.zeros(3),
            np.array([1.0, 0.0, 0.0, 0.0]),
        )
        self._ref_traj: np.ndarray | None = None  # [T_ref, n_pairs] reference distances
        self._ref_pair_labels: list[str] | None = None
        self._warned_joint_chunks = False
        self._pair_series_colors: dict[str, list[int]] = {}
        # The chunk currently being executed: [N, T, 3] positions and the
        # timestep it was issued at, so later steps can drop the executed part.
        # Chunk sets awaiting display, keyed by kind ("guided" is the chunk
        # the robot executes; "unguided" the counterfactual the policy would
        # have produced without guidance): [N, T, 3] positions and the
        # timestep they were issued at.
        self._chunk_sets: dict[str, tuple[np.ndarray, int]] = {}
        self._prev_aligned_ref_end: int = (
            0  # tracks last snippet end for clearing stale points
        )

        # Populated by ``set_env_spec`` so update() / send_blueprint can
        # iterate the live cameras instead of hardcoding LIBERO names.
        self._env_spec: Any = None

    def set_env_spec(self, env_spec: Any) -> None:
        """Tell the visualiser which cameras / robot the rollout uses.

        Called once by :class:`EpisodeRunner` at construction. Drives both
        ``update()`` (which entity paths to log per-frame) and
        ``send_blueprint`` (which space-views to register).
        """
        self._env_spec = env_spec

    def _chunks_are_joint_targets(self) -> bool:
        """Whether the env's actions are absolute joint targets (the UR5e)."""
        spec = self._env_spec
        if spec is None:
            return False
        layout = getattr(getattr(spec, "robot", None), "action_layout", None)
        return getattr(layout, "space", None) == "joint_absolute"

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the recording stream and its sink, once.

        Called by :meth:`setup` on the first episode, or earlier by a caller
        that needs the viewer before any episode exists — the demo window
        embeds the served web viewer, so its URL has to be known when the
        window is built. Idempotent.
        """
        if not self.first_time:
            return
        recording_id = f"{self._application_id}_{self._run_id}_boot"
        rr.init(self._application_id, recording_id=recording_id, spawn=False)
        self._open_sink(first=True)
        # Load robot model once
        if self._robot_model_path or self._robot_description_name:
            self._load_robot_model()
        self.first_time = False

    def _open_sink(self, *, first: bool) -> None:
        """Point the current recording at wherever this visualiser sends data."""
        if self._save_path:
            rr.save(self._save_path)
            if first:
                logger.info(f"[RerunVisualizer] Recording to {self._save_path}")
        elif self._serve:
            if first:
                import rerun.blueprint as rrb

                self._serve_url = rr.serve_grpc(
                    grpc_port=self._serve_grpc_port,
                    default_blueprint=self._blueprint(rrb),
                )
                # The server lives with the recording that created it; keep
                # that recording alive across the per-episode re-inits below.
                self._boot_recording = rr.get_global_data_recording()
                web_port = self._serve_web_port or 9090
                rr.serve_web_viewer(
                    web_port=web_port, open_browser=False, connect_to=self._serve_url
                )
                self.web_viewer_url = (
                    f"http://localhost:{web_port}/?url={self._serve_url}"
                )
                logger.info(
                    f"[RerunVisualizer] Serving {self._serve_url}; web viewer at "
                    f"{self.web_viewer_url}"
                )
            else:
                # A new recording joins the server this process already runs.
                assert self._serve_url is not None
                rr.connect_grpc(self._serve_url)
        elif self._connect_addr:
            if first:
                require_viewer(self._connect_addr)
            rr.connect_grpc(self._connect_addr)
            if first:
                logger.info(f"[RerunVisualizer] Connected to {self._connect_addr}")
        elif self._spawn:
            if first:
                # rr.spawn looks for the `rerun` binary on PATH; the SDK puts it
                # in the interpreter's bin dir, which a venv rarely exports.
                venv_bin = str(pathlib.Path(sys.executable).parent)
                if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
                    os.environ["PATH"] = (
                        venv_bin + os.pathsep + os.environ.get("PATH", "")
                    )
                rr.spawn()
                logger.info("[RerunVisualizer] Spawned Rerun viewer")
            else:
                rr.connect_grpc()

    def setup(
        self, task_description: str = "", *, policy_label: str | None = None
    ) -> None:
        """Start (or reset) the Rerun recording stream for one episode.

        The sink is opened only once (see :meth:`start`). Every call makes a
        new recording so the previous episode stays in the viewer as its own
        entry, re-logs the robot geometry into it, and logs the task text.

        Args:
            task_description: Human-readable task string shown in the UI.
            policy_label: Short label for the policy that is producing actions
                (e.g. ``"groot"``, ``"openpi"``, ``"replay"``). Logged as a
                static text document at ``"policy"`` and surfaced next to
                the Task panel in the blueprint.
        """

        # Flush any video stream encoders from the previous episode so the
        # new recording starts with a fresh codec stream.
        for encoder in self._video_streams.values():
            encoder.close()
        self._video_streams.clear()

        self.start()
        self._episode_count += 1
        recording_id = f"{self._application_id}_{self._run_id}_ep{self._episode_count}"
        # A new recording per episode, so the previous one is preserved in the
        # viewer as a separate recording.
        rr.init(self._application_id, recording_id=recording_id, spawn=False)
        self._open_sink(first=False)
        # Re-log the robot mesh into the new recording — log_model() only
        # writes to the active stream, so without this the second episode's
        # recording has joint poses but no geometry to render.
        if self.mjcf_logger is not None:
            if self._base_pos is not None:
                self._apply_base_to_model()
            self.mjcf_logger.log_model()
        logger.info(
            f"[RerunVisualizer] New recording for episode {self._episode_count}"
        )

        if task_description:
            rr.log("task", rr.TextDocument(task_description), static=True)
        if policy_label:
            rr.log("policy", rr.TextDocument(policy_label), static=True)

    # Gripper colour scheme matching ChunkProjector
    _COLOR_OPEN = np.array([200, 50, 50, 255], dtype=np.uint8)  # red  – fully open
    _COLOR_CLOSE = np.array([50, 200, 50, 255], dtype=np.uint8)  # green – fully closed
    _MAX_GS = 0.9
    _MIN_GS = -0.9

    @classmethod
    def _chunk_gripper_color(cls, gs: float) -> list[int]:
        """Interpolate open/close color by gripper state (matches ChunkProjector)."""
        span = cls._MAX_GS - cls._MIN_GS
        t = float(np.clip((float(gs) - cls._MIN_GS) / span, 0.0, 1.0))
        c = (cls._COLOR_CLOSE * (1 - t) + cls._COLOR_OPEN * t).astype(np.uint8)
        return c.tolist()

    def _log_image(
        self, entity: str, img: np.ndarray, *, timestep: int | None = None
    ) -> None:
        """Log an image, using video streaming if enabled.

        Falls back to compressed JPEG when streaming is disabled, when no
        timestep is supplied (e.g. static logs), or when the image isn't a
        3-channel uint8 array.
        """
        if (
            self._use_video_stream
            and timestep is not None
            and isinstance(img, np.ndarray)
            and img.ndim == 3
            and img.shape[2] == 3
        ):
            encoder = self._video_streams.get(entity)
            if encoder is None:
                encoder = _VideoStreamLogger(
                    entity, codec=self._video_codec, fps=self._video_fps
                )
                self._video_streams[entity] = encoder
            try:
                encoder.log(img.astype(np.uint8, copy=False), int(timestep))
                return
            except Exception as exc:
                logger.warning(
                    f"[RerunVisualizer] Video stream log failed for {entity!r}: {exc}; "
                    "falling back to JPEG."
                )
        rr.log(entity, rr.Image(img).compress(jpeg_quality=self.rgb_quality))

    def update(
        self,
        *,
        timestep: int,
        obs: dict[str, Any] | None = None,
        images: dict[str, np.ndarray] | None = None,
        encoded_obs: dict[str, Any] | None = None,
        robot_state: np.ndarray | None = None,
        robot_joints: np.ndarray | None = None,
        action: np.ndarray | None = None,
        action_chunks: np.ndarray | None = None,
        predicted_states: np.ndarray | None = None,
        action_chunk_states: np.ndarray | None = None,
        unguided_chunk_states: np.ndarray | None = None,
        reward: float | None = None,
        done: bool = False,
        camera_params: dict[str, dict] | None = None,
        abstraction_result=None,
        frame_bundle=None,
        guidance_info: dict | None = None,
        replan_steps: int | None = None,
    ) -> None:
        """Log one timestep of data to the Rerun stream."""

        rr.set_time("it", sequence=timestep)

        # --- Camera images ---------------------------------------------------
        # Iterate every camera in the env spec. ``images`` (pre-extracted,
        # already-upright frames keyed by camera name) is preferred; we fall
        # back to ``obs[cam.image_key]`` for legacy callers.
        if self._log_images and self._env_spec is not None:
            for cam in self._env_spec.cameras:
                entity = f"camera/{cam.name}"
                # Camera intrinsic / extrinsic
                cam_params = (camera_params or {}).get(cam.name)
                if cam_params is not None:
                    E = np.asarray(cam_params["extrinsic"], dtype=np.float64)
                    K = np.asarray(cam_params["intrinsic"], dtype=np.float64)
                    K3 = K[:3, :3] if K.shape == (4, 4) else K
                    rr.log(
                        entity,
                        rr.Transform3D(
                            translation=E[:3, 3],
                            mat3x3=E[:3, :3],
                        ),
                    )
                # Image: prefer the runner's pre-extracted dict, then obs.
                img_arr: np.ndarray | None = None
                if images is not None and cam.name in images:
                    img_arr = images[cam.name]
                elif obs is not None and cam.image_key in obs:
                    img_arr = np.ascontiguousarray(
                        cam.apply_image_transform(obs[cam.image_key])
                    )
                if img_arr is None:
                    continue
                if cam_params is not None:
                    h, w = img_arr.shape[:2]
                    rr.log(
                        entity,
                        rr.Pinhole(image_from_camera=K3, width=w, height=h),
                    )
                self._log_image(
                    entity, np.ascontiguousarray(img_arr), timestep=timestep
                )
            # Per-camera chunk overlays (any camera with one rendered).
            if frame_bundle is not None:
                overlays = getattr(frame_bundle, "chunk_overlays", None) or {}
                for cam_name, overlay_img in overlays.items():
                    if overlay_img is None:
                        continue
                    self._log_image(
                        f"camera/{cam_name}_overlay",
                        np.ascontiguousarray(overlay_img),
                        timestep=timestep,
                    )

        # --- Proprioceptive state --------------------------------------------
        if robot_state is not None:
            rr.log("robot/state", rr.Tensor(robot_state.astype(np.float32)))
            # log eef frame
            display_frame("robot/eef_frame", robot_state[:6])

        if robot_joints is not None:
            # Optionally render the robot model at the current state
            if self._mj_model is not None:
                self._render_robot(robot_joints)

        # --- Executed action --------------------------------------------------
        if action is not None:
            rr.log("robot/action", rr.Tensor(action.astype(np.float32)))

        # --- Action chunks 3D --------------------------------------------------
        if self._log_action_chunks:
            if action_chunk_states is not None or action_chunks is not None:
                self._remember_chunk(
                    timestep, action_chunk_states, action_chunks, robot_state
                )
            if unguided_chunk_states is not None:
                self._remember_chunk(
                    timestep, unguided_chunk_states, None, robot_state, kind="unguided"
                )
            self._draw_remaining_chunk(timestep)

        # --- Predicted states -------------------------------------------------
        if self._log_predicted_states and predicted_states is not None:
            rr.log(
                "dynamics/predicted_states",
                rr.Tensor(predicted_states.astype(np.float32)),
            )
            # Plot XYZ trajectory as a line strip (first 3 dims assumed to be pos)
            if predicted_states.shape[-1] >= 3:
                pts = predicted_states[:, :3]
                segments = [
                    [pts[i].tolist(), pts[i + 1].tolist()] for i in range(len(pts) - 1)
                ]
                colors = self._make_gradient_colors(len(segments))
                rr.log(
                    "dynamics/trajectory_3d",
                    rr.LineStrips3D(segments, colors=colors),
                )

        # --- Predicted action chunk states (legacy path — 3D logged above) ----
        # Kept for backwards compatibility; new code uses action_chunks block.

        # --- Abstraction / MTA -----------------------------------------------
        if abstraction_result is not None:
            for label, dist in abstraction_result.live_distances.items():
                rr.log(
                    f"abstraction/live/{label.replace(' ', '_')}",
                    rr.Scalars(float(dist)),
                )

            # Log aligned reference snippet: project a segment of ref_traj
            # starting from aligned_t onto the live timeline from the current
            # timestep.  The snippet length matches the replan interval so
            # consecutive snippets tile without gaps or overlapping discontinuities.
            aligned_t_val = int(abstraction_result.aligned_t)
            if (
                self._ref_traj is not None
                and self._ref_pair_labels is not None
                and replan_steps is not None
            ):
                T_ref = self._ref_traj.shape[0]
                snippet_len = min(T_ref - aligned_t_val, replan_steps)

                # Clear stale snippet points beyond the new snippet range
                new_end = timestep + snippet_len
                if new_end < self._prev_aligned_ref_end:
                    for stale_t in range(new_end, self._prev_aligned_ref_end):
                        rr.set_time("it", sequence=stale_t)
                        for lbl in self._ref_pair_labels:
                            rr.log(
                                f"abstraction/ref/{lbl.replace(' ', '_')}",
                                rr.clear(recursive=False),  # ty: ignore[unresolved-attribute]
                            )
                self._prev_aligned_ref_end = new_end

                for offset in range(snippet_len):
                    rr.set_time("it", sequence=timestep + offset)
                    for pair_idx, lbl in enumerate(self._ref_pair_labels):
                        val = float(self._ref_traj[aligned_t_val + offset, pair_idx])
                        if not np.isnan(val):
                            rr.log(
                                f"abstraction/ref/{lbl.replace(' ', '_')}",
                                rr.Scalars(val),
                            )

                # Tiny marker at the alignment point (ref value at aligned_t, plotted at current live time)
                rr.set_time("it", sequence=timestep)
                for pair_idx, lbl in enumerate(self._ref_pair_labels):
                    val = float(self._ref_traj[aligned_t_val, pair_idx])
                    if not np.isnan(val):
                        rr.log(
                            f"abstraction/ref_marker/{lbl.replace(' ', '_')}",
                            rr.Scalars(val),
                        )

                # Restore timeline to current timestep
                rr.set_time("it", sequence=timestep)

            rr.set_time("it", sequence=timestep)
            for label, err in abstraction_result.distance_errors.items():
                rr.log(
                    f"abstraction/error/{label.replace(' ', '_')}",
                    rr.Scalars(float(err)),
                )
            rr.log(
                "abstraction/aligned_t", rr.Scalars(float(abstraction_result.aligned_t))
            )
            rr.log(
                "abstraction/mta_progress",
                rr.Scalars(float(abstraction_result.mta_progress)),
            )
        # --- Frame bundle images ---------------------------------------------
        # Logged unconditionally (outside the abstraction_result gate) so that
        # ``--no-guide`` runs still surface the chunk-overlay agentview in the
        # "Agentview (annotated)" panel — when no abstractor is active, the
        # handler defaults ``live_image_annotated`` to the chunk overlay.
        if frame_bundle is not None:
            if frame_bundle.ref_image_annotated is not None:
                self._log_image(
                    "abstraction/ref_frame_annotated",
                    np.ascontiguousarray(frame_bundle.ref_image_annotated[:, ::-1]),
                    timestep=timestep,
                )
            # Per-non-primary-camera reference frames (typically the wrist,
            # but works for any auxiliary view the abstractor produced).
            ref_imgs = getattr(frame_bundle, "ref_images", None) or {}
            for cam_name, ref_img in ref_imgs.items():
                if cam_name == frame_bundle.primary_name or ref_img is None:
                    continue
                self._log_image(
                    f"abstraction/ref_{cam_name}_frame",
                    np.ascontiguousarray(ref_img[:, ::-1]),
                    timestep=timestep,
                )
            if frame_bundle.live_image_annotated is not None:
                self._log_image(
                    "abstraction/live_frame_annotated",
                    np.ascontiguousarray(frame_bundle.live_image_annotated[:, ::-1]),
                    timestep=timestep,
                )
            if frame_bundle.alignment_plot is not None:
                self._log_image(
                    "abstraction/alignment_plot",
                    frame_bundle.alignment_plot,
                    timestep=timestep,
                )

        # --- Constraint satisfaction ------------------------------------------
        if abstraction_result is not None and abstraction_result.constraint_costs:
            for key, cost in abstraction_result.constraint_costs.items():
                safe_key = key.replace(" ", "_")
                rr.log(f"constraints/{safe_key}/cost", rr.Scalars(float(cost)))
                rr.log(
                    f"constraints/{safe_key}/satisfied",
                    rr.Scalars(1.0 if cost <= 0.0 else 0.0),
                )
            panel = render_constraint_panel(
                abstraction_result.constraint_costs, self._constraints
            )
            if panel is not None:
                self._log_image("constraints/panel", panel, timestep=timestep)

        # --- Full-scene point cloud -------------------------------------------
        # The entire fused scene (table, background, every object), logged
        # separately from the per-object clouds below so it can be toggled
        # independently. Only present when the episode was recorded with
        # scene-pcd capture enabled.
        if abstraction_result is not None:
            scene_pcd = getattr(abstraction_result, "scene_pointcloud", None)
            if scene_pcd is not None and len(scene_pcd) > 0:
                scene_pcd = np.asarray(scene_pcd)
                xyz = scene_pcd[:, :3]
                colors = scene_pcd[:, 3:6].astype(np.uint8)
                rr.log(
                    "scene/full_pcd",
                    rr.Points3D(xyz, colors=colors, radii=0.0015),
                )

        # --- Object point clouds and centroids --------------------------------
        if abstraction_result is not None:
            if abstraction_result.object_pointclouds:
                for obj_name, obj_pcd in abstraction_result.object_pointclouds.items():
                    entity = f"scene/objects/{obj_name.replace(' ', '_')}"
                    xyz = obj_pcd[:, :3]
                    colors = obj_pcd[:, 3:6].astype(np.uint8)
                    rr.log(entity, rr.Points3D(xyz, colors=colors, radii=0.003))
            if abstraction_result.object_centroids:
                for obj_name, centroid in abstraction_result.object_centroids.items():
                    entity = f"scene/objects/{obj_name.replace(' ', '_')}/centroid"
                    rr.log(
                        entity,
                        rr.Points3D(
                            [centroid], radii=0.015, colors=[[255, 220, 0, 255]]
                        ),
                    )

        # --- Distance pairs as 3-D dimension lines ----------------------------
        if (
            abstraction_result is not None
            and abstraction_result.pairs
            and abstraction_result.entity_centroids
        ):
            self._log_distance_pairs(abstraction_result, robot_state)

        # --- Dynamic guidance metrics -----------------------------------------
        if guidance_info is not None:
            if "hl_mse" in guidance_info:
                rr.log("guidance/hl_mse", rr.Scalars(float(guidance_info["hl_mse"])))
            if "hl_factor" in guidance_info:
                rr.log(
                    "guidance/hl_factor", rr.Scalars(float(guidance_info["hl_factor"]))
                )
            if "ll_mse" in guidance_info:
                rr.log("guidance/ll_mse", rr.Scalars(float(guidance_info["ll_mse"])))
            if "ll_factor" in guidance_info:
                rr.log(
                    "guidance/ll_factor", rr.Scalars(float(guidance_info["ll_factor"]))
                )
            if "total_hl_relevance" in guidance_info:
                rr.log(
                    "guidance/total_hl_relevance",
                    rr.Scalars(float(guidance_info["total_hl_relevance"])),
                )
            if "guidance_factor" in guidance_info:
                rr.log(
                    "guidance/guidance_factor",
                    rr.Scalars(float(guidance_info["guidance_factor"])),
                )

        # --- Perception (predicted-seg) artefacts -----------------------------
        # Per-replan detection overlays + per-stage history grids from
        # OpenPerception. Surfaces the same content the example notebooks
        # render to disk via render_segmentation_video / render_history_gif.
        if abstraction_result is not None and self._log_images:
            view_names = (
                tuple(c.name for c in self._env_spec.cameras)
                if self._env_spec is not None
                else ("agentview", "wrist")
            )
            overlays = abstraction_result.perception_overlays
            if overlays:
                for v, img in enumerate(overlays):
                    name = view_names[v] if v < len(view_names) else f"view{v}"
                    self._log_image(
                        f"perception/{name}/overlay",
                        img,
                        timestep=timestep,
                    )
            grids = abstraction_result.perception_history_grids
            if grids:
                # Static log: the client emits stage-history grids only on the
                # first detect() call per episode, and we want them visible at
                # every timeline cursor position from then on.
                for v, img in enumerate(grids):
                    name = view_names[v] if v < len(view_names) else f"view{v}"
                    rr.log(
                        f"perception/{name}/history",
                        rr.Image(img),
                        static=True,
                    )
            debug_imgs = abstraction_result.perception_debug_images
            if debug_imgs:
                # Static — render_debug_images only fires on the first detect
                # frame (gated by ``debug_perception``); pin it across the
                # timeline so it stays visible while scrubbing.
                for v, img in enumerate(debug_imgs):
                    name = view_names[v] if v < len(view_names) else f"view{v}"
                    rr.log(
                        f"perception/{name}/debug",
                        rr.Image(img),
                        static=True,
                    )
            if abstraction_result.perception_timings_md:
                rr.log(
                    "perception/timings",
                    rr.TextDocument(
                        abstraction_result.perception_timings_md,
                        media_type="text/markdown",
                    ),
                )

        # --- Reward / done ----------------------------------------------------
        if reward is not None:
            rr.log("episode/reward", rr.Scalars(float(reward)))
        rr.log("episode/done", rr.Scalars(float(done)))

    def log_abstraction_setup(
        self,
        live_scene_image: np.ndarray | None = None,
        live_seg_image: np.ndarray | None = None,
        mapping=None,
        ref_traj: np.ndarray | None = None,
        pair_labels: list[str] | None = None,
        constraints=None,
        equivalence_conversation: dict | None = None,
        **_kwargs: Any,
    ) -> None:
        """Log episode-level abstraction setup info as static Rerun entities.

        Should be called once after :meth:`RolloutAbstractor.setup` succeeds.
        Static entities persist across the full episode timeline.

        Parameters
        ----------
        live_scene_image:
            uint8 RGB first live frame (upright).
        live_seg_image:
            uint8 RGB segmentation overlay for the live scene.
        mapping:
            :class:`~policy_abstraction.vlm.equivalence_mapper.ObjectMapping`
            with the VLM-determined object equivalences.
        ref_traj:
            ``[T_ref, n_pairs]`` float32 reference distance trajectory.
        pair_labels:
            Ordered list of pair label strings.
        """
        import json as _json

        if live_scene_image is not None:
            rr.log(
                "abstraction/live_scene",
                rr.Image(live_scene_image[:, ::-1]),
                static=True,
            )

        if live_seg_image is not None:
            rr.log(
                "abstraction/live_seg_overlay",
                rr.Image(live_seg_image[:, ::-1]),
                static=True,
            )

        if mapping is not None:
            mapping_dict = {
                "retrieved_objects": {
                    str(k): v for k, v in mapping.retrieved_objects.items()
                },
                "live_objects": {str(k): v for k, v in mapping.live_objects.items()},
                "id_map": {str(k): str(v) for k, v in mapping.id_map.items()},
                "unmapped_retrieved_ids": mapping.unmapped_retrieved_ids,
            }
            rr.log(
                "abstraction/object_mapping",
                rr.TextDocument(
                    _json.dumps(mapping_dict, indent=2),
                    media_type="application/json",
                ),
                static=True,
            )

            # -- Resolved mapping summary (compact view kept below the conversation) --
            equiv_md_parts: list[str] = [
                "## Equivalence Mapping (VLM)\n",
                f"### Resolved Mapping\n```json\n{_json.dumps(mapping_dict, indent=2)}\n```",
            ]
            rr.log(
                "vlm/equivalence_mapping",
                rr.TextDocument(
                    "\n".join(equiv_md_parts),
                    media_type="text/markdown",
                ),
                static=True,
            )

        # -- VLM equivalence full conversation (system + user text + images + assistant) --
        # Mirrors the retrieval/vlm_conversation rendering so that data: URI
        # images embed correctly inside the markdown TextDocument.
        equiv_conv_dict = equivalence_conversation
        if equiv_conv_dict is None and mapping is not None:
            conv_obj = getattr(mapping, "conversation", None)
            if conv_obj is not None:
                msgs = list(getattr(conv_obj, "messages", []) or [])
                user_msg = msgs[1] if len(msgs) > 1 else None
                user_imgs_b64: list[str] = []
                if user_msg is not None:
                    for img_arr in getattr(user_msg, "images", []) or []:
                        try:
                            pil_img = Image.fromarray(
                                np.asarray(img_arr).astype(np.uint8)
                            )
                            buf = io.BytesIO()
                            pil_img.save(buf, format="PNG")
                            user_imgs_b64.append(
                                base64.b64encode(buf.getvalue()).decode("utf-8")
                            )
                        except Exception:
                            continue
                equiv_conv_dict = {
                    "model": getattr(conv_obj, "model", "") or "",
                    "system_prompt": msgs[0].text if msgs else "",
                    "user_text": user_msg.text if user_msg is not None else "",
                    "user_images_b64": user_imgs_b64,
                    "assistant_response": msgs[2].text if len(msgs) > 2 else "",
                }

        if equiv_conv_dict:
            system_prompt = equiv_conv_dict.get("system_prompt", "") or ""
            user_text = equiv_conv_dict.get("user_text", "") or ""
            user_images_b64: list[str] = list(
                equiv_conv_dict.get("user_images_b64", []) or []
            )
            assistant = equiv_conv_dict.get("assistant_response", "") or ""
            model_name = equiv_conv_dict.get("model", "") or ""

            user_images_md = "".join(
                f"\n\n![equiv_image_{i}](data:image/png;base64,{b64})"
                for i, b64 in enumerate(user_images_b64)
            )
            convo_md = (
                f"*Model: {model_name}*\n\n"
                f"## System\n{system_prompt}\n\n"
                f"## User\n{user_text}{user_images_md}\n\n"
                f"## Assistant\n```json\n{assistant}\n```"
            )
            rr.log(
                "vlm/equivalence_conversation",
                rr.TextDocument(convo_md, media_type="text/markdown"),
                static=True,
            )

        if ref_traj is not None and pair_labels is not None:
            self._ref_traj = ref_traj.astype(np.float32)
            self._ref_pair_labels = list(pair_labels)
            self._prev_aligned_ref_end = 0
            rr.log(
                "abstraction/ref_traj",
                rr.Tensor(self._ref_traj),
                static=True,
            )
            # Match live/ref curve colors per label; keep ref curves darker.
            # Only "ee_to_soup_xy" series are visible by default.
            _DEFAULT_VISIBLE_LABEL = "ee_to_soup_xy"
            for label in pair_labels:
                safe_label = label.replace(" ", "_")
                live_entity = f"abstraction/live/{safe_label}"
                ref_entity = f"abstraction/ref_demo/{safe_label}"
                visible = _DEFAULT_VISIBLE_LABEL in label

                base_color = self._series_color_for_label(label)
                ref_color = self._darken_rgba(base_color)

                rr.log(
                    live_entity,
                    rr.SeriesLines(
                        colors=base_color,
                        widths=2.0,
                        names=label,
                        visible_series=[visible],
                    ),
                    static=True,
                )
                rr.log(
                    ref_entity,
                    rr.SeriesLines(
                        colors=ref_color,
                        widths=2.0,
                        names=f"{label} (ref)",
                        visible_series=[visible],
                    ),
                    static=True,
                )

                # Aligned-ref series: same hue as live but semi-transparent & thinner
                aligned_ref_entity = f"abstraction/ref/{safe_label}"
                aligned_ref_color = [base_color[0], base_color[1], base_color[2], 120]
                rr.log(
                    aligned_ref_entity,
                    rr.SeriesLines(
                        colors=aligned_ref_color,
                        widths=1.2,
                        names=f"{label} (aligned ref)",
                        visible_series=[visible],
                    ),
                    static=True,
                )

                # Tiny marker at the alignment point on the ref snippet
                marker_entity = f"abstraction/ref_marker/{safe_label}"
                rr.log(
                    marker_entity,
                    rr.SeriesPoints(
                        colors=aligned_ref_color,
                        marker_sizes=4.0,
                        names=f"{label} (align pt)",
                        visible_series=[visible],
                    ),
                    static=True,
                )

            # Log each reference pair channel on a separate ref_t timeline
            for pair_idx, label in enumerate(pair_labels):
                for it, val in enumerate(ref_traj[:, pair_idx]):
                    rr.set_time("it", sequence=it)
                    rr.log(
                        f"abstraction/ref_demo/{label.replace(' ', '_')}",
                        rr.Scalars(float(val)),
                    )
            rr.reset_time()

        if constraints:
            self._constraints = list(constraints)
            import json as _json

            # Log constraint metadata as a static text document
            rr.log(
                "constraints/metadata",
                rr.TextDocument(
                    _json.dumps(
                        [
                            {
                                "kind": c.kind,
                                "subtraj_idx": c.subtraj_idx,
                                "label": c.label,
                                "description": c.description,
                                "pairs_affected": c.pairs_affected,
                                "reasoning": c.reasoning,
                            }
                            for c in constraints
                        ],
                        indent=2,
                    ),
                    media_type="application/json",
                ),
                static=True,
            )
            # Register series colors: sub-goal = purple, path = orange
            _sg_color = [150, 80, 220, 255]
            _pc_color = [220, 140, 30, 255]
            for c in constraints:
                prefix = "sg" if c.kind == "sub_goal" else "pc"
                color = _sg_color if c.kind == "sub_goal" else _pc_color
                safe_label = c.label.replace(" ", "_")
                rr.log(
                    f"constraints/{prefix}/{safe_label}/cost",
                    rr.SeriesLines(colors=color, widths=2.0, names=f"{c.label} cost"),
                    static=True,
                )
                rr.log(
                    f"constraints/{prefix}/{safe_label}/satisfied",
                    rr.SeriesLines(
                        colors=[0, 200, 80, 255],
                        widths=1.5,
                        names=f"{c.label} satisfied",
                    ),
                    static=True,
                )

    # ------------------------------------------------------------------
    # Multi-demo distance statistics visualization
    # ------------------------------------------------------------------

    @staticmethod
    def plot_multi_demo_distances(stats: dict) -> np.ndarray | None:
        """Render per-pair distance statistics as a matplotlib figure.

        Creates one subplot per distance pair showing:
        - Solid line: top-1 reference trajectory
        - Shaded band: mean ± std across all aligned demos
        - Dashed lines: min / max bounds

        Parameters
        ----------
        stats:
            Dict returned by
            :func:`~policy_abstraction.evaluation.retrieval.compute_multi_demo_stats`
            with keys ``ref_traj``, ``mean``, ``std``, ``min``, ``max``,
            ``pair_labels``, ``n_demos``.

        Returns
        -------
        np.ndarray or None
            ``[H, W, 3]`` uint8 RGB image of the rendered figure, or ``None``
            if rendering fails.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            ref_traj = stats["ref_traj"]  # [T, n_pairs]
            mean = stats["mean"]  # [T, n_pairs]
            std = stats["std"]  # [T, n_pairs]
            mn = stats["min"]  # [T, n_pairs]
            mx = stats["max"]  # [T, n_pairs]
            labels = stats["pair_labels"]
            n_demos = stats["n_demos"]
            n_pairs = len(labels)
            T = ref_traj.shape[0]
            t = np.arange(T)

            ncols = min(n_pairs, 3)
            nrows = int(np.ceil(n_pairs / ncols))
            fig_w = max(6, ncols * 4)
            fig_h = max(3, nrows * 3)
            fig, axes = plt.subplots(
                nrows, ncols, figsize=(fig_w, fig_h), squeeze=False
            )
            fig.suptitle(
                f"Distance pairs — {n_demos} demos (DTW-aligned to top-1)",
                fontsize=10,
            )

            for idx in range(n_pairs):
                row, col = divmod(idx, ncols)
                ax = axes[row][col]
                lbl = labels[idx]

                ref_col = ref_traj[:, idx]
                mean_col = mean[:, idx]
                std_col = std[:, idx]
                mn_col = mn[:, idx]
                mx_col = mx[:, idx]

                # min / max range
                ax.fill_between(
                    t, mn_col, mx_col, alpha=0.15, color="steelblue", label="min–max"
                )
                # mean ± std
                ax.fill_between(
                    t,
                    mean_col - std_col,
                    mean_col + std_col,
                    alpha=0.35,
                    color="steelblue",
                    label="mean ± std",
                )
                ax.plot(t, mean_col, color="steelblue", linewidth=1.5, label="mean")
                # min / max dashed borders
                ax.plot(t, mn_col, "--", color="steelblue", linewidth=0.8, alpha=0.6)
                ax.plot(t, mx_col, "--", color="steelblue", linewidth=0.8, alpha=0.6)
                # top-1 reference
                ax.plot(
                    t, ref_col, color="darkorange", linewidth=1.8, label="top-1 ref"
                )

                ax.set_title(lbl, fontsize=8)
                ax.set_xlabel("timestep", fontsize=7)
                ax.set_ylabel("distance (m)", fontsize=7)
                ax.tick_params(labelsize=6)
                ax.legend(fontsize=6, loc="best")

            # Hide unused subplots
            for idx in range(n_pairs, nrows * ncols):
                row, col = divmod(idx, ncols)
                axes[row][col].set_visible(False)

            plt.tight_layout()

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)
            buf.seek(0)
            img_arr = np.array(Image.open(buf).convert("RGB"))
            return img_arr

        except Exception as exc:
            logger.warning(f"[RerunVisualizer] plot_multi_demo_distances failed: {exc}")
            return None

    def log_retrieval_info(
        self,
        query_grid: np.ndarray | None = None,
        seg_annotated_image: np.ndarray | None = None,
        retrieval_results: list[dict] | None = None,
        vlm_analysis: dict | None = None,
        vlm_scene_image: np.ndarray | None = None,
        vlm_subtraj_image: np.ndarray | None = None,
        vlm_conversation: dict | None = None,
        multi_demo_stats: dict | None = None,
        iterative_analysis: dict | None = None,
        **_kwargs: Any,
    ) -> None:
        """Log episode-level retrieval info as static Rerun entities.

        Should be called once after :meth:`setup`, before the replay loop.
        All entities are logged as ``static=True`` so they persist across the
        entire timeline without being tied to a specific timestep.

        Parameters
        ----------
        query_grid:
            uint8 RGB image of the retrieval figure (query + top-k demos).
        seg_annotated_image:
            uint8 RGB segmentation overlay from VLM demo analysis.
        retrieval_results:
            List of ranked retrieved-demo dicts (keys: rank, task, score, …).
        vlm_analysis:
            VLM analysis dict (keys: retrieved_task, objects, distance_pairs, …).
        vlm_scene_image:
            uint8 RGB raw scene image from the retrieved demo (sent to VLM).
        vlm_subtraj_image:
            uint8 RGB scene image with EE sub-trajectory paths overlaid (sent to VLM).
        vlm_conversation:
            Dict with keys ``model``, ``system_prompt``, ``user_text``,
            ``assistant_response`` from the pair-selection VLM call.
        """
        import json as _json

        if query_grid is not None:
            rr.log(
                "retrieval/query_grid",
                rr.Image(query_grid),
                static=True,
            )
            logger.info("[RerunVisualizer] Logged retrieval query grid.")

        if seg_annotated_image is not None:
            rr.log(
                "retrieval/seg_annotated_image",
                rr.Image(seg_annotated_image),
                static=True,
            )
            logger.info("[RerunVisualizer] Logged segmentation annotated image.")

        if retrieval_results:
            lines = ["# Retrieved Demos\n"]
            for r in retrieval_results:
                lines.append(
                    f"**#{r['rank']}** {r['task']}  \n"
                    f"score={r['score']:.3f} "
                    f"(visual={r.get('visual_score', 0):.3f}, "
                    f"text={r.get('text_score', 0):.3f})  \n"
                    f"`{r.get('demo_key', '')}` — "
                    f"[{r.get('source_kind', '?')}] "
                    f"`{r.get('source_path', '')}`\n"
                )
            rr.log(
                "retrieval/results",
                rr.TextDocument("\n".join(lines), media_type="text/markdown"),
                static=True,
            )
            logger.info(
                f"[RerunVisualizer] Logged {len(retrieval_results)} retrieval results."
            )

        if vlm_analysis:
            rr.log(
                "retrieval/vlm_analysis",
                rr.TextDocument(
                    _json.dumps(vlm_analysis, indent=2), media_type="application/json"
                ),
                static=True,
            )
            logger.info("[RerunVisualizer] Logged VLM analysis.")

        if vlm_scene_image is not None:
            rr.log(
                "retrieval/vlm_scene_image",
                rr.Image(vlm_scene_image),
                static=True,
            )
            logger.info("[RerunVisualizer] Logged VLM scene image.")

        if vlm_subtraj_image is not None:
            rr.log(
                "retrieval/vlm_subtraj_image",
                rr.Image(vlm_subtraj_image),
                static=True,
            )
            logger.info("[RerunVisualizer] Logged VLM sub-trajectory image.")

        if vlm_conversation:
            system_prompt = vlm_conversation.get("system_prompt", "")
            user_text = vlm_conversation.get("user_text", "")
            user_images_b64: list[str] = vlm_conversation.get("user_images_b64", [])
            assistant = vlm_conversation.get("assistant_response", "")
            model = vlm_conversation.get("model", "")
            cache_file = vlm_conversation.get("cache_file")
            if cache_file:
                source_banner = f"> **🟢 CACHED** — loaded from: `{cache_file}`\n\n"
            else:
                source_banner = "> **🔴 NEW** — live VLM query\n\n"
            user_images_md = "".join(
                f"\n\n![image_{i}](data:image/png;base64,{b64})"
                for i, b64 in enumerate(user_images_b64)
            )
            md = (
                f"{source_banner}"
                f"*Model: {model}*\n\n"
                f"## System\n{system_prompt}\n\n"
                f"## User\n{user_text}{user_images_md}\n\n"
                f"## Assistant\n```json\n{assistant}\n```"
            )
            rr.log(
                "retrieval/vlm_conversation",
                rr.TextDocument(md, media_type="text/markdown"),
                static=True,
            )
            logger.info("[RerunVisualizer] Logged VLM pair-selection conversation.")

        if multi_demo_stats is not None:
            img = self.plot_multi_demo_distances(multi_demo_stats)
            if img is not None:
                rr.log(
                    "retrieval/multi_demo_distances",
                    rr.Image(img),
                    static=True,
                )
                logger.info(
                    f"[RerunVisualizer] Logged multi-demo distance stats "
                    f"({multi_demo_stats.get('n_demos', '?')} demos)."
                )

        if iterative_analysis is not None:
            self._log_iterative_analysis(iterative_analysis)

    def _log_iterative_analysis(self, payload: dict) -> None:
        """Log the iterative-strategy iteration history under ``retrieval/iterative/...``.

        Renders, all ``static=True``:
        - per-iteration proposal text (Markdown)
        - per-iteration segment-score bar chart
        - per-iteration per-demo segmentation strip
        - aggregate-score-vs-iteration line plot
        - selected-iteration banner
        """
        import json as _json

        from policy_abstraction.visualization.plots import (
            plot_consistency_over_iterations,
            plot_demo_segmentation_strips,
            plot_segment_consistency,
        )

        iterations = payload.get("iterations") or []
        selected = payload.get("selected_iteration")
        boundaries = payload.get("selected_subtraj_boundaries")

        aggregates: list[float] = []
        for entry in iterations:
            agg = entry.get("score", {}).get("aggregate")
            aggregates.append(float(agg) if agg is not None else float("nan"))

        if aggregates:
            try:
                img = plot_consistency_over_iterations(aggregates, selected=selected)
                rr.log("retrieval/iterative/aggregate", rr.Image(img), static=True)
            except Exception as exc:
                logger.debug(f"[RerunVisualizer] aggregate plot failed: {exc}")

        for entry in iterations:
            i = entry.get("iteration", 0)
            score = entry.get("score", {})
            per_seg = score.get("per_segment", []) or []
            demo_breakdown = entry.get("demo_breakdown", []) or []

            # Proposal as a Markdown block.
            proposal_lines = [f"# Iteration {i}"]
            if entry.get("selected"):
                proposal_lines.append("> **selected**")
            agg = score.get("aggregate")
            metric = score.get("metric", "?")
            proposal_lines.append(
                f"- aggregate ({metric}): {agg if agg is not None else 'n/a':.4f}"
                if isinstance(agg, (int, float))
                else f"- aggregate ({metric}): n/a"
            )
            constraints = entry.get("proposal", {}).get("constraints", [])
            pairs = entry.get("proposal", {}).get("pairs", [])
            if constraints:
                proposal_lines.append("\n## Constraints")
                for c in constraints:
                    proposal_lines.append(
                        f"- **{c.get('label')}** ({c.get('kind')}, "
                        f"st{c.get('subtraj_idx')}): `{c.get('expression', '')}`"
                    )
            if pairs:
                proposal_lines.append("\n## Pairs")
                for p in pairs:
                    proposal_lines.append(
                        f"- **{p.get('label')}** "
                        f"({p.get('from_entity')} → {p.get('to_entity')}, "
                        f"{p.get('distance_type')}, st={p.get('subtraj_idx')})"
                    )
            rr.log(
                f"retrieval/iterative/iter_{i}/proposal",
                rr.TextDocument("\n".join(proposal_lines), media_type="text/markdown"),
                static=True,
            )

            try:
                img = plot_segment_consistency(per_seg)
                rr.log(
                    f"retrieval/iterative/iter_{i}/scores",
                    rr.Image(img),
                    static=True,
                )
            except Exception as exc:
                logger.debug(f"[RerunVisualizer] iter {i} score plot failed: {exc}")

            try:
                n_subtrajs = (
                    max((s.get("subtraj_idx", 0) for s in per_seg), default=-1) + 1
                )
                img = plot_demo_segmentation_strips(demo_breakdown, n_subtrajs)
                rr.log(
                    f"retrieval/iterative/iter_{i}/segmentation",
                    rr.Image(img),
                    static=True,
                )
            except Exception as exc:
                logger.debug(f"[RerunVisualizer] iter {i} strip plot failed: {exc}")

        if selected is not None:
            banner = [f"# Selected iteration: {selected}"]
            if boundaries is not None:
                banner.append(
                    f"\nVerified segment boundaries: `{_json.dumps(boundaries)}`"
                )
            rr.log(
                "retrieval/iterative/selected",
                rr.TextDocument("\n".join(banner), media_type="text/markdown"),
                static=True,
            )
        logger.info(
            f"[RerunVisualizer] Logged {len(iterations)} iterative-analysis iteration(s)."
        )

    def _build_perception_tab(self, rrb):
        """Construct the Perception tab dynamically from the env spec."""
        cam_names = (
            tuple(c.name for c in self._env_spec.cameras)
            if self._env_spec is not None
            else ("agentview", "wrist")
        )
        overlay_views = [
            rrb.Spatial2DView(name=f"{n} Overlay", origin=f"perception/{n}/overlay")
            for n in cam_names
        ]
        history_views = [
            rrb.Spatial2DView(
                name=f"{n} Stage History", origin=f"perception/{n}/history"
            )
            for n in cam_names
        ]
        debug_views = [
            rrb.Spatial2DView(name=f"{n} Debug Panel", origin=f"perception/{n}/debug")
            for n in cam_names
        ]
        return rrb.Vertical(
            name="Perception",
            contents=[
                rrb.Horizontal(*overlay_views),
                *history_views,
                *debug_views,
                rrb.TextDocumentView(name="Stage Timings", origin="perception/timings"),
            ],
        )

    def set_layout(self, layout: str) -> None:
        """Switch the viewer to *layout* now — see :attr:`LAYOUTS`.

        Safe to call from another thread while a rollout logs: a blueprint is
        one more message on the recording stream, and the viewer applies it
        without touching the data already there.
        """
        if layout not in self.LAYOUTS:
            raise ValueError(f"layout must be one of {self.LAYOUTS}; got {layout!r}")
        self.layout = layout
        if not self.first_time:
            self.send_blueprint()

    SCENE_BACKGROUND = (70, 73, 90)
    """The 3-D view's background: dark enough to sit in the demo window's
    theme, and deliberately not black — the cell's cables and bag are black,
    and a black background would swallow their point clouds."""

    def _scene_view(self, rrb):
        """The 3-D scene view both layouts share."""
        return rrb.Spatial3DView(
            name="3D Scene",
            origin="/",
            background=list(self.SCENE_BACKGROUND),
            # The EE frame triad is clutter next to the distance
            # pairs; it stays in the tree for whoever wants it.
            overrides={
                "robot/eef_frame": rrb.EntityBehavior(visible=False),
                "robot/eef_frame/axes": rrb.EntityBehavior(visible=False),
            },
        )

    def _scene_blueprint(self, rrb):
        """The 3-D scene and nothing else: no time panel, no entity tree, no
        selection panel. The robot model, the object clouds, the distance
        pairs and the action chunks fill the window."""
        return rrb.Blueprint(
            self._scene_view(rrb),
            collapse_panels=True,
        )

    def _blueprint(self, rrb):
        """The blueprint for :attr:`layout` (see :meth:`send_blueprint`)."""
        if self.layout == "scene":
            return self._scene_blueprint(rrb)
        return self._full_blueprint(rrb)

    def send_blueprint(self) -> None:
        """Send the preferred Rerun UI blueprint to the viewer.

        Which one depends on :attr:`layout`: ``"scene"`` sends the 3-D scene
        alone (:meth:`_scene_blueprint`); ``"full"`` sends the analysis layout
        described below.

        Layout
        ------
        * **Left (large)**: 3D scene view with a thin task-text strip at top,
          and the alignment plot beneath it.
        * **Right**: tabs for the live cameras, the retrieval artefacts, the
          VLM conversations, and the perception overlays.
        * **Right bottom (Live tab)**: five time-series charts — the live and
          MTA-aligned reference distances, their per-pair error, constraint
          costs, the guidance scalars, and MTA progress.

        Every entity this class logs is reachable from some view here. That is
        worth keeping true: an entity with no view is logged, recorded, and
        invisible, which reads in the viewer exactly like a value that was
        never computed.
        """
        try:
            import rerun.blueprint as rrb
        except ImportError:
            logger.warning(
                "[RerunVisualizer] rerun.blueprint not available; skipping blueprint."
            )
            return

        rr.send_blueprint(self._blueprint(rrb))

    def _full_blueprint(self, rrb):
        """The analysis layout described in :meth:`send_blueprint`."""
        # Camera-row content is built dynamically from the env spec so any
        # number of cameras with arbitrary names lights up. The primary
        # camera is replaced by the abstraction-annotated live frame when
        # available; the wrist camera (when present) maps to its overlay
        # tile so the chunk projection is visible by default.
        cam_row_views: list = []
        if self._env_spec is not None:
            primary_name = self._env_spec.primary_camera.name
            for cam in self._env_spec.cameras:
                if cam.name == primary_name:
                    cam_row_views.append(
                        rrb.Spatial2DView(
                            name=f"{cam.name} (live, annotated)",
                            origin="abstraction/live_frame_annotated",
                        )
                    )
                    cam_row_views.append(
                        rrb.Spatial2DView(
                            name="Retrieved Frame",
                            origin="abstraction/ref_frame_annotated",
                        )
                    )
                else:
                    overlay_origin = f"camera/{cam.name}_overlay"
                    raw_origin = f"camera/{cam.name}"
                    cam_row_views.append(
                        rrb.Spatial2DView(
                            name=f"{cam.name}",
                            origin=(
                                overlay_origin if cam.role == "wrist" else raw_origin
                            ),
                        )
                    )
                    cam_row_views.append(
                        rrb.Spatial2DView(
                            name=f"Ref {cam.name}",
                            origin=f"abstraction/ref_{cam.name}_frame",
                        )
                    )
        else:
            # Fallback: legacy 2-pane layout when no spec is available.
            cam_row_views = [
                rrb.Spatial2DView(
                    name="Agentview (annotated)",
                    origin="abstraction/live_frame_annotated",
                ),
                rrb.Spatial2DView(
                    name="Retrieved Frame",
                    origin="abstraction/ref_frame_annotated",
                ),
                rrb.Spatial2DView(name="Wrist", origin="camera/wrist_overlay"),
                rrb.Spatial2DView(
                    name="Ref Wrist", origin="abstraction/ref_wrist_frame"
                ),
            ]

        # Pair the views into rows of two for a compact grid.
        cam_grid_rows: list = []
        for i in range(0, len(cam_row_views), 2):
            cam_grid_rows.append(rrb.Horizontal(*cam_row_views[i : i + 2]))

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                # ---- Left column: task+policy (thin) + 3D scene (large) ----
                rrb.Vertical(
                    rrb.Horizontal(
                        rrb.TextDocumentView(name="Task", origin="task"),
                        rrb.TextDocumentView(name="Policy", origin="policy"),
                        column_shares=[3, 1],
                    ),
                    self._scene_view(rrb),
                    rrb.Spatial2DView(
                        name="Alignment Plot",
                        origin="abstraction/alignment_plot",
                    ),
                    row_shares=[2, 14, 6],
                ),
                # ---- Right column: tabs (live view | retrieval | VLM) ----
                rrb.Tabs(
                    # Tab 1: live cameras + progress charts
                    rrb.Vertical(
                        name="Live",
                        contents=[
                            rrb.Vertical(
                                *cam_grid_rows,
                                rrb.Spatial2DView(
                                    name="Constraints Panel", origin="constraints/panel"
                                ),
                            ),
                            # Charts: the measured-vs-reference distances the
                            # whole method turns on, their error, then the
                            # guidance and MTA scalars derived from them.
                            rrb.Vertical(
                                rrb.TimeSeriesView(
                                    name="Distances (live vs ref)",
                                    contents=[
                                        "abstraction/live/**",
                                        "abstraction/ref/**",
                                        "abstraction/ref_marker/**",
                                    ],
                                ),
                                rrb.TimeSeriesView(
                                    name="Distance Errors",
                                    contents=["abstraction/error/**"],
                                ),
                                rrb.TimeSeriesView(
                                    name="Constraint Costs",
                                    contents=["constraints/**"],
                                ),
                                rrb.TimeSeriesView(
                                    name="Guidance Values",
                                    contents=["guidance/**"],
                                ),
                                rrb.TimeSeriesView(
                                    name="MTA Progress",
                                    contents=[
                                        "abstraction/aligned_t",
                                        "abstraction/mta_progress",
                                        "episode/reward",
                                    ],
                                ),
                            ),
                        ],
                        row_shares=[18, 10],
                    ),
                    # Tab 2: retrieval images + results text
                    rrb.Vertical(
                        name="Retrieval",
                        contents=[
                            rrb.Horizontal(
                                rrb.Spatial2DView(
                                    name="Query Grid", origin="retrieval/query_grid"
                                ),
                                rrb.Spatial2DView(
                                    name="Seg Annotated",
                                    origin="retrieval/seg_annotated_image",
                                ),
                            ),
                            rrb.Horizontal(
                                rrb.Spatial2DView(
                                    name="VLM Scene", origin="retrieval/vlm_scene_image"
                                ),
                                rrb.Spatial2DView(
                                    name="VLM Sub-Traj",
                                    origin="retrieval/vlm_subtraj_image",
                                ),
                            ),
                            rrb.Spatial2DView(
                                name="Multi-Demo Distances (DTW)",
                                origin="retrieval/multi_demo_distances",
                            ),
                            rrb.TextDocumentView(
                                name="Retrieval Results", origin="retrieval/results"
                            ),
                        ],
                        row_shares=[7, 7, 8, 4],
                    ),
                    # Tab 3: iterative demo-analysis history. Only populated
                    # under --demo-analysis.strategy iterative; the views are
                    # registered regardless so the artefacts are not logged
                    # into a viewer that has nowhere to show them.
                    rrb.Vertical(
                        name="Iterative",
                        contents=[
                            rrb.Spatial2DView(
                                name="Aggregate vs Iteration",
                                origin="retrieval/iterative/aggregate",
                            ),
                            rrb.TextDocumentView(
                                name="Selected Iteration",
                                origin="retrieval/iterative/selected",
                            ),
                            rrb.Tabs(
                                name="Per-Iteration",
                                contents=[
                                    rrb.Spatial2DView(
                                        name="Scores",
                                        origin="retrieval/iterative",
                                        contents=["retrieval/iterative/**/scores"],
                                    ),
                                    rrb.Spatial2DView(
                                        name="Segmentation",
                                        origin="retrieval/iterative",
                                        contents=[
                                            "retrieval/iterative/**/segmentation"
                                        ],
                                    ),
                                    rrb.TextDocumentView(
                                        name="Proposals",
                                        origin="retrieval/iterative",
                                        contents=["retrieval/iterative/**/proposal"],
                                    ),
                                ],
                            ),
                        ],
                        row_shares=[5, 2, 8],
                    ),
                    # Tab 3: VLM conversation + analysis + equivalence mapping
                    rrb.Vertical(
                        name="VLM",
                        contents=[
                            rrb.TextDocumentView(
                                name="VLM Conversation",
                                origin="retrieval/vlm_conversation",
                            ),
                            rrb.TextDocumentView(
                                name="VLM Analysis", origin="retrieval/vlm_analysis"
                            ),
                            rrb.TextDocumentView(
                                name="Equivalence Conversation",
                                origin="vlm/equivalence_conversation",
                            ),
                            rrb.TextDocumentView(
                                name="Equivalence Mapping",
                                origin="vlm/equivalence_mapping",
                            ),
                        ],
                        row_shares=[3, 1, 3, 1],
                    ),
                    # Tab 4: open_perception detection / tracking artefacts
                    # (only populated under PerceptionMode.REAL_PERCEPTION).
                    # One overlay/history/debug strip per camera in the env
                    # spec.
                    self._build_perception_tab(rrb),
                ),
                column_shares=[3, 2],
            ),
            rrb.TimePanel(timeline="it"),
            collapse_panels=False,
        )
        return blueprint
        logger.info("[RerunVisualizer] Blueprint sent.")

    def close(self) -> None:
        """Flush the Rerun stream."""
        logger.info("[RerunVisualizer] Closing (stream will flush on exit).")
        # Drain any remaining video-stream encoder buffers before flushing
        # the rerun stream so trailing frames make it into the recording.
        for encoder in self._video_streams.values():
            encoder.close()
        self._video_streams.clear()
        # rerun-sdk flushes automatically when the process exits; explicit
        # flushing is only needed with a background thread.
        try:
            rr.flush()  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        except Exception:
            pass  # best-effort

    # ------------------------------------------------------------------
    # Static object point clouds
    # ------------------------------------------------------------------

    def log_static_object_pointclouds(self, clouds: dict[str, np.ndarray]) -> None:
        """Objects that stay put for the whole episode, e.g. the distractors
        of a disambiguation run that the rollout never tracked.

        Drawn under ``scene/objects/<name>`` in the same style as the tracked
        objects but logged once, statically, so they show at every timestep.
        ``clouds`` maps a name to an ``[M, 7]`` ``[x, y, z, r, g, b, id]``
        array; only the first six columns are used.
        """
        for name, pcd in clouds.items():
            pcd = np.asarray(pcd)
            if pcd.ndim != 2 or pcd.shape[0] == 0 or pcd.shape[1] < 6:
                continue
            entity = f"scene/objects/{name.replace(' ', '_')}"
            rr.log(
                entity,
                rr.Points3D(
                    pcd[:, :3], colors=pcd[:, 3:6].astype(np.uint8), radii=0.003
                ),
                static=True,
            )
            logger.info(
                f"[RerunVisualizer] Logged static cloud {name!r} ({len(pcd)} pts)"
            )

    # ------------------------------------------------------------------
    # Action chunk
    # ------------------------------------------------------------------

    _CHUNK_COLOR = [255, 220, 0, 255]  # bright yellow
    _CHUNK_COLOR_ALT = [255, 236, 130, 160]  # the other sampled chunks
    # The unguided counterfactual chunk: magenta, so it reads against the
    # yellow executed chunk and the red/blue sub-trajectory palette alike.
    _UNGUIDED_CHUNK_COLOR = [255, 60, 220, 255]
    _UNGUIDED_CHUNK_COLOR_ALT = [255, 160, 235, 160]
    _CHUNK_STYLES = {
        "guided": ("action_chunks", _CHUNK_COLOR, _CHUNK_COLOR_ALT),
        "unguided": (
            "action_chunks/unguided",
            _UNGUIDED_CHUNK_COLOR,
            _UNGUIDED_CHUNK_COLOR_ALT,
        ),
    }

    def _remember_chunk(
        self,
        timestep: int,
        action_chunk_states: np.ndarray | None,
        action_chunks: np.ndarray | None,
        robot_state: np.ndarray | None,
        *,
        kind: str = "guided",
    ) -> None:
        """Turn a freshly issued chunk into end-effector positions and keep it.

        Prefers pre-computed states (forward kinematics for a joint-target
        robot); otherwise integrates Cartesian delta actions from the current
        end-effector position. Absolute joint targets cannot be integrated
        that way — summing radians onto the gripper position walks the drawn
        chunk metres off the table — so those are dropped with a warning
        unless the caller ran them through a dynamics model.
        """
        if action_chunk_states is not None:
            states = np.asarray(action_chunk_states, dtype=np.float32)
            if states.ndim == 2:
                states = states[np.newaxis]  # [1, T, D]
            positions = states[:, :, :3]
        elif self._chunks_are_joint_targets() or robot_state is None:
            if not self._warned_joint_chunks:
                logger.warning(
                    "[RerunVisualizer] action_chunks are absolute joint "
                    "targets but no action_chunk_states were given; not "
                    "drawing the chunk. Pass states from a dynamics model."
                )
                self._warned_joint_chunks = True
            return
        else:
            chunks = np.asarray(action_chunks, dtype=np.float32)
            if chunks.ndim == 2:
                chunks = chunks[np.newaxis]
            N, T, _ = chunks.shape
            positions = np.zeros((N, T + 1, 3), dtype=np.float32)
            positions[:, 0, :] = robot_state[:3]
            for _t in range(T):
                positions[:, _t + 1, :] = positions[:, _t, :] + chunks[:, _t, :3] * (
                    1.0 / 50
                )
        if kind not in self._CHUNK_STYLES:
            raise ValueError(
                f"Unknown chunk kind {kind!r}; expected one of "
                f"{sorted(self._CHUNK_STYLES)}."
            )
        self._chunk_sets[kind] = (positions, int(timestep))

    def _draw_remaining_chunk(self, timestep: int) -> None:
        """Draw the part of each current chunk that is still ahead of the robot.

        Each step executed since a chunk was issued drops one point from its
        start, so the line always runs from the gripper forward and never
        trails behind it. The executed chunk is bright yellow under
        ``action_chunks/``; the unguided counterfactual, when one was given,
        is magenta under ``action_chunks/unguided/``. Within a set the first
        sampled chunk is solid and any others pale and translucent.
        """
        for kind, (positions, t0) in self._chunk_sets.items():
            root, color, color_alt = self._CHUNK_STYLES[kind]
            elapsed = max(int(timestep) - t0, 0)
            remaining = positions[:, elapsed:, :]
            for chunk_idx in range(remaining.shape[0]):
                path = f"{root}/chunk_{chunk_idx}"
                pts = remaining[chunk_idx]
                if len(pts) < 2:
                    rr.log(path, rr.Clear(recursive=False))
                    continue
                rr.log(
                    path,
                    rr.LineStrips3D(
                        [pts.tolist()],
                        colors=[color if chunk_idx == 0 else color_alt],
                    ),
                )

    # ------------------------------------------------------------------
    # Distance pairs as dimension lines
    # ------------------------------------------------------------------

    _DIM_RADIUS = 0.0015

    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        return v / max(float(np.linalg.norm(v)), 1e-9)

    def _log_dimension_line(
        self, path: str, a: np.ndarray, b: np.ndarray, *, color: list[int]
    ) -> None:
        """Draw ``a -> b`` as a dimension line in the pair's own colour: a
        thin line with an arrowhead at each end. The colour is the same one
        the pair's trace has in the distance panels, so no text is needed to
        tell the pairs apart."""
        delta = b - a
        if float(np.linalg.norm(delta)) < 1e-6:
            return
        mid = a + 0.5 * delta
        rr.log(
            path,
            rr.LineStrips3D(
                [[a.tolist(), b.tolist()]], colors=[color], radii=[self._DIM_RADIUS]
            ),
        )
        # Double-headed arrow: two arrows from the midpoint out to the ends.
        rr.log(
            f"{path}/arrows",
            rr.Arrows3D(
                origins=[mid.tolist(), mid.tolist()],
                vectors=[(a - mid).tolist(), (b - mid).tolist()],
                colors=[color] * 2,
                radii=self._DIM_RADIUS,
            ),
        )

    @staticmethod
    def _dashed(
        a: np.ndarray, b: np.ndarray, dash: float = 0.008, space: float = 0.005
    ) -> list[list[list[float]]]:
        """Split ``a -> b`` into dashes; Rerun has no native dashed lines."""
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            return []
        d = delta / length
        strips = []
        pos = 0.0
        while pos < length:
            end = min(pos + dash, length)
            strips.append([(a + d * pos).tolist(), (a + d * end).tolist()])
            pos = end + space
        return strips

    def _pair_colors(self, pairs: list) -> dict[str, list[int]]:
        """One RGBA per pair label — :func:`drawing.pair_colors` with alpha,
        so the 3-D lines and every 2-D view agree on a pair's colour."""
        from policy_abstraction.visualization.drawing import pair_colors

        return {
            label: [rgb[0], rgb[1], rgb[2], 255]
            for label, rgb in pair_colors(pairs).items()
        }

    def _sync_series_colors(self, colors: dict[str, list[int]]) -> None:
        """Recolour the live / reference distance traces to the pair colours
        used in the 3-D scene, so a line in the scene and its plot match."""
        for label, color in colors.items():
            if self._pair_series_colors.get(label) == color:
                continue
            self._pair_series_colors[label] = color
            safe_label = label.replace(" ", "_")
            rr.log(
                f"abstraction/live/{safe_label}",
                rr.SeriesLines(colors=color),
                static=True,
            )
            rr.log(
                f"abstraction/ref_demo/{safe_label}",
                rr.SeriesLines(colors=self._darken_rgba(color)),
                static=True,
            )

    def _log_distance_pairs(self, abstraction_result: Any, robot_state) -> None:
        """Render every distance pair of ``abstraction_result`` in 3-D."""
        from policy_abstraction.vlm.types import DistanceType

        ec = abstraction_result.entity_centroids
        colors = self._pair_colors(abstraction_result.pairs)
        self._sync_series_colors(colors)

        ee_R: np.ndarray | None = None
        if robot_state is not None and len(robot_state) >= 6:
            ee_R = R.from_rotvec(
                np.asarray(robot_state[3:6], dtype=np.float64)
            ).as_matrix()

        for pair in abstraction_result.pairs:
            from_pos = ec.get(pair.from_entity)
            to_pos = ec.get(pair.to_entity)
            if from_pos is None or to_pos is None:
                continue
            from_pos = np.asarray(from_pos, dtype=np.float64)
            to_pos = np.asarray(to_pos, dtype=np.float64)

            color = colors[pair.label]
            safe_label = pair.label.replace(" ", "_")
            path = f"scene/distance_pairs/{safe_label}"

            if pair.distance_type == DistanceType.ORIENTATION:
                self._log_orientation_pair(path, pair, from_pos, ee_R, color)
                continue

            delta = to_pos - from_pos
            far = to_pos
            residual: np.ndarray | None = None
            if pair.axis is not None:
                ax = self._unit(pair.axis)
                far = from_pos + float(np.dot(delta, ax)) * ax
                residual = to_pos
            elif pair.plane is not None:
                n = self._unit(pair.plane)
                far = from_pos + (delta - float(np.dot(delta, n)) * n)
                residual = to_pos

            self._log_dimension_line(path, from_pos, far, color=color)
            if residual is not None:
                self._log_residual(path, far, residual, color)

    def _log_residual(
        self, path: str, far: np.ndarray, to: np.ndarray, color: list[int]
    ) -> None:
        """The component a projected pair does not measure: a dashed support
        line in the pair's colour, so the reading still visibly connects to
        the entity without competing with the measurement itself."""
        strips = self._dashed(far, to)
        if not strips:
            return
        rr.log(
            f"{path}/residual",
            rr.LineStrips3D(
                strips,
                colors=[[color[0], color[1], color[2], 150]] * len(strips),
                radii=[0.001] * len(strips),
            ),
        )

    @staticmethod
    def _dashed_arc(
        origin: np.ndarray,
        axis: np.ndarray,
        start_dir: np.ndarray,
        radius: float,
        half_span_deg: float = 90.0,
        dash_deg: float = 5.0,
        space_deg: float = 4.0,
    ) -> list[list[list[float]]]:
        """Dashes along a circular arc about ``axis`` through ``origin``,
        symmetric about ``start_dir`` and spanning ``2 * half_span_deg``."""
        strips = []
        a = -half_span_deg
        while a < half_span_deg:
            b = min(a + dash_deg, half_span_deg)
            angles = np.radians(np.linspace(a, b, 3))
            pts = [
                (origin + R.from_rotvec(axis * t).apply(start_dir) * radius).tolist()
                for t in angles
            ]
            strips.append(pts)
            a = b + space_deg
        return strips

    _ARC_RADIUS = 0.04  # metres from the EE at which the angle is drawn
    _ARC_MIN_DEG = 0.5  # below this the pair is aligned and nothing is drawn

    def _log_orientation_pair(
        self,
        path: str,
        pair: Any,
        origin: np.ndarray,
        ee_R: np.ndarray | None,
        color: list[int],
    ) -> None:
        """An orientation pair as the angle it still has to close.

        Only the gap is drawn: an arc around the EE, about the axis that
        rotates the reference direction onto the tracked gripper vector,
        sweeping from the reference to the gripper with an arrowhead at the
        gripper end. It is in the pair's colour like every other pair, and it
        shrinks to nothing as the gripper turns into alignment.
        """
        if ee_R is None or pair.axis is None:
            return
        ref_dir = self._unit(pair.axis)
        live_dir = ee_R @ self._unit(pair.ee_vector or [0.0, 0.0, 1.0])
        axis = np.cross(ref_dir, live_dir)
        sin_a = float(np.linalg.norm(axis))
        cos_a = float(np.clip(np.dot(ref_dir, live_dir), -1.0, 1.0))
        angle = float(np.arctan2(sin_a, cos_a))
        aligned = np.degrees(angle) < self._ARC_MIN_DEG
        if aligned:
            # No rotation to speak of, so no axis either; pick any direction
            # perpendicular to the reference to lay the protractor in.
            for hint in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])):
                axis = np.cross(ref_dir, hint)
                if np.linalg.norm(axis) > 1e-6:
                    break
            axis = self._unit(axis)
        else:
            axis = axis / sin_a

        # A dashed 180-degree protractor, centred on the reference direction
        # and in the plane the gap rotates in, so a small gap still reads as
        # a fraction of something.
        protractor = self._dashed_arc(origin, axis, ref_dir, self._ARC_RADIUS)
        rr.log(
            f"{path}/protractor",
            rr.LineStrips3D(
                protractor,
                colors=[[color[0], color[1], color[2], 120]] * len(protractor),
                radii=[0.001] * len(protractor),
            ),
        )
        if aligned:
            for sub in ("", "/support", "/arrows"):
                rr.log(f"{path}{sub}", rr.Clear(recursive=False))
            return
        n_seg = max(4, int(np.degrees(angle) / 3.0))
        sweep = np.linspace(0.0, angle, n_seg + 1)
        dirs = np.stack([R.from_rotvec(axis * a).apply(ref_dir) for a in sweep])
        arc = origin + dirs * self._ARC_RADIUS
        rr.log(
            path,
            rr.LineStrips3D([arc.tolist()], colors=[color], radii=[0.002]),
        )
        # Dashed support lines from the rotation axis (through the EE) out to
        # the two tips of the arc, so the angle reads as a wedge.
        support = self._dashed(origin, arc[0]) + self._dashed(origin, arc[-1])
        rr.log(
            f"{path}/support",
            rr.LineStrips3D(
                support,
                colors=[[color[0], color[1], color[2], 150]] * len(support),
                radii=[0.001] * len(support),
            ),
        )
        # Arrowhead at the gripper end, along the arc's tangent there.
        tangent = self._unit(np.cross(axis, dirs[-1]))
        head = 0.012
        rr.log(
            f"{path}/arrows",
            rr.Arrows3D(
                origins=[(arc[-1] - tangent * head).tolist()],
                vectors=[(tangent * head).tolist()],
                colors=[color],
                radii=0.002,
            ),
        )

    def set_robot_base(
        self,
        base_pos: np.ndarray,
        base_ori: np.ndarray,
    ) -> None:
        """Apply the LIBERO robot base pose to the MuJoCo model.

        Patches the root body (body index 1) of the loaded MuJoCo model so
        that all subsequent forward-kinematics calls produce positions in the
        correct world frame, matching ``env.robots[0].base_pos/base_ori``.

        Args:
            base_pos: Robot base position ``[3]`` (world frame).
            base_ori: Robot base orientation ``[3, 3]`` rotation matrix
                      (world frame).
        """
        self._base_pos = np.asarray(base_pos, dtype=np.float64)
        self._base_ori = np.asarray(base_ori, dtype=np.float64)
        if self._mj_model is not None:
            self._apply_base_to_model()

    def _apply_base_to_model(self) -> None:
        """Place the model's root body (body 1) at the stored base pose.

        The base pose is *composed* with the root pose the model was authored
        with, not substituted for it. That authored pose can carry a frame
        convention: menagerie's UR5e declares ``quat="0 0 0 -1"`` — a 180°
        rotation about z — because the mesh is built in the ROS ``base_link``
        frame while the arm is driven in the controller's ``base`` frame.
        Overwriting it renders the robot facing backwards, which looks like a
        joint-order bug and isn't one. Panda's authored quat is the identity,
        so composing changes nothing for the simulator envs.
        """
        if self._mj_model is None or self._base_pos is None or self._base_ori is None:
            return
        if self._base_ori.shape == (3, 3):
            rot = R.from_matrix(self._base_ori)
        elif self._base_ori.shape == (4,):
            rot = R.from_quat(self._base_ori)  # assume it's already in xyzw
        else:
            raise ValueError("Unsupported base_ori shape for robot base orientation.")

        # body 0 = worldbody (fixed); body 1 = robot root link.
        pos0, quat0 = self._model_base_pose0
        wxyz = quat0
        authored = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])  # wxyz → xyzw

        self._mj_model.body_pos[1] = self._base_pos + rot.apply(pos0)
        xyzw = (rot * authored).as_quat()  # scipy gives xyzw
        self._mj_model.body_quat[1] = np.array(
            [xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64
        )
        logger.info(
            f"[RerunVisualizer] Robot base applied: pos={self._base_pos.tolist()} "
            f"(composed with the model's authored root quat {np.round(quat0, 3).tolist()})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_gradient_colors(
        n_segments: int,
        start_color: tuple[int, int, int, int] = (0, 0, 255, 255),  # blue  → beginning
        end_color: tuple[int, int, int, int] = (255, 0, 0, 255),  # red   → end
    ) -> list[list[int]]:
        """Return a list of RGBA colours interpolated from *start_color* to
        *end_color* over *n_segments* steps (one colour per segment)."""
        t = np.linspace(0.0, 1.0, max(n_segments, 1))
        return [
            [int(start_color[c] * (1.0 - ti) + end_color[c] * ti) for c in range(4)]
            for ti in t
        ]

    @staticmethod
    def _series_color_for_label(label: str) -> list[int]:
        """Return a deterministic, high-saturation RGBA color for a label."""
        digest = hashlib.md5(label.encode("utf-8")).digest()
        hue = digest[0] / 255.0
        sat = 0.72 + 0.18 * (digest[1] / 255.0)
        val = 0.88 + 0.10 * (digest[2] / 255.0)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        return [int(255 * r), int(255 * g), int(255 * b), 255]

    @staticmethod
    def _darken_rgba(color: list[int], factor: float = 0.55) -> list[int]:
        """Darken an RGBA color by scaling RGB channels while preserving alpha."""
        rgb = [int(np.clip(c * factor, 0, 255)) for c in color[:3]]
        alpha = int(color[3]) if len(color) > 3 else 255
        return [*rgb, alpha]

    # ------------------------------------------------------------------
    # Robot model helpers
    # ------------------------------------------------------------------

    def _resolve_model_path(self) -> str | None:
        """Return the MJCF/URDF file path when ``robot_model_path`` is set.

        Returns ``None`` if no file-path-based model is configured (e.g. when
        ``robot_description_name`` is used instead, which yields a model
        directly without an accessible path).
        """
        import os

        if self._robot_model_path:
            path = self._robot_model_path
            # Check if it's a plain name (not a file path)
            if not os.path.exists(path):
                # Treat as mujoco_menagerie robot name
                menagerie_root = os.path.expanduser("~/.mujoco_menagerie")
                candidate = os.path.join(menagerie_root, path, f"{path}.xml")
                if os.path.exists(candidate):
                    logger.info(
                        f"[RerunVisualizer] Resolved menagerie model: {candidate}"
                    )
                    return candidate
                else:
                    logger.warning(
                        f"[RerunVisualizer] Could not find mujoco_menagerie model '{path}' "
                        f"at {candidate}.  Expected a clone at {menagerie_root}."
                    )
                    return None
            return path

        return None

    def _load_robot_model(self) -> None:
        """Load MuJoCo model for robot rendering."""
        try:
            import mujoco

        except ImportError:
            logger.warning(
                "[RerunVisualizer] mujoco package not found; robot rendering disabled. "
                "Install with: pip install mujoco"
            )
            return

        try:
            self._mj = mujoco

            if self._robot_description_name and self._gripper_model_name:
                # Menagerie publishes arms without end-effectors, so the
                # gripper is a separate model grafted onto the tool flange.
                from policy_abstraction.visualization.robot_assembly import (
                    compose_arm_with_gripper,
                )

                assembled = compose_arm_with_gripper(
                    self._robot_description_name,
                    self._gripper_model_name,
                    mount_yaw_deg=self._gripper_mount_yaw_deg,
                )
                self._mj_model = assembled.model
                self._mj_data = mujoco.MjData(self._mj_model)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
                self._gripper_linkage = assembled.linkage
                self._arm_nq = assembled.arm_nq
                self._gripper_geometry = self._resolve_gripper_geometry()
            elif self._robot_description_name:
                # robot_descriptions mujoco loader returns a MjModel directly.
                import robot_descriptions.loaders.mujoco as mj_loader

                self._mj_model = mj_loader.load_robot_description(
                    self._robot_description_name
                )
                self._mj_data = mujoco.MjData(self._mj_model)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
                logger.info(
                    f"[RerunVisualizer] Loaded robot_descriptions model "
                    f"'{self._robot_description_name}' via mujoco loader"
                )
            else:
                path = self._resolve_model_path()
                if path is None:
                    return
                self._mj_model = mujoco.MjModel.from_xml_path(path)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
                self._mj_data = mujoco.MjData(self._mj_model)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
                logger.info(f"[RerunVisualizer] Loaded MuJoCo model from {path}")

            # Snapshot the root pose the model was authored with, before
            # anything patches it — `_apply_base_to_model` composes onto this
            # rather than replacing it, and is called once per episode.
            self._model_base_pose0 = (
                self._mj_model.body_pos[1].copy(),
                self._mj_model.body_quat[1].copy(),
            )

            self.mjcf_logger = rerun_loader_mjcf.MJCFLogger(
                self._mj_model, entity_path_prefix="robot/model"
            )

            # Apply base pose before logging so initial geometry is at the
            # correct world position.
            if self._base_pos is not None:
                self._apply_base_to_model()

            self.mjcf_logger.log_model()

        except Exception as exc:
            logger.warning(f"[RerunVisualizer] Failed to load MuJoCo model: {exc}")
            self._mj_model = None
            self._mj_data = None

    def _resolve_gripper_geometry(self):
        """Geometry that converts observed gripper qpos into metres.

        Prefers whatever the env says it is running, so the drawn pads land on
        the same fingertips ``policy_steering``'s FK measures distances from.
        Falls back to the rendered model's own name, which is the same string
        for every gripper that has both a model and a geometry config.
        """
        from policy_steering.robot_config import load_gripper_geometry

        name = self._gripper_model_name
        spec = getattr(self._env_spec, "robot", None)
        spec_id = getattr(getattr(spec, "gripper", None), "geometry_id", None)
        if spec_id:
            name = spec_id
        try:
            return load_gripper_geometry(name)
        except (FileNotFoundError, KeyError) as exc:
            logger.warning(
                f"[RerunVisualizer] No gripper geometry {name!r} ({exc}); the "
                "rendered gripper will treat its input as metres directly."
            )
            return None

    def _pose_arm_and_gripper(self, data, robot_joints: np.ndarray) -> None:
        """Write arm joints, then the whole closed-chain gripper, into ``qpos``.

        The gripper's eight joints are not independent — the four-bar's loop is
        held shut by equality constraints that ``mj_kinematics`` does not solve
        — so they are read out of the pre-solved linkage table rather than
        written one by one.
        """
        arm_nq = int(self._arm_nq)
        n_arm = min(arm_nq, len(robot_joints))
        data.qpos[:n_arm] = robot_joints[:n_arm]

        if len(robot_joints) <= arm_nq:
            return  # arm-only observation; leave the fingers where they were

        raw = float(robot_joints[arm_nq])
        geom = self._gripper_geometry
        aperture = geom.qpos_to_metric(raw) if geom is not None else raw

        lo, hi = self._gripper_linkage.aperture_range
        if not self._aperture_clamped and not (lo - 1e-9 <= aperture <= hi + 1e-9):
            self._aperture_clamped = True
            logger.warning(
                f"[RerunVisualizer] Gripper aperture {aperture * 1000:.1f} mm is "
                f"outside what the model's linkage spans "
                f"({lo * 1000:.1f}–{hi * 1000:.1f} mm); clamping. Further "
                "occurrences are not logged."
            )
        self._gripper_linkage.write_into(data.qpos, aperture)

    def _render_robot(self, robot_joints: np.ndarray) -> None:
        """Forward-simulate the robot model and log joint positions.

        ``robot_joints`` is ``[arm_qpos..., gripper_qpos...]`` where the
        gripper portion is either:

        * 2 per-finger qpos values (LIBERO/RoboCasa convention) — written
          straight into the model with the trailing finger negated to match
          menagerie's mirrored second-finger axis.
        * 1 aperture (full opening width) value (CALVIN convention) — split
          symmetrically: ``aperture/2`` and ``-aperture/2`` are written into
          the two finger qpos slots so the rendered gripper opens to half
          the full width on each side instead of the full width on one
          side.
        * 1 normalised openness, when a closed-chain gripper was bolted on
          (:attr:`_gripper_linkage`) — the eight linkage joints cannot be
          written independently, so the whole finger pose is looked up from
          the aperture that openness corresponds to.
        """
        if self._mj_model is None or self._mj_data is None:
            return

        mj = self._mj
        data = self._mj_data

        nq = self._mj_model.nq
        joints_len = len(robot_joints)
        if self._gripper_linkage is not None and self._arm_nq is not None:
            self._pose_arm_and_gripper(data, robot_joints)
        elif joints_len == nq:
            # Two per-finger qpos values; negate the trailing one so that a
            # symmetric (both positive) input opens the menagerie panda
            # gripper symmetrically.
            data.qpos[:nq] = robot_joints[:nq]
            data.qpos[-1] = -float(robot_joints[-1])
        elif joints_len == nq - 1 and nq >= 2:
            # Single full-aperture value; split into ±half-aperture.
            data.qpos[: nq - 2] = robot_joints[: nq - 2]
            half = 0.5 * float(robot_joints[-1])
            data.qpos[nq - 2] = half
            data.qpos[nq - 1] = -half
        else:
            fill_len = min(joints_len, nq)
            data.qpos[:fill_len] = robot_joints[:fill_len]
        mj.mj_kinematics(self._mj_model, data)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
        self.mjcf_logger.log_data(data)

        # Log Cartesian positions of all named bodies
        for body_i in range(self._mj_model.nbody):
            name = mj.mj_id2name(self._mj_model, mj.mjtObj.mjOBJ_BODY, body_i)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
            if name:
                pos = data.xpos[body_i]
                rr.log(
                    f"robot/body/{name}/position",
                    rr.Points3D([pos.tolist()]),
                )

        # Log end-effector site positions (if any)
        for site_i in range(self._mj_model.nsite):
            name = mj.mj_id2name(self._mj_model, mj.mjtObj.mjOBJ_SITE, site_i)  # type: ignore[attr-defined]  # ty: ignore[unresolved-attribute]
            if name:
                pos = data.site_xpos[site_i]
                rr.log(
                    f"robot/site/{name}/position",
                    rr.Points3D([pos.tolist()]),
                )
