"""WebSocket server wrapping the SAM 3D Objects inference pipeline.

Run this in the sam-3d-objects venv *before* launching the demo:

    # Terminal 1 — SAM3D Python (sam-3d-objects venv)
    third_party/sam-3d-objects/.venv/bin/python scripts/servers/serve_sam3d.py \\
        --config third_party/sam-3d-objects/checkpoints/hf/pipeline.yaml

    # Terminal 2 — policy Python (main venv)
    python examples/sam3d_libero_demo.py \\
        --demo datasets/libero_object_new/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5

Protocol (all frames are pickle-encoded bytes):
  Client → Server:  {"cmd": "<command>", "kwargs": {...}}
  Server → Client:  {"result": <value>, "error": None | "<traceback string>"}

Handshake:
  On connect the server sends {"server": "SAM3DServer", "version": "1.0"}.

Commands
--------
reconstruct
    kwargs: {
        "image":        np.ndarray [H, W, 3] uint8,
        "mask":         np.ndarray [H, W] bool,
        "seed":         int (optional, default 42),
        "return_mesh":  bool (optional, default False) — include GLB mesh bytes,
        "pointmap":     np.ndarray [H, W, 3] float32 (optional) — pre-computed
                        3-D point map in PyTorch3D camera space.  When provided
                        MoGe depth estimation is skipped and this map is used
                        directly to condition the diffusion.  Pass the
                        ``pointmap`` field from a previous reconstruction to
                        re-use the same camera/depth.
    }
    result: {
        "gs_xyz":         np.ndarray [N, 3] float32,  canonical Gaussian positions
        "gs_colors":      np.ndarray [N, 3] float32,  RGB in [0, 1]
        "gs_opacity":     np.ndarray [N, 1] float32,  opacity in [0, 1]
        "gs_scaling":     np.ndarray [N, 3] float32,  Gaussian scale
        "gs_rotation":    np.ndarray [N, 4] float32,  per-splat quaternion (w,x,y,z)
        "rotation":       np.ndarray [4]    float32,  object rotation quaternion (w,x,y,z)
        "translation":    np.ndarray [3]    float32,  object translation (MoGe cam-space)
        "scale":          np.ndarray [3]    float32,  object scale
        "pointmap":       np.ndarray [H, W, 3] float32,  MoGe 3D point map (PyTorch3D space)
        "mesh_glb_bytes": bytes | None — raw GLB file bytes (only when return_mesh=True)
    }

Mesh notes
----------
The mesh is generated via FlexiCubes from the same latent features as the
Gaussians (decoder_mesh.py).  It uses vertex colours (no texture baking) so
it is fast but does not require UV unwrapping.  Save the bytes directly:

    with open("object.glb", "wb") as f:
        f.write(result["mesh_glb_bytes"])

Or use the :func:`SAM3DClient.save_mesh` helper.
"""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import os
import pickle
import sys
import traceback as tb

# ── Env setup for sam3d-objects notebook ────────────────────────────────────
if "CONDA_PREFIX" not in os.environ:
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("CUDA_HOME", cuda_home)
    os.environ["CONDA_PREFIX"] = cuda_home

os.environ.setdefault("LIDRA_SKIP_INIT", "true")

# Put the sam-3d-objects notebook dir on the path so `inference.py` is importable.
_repo_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_sam3d_root = os.path.join(_repo_root, "third_party", "sam-3d-objects")
_notebook_dir = os.path.join(_sam3d_root, "notebook")
for _p in (_notebook_dir, _sam3d_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import websockets.exceptions  # noqa: E402
import websockets.server as _ws_server  # noqa: E402

logger = logging.getLogger(__name__)

# SH 0th-degree coefficient for converting DC features → RGB
_SH_C0 = 0.28209479177387814


def _extract_gaussians(output: dict) -> dict:
    """Extract plain numpy arrays from a SAM3D Inference output dict."""
    import torch

    gs = output["gaussian"][0]

    # Canonical Gaussian positions (local space, scaled by aabb)
    gs_xyz = gs.get_xyz.detach().cpu().float().numpy()  # [N, 3]
    gs_opacity = gs.get_opacity.detach().cpu().float().numpy()  # [N, 1]
    gs_scaling = gs.get_scaling.detach().cpu().float().numpy()  # [N, 3]
    gs_rotation = gs.get_rotation.detach().cpu().float().numpy()  # [N, 4] (w,x,y,z)

    # DC SH → RGB: color = SH_C0 * dc + 0.5, clamped to [0, 1]
    dc = gs._features_dc.detach().cpu().float()  # [N, 3, 1]
    gs_colors = (_SH_C0 * dc.squeeze(2) + 0.5).clamp(0, 1).numpy()  # [N, 3]

    # Pose
    rotation = (
        output["rotation"].detach().cpu().float().numpy().reshape(4)
    )  # [4]  (w,x,y,z)
    translation = output["translation"].detach().cpu().float().numpy().reshape(3)  # [3]
    scale = output["scale"].detach().cpu().float().numpy().reshape(3)  # [3]

    # Pointmap (downsampled scene geometry in PyTorch3D camera space)
    pointmap = output["pointmap"]
    if isinstance(pointmap, torch.Tensor):
        pointmap = pointmap.detach().cpu().float().numpy()
    pointmap = pointmap.astype(np.float32)  # [H, W, 3]

    return {
        "gs_xyz": gs_xyz.astype(np.float32),
        "gs_colors": gs_colors.astype(np.float32),
        "gs_opacity": gs_opacity.astype(np.float32),
        "gs_scaling": gs_scaling.astype(np.float32),
        "gs_rotation": gs_rotation.astype(np.float32),
        "rotation": rotation.astype(np.float32),
        "translation": translation.astype(np.float32),
        "scale": scale.astype(np.float32),
        "pointmap": pointmap,
    }


def _extract_mesh_bytes(output: dict) -> bytes | None:
    """Serialise the GLB trimesh from a SAM3D output dict to raw bytes.

    Returns ``None`` if no mesh was produced (``output["glb"]`` is None).
    """
    glb = output.get("glb", None)
    if glb is None:
        return None
    # trimesh.Trimesh.export() returns bytes when file_type is given
    return glb.export(file_type="glb")


class SAM3DServer:
    """Serves the SAM 3D Objects inference pipeline over a WebSocket."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8766,
        config_path: str | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._config_path = config_path or os.path.join(
            _sam3d_root, "checkpoints", "hf", "pipeline.yaml"
        )
        self._inference = None  # loaded lazily on first connection

    # ------------------------------------------------------------------
    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        logger.info("Loading SAM3D inference pipeline …")
        from inference import Inference  # notebook/inference.py

        self._inference = Inference(self._config_path, compile=False)
        logger.info("SAM3D inference pipeline ready.")

        async with _ws_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ) as server:
            logger.info(f"SAM3DServer listening on ws://{self._host}:{self._port}")
            await server.serve_forever()

    @staticmethod
    async def _health_check(path, request_headers):
        if path == "/healthz":
            return http.HTTPStatus.OK, {}, b"OK\n"
        return None

    async def _handler(self, websocket) -> None:
        logger.info(f"Connection from {websocket.remote_address}")

        # Handshake
        await websocket.send(pickle.dumps({"server": "SAM3DServer", "version": "1.0"}))

        try:
            async for raw in websocket:
                msg = pickle.loads(raw)
                cmd: str = msg.get("cmd", "")
                kwargs: dict = msg.get("kwargs", {})

                try:
                    result = await self._dispatch(cmd, kwargs)
                    await websocket.send(
                        pickle.dumps({"result": result, "error": None})
                    )
                except Exception:
                    err = tb.format_exc()
                    logger.error(f"Error handling cmd={cmd!r}:\n{err}")
                    await websocket.send(pickle.dumps({"result": None, "error": err}))

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Connection from {websocket.remote_address} closed")

    async def _dispatch(self, cmd: str, kwargs: dict):
        if cmd == "reconstruct":
            return await asyncio.get_event_loop().run_in_executor(
                None, self._reconstruct, kwargs
            )
        raise ValueError(f"Unknown command: {cmd!r}")

    def _reconstruct(self, kwargs: dict) -> dict:
        import torch

        image = np.asarray(kwargs["image"], dtype=np.uint8)  # [H, W, 3]
        mask = np.asarray(kwargs["mask"], dtype=bool)  # [H, W]
        seed = int(kwargs.get("seed", 42))
        return_mesh = bool(kwargs.get("return_mesh", False))

        # Mesh post-processing controls (forwarded to to_glb())
        with_mesh_postprocess = bool(kwargs.get("with_mesh_postprocess", False))
        with_texture_baking = bool(kwargs.get("with_texture_baking", False))
        use_vertex_color = bool(kwargs.get("use_vertex_color", True))

        # Optional pre-computed point map in PyTorch3D camera space [H, W, 3].
        # When provided, MoGe depth estimation is skipped.
        raw_pointmap = kwargs.get("pointmap", None)
        pointmap_tensor = None
        if raw_pointmap is not None:
            pointmap_tensor = torch.from_numpy(
                np.asarray(raw_pointmap, dtype=np.float32)
            )  # [H, W, 3]

        if self._inference is None:
            raise RuntimeError("Inference pipeline not loaded yet.")

        # Call the pipeline directly so we can pass mesh options that
        # Inference.__call__ hard-codes.
        rgba = self._inference.merge_mask_to_rgba(image, mask)
        output = self._inference._pipeline.run(
            rgba,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=with_mesh_postprocess,
            with_texture_baking=with_texture_baking,
            with_layout_postprocess=False,
            use_vertex_color=use_vertex_color,
            stage1_inference_steps=None,
            pointmap=pointmap_tensor,
        )

        result = _extract_gaussians(output)
        result["mesh_glb_bytes"] = _extract_mesh_bytes(output) if return_mesh else None
        return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SAM3D WebSocket inference server.")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind to (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8766,
        help="WebSocket port (default: 8766).",
    )
    parser.add_argument(
        "--config",
        default=os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "third_party",
            "sam-3d-objects",
            "checkpoints",
            "hf",
            "pipeline.yaml",
        ),
        help="Path to the SAM3D pipeline.yaml config file.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    server = SAM3DServer(host=args.host, port=args.port, config_path=args.config)
    server.serve_forever()


if __name__ == "__main__":
    main()
