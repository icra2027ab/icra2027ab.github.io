"""Hand-eye calibration of the UR5e cell's cameras, built on Kalib's tracker.

The pipeline is ``capture -> click -> track -> solve -> verify -> save``:

* **capture** frames from one RealSense while the operator freedrives the arm,
  recording the controller's TCP pose with every frame (:mod:`.session`,
  :mod:`.dataset`);
* **click** the tracked point once in the first frame (the GUI in :mod:`.gui`);
* **track** it through the whole capture with Kalib's SpaTracker, run inside
  the ``kalib`` conda env (:mod:`.kalib_runner`);
* **solve** the camera pose with a RANSAC PnP over the settled frames
  (:mod:`.solve`);
* **verify** it against the camera's own depth and against Kalib's PnP;
* **save** it in the layout ``Ur5EnvServer`` loads (:mod:`.output`).

Two mountings are supported (:class:`~.geometry.CalibrationMode`). For the
*exterior* camera the tracked point is the TCP itself and the result is the
camera's pose in the robot ``base`` frame. For the *wrist* camera the tracked
point is a fixed mark on the table whose base-frame position the operator
records by touching it, and the result is the camera's pose in the TCP frame.

Everything but :mod:`.gui` is numpy/OpenCV only, so the package imports in the
``physical-ai-sdk`` venv that owns the RealSense and RTDE bindings.
"""

from __future__ import annotations

__all__: list[str] = []
