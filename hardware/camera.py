"""RealSense + ArUco in-plane finger-tracking camera module.

This module opens an Intel RealSense color stream, detects four ArUco markers,
and computes — for each marker — its IN-PLANE roll angle ``phi`` (rotation about
the camera optical axis) directly from the detected image corners.

Why the in-plane angle (and NOT ``solvePnP``)
---------------------------------------------
For a small, near-fronto-parallel planar marker viewed by a fixed camera, the
full 6-DoF pose returned by ``cv2.aruco.estimatePoseSingleMarkers`` /
``solvePnP`` is poorly conditioned: the out-of-plane tilt and the translation
along the optical axis trade off against each other and are sensitive to corner
noise and to the (often imperfect) intrinsics/marker-size calibration. The one
quantity that *is* well conditioned is the rotation of the marker square inside
the image plane — i.e. the angle of its top edge in pixel coordinates. That
angle is exactly the roll about the optical axis and is what a finger joint's
flexion maps to on this rig. So we read it straight from the (sub-pixel refined)
corners with a single ``atan2`` and skip pose estimation entirely.

ID -> body mapping
------------------
The rig glues one marker to each finger segment::

    0 = base     (M0)
    1 = proximal (M1)
    2 = middle   (M2)
    3 = distal   (M3)

``joints.py`` downstream is responsible for any sign convention, zeroing, and
relative-angle bookkeeping, so this module returns a raw, continuous ``atan2``
value in degrees in ``(-180, 180]`` and does NOT try to flip or unwrap it.

A :class:`MockCamera` with the same interface is provided so the dashboard / UI
can run with no RealSense attached.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np

# --- Lazy / deferred hardware imports -------------------------------------
# We attempt the imports at module load so callers can check availability, but
# any failure is deferred: the module still imports fine, and a clear error is
# only raised when ``start()`` is actually called on the hardware-backed class.
try:  # pragma: no cover - depends on environment
    import cv2  # type: ignore
    _CV2_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2_IMPORT_ERROR = exc

try:  # pragma: no cover - depends on environment
    import pyrealsense2 as rs  # type: ignore
    _RS_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover
    rs = None  # type: ignore
    _RS_IMPORT_ERROR = exc


ARUCO_DICT_NAME = "DICT_4X4_50"

# Marker ID -> body mapping (used for labels in the annotated frame).
#   0 = base (M0), 1 = proximal (M1), 2 = middle (M2), 3 = distal (M3)
MARKER_LABELS: Dict[int, str] = {0: "base", 1: "prox", 2: "mid", 3: "dist"}

# BGR colors for the status legend.
_COLOR_VISIBLE = (0, 200, 0)    # green
_COLOR_MISSING = (0, 0, 220)    # red
_COLOR_TEXT = (255, 255, 255)   # white


def _phi_from_corners(corners: np.ndarray) -> float:
    """Compute the in-plane roll angle (degrees) from a marker's 4 corners.

    OpenCV returns each marker's corners as a ``(4, 2)`` array ordered
    ``[top-left, top-right, bottom-right, bottom-left]`` (clockwise, image
    coordinates with +x right and +y down).

    We take the top edge vector ``v = corners[1] - corners[0]``
    (top-left -> top-right) and return ``degrees(atan2(v_y, v_x))``. Because
    image +y points DOWN, this gives a consistent in-plane roll. The value is a
    continuous ``atan2`` result in ``(-180, 180]``; no flipping/normalizing is
    applied here (the rig's ``joints.py`` handles sign and zeroing).
    """
    pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    v = pts[1] - pts[0]  # top-left -> top-right
    return math.degrees(math.atan2(float(v[1]), float(v[0])))


class RealSenseAruco:
    """Open a RealSense color stream and report per-marker in-plane roll angles."""

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        marker_ids: Tuple[int, ...] = (0, 1, 2, 3),
        marker_size_mm: float = 12.0,
        dict_name: str = "DICT_4X4_50",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.marker_ids: Tuple[int, ...] = tuple(int(i) for i in marker_ids)
        self.marker_size_mm = float(marker_size_mm)
        self.dict_name = str(dict_name)

        # Populated on start().
        self._pipeline = None
        self._profile = None
        self._dictionary = None
        self._params = None
        self._detector = None        # new-API ArucoDetector (if available)
        self._use_new_api = False
        self._started = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        """Open the RealSense pipeline and build the ArUco detector.

        Raises ``RuntimeError`` with a helpful message if ``pyrealsense2`` or
        ``cv2`` are unavailable, or if no RealSense device can be opened.
        """
        if cv2 is None:
            raise RuntimeError(
                "OpenCV (cv2) is required but could not be imported: "
                f"{_CV2_IMPORT_ERROR!r}. Install opencv-contrib-python."
            )
        if not hasattr(cv2, "aruco"):
            raise RuntimeError(
                "cv2.aruco is unavailable. Install 'opencv-contrib-python' "
                "(the plain 'opencv-python' wheel does not include aruco)."
            )
        if rs is None:
            raise RuntimeError(
                "pyrealsense2 is required but could not be imported: "
                f"{_RS_IMPORT_ERROR!r}. Install the Intel RealSense SDK / "
                "pyrealsense2."
            )

        # Build the ArUco dictionary + detector params, picking the API path.
        self._build_detector()

        # Open the color-only pipeline.
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )
            self._profile = pipeline.start(config)
            self._pipeline = pipeline
        except Exception as exc:  # device missing / busy / bad config
            self._pipeline = None
            self._profile = None
            raise RuntimeError(
                "Failed to start RealSense color stream "
                f"({self.width}x{self.height}@{self.fps}). Is a device "
                f"connected and free? Underlying error: {exc!r}"
            ) from exc

        self._started = True

    def stop(self) -> None:
        """Stop the pipeline safely (no-op if it was never started)."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = None
        self._profile = None
        self._started = False

    # -- detector construction --------------------------------------------
    def _resolve_dict_id(self) -> int:
        """Map ``self.dict_name`` to the cv2.aruco predefined dictionary id."""
        dict_id = getattr(cv2.aruco, self.dict_name, None)
        if dict_id is None:
            dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME, None)
        if dict_id is None:
            raise RuntimeError(
                f"Unknown ArUco dictionary name: {self.dict_name!r}."
            )
        return dict_id

    def _build_detector(self) -> None:
        """Construct dictionary + params, supporting both new and old APIs."""
        dict_id = self._resolve_dict_id()

        # --- dictionary -----------------------------------------------------
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            self._dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        else:  # very old API
            self._dictionary = cv2.aruco.Dictionary_get(dict_id)

        # --- detector parameters -------------------------------------------
        if hasattr(cv2.aruco, "DetectorParameters") and callable(
            getattr(cv2.aruco, "DetectorParameters")
        ):
            # New API: DetectorParameters() is a constructor.
            try:
                self._params = cv2.aruco.DetectorParameters()
            except Exception:
                # Some builds expose the *_create factory instead.
                self._params = cv2.aruco.DetectorParameters_create()
        else:
            self._params = cv2.aruco.DetectorParameters_create()

        # Enable sub-pixel corner refinement (well-conditioned phi depends on it).
        refine = getattr(cv2.aruco, "CORNER_REFINE_SUBPIX", None)
        if refine is not None:
            self._params.cornerRefinementMethod = refine

        # --- detector object (new API) -------------------------------------
        if hasattr(cv2.aruco, "ArucoDetector"):
            self._detector = cv2.aruco.ArucoDetector(self._dictionary, self._params)
            self._use_new_api = True
        else:
            self._detector = None
            self._use_new_api = False

    def _detect_markers(self, gray: np.ndarray):
        """Run marker detection through whichever API is available."""
        if self._use_new_api and self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, self._dictionary, parameters=self._params
            )
        return corners, ids

    # -- main API ----------------------------------------------------------
    def detect(self) -> dict:
        """Grab one color frame, detect markers, compute per-marker ``phi``.

        Returns a dict of the shape::

            {
              "phi":     {id: float|None, ...},   # degrees, None if unseen
              "visible": {id: bool, ...},
              "frame":   np.ndarray,              # BGR uint8, annotated
            }
        """
        if not self._started or self._pipeline is None:
            raise RuntimeError("Camera not started. Call start() first.")

        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            # No color frame this poll; return an all-missing result.
            blank = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            return self._empty_result(blank)

        frame = np.asanyarray(color_frame.get_data())
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        frame = np.ascontiguousarray(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = self._detect_markers(gray)

        return self._build_result(frame, corners, ids)

    # -- result assembly ---------------------------------------------------
    def _empty_result(self, frame: np.ndarray) -> dict:
        phi = {mid: None for mid in self.marker_ids}
        visible = {mid: False for mid in self.marker_ids}
        self._draw_legend(frame, visible)
        return {"phi": phi, "visible": visible, "frame": frame}

    def _build_result(self, frame: np.ndarray, corners, ids) -> dict:
        phi: Dict[int, Optional[float]] = {mid: None for mid in self.marker_ids}
        visible: Dict[int, bool] = {mid: False for mid in self.marker_ids}

        if ids is not None and len(ids) > 0:
            # Draw all detected markers (outline + id).
            try:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            except Exception:
                pass

            ids_flat = np.asarray(ids).reshape(-1)
            for marker_corners, marker_id in zip(corners, ids_flat):
                mid = int(marker_id)
                pts = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
                angle = _phi_from_corners(pts)

                if mid in phi:
                    phi[mid] = angle
                    visible[mid] = True

                self._annotate_marker(frame, pts, mid, angle)

        self._draw_legend(frame, visible)
        return {"phi": phi, "visible": visible, "frame": frame}

    # -- drawing helpers ---------------------------------------------------
    def _annotate_marker(
        self, frame: np.ndarray, pts: np.ndarray, mid: int, angle: float
    ) -> None:
        """Overlay 'id:label  phi=NN.N' near the marker center."""
        center = pts.mean(axis=0)
        cx, cy = int(round(center[0])), int(round(center[1]))
        label = MARKER_LABELS.get(mid, "?")
        text = f"{mid}:{label}  phi={angle:5.1f}"
        org = (max(0, cx - 60), max(15, cy))
        # Dark outline for readability over any background, then bright text.
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_TEXT, 1,
            cv2.LINE_AA,
        )

    def _draw_legend(self, frame: np.ndarray, visible: Dict[int, bool]) -> None:
        """Small per-marker visibility legend in the top-left corner."""
        x0, y0 = 10, 24
        cv2.putText(
            frame, "markers:", (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            _COLOR_TEXT, 1, cv2.LINE_AA,
        )
        x = x0 + 110
        for mid in self.marker_ids:
            seen = bool(visible.get(mid, False))
            color = _COLOR_VISIBLE if seen else _COLOR_MISSING
            label = MARKER_LABELS.get(mid, "?")
            tag = f"{mid}:{label}"
            cv2.circle(frame, (x, y0 - 5), 6, color, -1)
            cv2.putText(
                frame, tag, (x + 12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color, 1, cv2.LINE_AA,
            )
            x += 12 + 10 * len(tag) + 20

    # -- intrinsics --------------------------------------------------------
    def get_intrinsics(self) -> dict:
        """Return factory color intrinsics from the SDK profile.

        Keys: ``fx, fy, ppx, ppy, width, height``. Values are ``None`` if the
        pipeline has not been started yet.
        """
        empty = {
            "fx": None, "fy": None, "ppx": None, "ppy": None,
            "width": self.width, "height": self.height,
        }
        if self._profile is None or rs is None:
            return empty
        try:
            color_stream = self._profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            return {
                "fx": float(intr.fx),
                "fy": float(intr.fy),
                "ppx": float(intr.ppx),
                "ppy": float(intr.ppy),
                "width": int(intr.width),
                "height": int(intr.height),
            }
        except Exception:
            return empty


class MockCamera:
    """Hardware-free stand-in for :class:`RealSenseAruco` (same interface).

    ``detect()`` returns a synthetic 720x1280 BGR frame and slowly rotating
    ``phi`` values for ids 0..3, all marked visible. Useful for running the
    dashboard / UI without a RealSense connected.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        marker_ids: Tuple[int, ...] = (0, 1, 2, 3),
        marker_size_mm: float = 12.0,
        dict_name: str = "DICT_4X4_50",
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.marker_ids: Tuple[int, ...] = tuple(int(i) for i in marker_ids)
        self.marker_size_mm = float(marker_size_mm)
        self.dict_name = str(dict_name)

        self._started = False
        self._t0 = 0.0

    def start(self) -> None:
        self._t0 = time.time()
        self._started = True

    def stop(self) -> None:
        self._started = False

    def detect(self) -> dict:
        if not self._started:
            raise RuntimeError("MockCamera not started. Call start() first.")

        t = time.time() - self._t0
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        phi: Dict[int, Optional[float]] = {}
        visible: Dict[int, bool] = {}

        # Slowly rotating angles, phase-offset per marker, wrapped to (-180,180].
        for k, mid in enumerate(self.marker_ids):
            raw = (30.0 * t + 90.0 * k) % 360.0
            angle = raw - 360.0 if raw > 180.0 else raw
            phi[mid] = float(angle)
            visible[mid] = True

        self._draw_mock(frame, phi)
        return {"phi": phi, "visible": visible, "frame": frame}

    def get_intrinsics(self) -> dict:
        # Plausible synthetic pinhole intrinsics centered on the frame.
        return {
            "fx": float(self.width),
            "fy": float(self.width),
            "ppx": self.width / 2.0,
            "ppy": self.height / 2.0,
            "width": self.width,
            "height": self.height,
        }

    # -- drawing ----------------------------------------------------------
    def _draw_mock(self, frame: np.ndarray, phi: Dict[int, Optional[float]]) -> None:
        try:
            import cv2 as _cv2  # local import; mock can draw if cv2 exists
        except Exception:
            return  # no cv2 -> just return the blank frame

        _cv2.putText(
            frame, "MOCK CAMERA (no hardware)", (20, 60),
            _cv2.FONT_HERSHEY_SIMPLEX, 1.0, _COLOR_TEXT, 2, _cv2.LINE_AA,
        )

        # Draw a little rotating square per marker so the UI has something live.
        n = max(1, len(self.marker_ids))
        spacing = self.width // (n + 1)
        cy = self.height // 2
        half = 70
        base_sq = np.array(
            [[-half, -half], [half, -half], [half, half], [-half, half]],
            dtype=np.float64,
        )
        for k, mid in enumerate(self.marker_ids):
            angle = phi.get(mid) or 0.0
            rad = math.radians(angle)
            rot = np.array(
                [[math.cos(rad), -math.sin(rad)],
                 [math.sin(rad), math.cos(rad)]]
            )
            cx = spacing * (k + 1)
            poly = (base_sq @ rot.T) + np.array([cx, cy])
            poly_i = poly.astype(np.int32).reshape(-1, 1, 2)
            _cv2.polylines(frame, [poly_i], True, _COLOR_VISIBLE, 2, _cv2.LINE_AA)

            label = MARKER_LABELS.get(mid, "?")
            text = f"{mid}:{label}  phi={angle:5.1f}"
            _cv2.putText(
                frame, text, (cx - 80, cy + half + 30),
                _cv2.FONT_HERSHEY_SIMPLEX, 0.6, _COLOR_TEXT, 1, _cv2.LINE_AA,
            )


__all__ = [
    "ARUCO_DICT_NAME",
    "MARKER_LABELS",
    "RealSenseAruco",
    "MockCamera",
]
