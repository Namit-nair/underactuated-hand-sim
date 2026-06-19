#!/usr/bin/env python3
"""PhaseSpace mocap finger tracker (drop-in replacement for the ArUco camera).

Replaces ``hardware/camera.py`` (RealSense + ArUco) as the joint-angle source for
the validation rig. Instead of a single planar marker per segment, the
PhaseSpace system streams the 3D position of TWO LED markers per segment; this
module turns each segment's two points into a direction vector and reports the
segment's in-plane orientation ``phi`` — exactly the quantity ``hardware/joints.py``
already differences into MCP / PIP / DIP angles. So the whole downstream pipeline
(JointAngles zeroing, predictor overlay, CSV logging, plots) runs unchanged.

Contract (matches ``RealSenseAruco`` so the dashboard treats it identically)::

    start() / stop()
    detect() -> {"phi": {0..3: deg|None}, "visible": {0..3: bool}, "frame": ndarray|None}
    set_alignment_reference(phi) / clear_alignment_reference() / set_show_reference(on)
    attributes: _started (bool), show_reference (bool)

``phi`` keys 0..3 are the four SEGMENTS (0=base, 1=prox, 2=mid, 3=dist), so
``joints.py`` computes mcp = phi1 - phi0, pip = phi2 - phi1, dip = phi3 - phi2.

Flexion-plane handling
----------------------
The finger flexes in one plane, but the LEDs are not mounted perfectly — a
segment's two markers may be pitched/yawed off the link's long axis. We therefore
PROJECT each 3D segment vector onto the calibrated flexion plane (normal ``n``)
before measuring its angle, which removes the out-of-plane tilt. Any residual
constant mounting offset is cancelled by the straight-pose "Set Zero" in
``joints.py``. The plane basis ``(n, u_axis, u_perp)`` comes from a one-time
calibration flex (see :meth:`finalize_calibration`); absent a calibration file we
fall back to assuming the mocap world XY plane is the flexion plane.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from collections import namedtuple
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2  # only for rendering the stick-figure preview; optional
except Exception:  # pragma: no cover - cv2 should be installed, but degrade gracefully
    cv2 = None

# Vendored PhaseSpace OWL2 SDK (mocap/owl.py).
import owl  # noqa: E402

_EPS = 1e-9
_NSEG = 4  # base, prox, mid, dist

# Lightweight marker record (id-keyed snapshot value).
_M = namedtuple("_M", ["x", "y", "z", "cond"])


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    if n < _EPS:
        return None
    return v / n


class _BaseTracker:
    """Shared geometry + camera-contract logic; subclasses supply marker data."""

    def __init__(self, segment_marker_ids, calib_path: Optional[str] = None):
        self.segment_marker_ids: Tuple[Tuple[int, int], ...] = tuple(
            (int(a), int(b)) for a, b in segment_marker_ids)
        if len(self.segment_marker_ids) != _NSEG:
            raise ValueError(f"need {_NSEG} (near,far) segment id pairs, "
                             f"got {len(self.segment_marker_ids)}")
        self.calib_path = calib_path

        self._started = False
        self.show_reference = True
        self.render_enabled = True   # dashboard turns this off (draws its own pose)
        self._align_ref: Optional[Dict[int, Optional[float]]] = None

        # Flexion-plane basis. Default: mocap world XY plane is the flexion plane
        # (normal +Z, finger long axis +Y). Calibration overwrites these.
        self.n = np.array([0.0, 0.0, 1.0])
        self.u_axis = np.array([0.0, 1.0, 0.0])
        self.u_perp = np.cross(self.n, self.u_axis)
        self.calibrated = False

        # Calibration accumulation (raw per-segment unit vectors per recorded frame).
        self._calib_samples: List[Dict[int, np.ndarray]] = []

        if calib_path and os.path.exists(calib_path):
            try:
                self.load_calibration(calib_path)
            except Exception:  # noqa: BLE001 - bad/old file shouldn't block startup
                pass

    # ----- lifecycle (subclasses extend) -----------------------------------
    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    # Subclasses return the latest marker snapshot: {id: _M(x,y,z,cond)}.
    def _snapshot(self) -> Dict[int, _M]:
        raise NotImplementedError

    # ----- core: markers -> per-segment vectors / phi -----------------------
    def segment_vectors(self) -> Dict[int, Optional[np.ndarray]]:
        """Raw (un-projected) 3D unit vector per segment, or None if unseen.

        Both markers of a segment must have cond>0 for the segment to resolve.
        """
        snap = self._snapshot()
        out: Dict[int, Optional[np.ndarray]] = {}
        for si, (near_id, far_id) in enumerate(self.segment_marker_ids):
            mn = snap.get(near_id)
            mf = snap.get(far_id)
            if mn is None or mf is None or mn.cond <= 0 or mf.cond <= 0:
                out[si] = None
                continue
            out[si] = _unit(np.array([mf.x - mn.x, mf.y - mn.y, mf.z - mn.z]))
        return out

    def _phi_from_vec(self, u: Optional[np.ndarray]) -> Optional[float]:
        """In-plane orientation [deg] of a segment unit vector, projected onto
        the calibrated flexion plane. None if the vector is missing or lies along
        the plane normal (degenerate)."""
        if u is None:
            return None
        vp = u - float(u @ self.n) * self.n      # project out the normal component
        vpn = _unit(vp)
        if vpn is None:
            return None
        return math.degrees(math.atan2(float(vpn @ self.u_perp),
                                       float(vpn @ self.u_axis)))

    def detect(self) -> dict:
        """One frame -> {phi, visible, frame} (camera-contract compatible)."""
        vecs = self.segment_vectors()
        phi: Dict[int, Optional[float]] = {}
        visible: Dict[int, bool] = {}
        for si in range(_NSEG):
            p = self._phi_from_vec(vecs.get(si))
            phi[si] = p
            visible[si] = p is not None
        frame = (self._render(vecs, phi, visible)
                 if cv2 is not None and self.render_enabled else None)
        return {"phi": phi, "visible": visible, "frame": frame}

    # ----- alignment reference (overlay only; never alters raw phi) ---------
    def set_alignment_reference(self, phi: Dict[int, Optional[float]]) -> bool:
        if any(phi.get(i) is None for i in range(_NSEG)):
            return False
        self._align_ref = dict(phi)
        return True

    def clear_alignment_reference(self) -> None:
        self._align_ref = None

    def set_show_reference(self, on: bool) -> None:
        self.show_reference = bool(on)

    # ----- calibration ------------------------------------------------------
    def begin_calibration(self) -> None:
        """Start a fresh calibration flex (clears any recorded samples)."""
        self._calib_samples = []

    def record_calibration_sample(self) -> int:
        """Record the current per-segment vectors. Call repeatedly while the
        operator flexes the finger through its range. Returns the running count.
        Only frames where ALL four segments resolve are kept (so every sample is
        a complete pose)."""
        vecs = self.segment_vectors()
        if any(vecs.get(i) is None for i in range(_NSEG)):
            return len(self._calib_samples)
        self._calib_samples.append({i: vecs[i] for i in range(_NSEG)})
        return len(self._calib_samples)

    def finalize_calibration(self, save: bool = True) -> bool:
        """Fit the flexion-plane basis from the recorded sweep and (optionally)
        persist it.

        * ``n``      = flexion-plane normal = least-variance direction of the
                       stacked moving-segment (prox/mid/dist) vectors (SVD).
        * ``u_axis`` = base-segment direction at the straight pose (mean base
                       vector), projected into the plane = the finger long axis.
        * ``u_perp`` = n x u_axis (the in-plane flexion direction).

        The base markers sit on a known horizontal plane, so the mean base vector
        is a stable long-axis reference; ``u_perp``'s sign sets which curl
        direction reads positive (the dashboard's "FLIP theta SIGN" can flip it).
        """
        if len(self._calib_samples) < 3:
            return False

        # Stack the moving segments' vectors (these carry the flexion sweep).
        moving = np.array([s[si] for s in self._calib_samples for si in (1, 2, 3)])
        # Best-fit plane through the origin: normal = smallest right-singular vec.
        _, _, vt = np.linalg.svd(moving, full_matrices=False)
        n = _unit(vt[-1])
        if n is None:
            return False

        base_mean = np.mean([s[0] for s in self._calib_samples], axis=0)
        # Long axis = base direction with the normal component removed.
        axis = base_mean - float(base_mean @ n) * n
        u_axis = _unit(axis)
        if u_axis is None:
            return False
        u_perp = _unit(np.cross(n, u_axis))
        if u_perp is None:
            return False

        # Orient u_perp so that flexion (distal curling away from base) is +.
        # Mean signed swing of the distal segment about u_axis over the sweep:
        d = np.array([s[3] for s in self._calib_samples])
        d_perp = d @ u_perp
        if float(np.mean(d_perp)) < 0:
            u_perp = -u_perp

        self.n, self.u_axis, self.u_perp = n, u_axis, u_perp
        self.calibrated = True
        if save and self.calib_path:
            self.save_calibration(self.calib_path)
        return True

    def save_calibration(self, path: str) -> None:
        data = {
            "n": self.n.tolist(),
            "u_axis": self.u_axis.tolist(),
            "u_perp": self.u_perp.tolist(),
            "n_samples": len(self._calib_samples),
            "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_calibration(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.n = _unit(np.array(data["n"], dtype=float))
        self.u_axis = _unit(np.array(data["u_axis"], dtype=float))
        self.u_perp = _unit(np.array(data["u_perp"], dtype=float))
        self.calibrated = True

    # ----- stick-figure preview (replaces the dead camera video) ------------
    def _render(self, vecs, phi, visible, size=(380, 640)):
        """Draw the finger as a chain of in-plane segment vectors on a dark BGR
        image, so the dashboard's existing preview pane shows the live pose."""
        h, w = size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (13, 17, 23)  # match the dashboard #0d1117

        seg_colors = [(0x8b, 0x94, 0x9e), (0xff, 0xa6, 0x58),
                      (0x87, 0xe7, 0x7e), (0x72, 0x7b, 0xff)]  # BGR-ish
        labels = ("base", "prox", "mid", "dist")

        # Chain the unit segment directions (each drawn at a fixed pixel length)
        # using the in-plane angle phi, starting from a left-center origin.
        origin = np.array([w * 0.18, h * 0.5])
        seg_len = min(w, h) * 0.20
        p = origin.copy()
        cv2.circle(img, (int(p[0]), int(p[1])), 4, (200, 200, 200), -1)
        for si in range(_NSEG):
            a = phi.get(si)
            ok = visible.get(si, False)
            col = seg_colors[si] if ok else (60, 60, 60)
            if a is None:
                # unseen segment: short stub so the gap is visible
                nxt = p + np.array([seg_len * 0.4, 0.0])
            else:
                rad = math.radians(a)
                # +x right, +y down in image space; draw along (cos, -sin)
                nxt = p + seg_len * np.array([math.cos(rad), -math.sin(rad)])
            cv2.line(img, (int(p[0]), int(p[1])), (int(nxt[0]), int(nxt[1])),
                     col, 4, cv2.LINE_AA)
            cv2.circle(img, (int(nxt[0]), int(nxt[1])), 5, col, -1)
            cv2.putText(img, labels[si], (int(nxt[0]) + 6, int(nxt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
            p = nxt

        status = "CALIBRATED" if self.calibrated else "uncalibrated (world plane)"
        cv2.putText(img, f"PhaseSpace  -  {status}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
        return img


class PhaseSpaceTracker(_BaseTracker):
    """Live PhaseSpace tracker over the OWL2 SDK.

    A background thread drains ``Context.nextEvent`` into a latest-frame marker
    snapshot, decoupling the fast mocap stream from the dashboard's 20 Hz poll.
    """

    def __init__(self, server: str, segment_marker_ids,
                 calib_path: Optional[str] = None,
                 timeout_us: int = 1_000_000, slave: bool = True):
        super().__init__(segment_marker_ids, calib_path)
        self.server = server
        self.timeout_us = int(timeout_us)
        self.slave = bool(slave)
        self._ctx: Optional[owl.Context] = None
        self._lock = threading.Lock()
        self._latest: Dict[int, _M] = {}
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        ctx = owl.Context()
        ctx.open(self.server)
        opts = "event.markers=1 event.rigids=0"
        if self.slave:
            opts += " slave=1"
        ctx.initialize(opts)
        ctx.streaming(1)
        self._ctx = ctx
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self._started = True

    def _reader(self) -> None:
        ctx = self._ctx
        while self._running and ctx is not None and ctx.isOpen():
            try:
                evt = ctx.nextEvent(self.timeout_us)
            except Exception:  # noqa: BLE001 - socket error -> stop cleanly
                break
            if not evt:
                continue
            if evt.type_id == owl.Type.FRAME and "markers" in evt:
                snap = {m.id: _M(m.x, m.y, m.z, m.cond) for m in evt.markers}
                with self._lock:
                    self._latest = snap

    def _snapshot(self) -> Dict[int, _M]:
        with self._lock:
            return dict(self._latest)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ctx is not None:
            try:
                self._ctx.done()
                self._ctx.close()
            except Exception:  # noqa: BLE001
                pass
            self._ctx = None
        self._started = False

    @staticmethod
    def list_devices() -> list:  # parity with RealSenseAruco; not meaningful here
        return []


class MockTracker(_BaseTracker):
    """Synthetic tracker for no-hardware development (mirrors MockCamera).

    Generates 8 LED positions for a finger slowly flexing in a known plane, with
    a small per-marker offset so the LEDs are NOT perfectly on the link axis —
    exercising the projection math. Pre-sets a trivial calibration so ``detect``
    works immediately without a calibration flex.
    """

    def __init__(self, segment_marker_ids, calib_path: Optional[str] = None,
                 link_len: float = 50.0):
        super().__init__(segment_marker_ids, calib_path)
        self.link_len = float(link_len)
        self._t0 = time.monotonic()
        # The mock finger lies in the world XY plane (normal +Z), long axis +Y;
        # the default basis already matches, so detect() works pre-calibration.
        self.calibrated = True

    def _snapshot(self) -> Dict[int, _M]:
        t = time.monotonic() - self._t0
        # Smoothly varying flex, distal joints curling more than proximal.
        s = 0.5 * (1.0 - math.cos(t * 0.5))           # 0..1 drive
        joint = np.radians(np.array([60.0, 75.0, 80.0]) * s)  # mcp,pip,dip
        # Build the 4 segment endpoints in the XY plane, base along +Y.
        snap: Dict[int, _M] = {}
        pos = np.array([0.0, 0.0, 0.0])
        cum = 0.0
        # base segment points straight up (+Y), no joint before it
        seg_dirs = []
        cum = 0.0
        for si in range(_NSEG):
            if si > 0:
                cum += float(joint[si - 1])
            d = np.array([math.sin(cum), math.cos(cum), 0.0])  # +Y at cum=0
            seg_dirs.append(d)
        for si, (near_id, far_id) in enumerate(self.segment_marker_ids):
            d = seg_dirs[si]
            near = pos.copy()
            far = pos + self.link_len * d
            # imperfect mounting: nudge each LED slightly out of plane / sideways
            jitter_n = np.array([0.0, 0.0, 3.0 * math.sin(0.7 * si + 1.0)])
            jitter_f = np.array([0.0, 0.0, 3.0 * math.cos(0.9 * si + 0.3)])
            snap[near_id] = _M(*(near + jitter_n), 1.0)
            snap[far_id] = _M(*(far + jitter_f), 1.0)
            pos = far
        return snap


def build_tracker(mock: bool = False, *, server=None, segment_marker_ids=None,
                  calib_path=None, timeout_us=1_000_000, slave=True):
    """Factory used by the dashboard / CLI."""
    if mock:
        return MockTracker(segment_marker_ids, calib_path=calib_path)
    return PhaseSpaceTracker(server, segment_marker_ids, calib_path=calib_path,
                             timeout_us=timeout_us, slave=slave)
