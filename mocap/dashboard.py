#!/usr/bin/env python3
"""PhaseSpace mocap validation dashboard for the tendon-driven 3R finger.

Separate from ``hardware/dashboard.py`` (which it reuses wholesale) — same window,
same full manual servo control, same auto-sweep + CSV validation workflow, but the
joint-angle source is the PhaseSpace mocap (``mocap/tracker.py``) instead of the
RealSense + ArUco camera, and all results are written inside ``mocap/results/``.

What is reused unchanged (imported from the hardware rig):
  * the entire ``Dashboard`` window — tick loop, plots, readouts, jog/e-stop,
    Set Zero, ΔL ramp, auto-sweep, settle detection;
  * ``servo.py`` (full manual servo control for testing + setting base tension),
    ``logger.py`` (CSV), ``predictor.py`` (analytical overlay), ``joints.py``.

What this file adds:
  * a PhaseSpace tracker (or a synthetic ``--mock`` tracker) in the camera slot;
  * a one-time CALIBRATION FLEX to fit the flexion plane (toolbar);
  * CSV output redirected to ``mocap/results/`` with a ``mocap_validation`` prefix.

Run:
    python mocap/dashboard.py --mock              # no hardware (synthetic mocap + servo)
    python mocap/dashboard.py                     # PhaseSpace + real Dynamixel
    python mocap/dashboard.py --server 192.168.1.230 --port auto --id 15
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import math
import os
import sys

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

# --- make the repo root + the hardware rig importable -----------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_HW_DIR = os.path.join(_REPO_ROOT, "hardware")
for _p in (_HERE, _REPO_ROOT, _HW_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import config  # noqa: E402  — root single source of truth (physical params)
import mocap_config as mcfg  # noqa: E402  — mocap-only knobs
import predictor  # noqa: E402  (hardware rig)
import servo as hwservo  # noqa: E402  (hardware rig)
from logger import CsvLogger  # noqa: E402
from servo import MockServo, Servo  # noqa: E402

from tracker import build_tracker  # noqa: E402  (mocap/)


# --- cross-platform servo auto-detection ------------------------------------
# servo.autodetect_servo already scans every port + baud and binds whatever
# Dynamixel answers (any id), so port="auto" recognises the motor regardless of
# COM port or id. The only gap is that servo.list_serial_ports() globs the Linux
# /dev/ttyUSB* paths; on Windows the U2D2 shows up as COMx. We patch the port
# enumerator (in-process only — servo.py is untouched) to use pyserial's
# cross-platform comports(), with the original Linux globs as a fallback.
def _list_serial_ports_xplat():
    try:
        from serial.tools import list_ports
        ports = sorted(p.device for p in list_ports.comports())
        if ports:
            return ports
    except Exception:  # noqa: BLE001
        pass
    import glob
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


hwservo.list_serial_ports = _list_serial_ports_xplat


def _load_hw_dashboard():
    """Load hardware/dashboard.py under a UNIQUE module name.

    This file is also called ``dashboard.py`` and sits first on sys.path, so a
    plain ``import dashboard`` would re-import THIS module. Load the hardware one
    explicitly by path as ``hw_dashboard`` to reuse its ``Dashboard`` window.
    """
    path = os.path.join(_HW_DIR, "dashboard.py")
    spec = importlib.util.spec_from_file_location("hw_dashboard", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hw_dashboard"] = mod
    spec.loader.exec_module(mod)
    return mod


hwdash = _load_hw_dashboard()
Dashboard = hwdash.Dashboard
_btn = hwdash._btn

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QFont  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QToolBar,
    QWidget,
)


class MocapDashboard(Dashboard):
    """Hardware ``Dashboard`` with a PhaseSpace tracker + flexion calibration."""

    def __init__(self, tracker, servo, *, geom_r, geom_note,
                 geom_link_lengths=None, results_dir=mcfg.MOCAP_RESULTS_DIR):
        # Factory that pins every CSV this rig opens to mocap/results/ with a
        # mocap-specific prefix. The base _capture() constructs the logger via the
        # module-global name ``CsvLogger``; we swap it in for the call (see
        # _capture below) so none of the base capture logic is duplicated.
        self._logger_factory = functools.partial(
            CsvLogger, out_dir=results_dir, filename_prefix="mocap_validation")
        self._calibrating = False

        super().__init__(tracker, servo, geom_r=geom_r, geom_note=geom_note,
                         geom_link_lengths=geom_link_lengths)

        self.setWindowTitle("Finger MOCAP Validation Rig — PhaseSpace — IITGN")
        self.btn_conn_cam.setText("CONNECT MOCAP")

        # Raise the ΔL ceiling well past 70 mm (the base rig caps the target at
        # 25 mm). Lift both the spinbox max and the servo soft cap so manual GO /
        # jog can drive the tendon to MOCAP_MAX_DELTA_MM.
        self.delta_spin.setMaximum(mcfg.MOCAP_MAX_DELTA_MM)
        try:
            self.servo.soft_delta_l_cap_mm = float(mcfg.MOCAP_MAX_DELTA_MM)
        except Exception:  # noqa: BLE001
            pass

        # The preview pane shows the ZEROED kinematic pose (drawn here from the
        # joint angles), not the tracker's raw-marker figure — so a straight
        # finger draws straight after Set Zero and bends to match real flexion.
        if hasattr(self.cam, "render_enabled"):
            self.cam.render_enabled = False

        self._build_calib_toolbar()

    # -----------------------------------------------------------------
    # calibration flex (added on top of the inherited UI, as a toolbar so the
    # base central layout is untouched)
    # -----------------------------------------------------------------
    def _build_calib_toolbar(self):
        tb = QToolBar("Calibration")
        tb.setMovable(False)
        tb.setStyleSheet("QToolBar{background:#161b22;border:0;spacing:8px;padding:6px;}")
        self.addToolBar(Qt.TopToolBarArea, tb)

        self.btn_calib = _btn("◎ CALIBRATE FLEX", "#0d1117", "#d2a8ff", "#d2a8ff")
        self.btn_calib.clicked.connect(self._toggle_calibration)
        tb.addWidget(self.btn_calib)

        self.lbl_calib = QLabel("")
        self.lbl_calib.setStyleSheet("color:#8b949e; font-family:Consolas; font-size:11px;")
        self._refresh_calib_label()
        tb.addWidget(self.lbl_calib)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tb.addWidget(spacer)

    def _refresh_calib_label(self):
        if self._calibrating:
            n = len(getattr(self.cam, "_calib_samples", []))
            self.lbl_calib.setText(f"recording… flex the finger fully  ({n} samples)")
            self.lbl_calib.setStyleSheet("color:#d29922; font-family:Consolas; font-size:11px;")
        elif getattr(self.cam, "calibrated", False):
            self.lbl_calib.setText("flexion plane: CALIBRATED ✓")
            self.lbl_calib.setStyleSheet("color:#56d364; font-family:Consolas; font-size:11px;")
        else:
            self.lbl_calib.setText("flexion plane: uncalibrated (assuming world plane)")
            self.lbl_calib.setStyleSheet("color:#8b949e; font-family:Consolas; font-size:11px;")

    def _toggle_calibration(self):
        if not getattr(self.cam, "_started", False):
            QMessageBox.warning(self, "Calibrate", "Connect the mocap first.")
            return
        if not self._calibrating:
            # start
            self.cam.begin_calibration()
            self._calibrating = True
            self.btn_calib.setText("■ FINISH CALIBRATION")
        else:
            # finish + fit
            self._calibrating = False
            self.btn_calib.setText("◎ CALIBRATE FLEX")
            ok = self.cam.finalize_calibration(save=True)
            if ok:
                QMessageBox.information(
                    self, "Calibrate",
                    "Flexion plane fitted and saved to\n"
                    f"{getattr(self.cam, 'calib_path', '?')}\n\n"
                    "Now press ◎ SET ZERO at the straight pose.")
            else:
                QMessageBox.warning(
                    self, "Calibrate",
                    "Not enough complete samples (need all 4 segments visible "
                    "across the flex). Try again with a slower, fuller flex.")
            self._redraw_plots()
        self._refresh_calib_label()
        self._grab_focus()

    # -----------------------------------------------------------------
    # overrides
    # -----------------------------------------------------------------
    def _tick(self):
        super()._tick()
        if self._calibrating:
            self.cam.record_calibration_sample()
            self._refresh_calib_label()
        img = self._render_live_pose()
        if img is not None:
            self._show_frame(img)

    # -----------------------------------------------------------------
    # live finger-pose preview (zeroed joint angles -> forward kinematics)
    # -----------------------------------------------------------------
    def _render_live_pose(self, w=640, h=380):
        """Draw the live finger as a 3-link chain bent by the ZEROED joint
        angles, so a straight finger renders straight (after Set Zero) and the
        drawing tracks real flexion — unlike the raw per-marker orientations."""
        if cv2 is None:
            return None
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (23, 17, 13)  # BGR of #0d1117

        th = self._last_theta
        vals = [th.get(j) for j in ("mcp", "pip", "dip")]
        zeroed = self.joints.is_zeroed()

        origin = np.array([w * 0.5, h * 0.84])
        seg_len = min(w, h) * 0.21
        # palm stub (fixed, below the MCP pivot)
        palm = origin + np.array([0.0, seg_len * 0.5])
        cv2.line(img, tuple(origin.astype(int)), tuple(palm.astype(int)),
                 (140, 148, 139), 3, cv2.LINE_AA)

        colors = [(0xff, 0xa6, 0x58), (0x7e, 0xe7, 0x87), (0x72, 0x7b, 0xff)]
        names = ("MCP", "PIP", "DIP")
        p = origin.copy()
        cum = 0.0
        lost = False
        for i in range(3):
            cv2.circle(img, tuple(p.astype(int)), 6, colors[i], -1, cv2.LINE_AA)
            if vals[i] is None:
                lost = True
                break
            cum += math.radians(vals[i])
            # straight (cum=0) points up: (sin, -cos)
            nxt = p + seg_len * np.array([math.sin(cum), -math.cos(cum)])
            cv2.line(img, tuple(p.astype(int)), tuple(nxt.astype(int)),
                     colors[i], 5, cv2.LINE_AA)
            cv2.putText(img, f"{names[i]} {vals[i]:+5.1f}",
                        (int(p[0]) + 10, int(p[1])), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, colors[i], 1, cv2.LINE_AA)
            p = nxt
        if not lost:
            cv2.circle(img, tuple(p.astype(int)), 6, (255, 255, 255), -1, cv2.LINE_AA)

        if lost:
            cv2.putText(img, "marker(s) lost", (10, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 1, cv2.LINE_AA)
        msg = ("live finger pose (zeroed)" if zeroed
               else "RAW pose - press SET ZERO to align")
        cv2.putText(img, msg, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (180, 180, 180) if zeroed else (80, 160, 255), 1, cv2.LINE_AA)
        return img

    def _capture(self, auto=False):
        # Redirect the CSV the base capture opens into mocap/results/ without
        # duplicating the (long) capture logic: swap the module-global CsvLogger
        # name the base method resolves, just for this call.
        orig = hwdash.CsvLogger
        hwdash.CsvLogger = self._logger_factory
        try:
            return super()._capture(auto=auto)
        finally:
            hwdash.CsvLogger = orig


# =====================================================================
# entry point
# =====================================================================
def main():
    p = argparse.ArgumentParser(description="PhaseSpace mocap finger-validation dashboard")
    p.add_argument("--mock", action="store_true",
                   help="no hardware (synthetic mocap + servo)")
    p.add_argument("--mock-servo", action="store_true",
                   help="real mocap, synthetic servo")
    p.add_argument("--server", default=mcfg.MOCAP_SERVER_IP,
                   help=f"PhaseSpace server IP (default {mcfg.MOCAP_SERVER_IP})")
    p.add_argument("--no-slave", action="store_true",
                   help="connect as the primary client (default: slave, so the "
                        "Master Client can stay open)")
    p.add_argument("--port", default="auto",
                   help="servo serial port; 'auto' (default) scans every COM/tty "
                        "port and baud and binds whatever Dynamixel answers")
    p.add_argument("--id", type=int, default=config.FINGER_A_DXL_ID,
                   help="preferred Dynamixel id; ignored if absent — auto-detect "
                        "then binds the id that actually responds")
    p.add_argument("--baud", type=int, default=57600)
    p.add_argument("--spool-radius", type=float, default=config.SPOOL_RADIUS,
                   help=f"tendon spool RADIUS [m] (default {config.SPOOL_RADIUS})")
    args = p.parse_args()

    # Moment arm for the analytical prediction. The analytical model applies the
    # ANGLE-DEPENDENT arm r(θ) = r0 + measured-curve(|θ|) internally (Picard fixed
    # point) whenever config.MOMENT_ARM_ANGLE_DEPENDENT is on — so what we supply
    # here is only the 0° base arm r0. It comes from the high-fidelity model when
    # finger.xml is present, else from config.SHEATH_MOMENT_ARM (= the curve's 0°
    # value). The θ-growth is applied regardless of which base is used.
    if getattr(config, "MOMENT_ARM_ANGLE_DEPENDENT", False):
        lo = config.MOMENT_ARM_CURVE_MM[0]
        hi = config.MOMENT_ARM_CURVE_MM[-1]
        ma_note = (f"  angle-dependent r(θ) ACTIVE: {lo:.1f}→{hi:.2f} mm measured "
                   f"curve over 0–{config.MOMENT_ARM_CURVE_DEG[-1]:.0f}°.")
    else:
        ma_note = "  ⚠ angle-dependence OFF (constant arm)."

    link_lengths = None
    try:
        r, link_lengths = predictor.get_geometry()
        geom_note = (f"base r0 (0°) from high-fidelity model: "
                     f"[{r[0]*1000:.2f}, {r[1]*1000:.2f}, {r[2]*1000:.2f}] mm." + ma_note)
    except Exception as e:  # noqa: BLE001
        r = np.full(3, config.SHEATH_MOMENT_ARM)
        geom_note = (f"base r0 = {config.SHEATH_MOMENT_ARM*1000:.1f} mm "
                     f"(config.SHEATH_MOMENT_ARM; finger.xml not built)." + ma_note)

    tracker = build_tracker(
        mock=args.mock,
        server=args.server,
        segment_marker_ids=mcfg.MOCAP_SEGMENT_MARKER_IDS,
        calib_path=mcfg.MOCAP_CALIB_PATH,
        timeout_us=mcfg.MOCAP_EVENT_TIMEOUT_US,
        slave=(not args.no_slave) and mcfg.MOCAP_SLAVE,
    )
    if args.mock or args.mock_servo:
        servo = MockServo(spool_radius_m=args.spool_radius)
    else:
        servo = Servo(port=args.port, baud=args.baud, dxl_id=args.id,
                      spool_radius_m=args.spool_radius)

    app = QApplication(sys.argv)
    app.setFont(QFont("Consolas", 10))
    win = MocapDashboard(tracker, servo, geom_r=r, geom_note=geom_note,
                         geom_link_lengths=link_lengths)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
