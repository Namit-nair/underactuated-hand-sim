#!/usr/bin/env python3
"""Hardware-validation dashboard for the tendon-driven 3R finger.

Single PySide6 window that:
  • shows the live RealSense preview with the ArUco overlay + per-marker status,
  • lets you jog the Dynamixel (←/→ or A/D, held), e-stop on Space,
  • captures the zero / reference pose,
  • slow-ramps the tendon to a commanded ΔL (5/10/15/20 mm presets or custom),
  • detects quasi-static settle, then on "Capture" reads the ArUco hardware
    joint angles, computes the analytical prediction (same geometry as
    high_fidelity/validation.py), logs a CSV row, and updates the live plots,
  • optionally AUTO-SWEEPs the ΔL list for repeat trials.

Everything physical is sourced from config.py and analytical_model.py — no
joint limit, moment arm, or spring value is hard-coded here.

Run:
    python3 dashboard.py                 # real RealSense + real Dynamixel
    python3 dashboard.py --mock          # no hardware (synthetic cam + servo)
    python3 dashboard.py --mock-camera   # real servo, fake camera (etc.)
    python3 dashboard.py --port /dev/ttyUSB0 --id 15 --baud 57600
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np

# --- repo single-source-of-truth ---------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402

# --- rig modules (same folder) ----------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import predictor  # noqa: E402
from camera import MARKER_LABELS, MockCamera, RealSenseAruco  # noqa: E402
from joints import JointAngles  # noqa: E402
from logger import CsvLogger  # noqa: E402
from servo import MockServo, Servo  # noqa: E402
from state_machine import AutoSweep, SettleDetector, State  # noqa: E402

# --- Qt / matplotlib ---------------------------------------------------------
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QFont, QImage, QKeyEvent, QPixmap  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

JOINTS = ("mcp", "pip", "dip")
JCOLORS = {"mcp": "#58a6ff", "pip": "#7ee787", "dip": "#ff7b72"}
DELTA_PRESETS = (5.0, 10.0, 15.0, 20.0)
TICK_MS = 50  # 20 Hz main loop


# =====================================================================
# Small styled helpers
# =====================================================================
def _btn(text, color="#c9d1d9", bg="#21262d", border="#30363d"):
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton {{ background:{bg}; color:{color}; border:1px solid {border};"
        f" border-radius:4px; padding:7px; font-family:Consolas; font-size:11px; }}"
        f" QPushButton:hover {{ background:#30363d; }}"
        f" QPushButton:disabled {{ color:#484f58; border-color:#21262d; }}")
    return b


def _group(title, color):
    g = QGroupBox(title)
    g.setStyleSheet(
        f"QGroupBox {{ color:{color}; border:1px solid {color}; border-radius:6px;"
        f" margin-top:10px; font-weight:bold; font-size:12px; }}"
        f" QGroupBox::title {{ subcontrol-origin: margin; left:10px; }}"
        f" QLabel {{ color:#c9d1d9; font-family:Consolas; font-size:11px; }}")
    return g


def _dspin(lo, hi, val, step, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setDecimals(4 if step < 0.01 else 2)
    if suffix:
        s.setSuffix(suffix)
    s.setStyleSheet("QDoubleSpinBox{background:#0d1117;color:#c9d1d9;"
                    "border:1px solid #30363d;border-radius:3px;padding:2px;}")
    return s


# =====================================================================
# Main window
# =====================================================================
class Dashboard(QMainWindow):
    def __init__(self, camera, servo, *, geom_r, geom_note):
        super().__init__()
        self.cam = camera
        self.servo = servo
        self.joints = JointAngles(flexion_sign=+1)
        self.settle = SettleDetector(vel_thresh_deg_s=2.0, hold_s=0.5, timeout_s=8.0)
        self.sweep = AutoSweep(DELTA_PRESETS, n_trials=5)

        self.r = np.asarray(geom_r, dtype=float)
        self.geom_note = geom_note
        self.state = State.IDLE
        self.logger = None
        self.captures = []          # list of {delta_L, exp{}, ana[]}
        self.last_capture = None
        self.target_mm = DELTA_PRESETS[0]
        self.settle_status = SettleDetector.SETTLING
        self.settle_time = float("nan")
        self.auto_active = False

        self._jog_dir = 0
        self._last_phi = {i: None for i in range(4)}
        self._last_visible = {i: False for i in range(4)}
        self._last_theta = {j: None for j in JOINTS}
        self._frame_buf = None
        self._t_prev = time.monotonic()

        self.setWindowTitle("Finger Hardware-Validation Rig — IITGN")
        self.setStyleSheet("background-color:#0d1117; color:#c9d1d9;")
        self.setMinimumSize(1320, 860)
        self.setFocusPolicy(Qt.StrongFocus)

        self._build_ui()

        # main 20 Hz loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(TICK_MS)

        self._refresh_spring_readout()
        self._set_state(State.IDLE)

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        # ---- LEFT: camera preview + plots ----
        left = QVBoxLayout()
        self.preview = QLabel("camera preview")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(640, 380)
        self.preview.setStyleSheet("background:#000; border:1px solid #30363d;")
        left.addWidget(self.preview)

        # marker status strip
        strip = QHBoxLayout()
        self.marker_dots = {}
        for i in range(4):
            lab = QLabel(f"  ●  M{i} {MARKER_LABELS[i]}  ")
            lab.setStyleSheet("color:#da3633; font-family:Consolas; font-size:11px;")
            strip.addWidget(lab)
            self.marker_dots[i] = lab
        strip.addStretch()
        left.addLayout(strip)

        # embedded plots
        self.fig = Figure(figsize=(7, 3.4), facecolor="#0d1117")
        self.canvas = FigureCanvas(self.fig)
        self.ax_bar = self.fig.add_subplot(1, 2, 1)
        self.ax_curve = self.fig.add_subplot(1, 2, 2)
        self.fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.16, wspace=0.3)
        left.addWidget(self.canvas)
        root.addLayout(left, stretch=3)

        # ---- RIGHT: controls ----
        right = QVBoxLayout()
        right.addWidget(self._connect_group())
        right.addWidget(self._spring_group())
        right.addWidget(self._jog_zero_group())
        right.addWidget(self._delta_group())
        right.addWidget(self._capture_group())
        right.addWidget(self._readout_group())
        right.addStretch()
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(430)
        root.addWidget(rw)

        self.setCentralWidget(central)
        self._init_plots()

    def _connect_group(self):
        g = _group("CONNECTION", "#00d4ff")
        lay = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_conn_cam = _btn("CONNECT CAMERA", "#58a6ff", border="#58a6ff")
        self.btn_conn_servo = _btn("CONNECT SERVO", "#58a6ff", border="#58a6ff")
        self.btn_conn_cam.clicked.connect(self._connect_camera)
        self.btn_conn_servo.clicked.connect(self._connect_servo)
        row.addWidget(self.btn_conn_cam)
        row.addWidget(self.btn_conn_servo)
        lay.addLayout(row)
        self.lbl_conn = QLabel("camera: —   servo: —")
        self.lbl_conn.setStyleSheet("color:#8b949e;")
        lay.addWidget(self.lbl_conn)
        if self.geom_note:
            note = QLabel(self.geom_note)
            note.setWordWrap(True)
            note.setStyleSheet("color:#d29922; font-size:10px;")
            lay.addWidget(note)
        g.setLayout(lay)
        return g

    def _spring_group(self):
        g = _group("INSTALLED SPRINGS  (custom k, N·m/rad)", "#d2a8ff")
        grid = QGridLayout()
        grid.addWidget(QLabel("label"), 0, 0)
        self.lbl_label = QComboBox()
        self.lbl_label.setEditable(True)
        self.lbl_label.addItems(["custom", "uniform", "proximal_dominant",
                                 "distal_dominant"])
        self.lbl_label.setStyleSheet("QComboBox{background:#0d1117;color:#c9d1d9;"
                                     "border:1px solid #30363d;padding:2px;}")
        grid.addWidget(self.lbl_label, 0, 1, 1, 3)

        grid.addWidget(QLabel("k_mcp (k1)"), 1, 0)
        grid.addWidget(QLabel("k_pip (k2)"), 1, 1)
        grid.addWidget(QLabel("k_dip (k3)"), 1, 2)
        self.k1 = _dspin(0.001, 5.0, config.SPRING_2, 0.001)
        self.k2 = _dspin(0.001, 5.0, config.SPRING_2, 0.001)
        self.k3 = _dspin(0.001, 5.0, config.SPRING_2, 0.001)
        for s in (self.k1, self.k2, self.k3):
            s.valueChanged.connect(self._refresh_spring_readout)
        grid.addWidget(self.k1, 2, 0)
        grid.addWidget(self.k2, 2, 1)
        grid.addWidget(self.k3, 2, 2)
        self.lbl_rho = QLabel("ρ1 = —   ρ3 = —")
        self.lbl_rho.setStyleSheet("color:#d2a8ff;")
        grid.addWidget(self.lbl_rho, 3, 0, 1, 4)
        g.setLayout(grid)
        return g

    def _jog_zero_group(self):
        g = _group("JOG  &  ZERO", "#ffd33d")
        lay = QVBoxLayout()
        hint = QLabel("←/A = CCW   →/D = CW (hold to move)   Space = E-STOP")
        hint.setStyleSheet("color:#8b949e; font-size:10px;")
        lay.addWidget(hint)
        row = QHBoxLayout()
        self.btn_ccw = _btn("◀ CCW")
        self.btn_cw = _btn("CW ▶")
        self.btn_estop = _btn("E-STOP", "white", "#da3633", "#da3633")
        self.btn_ccw.pressed.connect(lambda: self._set_jog(-1))
        self.btn_ccw.released.connect(lambda: self._set_jog(0))
        self.btn_cw.pressed.connect(lambda: self._set_jog(+1))
        self.btn_cw.released.connect(lambda: self._set_jog(0))
        self.btn_estop.clicked.connect(self._do_estop)
        row.addWidget(self.btn_ccw)
        row.addWidget(self.btn_cw)
        row.addWidget(self.btn_estop)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("step (rev)"))
        self.jog_step = _dspin(0.001, 0.2, 0.01, 0.001)
        row2.addWidget(self.jog_step)
        self.btn_flip_pull = _btn("FLIP PULL DIR")
        self.btn_flip_sign = _btn("FLIP θ SIGN")
        self.btn_flip_pull.clicked.connect(self._flip_pull)
        self.btn_flip_sign.clicked.connect(self._flip_sign)
        row2.addWidget(self.btn_flip_pull)
        row2.addWidget(self.btn_flip_sign)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_enable = _btn("TORQUE ON", "white", "#238636", "#238636")
        self.btn_zero = _btn("◎ SET ZERO", "#0d1117", "#ffd33d", "#ffd33d")
        self.btn_enable.clicked.connect(self._enable_torque)
        self.btn_zero.clicked.connect(self._set_zero)
        row3.addWidget(self.btn_enable)
        row3.addWidget(self.btn_zero)
        lay.addLayout(row3)
        self.lbl_zero = QLabel("not zeroed")
        self.lbl_zero.setStyleSheet("color:#8b949e;")
        lay.addWidget(self.lbl_zero)
        g.setLayout(lay)
        return g

    def _delta_group(self):
        g = _group("ΔL CONTROL", "#79c0ff")
        lay = QVBoxLayout()
        row = QHBoxLayout()
        self.preset_btns = {}
        for d in DELTA_PRESETS:
            b = _btn(f"{d:.0f}")
            b.clicked.connect(lambda _=False, v=d: self._pick_delta(v))
            row.addWidget(b)
            self.preset_btns[d] = b
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("target ΔL"))
        self.delta_spin = _dspin(0.0, config.MAX_DELTA_L * 1000, DELTA_PRESETS[0], 0.5, " mm")
        self.delta_spin.setMaximum(25.0)  # soft cap
        self.delta_spin.valueChanged.connect(lambda v: setattr(self, "target_mm", v))
        row2.addWidget(self.delta_spin)
        row2.addWidget(QLabel("speed"))
        self.speed_spin = _dspin(0.2, 10.0, 2.0, 0.1, " mm/s")
        row2.addWidget(self.speed_spin)
        lay.addLayout(row2)
        row3 = QHBoxLayout()
        self.btn_go = _btn("▶ GO (ramp)", "#0d1117", "#79c0ff", "#79c0ff")
        self.btn_go.clicked.connect(self._go_ramp)
        row3.addWidget(self.btn_go)
        self.lbl_settle = QLabel("—")
        self.lbl_settle.setStyleSheet("color:#8b949e; font-weight:bold;")
        row3.addWidget(self.lbl_settle)
        lay.addLayout(row3)
        g.setLayout(lay)
        return g

    def _capture_group(self):
        g = _group("CAPTURE  &  AUTO-SWEEP", "#56d364")
        lay = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_capture = _btn("◉ CAPTURE", "white", "#238636", "#238636")
        self.btn_capture.clicked.connect(self._capture)
        row.addWidget(self.btn_capture)
        self.lbl_trial = QLabel("rows: 0")
        self.lbl_trial.setStyleSheet("color:#8b949e;")
        row.addWidget(self.lbl_trial)
        lay.addLayout(row)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("trials"))
        self.trial_spin = QSpinBox()
        self.trial_spin.setRange(1, 50)
        self.trial_spin.setValue(5)
        self.trial_spin.setStyleSheet("QSpinBox{background:#0d1117;color:#c9d1d9;"
                                      "border:1px solid #30363d;padding:2px;}")
        row2.addWidget(self.trial_spin)
        self.btn_sweep = _btn("AUTO SWEEP", "#0d1117", "#56d364", "#56d364")
        self.btn_sweep.clicked.connect(self._toggle_sweep)
        row2.addWidget(self.btn_sweep)
        lay.addLayout(row2)
        self.lbl_sweep = QLabel("sweep idle")
        self.lbl_sweep.setStyleSheet("color:#8b949e; font-size:10px;")
        lay.addWidget(self.lbl_sweep)
        g.setLayout(lay)
        return g

    def _readout_group(self):
        g = _group("READOUTS", "#58a6ff")
        grid = QGridLayout()

        def row(name, r):
            grid.addWidget(QLabel(name), r, 0)
            v = QLabel("--")
            v.setAlignment(Qt.AlignRight)
            v.setStyleSheet("color:#c9d1d9; font-family:Consolas;")
            grid.addWidget(v, r, 1)
            return v

        self.ro_state = row("state", 0)
        self.ro_delta = row("ΔL now (mm)", 1)
        self.ro_pos = row("servo pos (rev)", 2)
        self.ro_cur = row("servo current (mA)", 3)
        self.ro_theta = row("θ exp mcp/pip/dip", 4)
        self.ro_ana = row("θ ana mcp/pip/dip", 5)
        self.ro_csv = row("csv", 6)
        g.setLayout(grid)
        return g

    # -----------------------------------------------------------------
    # plotting
    # -----------------------------------------------------------------
    def _style_ax(self, ax):
        ax.set_facecolor("#0d1117")
        for s in ax.spines.values():
            s.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)
        ax.xaxis.label.set_color("#c9d1d9")
        ax.yaxis.label.set_color("#c9d1d9")
        ax.title.set_color("#c9d1d9")
        ax.grid(True, color="#21262d", linestyle="--", linewidth=0.5)

    def _init_plots(self):
        self.ax_bar.clear()
        self.ax_curve.clear()
        self.ax_bar.set_title("current capture", fontsize=9)
        self.ax_bar.set_ylabel("angle (deg)", fontsize=8)
        self.ax_curve.set_title("ΔL vs angle", fontsize=9)
        self.ax_curve.set_xlabel("ΔL (mm)", fontsize=8)
        self.ax_curve.set_ylabel("angle (deg)", fontsize=8)
        for ax in (self.ax_bar, self.ax_curve):
            self._style_ax(ax)
        self.canvas.draw_idle()

    def _redraw_plots(self):
        # ---- (a) grouped bar of current capture ----
        self.ax_bar.clear()
        self._style_ax(self.ax_bar)
        self.ax_bar.set_title("current capture", fontsize=9)
        self.ax_bar.set_ylabel("angle (deg)", fontsize=8)
        x = np.arange(3)
        w = 0.36
        if self.last_capture is not None:
            exp = [self.last_capture["exp"][j] for j in JOINTS]
            ana = list(self.last_capture["ana"])
            self.ax_bar.bar(x - w / 2, exp, w, color="#C0392B", label="exp")
            self.ax_bar.bar(x + w / 2, ana, w, color="#1F4E79", label="ana")
            for i, j in enumerate(JOINTS):
                e = (exp[i] or 0.0) - ana[i]
                top = max(exp[i] or 0.0, ana[i])
                self.ax_bar.text(i, top + 1.5, f"Δ{e:+.1f}°", ha="center",
                                 color="#8b949e", fontsize=7.5)
            self.ax_bar.legend(fontsize=7, facecolor="#161b22",
                               edgecolor="#30363d", labelcolor="#c9d1d9")
        self.ax_bar.set_xticks(x)
        self.ax_bar.set_xticklabels([j.upper() for j in JOINTS], fontsize=8)

        # ---- (b) accumulating ΔL vs angle: analytical curves + exp points ----
        self.ax_curve.clear()
        self._style_ax(self.ax_curve)
        self.ax_curve.set_title("ΔL vs angle", fontsize=9)
        self.ax_curve.set_xlabel("ΔL (mm)", fontsize=8)
        self.ax_curve.set_ylabel("angle (deg)", fontsize=8)
        k_vec = self._k_vec()
        dl = np.linspace(0, 25, 26)
        try:
            curve = np.array([predictor.predict(d, k_vec, r=self.r) for d in dl]).T
            for i, j in enumerate(JOINTS):
                self.ax_curve.plot(dl, curve[i], "-", color=JCOLORS[j],
                                   lw=1.6, label=f"{j.upper()} ana")
        except Exception:
            pass
        # experimental scatter
        for j in JOINTS:
            xs = [c["delta_L"] for c in self.captures if c["exp"][j] is not None]
            ys = [c["exp"][j] for c in self.captures if c["exp"][j] is not None]
            if xs:
                self.ax_curve.plot(xs, ys, "o", color=JCOLORS[j], ms=5,
                                   mfc="white", mew=1.2)
        self.ax_curve.legend(fontsize=6.5, facecolor="#161b22",
                             edgecolor="#30363d", labelcolor="#c9d1d9", loc="upper left")
        self.canvas.draw_idle()

    # -----------------------------------------------------------------
    # spring helpers
    # -----------------------------------------------------------------
    def _k_vec(self):
        return np.array([self.k1.value(), self.k2.value(), self.k3.value()])

    def _refresh_spring_readout(self):
        k = self._k_vec()
        rho1 = k[0] / k[1] if k[1] else float("nan")
        rho3 = k[2] / k[1] if k[1] else float("nan")
        self.lbl_rho.setText(f"ρ1 = {rho1:.3f}   ρ3 = {rho3:.3f}")
        self._redraw_plots()

    # -----------------------------------------------------------------
    # connection
    # -----------------------------------------------------------------
    def _connect_camera(self):
        try:
            self.cam.start()
            self.btn_conn_cam.setText("CAMERA ✓")
            self.btn_conn_cam.setEnabled(False)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Camera", str(e))
        self._update_conn_label()

    def _connect_servo(self):
        ok, msg = self.servo.connect()
        if not ok:
            QMessageBox.critical(self, "Servo", msg or "connect failed")
        else:
            self.btn_conn_servo.setText("SERVO ✓")
            self.btn_conn_servo.setEnabled(False)
            if self.state == State.IDLE:
                self._set_state(State.JOG)
        self._update_conn_label()

    def _update_conn_label(self):
        cam_ok = getattr(self.cam, "_started", None)
        st = self.servo.get_state()
        self.lbl_conn.setText(
            f"camera: {'on' if self.btn_conn_cam.text().endswith('✓') else '—'}   "
            f"servo: {'on' if st.get('connected') else '—'}")

    # -----------------------------------------------------------------
    # jog / safety
    # -----------------------------------------------------------------
    def _set_jog(self, d):
        self._jog_dir = d

    def _enable_torque(self):
        if self.servo.get_state().get("connected"):
            self.servo.enable()

    def _do_estop(self):
        self.servo.e_stop()
        self._jog_dir = 0
        self.auto_active = False
        self.lbl_settle.setText("E-STOP")
        self.lbl_settle.setStyleSheet("color:#f85149; font-weight:bold;")

    def _flip_pull(self):
        st = self.servo.get_state()
        sign = -1 if getattr(self.servo, "pull_sign", 1) > 0 else +1
        self.servo.set_pull_direction(sign)
        QMessageBox.information(self, "Pull direction",
                                f"pull_sign set to {sign}. Re-jog to confirm a "
                                f"CW nudge flexes the finger.")

    def _flip_sign(self):
        self.joints.set_flexion_sign(-self.joints.flexion_sign)
        QMessageBox.information(self, "θ sign",
                                f"flexion_sign = {self.joints.flexion_sign} "
                                f"(flexion should read positive).")

    def _set_zero(self):
        phi = self._last_phi
        ok = self.joints.set_zero(phi)
        if not ok:
            QMessageBox.warning(self, "Set Zero",
                                "Need all 4 markers visible to capture the "
                                "reference pose. Check the preview.")
            return
        if self.servo.get_state().get("connected"):
            self.servo.set_zero()
        self.lbl_zero.setText("zeroed ✓  (θ=0, ΔL=0)")
        self.lbl_zero.setStyleSheet("color:#56d364;")
        self._set_state(State.ZEROED)

    # -----------------------------------------------------------------
    # ΔL ramp + settle
    # -----------------------------------------------------------------
    def _pick_delta(self, v):
        self.target_mm = v
        self.delta_spin.setValue(v)

    def _go_ramp(self):
        if not self.joints.is_zeroed():
            QMessageBox.warning(self, "Go", "Set Zero first.")
            return
        if not self.servo.get_state().get("connected"):
            QMessageBox.warning(self, "Go", "Connect the servo first.")
            return
        target = float(self.delta_spin.value())
        ok = self.servo.start_ramp(target, speed_mm_s=float(self.speed_spin.value()))
        if not ok:
            QMessageBox.warning(self, "Go",
                                f"Ramp refused (ΔL {target} mm beyond soft cap or "
                                f"e-stopped). Re-enable torque / lower ΔL.")
            return
        self.target_mm = target
        self._set_state(State.RAMP)

    # -----------------------------------------------------------------
    # capture
    # -----------------------------------------------------------------
    def _capture(self, auto=False):
        all_vis = all(self._last_visible.values())
        exp = dict(self._last_theta)
        if not all_vis or any(exp[j] is None for j in JOINTS):
            if not auto:
                QMessageBox.warning(self, "Capture",
                                    "Not all markers visible — cannot read all "
                                    "joint angles. Adjust the finger/camera.")
            return False

        k_vec = self._k_vec()
        try:
            ana = np.asarray(predictor.predict(self.target_mm, k_vec, r=self.r), float)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Analytical", f"predict() failed: {e}")
            return False

        exp_arr = np.array([exp[j] for j in JOINTS], float)
        m12_e, m32_e = predictor.metrics(exp_arr)
        m12_a, m32_a = predictor.metrics(ana)
        st = self.servo.get_state()

        if self.logger is None:
            self.logger = CsvLogger(self.lbl_label.currentText().strip() or "custom")
            self.ro_csv.setText(os.path.basename(self.logger.filepath))

        rho1 = k_vec[0] / k_vec[1] if k_vec[1] else float("nan")
        rho3 = k_vec[2] / k_vec[1] if k_vec[1] else float("nan")
        trial = self.sweep.trial_idx if self.auto_active else 0
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "spring_set_label": self.lbl_label.currentText().strip() or "custom",
            "rho1": rho1, "rho3": rho3,
            "k_mcp": k_vec[0], "k_pip": k_vec[1], "k_dip": k_vec[2],
            "delta_L_mm": self.target_mm,
            "servo_pos": st.get("pos_rev"), "servo_current": st.get("current_ma"),
            "theta_mcp_exp": exp["mcp"], "theta_pip_exp": exp["pip"], "theta_dip_exp": exp["dip"],
            "theta_mcp_ana": ana[0], "theta_pip_ana": ana[1], "theta_dip_ana": ana[2],
            "err_mcp": exp["mcp"] - ana[0],
            "err_pip": exp["pip"] - ana[1],
            "err_dip": exp["dip"] - ana[2],
            "M12_exp": m12_e, "M32_exp": m32_e, "M12_ana": m12_a, "M32_ana": m32_a,
            "markers_all_visible": all_vis,
            "settle_time_s": self.settle_time,
            "trial_idx": trial,
        }
        self.logger.log(row)
        self.captures.append({"delta_L": self.target_mm, "exp": dict(exp), "ana": ana})
        self.last_capture = self.captures[-1]
        self.lbl_trial.setText(f"rows: {self.logger.n_rows()}")
        self._set_state(State.CAPTURE)
        self._redraw_plots()
        return True

    # -----------------------------------------------------------------
    # auto sweep
    # -----------------------------------------------------------------
    def _toggle_sweep(self):
        if self.auto_active:
            self.auto_active = False
            self.btn_sweep.setText("AUTO SWEEP")
            self.lbl_sweep.setText("sweep stopped")
            return
        if not self.joints.is_zeroed() or not self.servo.get_state().get("connected"):
            QMessageBox.warning(self, "Auto sweep", "Connect servo and Set Zero first.")
            return
        self.sweep = AutoSweep(DELTA_PRESETS, n_trials=self.trial_spin.value())
        self.auto_active = True
        self.btn_sweep.setText("STOP SWEEP")
        self._sweep_next()

    def _sweep_next(self):
        tgt = self.sweep.current_target()
        if tgt is None:
            self.auto_active = False
            self.btn_sweep.setText("AUTO SWEEP")
            self.lbl_sweep.setText("sweep complete")
            return
        self.delta_spin.setValue(tgt)
        self.target_mm = tgt
        self.servo.start_ramp(tgt, speed_mm_s=float(self.speed_spin.value()))
        self.lbl_sweep.setText(self.sweep.progress())
        self._set_state(State.RAMP)

    # -----------------------------------------------------------------
    # state
    # -----------------------------------------------------------------
    def _set_state(self, s):
        self.state = s
        self.ro_state.setText(s.label())
        if s in (State.SETTLING,):
            self.lbl_settle.setText("SETTLING…")
            self.lbl_settle.setStyleSheet("color:#d29922; font-weight:bold;")
        elif s == State.SETTLED:
            self.lbl_settle.setText("SETTLED ✓")
            self.lbl_settle.setStyleSheet("color:#56d364; font-weight:bold;")
        elif s == State.RAMP:
            self.lbl_settle.setText("ramping…")
            self.lbl_settle.setStyleSheet("color:#79c0ff; font-weight:bold;")

    # -----------------------------------------------------------------
    # main loop
    # -----------------------------------------------------------------
    def _tick(self):
        now = time.monotonic()
        dt = now - self._t_prev
        self._t_prev = now

        # 1) camera
        try:
            det = self.cam.detect()
            self._last_phi = det["phi"]
            self._last_visible = det["visible"]
            self._show_frame(det["frame"])
        except Exception:
            pass
        self._update_marker_dots()

        # 2) servo service (advances ramp + telemetry + overcurrent estop)
        try:
            st = self.servo.service(dt=dt)
        except Exception:
            st = self.servo.get_state()

        # 3) joint angles
        self._last_theta = self.joints.compute(self._last_phi)

        # 4) jog while held
        if self._jog_dir and st.get("connected") and not st.get("estop"):
            if self.state in (State.JOG, State.ZEROED, State.SETTLED, State.CAPTURE):
                self.servo.jog(self._jog_dir, step_rev=float(self.jog_step.value()))

        # 5) state transitions
        if self.state == State.RAMP and not self.servo.is_ramping():
            self.settle.start(now)
            self._set_state(State.SETTLING)
        elif self.state == State.SETTLING:
            res = self.settle.update(self._last_theta, now)
            self.settle_time = self.settle.elapsed(now)
            if res == SettleDetector.SETTLED:
                self.settle_status = res
                self._set_state(State.SETTLED)
            elif res == SettleDetector.TIMEOUT:
                self.settle_status = res
                self._set_state(State.SETTLED)
                self.lbl_settle.setText("SETTLED (timeout)")
                self.lbl_settle.setStyleSheet("color:#d29922; font-weight:bold;")
        elif self.state == State.SETTLED and self.auto_active:
            # auto-capture once quasi-static; retries next tick if markers were hidden
            if self._capture(auto=True):
                self.sweep.advance()
                self._sweep_next()

        # 6) overcurrent surfaced
        if st.get("over_current"):
            self.lbl_settle.setText("OVER-CURRENT E-STOP")
            self.lbl_settle.setStyleSheet("color:#f85149; font-weight:bold;")

        self._update_readouts(st)

    def _show_frame(self, frame):
        if frame is None:
            return
        rgb = frame[:, :, ::-1].copy()  # BGR->RGB, contiguous
        self._frame_buf = rgb
        h, w = rgb.shape[:2]
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img).scaled(
            self.preview.width(), self.preview.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setPixmap(pix)

    def _update_marker_dots(self):
        for i in range(4):
            vis = self._last_visible.get(i, False)
            color = "#56d364" if vis else "#da3633"
            self.marker_dots[i].setStyleSheet(
                f"color:{color}; font-family:Consolas; font-size:11px;")

    def _update_readouts(self, st):
        self.ro_delta.setText(f"{st.get('delta_L_mm', float('nan')):.2f}")
        self.ro_pos.setText(f"{st.get('pos_rev', float('nan')):.3f}")
        cur = st.get("current_ma", float("nan"))
        self.ro_cur.setText(f"{cur:.0f}")
        t = self._last_theta
        self.ro_theta.setText(" / ".join(
            "--" if t[j] is None else f"{t[j]:.1f}" for j in JOINTS))
        try:
            ana = predictor.predict(self.target_mm, self._k_vec(), r=self.r)
            self.ro_ana.setText(" / ".join(f"{a:.1f}" for a in ana))
        except Exception:
            self.ro_ana.setText("-- (no mujoco)")

    # -----------------------------------------------------------------
    # keyboard
    # -----------------------------------------------------------------
    def keyPressEvent(self, e: QKeyEvent):
        if e.isAutoRepeat():
            return
        if e.key() in (Qt.Key_Right, Qt.Key_D):
            self._set_jog(+1)
        elif e.key() in (Qt.Key_Left, Qt.Key_A):
            self._set_jog(-1)
        elif e.key() == Qt.Key_Space:
            self._do_estop()
        else:
            super().keyPressEvent(e)

    def keyReleaseEvent(self, e: QKeyEvent):
        if e.isAutoRepeat():
            return
        if e.key() in (Qt.Key_Right, Qt.Key_D, Qt.Key_Left, Qt.Key_A):
            self._set_jog(0)
        else:
            super().keyReleaseEvent(e)

    def closeEvent(self, e):
        try:
            self.servo.disable()
            self.servo.disconnect()
        except Exception:
            pass
        try:
            self.cam.stop()
        except Exception:
            pass
        if self.logger:
            self.logger.close()
        super().closeEvent(e)


# =====================================================================
# entry point
# =====================================================================
def _build_devices(args):
    if args.mock or args.mock_camera:
        cam = MockCamera()
    else:
        cam = RealSenseAruco(width=args.width, height=args.height, fps=args.fps)
    if args.mock or args.mock_servo:
        servo = MockServo(spool_radius_m=args.spool_radius)
    else:
        servo = Servo(port=args.port, baud=args.baud, dxl_id=args.id,
                      spool_radius_m=args.spool_radius)
    return cam, servo


def main():
    p = argparse.ArgumentParser(description="Finger hardware-validation dashboard")
    p.add_argument("--mock", action="store_true", help="no hardware (mock cam+servo)")
    p.add_argument("--mock-camera", action="store_true")
    p.add_argument("--mock-servo", action="store_true")
    p.add_argument("--port", default="/dev/ttyUSB0")
    p.add_argument("--id", type=int, default=15)
    p.add_argument("--baud", type=int, default=57600)
    p.add_argument("--spool-radius", type=float, default=0.0125,
                   help="tendon spool RADIUS [m] (diameter 25mm -> 0.0125)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    # geometry (same source of truth as validation.py); fall back to a constant
    # sheath arm if mujoco is unavailable so the rig is still usable.
    geom_note = ""
    try:
        r, _ = predictor.get_geometry()
        geom_note = (f"r (moment arms) from high-fidelity model: "
                     f"[{r[0]*1000:.2f}, {r[1]*1000:.2f}, {r[2]*1000:.2f}] mm")
    except Exception as e:  # noqa: BLE001
        r = np.full(3, config.SHEATH_MOMENT_ARM)
        geom_note = (f"⚠ mujoco/finger.xml unavailable ({type(e).__name__}); "
                     f"using constant r = {config.SHEATH_MOMENT_ARM*1000:.1f} mm.")

    app = QApplication(sys.argv)
    app.setFont(QFont("Consolas", 10))
    cam, servo = _build_devices(args)
    win = Dashboard(cam, servo, geom_r=r, geom_note=geom_note)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
