#!/usr/bin/env python3
"""Load-carrying (pull-out) hardware test dashboard.

Physical counterpart of the *simulated* two-finger load test
(``gripper/interactive_load_test.py``). All three Dynamixels share ONE U2D2:
the two tendon-driven fingers (A + B) are daisy-chained on one hub port and the
pull servo hangs off another hub port — electrically the same serial bus, so the
motors are told apart purely by Dynamixel ID. The fingers close around an object
while the pull servo winds a stainless string that pulls the object out through a
**Futek LCM300** axial load cell read over a **USB220** serial module. We tension
the tendons, zero the cell, grip, then ramp the pull and record the **peak force
at which the grip releases** — the load-carrying capacity. That measured force is
the ground truth the analytical grip-force model will be validated against.

Run::

    python3 load_test_dashboard.py --mock                 # no hardware at all
    python3 load_test_dashboard.py --mock-servo           # real cell, mock motors
    python3 load_test_dashboard.py \
        --finger-port /dev/serial/by-id/...U2D2 \
        --loadcell-port /dev/ttyUSB1

Workflow (top to bottom in the panel):
    1. CONNECT fingers + pull (all on the one U2D2 bus) and the load cell.
    2. TENDON TENSIONING — jog each finger to take up slack, then SET ZERO it.
    3. LOAD SENSOR — TARE / ZERO the cell with no load on the string.
    4. GRIP — ramp both fingers to the grip ΔL to close on the object.
    5. PULL TEST — START PULL; the cell force rises until the grip releases;
       the peak force is latched as the load capacity and logged to CSV.

Reuses the safety-wrapped Servo (hardware/servo.py), the tolerant CsvLogger
(hardware/logger.py), and the dashboard widget helpers (_btn/_dspin/_group).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import deque
from datetime import datetime

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import load_cell  # noqa: E402  (imported as a module so the patch below sticks)
import servo  # noqa: E402  (ditto — autodetect_* call the module-level enumerator)
from dashboard import _btn, _dspin, _group  # noqa: E402  (reuse widget helpers)
from load_cell import LoadCell, MockLoadCell  # noqa: E402
from logger import CsvLogger  # noqa: E402
from servo import MockServo, Servo, install_emergency_shutdown, open_bus  # noqa: E402


# --- cross-platform serial-port enumeration ---------------------------------
# servo.autodetect_servo / load_cell.autodetect_loadcell scan every port + baud,
# so port="auto" binds the device regardless of which port it landed on. The one
# gap is that both modules' list_serial_ports() globs the Linux /dev/ttyUSB*
# paths; on Windows the U2D2 and the USB220 enumerate as COMx. We patch the port
# enumerator (in-process only — servo.py / load_cell.py are untouched) to use
# pyserial's cross-platform comports(), falling back to the Linux globs.
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


servo.list_serial_ports = _list_serial_ports_xplat
load_cell.list_serial_ports = _list_serial_ports_xplat

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PySide6.QtCore import Qt, QTimer  # noqa: E402
from PySide6.QtGui import QFont, QKeyEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

TICK_MS = 30
PLOT_WINDOW_S = 30.0           # rolling time axis
_REDRAW_EVERY = 3              # redraw plots every Nth tick (~10 Hz)

# Mock pull-out plant (only used with --mock / --mock-loadcell): the grip can
# hold a force proportional to the grip ΔL; the string tension rises with the
# pull ΔL until it exceeds that hold force and the object slips.
_MOCK_GRIP_N_PER_MM = 9.0
_MOCK_PULL_N_PER_MM = 14.0
_MOCK_PULL_SLACK_MM = 6.0
_MOCK_RESIDUAL_N = 0.4

_LOG_COLUMNS = [
    "timestamp", "trial_idx", "event", "state",
    "force_n", "force_raw_n", "peak_n", "capacity_n",
    "grip_target_mm",
    "finger_a_dL_mm", "finger_b_dL_mm",
    "finger_a_current_ma", "finger_b_current_ma",
    "pull_dL_mm", "pull_current_ma",
    "k_mcp", "k_pip", "k_dip",
]


class TestState:
    IDLE = "IDLE"
    TENSIONED = "TENSIONED"
    GRIPPED = "GRIPPED"
    PULLING = "PULLING"
    RELEASED = "RELEASED"


class LoadTestDashboard(QMainWindow):
    """Three servos (finger A/B + pull) plus a load cell, in one control panel."""

    def __init__(self, finger_a, finger_b, pull, cell, *,
                 finger_cap_mm, pull_cap_mm, finger_speed_mm_s, pull_speed_mm_s,
                 finger_port="auto", finger_ids=(15, 16)):
        super().__init__()
        self.fa = finger_a
        self.fb = finger_b
        self.pull = pull
        self.cell = cell
        self.finger_cap = float(finger_cap_mm)
        self.pull_cap = float(pull_cap_mm)
        self.finger_speed = float(finger_speed_mm_s)
        self.pull_speed_default = float(pull_speed_mm_s)
        self.finger_port = finger_port
        self.finger_ids = tuple(finger_ids)
        # All three servos (A, B, pull) live on the SAME U2D2 bus, told apart by
        # Dynamixel ID. We open that bus once and attach every motor to it.
        self.bus_ids = self.finger_ids + (
            getattr(pull, "dxl_id", config.PULL_DXL_ID),)

        self.state = TestState.IDLE
        self.logger = None
        # Persistent trial counter: each trial gets its own CSV (trialN). On
        # startup we resume from one past the highest trialN already in results/
        # so a fresh dashboard continues the numbering instead of overwriting.
        self.trial_idx = self._next_trial_number()
        self.capacity_n = float("nan")     # latched load capacity (last release)
        self._run_peak = 0.0               # running peak force during a pull
        self._bus_handles = None           # (port_handler, packet_handler) if shared
        self._mock_released = False
        self._t0 = time.monotonic()
        self._t_prev = self._t0
        self._tick_n = 0

        # rolling plot buffers
        self.buf_t = deque(maxlen=2000)
        self.buf_force = deque(maxlen=2000)
        self.buf_fa = deque(maxlen=2000)
        self.buf_fb = deque(maxlen=2000)
        self.buf_pull = deque(maxlen=2000)
        self.buf_ica = deque(maxlen=2000)
        self.buf_icb = deque(maxlen=2000)
        self.buf_icp = deque(maxlen=2000)

        self.setWindowTitle("Load-Carrying (Pull-Out) Test — IITGN")
        self.setStyleSheet("background-color:#0d1117; color:#c9d1d9;")
        self.setMinimumSize(1280, 860)
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(TICK_MS)

        # Kill all three servos on ANY exit (Ctrl-C / SIGTERM too), not just a
        # clicked window close — otherwise the motors stay energised after the
        # process dies. _shutdown is idempotent (closeEvent + atexit + signal).
        self._shutdown_done = False
        install_emergency_shutdown(self._shutdown)

    # =================================================================
    # UI
    # =================================================================
    def _build_ui(self):
        central = QWidget()
        root = QHBoxLayout(central)

        # ---- LEFT: plots ----
        left = QVBoxLayout()
        self.fig = Figure(figsize=(7, 7.6), facecolor="#0d1117")
        self.canvas = FigureCanvas(self.fig)
        self.ax_force = self.fig.add_subplot(3, 1, 1)
        self.ax_dl = self.fig.add_subplot(3, 1, 2)
        self.ax_cur = self.fig.add_subplot(3, 1, 3)
        self.fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.07,
                                 hspace=0.45)
        left.addWidget(self.canvas)
        root.addLayout(left, stretch=3)

        # ---- RIGHT: controls (scrollable) ----
        right = QVBoxLayout()
        right.addWidget(self._banner())
        right.addWidget(self._connect_group())
        right.addWidget(self._stiffness_group())
        right.addWidget(self._tension_group())
        right.addWidget(self._sensor_group())
        right.addWidget(self._grip_group())
        right.addWidget(self._pull_group())
        right.addWidget(self._readout_group())
        right.addStretch()
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(450)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(rw)
        scroll.setFixedWidth(470)
        scroll.setStyleSheet("QScrollArea{border:none;}")
        root.addWidget(scroll)

        self.setCentralWidget(central)
        self._init_plots()

    def _banner(self):
        a_id, b_id = self.finger_ids
        pull_id = getattr(self.pull, "dxl_id", config.PULL_DXL_ID)
        lbl = QLabel(
            f"Fingers {config.GRIPPER_APERTURE_MAX_MM:.0f} mm apart   ·   "
            f"A=id{a_id}  B=id{b_id}  pull=id{pull_id} (one U2D2)   ·   "
            f"Futek LCM300")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            "color:#8b949e; background-color:#161b22; border:1px solid #30363d;"
            "border-radius:4px; padding:6px; font-family:Consolas; font-size:11px;")
        return lbl

    def _connect_group(self):
        g = _group("CONNECTION", "#00d4ff")
        lay = QVBoxLayout()
        row = QHBoxLayout()
        self.btn_conn_fingers = _btn("CONNECT FINGERS", "#58a6ff", border="#58a6ff")
        self.btn_conn_pull = _btn("CONNECT PULL", "#58a6ff", border="#58a6ff")
        self.btn_conn_cell = _btn("CONNECT CELL", "#58a6ff", border="#58a6ff")
        self.btn_conn_fingers.clicked.connect(self._connect_fingers)
        self.btn_conn_pull.clicked.connect(self._connect_pull)
        self.btn_conn_cell.clicked.connect(self._connect_cell)
        row.addWidget(self.btn_conn_fingers)
        row.addWidget(self.btn_conn_pull)
        row.addWidget(self.btn_conn_cell)
        lay.addLayout(row)
        self.lbl_conn = QLabel("fingers: —   pull: —   cell: —")
        self.lbl_conn.setStyleSheet("color:#8b949e;")
        lay.addWidget(self.lbl_conn)
        self.btn_estop = _btn("■  E-STOP ALL  (space)", "#f85149", "#3d1416", "#f85149")
        self.btn_estop.clicked.connect(self._estop_all)
        lay.addWidget(self.btn_estop)
        g.setLayout(lay)
        return g

    def _stiffness_group(self):
        g = _group("STIFFNESS  (which hardware springs are installed)", "#bc8cff")
        grid = QGridLayout()
        grid.addWidget(QLabel("spring set"), 0, 0)
        self.stiffness_combo = QComboBox()
        self.stiffness_combo.addItems(list(config.LOAD_TEST_STIFFNESS_CONFIGS))
        self.stiffness_combo.setStyleSheet(
            "QComboBox{background:#161b22; color:#c9d1d9; border:1px solid #30363d;"
            "padding:2px;}")
        if config.LOAD_TEST_STIFFNESS_DEFAULT in config.LOAD_TEST_STIFFNESS_CONFIGS:
            self.stiffness_combo.setCurrentText(config.LOAD_TEST_STIFFNESS_DEFAULT)
        self.stiffness_combo.currentTextChanged.connect(self._on_stiffness_changed)
        grid.addWidget(self.stiffness_combo, 0, 1, 1, 3)
        # Resolved per-joint values for the chosen set — these are what get logged.
        self.lbl_stiffness = QLabel()
        self.lbl_stiffness.setStyleSheet("color:#8b949e; font-size:10px;")
        self.lbl_stiffness.setWordWrap(True)
        grid.addWidget(self.lbl_stiffness, 1, 0, 1, 4)
        g.setLayout(grid)
        self._update_stiffness_label()
        return g

    def _selected_stiffness(self):
        """(k_mcp, k_pip, k_dip) [N·m/rad] for the chosen spring set.

        Falls back to the config.py per-joint defaults if the selected key is
        somehow missing (e.g. the dict was edited while the panel was open).
        """
        key = self.stiffness_combo.currentText()
        return config.LOAD_TEST_STIFFNESS_CONFIGS.get(
            key, (config.MCP_STIFFNESS, config.PIP_STIFFNESS, config.DIP_STIFFNESS))

    def _update_stiffness_label(self):
        k_mcp, k_pip, k_dip = self._selected_stiffness()
        self.lbl_stiffness.setText(
            f"MCP {k_mcp:.3f} · PIP {k_pip:.3f} · DIP {k_dip:.3f}  N·m/rad "
            f"(logged as k_mcp/k_pip/k_dip)")

    def _on_stiffness_changed(self, _text):
        self._update_stiffness_label()
        self.statusBar().showMessage(
            f"stiffness set → {self.stiffness_combo.currentText()}", 3000)

    def _tension_group(self):
        g = _group("TENSIONING  (take up slack on each tendon + string, then SET ZERO)",
                   "#3fb950")
        grid = QGridLayout()
        grid.addWidget(QLabel("jog step"), 0, 0)
        self.jog_step = _dspin(0.002, 0.5, 0.01, 0.002, " rev")
        grid.addWidget(self.jog_step, 0, 1)
        b_step_down = _btn("−")
        b_step_up = _btn("+")
        b_step_down.setFixedWidth(28)
        b_step_up.setFixedWidth(28)
        b_step_down.clicked.connect(self._jog_step_down)
        b_step_up.clicked.connect(self._jog_step_up)
        grid.addWidget(b_step_down, 0, 2)
        grid.addWidget(b_step_up, 0, 3)

        def finger_row(name, key, r):
            grid.addWidget(QLabel(name), r, 0)
            b_minus = _btn("◀ −")
            b_plus = _btn("+ ▶")
            b_zero = _btn("◎ SET ZERO", "#3fb950", border="#3fb950")
            b_minus.clicked.connect(lambda: self._jog_finger(key, -1))
            b_plus.clicked.connect(lambda: self._jog_finger(key, +1))
            b_zero.clicked.connect(lambda: self._zero_finger(key))
            grid.addWidget(b_minus, r, 1)
            grid.addWidget(b_plus, r, 2)
            grid.addWidget(b_zero, r, 3)

        finger_row("finger A", "a", 1)
        finger_row("finger B", "b", 2)
        b_both = _btn("◎ SET ZERO BOTH (tensioned)", "#3fb950", border="#3fb950")
        b_both.clicked.connect(self._zero_both)
        grid.addWidget(b_both, 3, 0, 1, 4)

        # Pull-string (sensor-end) servo: same pre-tension UX as the fingers —
        # jog to take up string slack so the cell reads ~0 at the start, then
        # SET ZERO so the pull ΔL begins at the object.
        grid.addWidget(QLabel("pull string"), 4, 0)
        b_pminus = _btn("◀ −")
        b_pplus = _btn("+ ▶")
        b_pzero = _btn("◎ SET ZERO", "#3fb950", border="#3fb950")
        b_pminus.clicked.connect(lambda: self._jog_pull(-1))
        b_pplus.clicked.connect(lambda: self._jog_pull(+1))
        b_pzero.clicked.connect(self._zero_pull)
        grid.addWidget(b_pminus, 4, 1)
        grid.addWidget(b_pplus, 4, 2)
        grid.addWidget(b_pzero, 4, 3)

        # Keyboard jog: ←/→ jog the selected target, ↑/↓ scale the step, space
        # E-stops. The selector picks what the arrows drive.
        grid.addWidget(QLabel("⌨ jog target"), 5, 0)
        self.kbd_target = QComboBox()
        self.kbd_target.addItems(["both fingers", "finger A", "finger B", "pull"])
        self.kbd_target.setStyleSheet(
            "QComboBox{background:#161b22; color:#c9d1d9; border:1px solid #30363d;"
            "padding:2px;}")
        grid.addWidget(self.kbd_target, 5, 1, 1, 2)
        hint = QLabel("←/→ jog · ↑/↓ step · space=stop")
        hint.setStyleSheet("color:#6e7681; font-size:10px;")
        grid.addWidget(hint, 5, 3)

        # Per-servo direction flip (same as the sweep dashboard's FLIP PULL DIR):
        # toggles each motor's pull_sign so a +jog flexes/winds the right way
        # regardless of how the horn/spool was wound. Re-jog after flipping.
        grid.addWidget(QLabel("flip dir"), 6, 0)
        b_flip_a = _btn("⇄ A")
        b_flip_b = _btn("⇄ B")
        b_flip_p = _btn("⇄ pull")
        b_flip_a.clicked.connect(lambda: self._flip_dir("a"))
        b_flip_b.clicked.connect(lambda: self._flip_dir("b"))
        b_flip_p.clicked.connect(lambda: self._flip_dir("pull"))
        grid.addWidget(b_flip_a, 6, 1)
        grid.addWidget(b_flip_b, 6, 2)
        grid.addWidget(b_flip_p, 6, 3)
        g.setLayout(grid)
        return g

    def _sensor_group(self):
        g = _group("LOAD SENSOR  (Futek LCM300)", "#d29922")
        lay = QVBoxLayout()
        self.lbl_force = QLabel("force:  --  N    (--  kg)")
        self.lbl_force.setStyleSheet("color:#f0f6fc; font-size:15px; font-weight:bold;")
        lay.addWidget(self.lbl_force)
        self.lbl_peak = QLabel("peak: -- N    rate: -- Hz")
        self.lbl_peak.setStyleSheet("color:#8b949e;")
        lay.addWidget(self.lbl_peak)
        row = QHBoxLayout()
        b_tare = _btn("⊘ TARE / ZERO SENSOR", "#d29922", border="#d29922")
        b_reset = _btn("reset peak")
        b_tare.clicked.connect(self._tare_cell)
        b_reset.clicked.connect(self._reset_peak)
        row.addWidget(b_tare)
        row.addWidget(b_reset)
        lay.addLayout(row)
        g.setLayout(lay)
        return g

    def _grip_group(self):
        g = _group("GRIP  (close both fingers on the object)", "#58a6ff")
        grid = QGridLayout()
        grid.addWidget(QLabel("grip target ΔL"), 0, 0)
        self.grip_spin = _dspin(0.0, self.finger_cap, min(20.0, self.finger_cap),
                                1.0, " mm")
        grid.addWidget(self.grip_spin, 0, 1)
        grid.addWidget(QLabel("speed"), 0, 2)
        self.grip_speed = _dspin(0.5, 20.0, self.finger_speed, 0.5, " mm/s")
        grid.addWidget(self.grip_speed, 0, 3)
        b_go = _btn("▶ GO GRIP", "#3fb950", border="#3fb950")
        b_open = _btn("OPEN (ΔL→0)")
        b_go.clicked.connect(self._go_grip)
        b_open.clicked.connect(self._open_grip)
        grid.addWidget(b_go, 1, 0, 1, 2)
        grid.addWidget(b_open, 1, 2, 1, 2)
        g.setLayout(grid)
        return g

    def _pull_group(self):
        g = _group("PULL TEST  (wind string until the grip releases)", "#ff7b72")
        grid = QGridLayout()
        grid.addWidget(QLabel("pull target ΔL"), 0, 0)
        self.pull_spin = _dspin(0.0, self.pull_cap, self.pull_cap, 1.0, " mm")
        grid.addWidget(self.pull_spin, 0, 1)
        grid.addWidget(QLabel("speed"), 0, 2)
        self.pull_speed = _dspin(0.2, 20.0, self.pull_speed_default, 0.2, " mm/s")
        grid.addWidget(self.pull_speed, 0, 3)
        grid.addWidget(QLabel("release drop"), 1, 0)
        self.drop_spin = _dspin(5.0, 90.0, config.RELEASE_DROP_FRAC * 100.0,
                                5.0, " %")
        grid.addWidget(self.drop_spin, 1, 1)
        b_start = _btn("▶ START PULL", "#ff7b72", border="#ff7b72")
        b_stop = _btn("STOP PULL")
        b_start.clicked.connect(self._start_pull)
        b_stop.clicked.connect(self._stop_pull)
        grid.addWidget(b_start, 1, 2)
        grid.addWidget(b_stop, 1, 3)
        self.lbl_pull = QLabel("idle")
        self.lbl_pull.setStyleSheet("color:#8b949e;")
        grid.addWidget(self.lbl_pull, 2, 0, 1, 4)
        self.lbl_cap = QLabel("load capacity (last release): —")
        self.lbl_cap.setStyleSheet("color:#f0f6fc; font-weight:bold;")
        grid.addWidget(self.lbl_cap, 3, 0, 1, 4)
        g.setLayout(grid)
        return g

    def _readout_group(self):
        g = _group("READOUTS", "#8b949e")
        grid = QGridLayout()

        def row(name, r):
            grid.addWidget(QLabel(name), r, 0)
            v = QLabel("--")
            v.setAlignment(Qt.AlignRight)
            v.setStyleSheet("color:#c9d1d9; font-family:Consolas;")
            grid.addWidget(v, r, 1)
            return v

        self.ro_state = row("state", 0)
        self.ro_fa = row("finger A  ΔL / I", 1)
        self.ro_fb = row("finger B  ΔL / I", 2)
        self.ro_pull = row("pull  ΔL / I", 3)
        self.ro_force = row("force (N)", 4)
        self.ro_pred = row("pred. grip force", 5)
        self.ro_csv = row("csv", 6)
        # The analytical grip-force mapping is the "validate later" piece.
        self.ro_pred.setText("— (analytical mapping TODO)")
        b_new = _btn("🗂 NEW TRIAL → NEW CSV", "#3fb950", border="#3fb950")
        b_new.clicked.connect(self._new_trial)
        grid.addWidget(b_new, 7, 0, 1, 2)
        g.setLayout(grid)
        return g

    # =================================================================
    # plotting
    # =================================================================
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
        for ax, title, ylab in (
            (self.ax_force, "axial force vs time", "force [N]"),
            (self.ax_dl, "tendon / pull ΔL vs time", "ΔL [mm]"),
            (self.ax_cur, "servo current vs time", "current [mA]"),
        ):
            ax.clear()
            ax.set_title(title, fontsize=9)
            ax.set_ylabel(ylab, fontsize=8)
            self._style_ax(ax)
        self.ax_cur.set_xlabel("t [s]", fontsize=8)
        self.canvas.draw_idle()

    def _redraw_plots(self):
        if not self.buf_t:
            return
        t = np.fromiter(self.buf_t, float)
        for ax, title, ylab in (
            (self.ax_force, "axial force vs time", "force [N]"),
            (self.ax_dl, "tendon / pull ΔL vs time", "ΔL [mm]"),
            (self.ax_cur, "servo current vs time", "current [mA]"),
        ):
            ax.clear()
            ax.set_title(title, fontsize=9)
            ax.set_ylabel(ylab, fontsize=8)
            self._style_ax(ax)
        self.ax_cur.set_xlabel("t [s]", fontsize=8)

        self.ax_force.plot(t, np.fromiter(self.buf_force, float),
                           color="#ff7b72", lw=1.6, label="force")
        if np.isfinite(self.capacity_n):
            self.ax_force.axhline(self.capacity_n, color="#f0f6fc", lw=0.9,
                                  ls="--", label=f"capacity {self.capacity_n:.1f} N")
        self.ax_force.legend(fontsize=7, loc="upper left", facecolor="#161b22",
                             edgecolor="#30363d", labelcolor="#c9d1d9")

        self.ax_dl.plot(t, np.fromiter(self.buf_fa, float), color="#58a6ff",
                        lw=1.4, label="finger A")
        self.ax_dl.plot(t, np.fromiter(self.buf_fb, float), color="#79c0ff",
                        lw=1.4, label="finger B")
        self.ax_dl.plot(t, np.fromiter(self.buf_pull, float), color="#ff7b72",
                        lw=1.4, label="pull")
        self.ax_dl.legend(fontsize=7, loc="upper left", facecolor="#161b22",
                          edgecolor="#30363d", labelcolor="#c9d1d9")

        self.ax_cur.plot(t, np.fromiter(self.buf_ica, float), color="#58a6ff",
                         lw=1.2, label="finger A")
        self.ax_cur.plot(t, np.fromiter(self.buf_icb, float), color="#79c0ff",
                         lw=1.2, label="finger B")
        self.ax_cur.plot(t, np.fromiter(self.buf_icp, float), color="#ff7b72",
                         lw=1.2, label="pull")
        self.ax_cur.legend(fontsize=7, loc="upper left", facecolor="#161b22",
                           edgecolor="#30363d", labelcolor="#c9d1d9")
        self.canvas.draw_idle()

    # =================================================================
    # connection
    # =================================================================
    def _is_mock_finger(self):
        return isinstance(self.fa, MockServo)

    def _ensure_bus(self):
        """Open the single shared U2D2 bus once and attach all three servos.

        Fingers A/B and the pull servo are daisy-chained on one U2D2 (fingers on
        one hub port, the pull string on another — the same serial bus), so we
        open ONE ``PortHandler`` and attach every motor to it. Idempotent: the
        first CONNECT button to be pressed opens the bus; the rest reuse it.
        Returns ``(ok, message)``. A no-op for mock servos (no real port).
        """
        if self._is_mock_finger():
            return True, "mock (no bus)"
        if self._bus_handles is not None:
            return True, "bus already open"
        ok, ph, pk, port, baud, ids, msg = open_bus(
            port=self.finger_port, baud=self.fa.baud,
            expected_ids=self.bus_ids)
        if not ok:
            return False, msg
        self._bus_handles = (ph, pk)
        for s in (self.fa, self.fb, self.pull):
            s.attach_bus(ph, pk, port, baud)
        return True, msg

    def _connect_fingers(self):
        if self._is_mock_finger():
            for s in (self.fa, self.fb):
                s.connect()
            self.btn_conn_fingers.setText("FINGERS ✓ (mock)")
            self.btn_conn_fingers.setEnabled(False)
            self._update_conn_label()
            return
        ok, msg = self._ensure_bus()
        if not ok:
            QMessageBox.critical(self, "Fingers", msg)
            return
        ok_a, msg_a = self.fa.connect()
        ok_b, msg_b = self.fb.connect()
        if not (ok_a and ok_b):
            QMessageBox.critical(self, "Fingers",
                                 f"A: {msg_a}\nB: {msg_b}")
            return
        self.btn_conn_fingers.setText("FINGERS ✓")
        self.btn_conn_fingers.setEnabled(False)
        self._update_conn_label()

    def _connect_pull(self):
        ok, msg = self._ensure_bus()
        if not ok:
            QMessageBox.critical(self, "Pull servo", msg or "connect failed")
            return
        ok, msg = self.pull.connect()
        if not ok:
            QMessageBox.critical(self, "Pull servo", msg or "connect failed")
            return
        self.btn_conn_pull.setText("PULL ✓")
        self.btn_conn_pull.setEnabled(False)
        self._update_conn_label()

    def _connect_cell(self):
        # Steer the cell's port autodetect away from ports the servos hold.
        exclude = []
        for s in (self.fa, self.fb, self.pull):
            p = getattr(s, "port", None)
            if isinstance(p, str) and p.startswith("/dev/"):
                exclude.append(p)
        if hasattr(self.cell, "exclude_ports"):
            self.cell.exclude_ports = exclude
        ok, msg = self.cell.connect()
        if not ok:
            QMessageBox.critical(self, "Load cell", msg or "connect failed")
            return
        self.btn_conn_cell.setText("CELL ✓")
        self.btn_conn_cell.setEnabled(False)
        self._update_conn_label()

    def _update_conn_label(self):
        def on(dev):
            try:
                return dev.get_state().get("connected")
            except Exception:
                return False
        self.lbl_conn.setText(
            f"fingers: {'on' if on(self.fa) and on(self.fb) else '—'}   "
            f"pull: {'on' if on(self.pull) else '—'}   "
            f"cell: {'on' if on(self.cell) else '—'}")

    # =================================================================
    # tensioning / grip / pull actions
    # =================================================================
    def _finger(self, key):
        return self.fa if key == "a" else self.fb

    def _jog_finger(self, key, direction):
        s = self._finger(key)
        if not s.get_state().get("connected"):
            return
        ok, msg = s.jog(direction, step_rev=float(self.jog_step.value()))
        if not ok:
            self.statusBar().showMessage(f"finger {key.upper()}: {msg}", 2500)

    def _zero_finger(self, key):
        s = self._finger(key)
        if s.get_state().get("connected"):
            s.set_zero()
            self._maybe_tensioned()

    def _zero_both(self):
        for s in (self.fa, self.fb):
            if s.get_state().get("connected"):
                s.set_zero()
        self._maybe_tensioned()

    def _jog_pull(self, direction):
        if not self.pull.get_state().get("connected"):
            return
        ok, msg = self.pull.jog(direction, step_rev=float(self.jog_step.value()))
        if not ok:
            self.statusBar().showMessage(f"pull: {msg}", 2500)

    def _zero_pull(self):
        if self.pull.get_state().get("connected"):
            self.pull.set_zero()

    def _flip_dir(self, key):
        """Flip a servo's pull direction (pull_sign), like the sweep dashboard.

        Mirrors hardware/dashboard.py::_flip_pull but per-target (finger A/B or
        pull). The zero reference is unchanged, so re-jog to confirm a +jog now
        flexes/winds the intended way.
        """
        s = self.pull if key == "pull" else self._finger(key)
        new_sign = -1 if getattr(s, "pull_sign", 1) > 0 else +1
        s.set_pull_direction(new_sign)
        label = {"a": "finger A", "b": "finger B", "pull": "pull"}[key]
        self.statusBar().showMessage(
            f"{label} direction flipped: pull_sign = {new_sign} "
            f"(re-jog to confirm)", 3000)

    def _jog_step_up(self):
        self.jog_step.setValue(
            min(self.jog_step.value() * 2, self.jog_step.maximum()))

    def _jog_step_down(self):
        self.jog_step.setValue(
            max(self.jog_step.value() / 2, self.jog_step.minimum()))

    def _maybe_tensioned(self):
        if (self.fa.get_state().get("connected")
                and self.fb.get_state().get("connected")
                and self.state == TestState.IDLE):
            self.state = TestState.TENSIONED

    def _tare_cell(self):
        self.cell.tare()
        self._run_peak = 0.0

    def _reset_peak(self):
        self.cell.reset_peak()
        self._run_peak = 0.0

    def _new_trial(self):
        """Close the current trial's CSV and open a fresh one for the next trial.

        Each trial (10 per stiffness combination) gets its own isolated CSV named
        ``…_trialN_…``. Pressing this advances the counter and starts a clean file
        so one set of stiffness tuning can drive many trials without restarting
        the dashboard. The per-trial latched state (capacity, running/measured
        peak) is reset so the new trial starts from a clean slate.
        """
        # If a CSV is already open for the current trial, close it and step the
        # counter forward; otherwise reuse the (startup-resolved) trial number so
        # the very first trial keeps the number we resumed at — no gaps.
        if self.logger is not None:
            self.logger.close()
            self.logger = None
            self.trial_idx += 1
        self.capacity_n = float("nan")
        self._run_peak = 0.0
        self._peak_count = 0
        self.cell.reset_peak()
        self.lbl_cap.setText("load capacity (last release): —")
        self._ensure_logger()
        self.lbl_pull.setText(f"ready — trial {self.trial_idx}")
        self.lbl_pull.setStyleSheet("color:#8b949e;")
        self.statusBar().showMessage(
            f"new trial → {os.path.basename(self.logger.filepath)} "
            f"(trial {self.trial_idx})", 5000)

    def _go_grip(self):
        if not (self.fa.get_state().get("connected")
                and self.fb.get_state().get("connected")):
            QMessageBox.warning(self, "Grip", "Connect + zero both fingers first.")
            return
        tgt = float(self.grip_spin.value())
        spd = float(self.grip_speed.value())
        self.fa.start_ramp(tgt, speed_mm_s=spd)
        self.fb.start_ramp(tgt, speed_mm_s=spd)
        self.state = TestState.GRIPPED
        self._mock_released = False

    def _open_grip(self):
        spd = float(self.grip_speed.value())
        for s in (self.fa, self.fb):
            s.start_ramp(0.0, speed_mm_s=spd)
        self.state = TestState.TENSIONED
        self._mock_released = False

    def _start_pull(self):
        if not self.pull.get_state().get("connected"):
            QMessageBox.warning(self, "Pull", "Connect the pull servo first.")
            return
        if self.state not in (TestState.GRIPPED, TestState.RELEASED,
                              TestState.PULLING):
            QMessageBox.warning(self, "Pull", "Grip the object first (GO GRIP).")
            return
        self._run_peak = 0.0
        self._armed = True            # looking for the next downward drop
        self._valley = float("inf")   # lowest force since the last peak
        self._peak_count = 0
        self.cell.reset_peak()
        self._mock_released = False
        self.pull.start_ramp(float(self.pull_spin.value()),
                             speed_mm_s=float(self.pull_speed.value()))
        self.state = TestState.PULLING
        self.lbl_pull.setText(f"pulling — trial {self.trial_idx}…")
        self._ensure_logger()

    def _stop_pull(self):
        # Freeze the pull servo at its current ΔL (don't drop torque).
        self.pull.start_ramp(self.pull.current_delta_L_mm(),
                             speed_mm_s=float(self.pull_speed.value()))
        if self.state == TestState.PULLING:
            self.state = TestState.GRIPPED
            self.lbl_pull.setText("pull stopped")

    def _estop_all(self):
        for s in (self.fa, self.fb, self.pull):
            try:
                s.e_stop()
            except Exception:
                pass
        self.state = TestState.IDLE
        self.lbl_pull.setText("E-STOP")
        self.lbl_pull.setStyleSheet("color:#f85149; font-weight:bold;")

    # =================================================================
    # logging
    # =================================================================
    def _results_dir(self):
        return os.path.join(_HERE, "results")

    def _next_trial_number(self):
        """Scan results/ for existing ``trialN`` CSVs; return one past the max.

        Lets a freshly opened dashboard resume the numbering (last file trial15 →
        next trial16). Starts at 1 when no such CSVs exist (or the dir is absent).
        """
        out_dir = self._results_dir()
        pat = re.compile(r"trial(\d+)", re.IGNORECASE)
        max_n = 0
        if os.path.isdir(out_dir):
            for fn in os.listdir(out_dir):
                if not fn.lower().endswith(".csv"):
                    continue
                m = pat.search(fn)
                if m:
                    max_n = max(max_n, int(m.group(1)))
        return max_n + 1

    def _ensure_logger(self):
        if self.logger is not None:
            return
        self.logger = CsvLogger(
            spring_set_label=f"trial{self.trial_idx}",
            out_dir=self._results_dir(),
            columns=_LOG_COLUMNS,
            filename_prefix="hw_loadtest",
        )
        self.ro_csv.setText(os.path.basename(self.logger.filepath))

    def _log_row(self, event, force, raw, peak, sa, sb, sp):
        if self.logger is None:
            return
        k_mcp, k_pip, k_dip = self._selected_stiffness()
        self.logger.log({
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "trial_idx": self.trial_idx,
            "event": event,
            "state": self.state,
            "force_n": force,
            "force_raw_n": raw,
            "peak_n": peak,
            "capacity_n": self.capacity_n,
            "grip_target_mm": float(self.grip_spin.value()),
            "finger_a_dL_mm": sa.get("delta_L_mm"),
            "finger_b_dL_mm": sb.get("delta_L_mm"),
            "finger_a_current_ma": sa.get("current_ma"),
            "finger_b_current_ma": sb.get("current_ma"),
            "pull_dL_mm": sp.get("delta_L_mm"),
            "pull_current_ma": sp.get("current_ma"),
            "k_mcp": k_mcp,
            "k_pip": k_pip,
            "k_dip": k_dip,
        })

    # =================================================================
    # main loop
    # =================================================================
    def _tick(self):
        now = time.monotonic()
        dt = max(1e-3, now - self._t_prev)
        self._t_prev = now

        sa = self.fa.service(dt)
        sb = self.fb.service(dt)
        sp = self.pull.service(dt)

        # Mock plant: synthesise the axial force from grip vs pull ΔL so the real
        # release-detection path (force-drop) can be exercised with no hardware.
        if isinstance(self.cell, MockLoadCell):
            grip_dl = min(sa.get("delta_L_mm", 0.0), sb.get("delta_L_mm", 0.0))
            f_hold = _MOCK_GRIP_N_PER_MM * max(0.0, grip_dl)
            pull_dl = sp.get("delta_L_mm", 0.0)
            f = max(0.0, _MOCK_PULL_N_PER_MM * (pull_dl - _MOCK_PULL_SLACK_MM))
            if (not self._mock_released and f_hold > config.RELEASE_MIN_FORCE_N
                    and f >= f_hold):
                self._mock_released = True
            self.cell.set_sim_force(_MOCK_RESIDUAL_N if self._mock_released else f)

        cst = self.cell.get_state()
        force = cst.get("force_n", 0.0)
        raw = cst.get("raw_n", 0.0)

        # Peak detection (force drops from a running peak while pulling). We do
        # NOT halt on a peak — pulling continues so further peaks can show up as
        # the grip morphology shifts around the object; the operator stops
        # manually. Hysteresis (arm → valley → re-arm) keeps one big slip from
        # registering as many peaks during a single long fall: after a peak we
        # disarm and watch the valley, re-arming only once the force has climbed
        # back out of it.
        if self.state == TestState.PULLING:
            drop = float(self.drop_spin.value()) / 100.0
            if self._armed:
                self._run_peak = max(self._run_peak, force)
                if (self._run_peak > config.RELEASE_MIN_FORCE_N
                        and force < (1.0 - drop) * self._run_peak):
                    self._on_release(sa, sb, sp)
            else:
                self._valley = min(self._valley, force)
                rearm_margin = drop * max(self._run_peak,
                                          config.RELEASE_MIN_FORCE_N)
                if force > self._valley + rearm_margin:
                    self._armed = True
                    self._run_peak = force

        self._update_readouts(sa, sb, sp, cst)

        # buffers + CSV
        t = now - self._t0
        self.buf_t.append(t)
        self.buf_force.append(force)
        self.buf_fa.append(sa.get("delta_L_mm", 0.0))
        self.buf_fb.append(sb.get("delta_L_mm", 0.0))
        self.buf_pull.append(sp.get("delta_L_mm", 0.0))
        self.buf_ica.append(sa.get("current_ma", 0.0))
        self.buf_icb.append(sb.get("current_ma", 0.0))
        self.buf_icp.append(sp.get("current_ma", 0.0))
        while self.buf_t and (t - self.buf_t[0]) > PLOT_WINDOW_S:
            for b in (self.buf_t, self.buf_force, self.buf_fa, self.buf_fb,
                      self.buf_pull, self.buf_ica, self.buf_icb, self.buf_icp):
                b.popleft()

        if self.state == TestState.PULLING:
            self._log_row("", force, raw, self._run_peak, sa, sb, sp)

        self._tick_n += 1
        if self._tick_n % _REDRAW_EVERY == 0:
            self._redraw_plots()

    def _on_release(self, sa, sb, sp):
        # A force drop from the running peak: record it as a (local) peak but
        # KEEP PULLING and stay in PULLING so subsequent peaks can appear. The
        # largest peak across the continuous pull is latched as the capacity;
        # the operator halts manually (STOP PULL / space) when done.
        self._peak_count += 1
        peak = self._run_peak
        if not np.isfinite(self.capacity_n) or peak > self.capacity_n:
            self.capacity_n = peak
        self.lbl_cap.setText(
            f"load capacity (max peak): {self.capacity_n:.1f} N "
            f"({self.capacity_n / config.KGF_TO_N:.3f} kg)")
        self.lbl_pull.setText(
            f"⚠ peak #{self._peak_count}: {peak:.1f} N (max "
            f"{self.capacity_n:.1f} N) — still pulling, STOP PULL / space to halt")
        self._log_row("peak", self.cell.get_state().get("force_n", 0.0),
                      self.cell.get_state().get("raw_n", 0.0),
                      peak, sa, sb, sp)
        # Disarm and start tracking the post-peak valley; the tick re-arms once
        # the force climbs back out, so the next local maximum is found cleanly.
        self._armed = False
        self._valley = self.cell.get_state().get("force_n", 0.0)

    def _update_readouts(self, sa, sb, sp, cst):
        self.ro_state.setText(self.state)
        self.ro_fa.setText(f"{sa.get('delta_L_mm', 0):.1f} mm / "
                           f"{sa.get('current_ma', 0):.0f} mA")
        self.ro_fb.setText(f"{sb.get('delta_L_mm', 0):.1f} mm / "
                           f"{sb.get('current_ma', 0):.0f} mA")
        self.ro_pull.setText(f"{sp.get('delta_L_mm', 0):.1f} mm / "
                             f"{sp.get('current_ma', 0):.0f} mA")
        force = cst.get("force_n", 0.0)
        self.ro_force.setText(f"{force:.2f}")
        self.lbl_force.setText(
            f"force:  {force:7.2f}  N    ({cst.get('force_kg', 0.0):6.3f}  kg)")
        self.lbl_peak.setText(
            f"peak: {cst.get('peak_n', 0.0):.1f} N    "
            f"rate: {cst.get('rate_hz', 0.0):.0f} Hz")
        # E-stop / overcurrent surfacing on the connection line.
        flags = []
        for nm, s in (("A", sa), ("B", sb), ("pull", sp)):
            if s.get("estop"):
                flags.append(f"{nm}:ESTOP")
            elif s.get("over_current"):
                flags.append(f"{nm}:OVERCUR")
        if flags:
            self.lbl_conn.setText("  ".join(flags))

    # =================================================================
    # keyboard
    # =================================================================
    def _kbd_jog(self, direction):
        target = self.kbd_target.currentText()
        if target == "finger A":
            self._jog_finger("a", direction)
        elif target == "finger B":
            self._jog_finger("b", direction)
        elif target == "pull":
            self._jog_pull(direction)
        else:  # both fingers (default)
            self._jog_finger("a", direction)
            self._jog_finger("b", direction)

    def keyPressEvent(self, e: QKeyEvent):
        if e.isAutoRepeat():
            return
        k = e.key()
        if k == Qt.Key_Space:
            self._estop_all()
        elif k == Qt.Key_Right:
            self._kbd_jog(+1)
        elif k == Qt.Key_Left:
            self._kbd_jog(-1)
        elif k == Qt.Key_Up:
            self._jog_step_up()
        elif k == Qt.Key_Down:
            self._jog_step_down()
        else:
            super().keyPressEvent(e)

    def _shutdown(self):
        """Torque-off + disconnect all three servos, close the shared bus and
        the load cell. Idempotent: safe from closeEvent, atexit, and signals."""
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        for s in (self.fa, self.fb, self.pull):
            try:
                s.disable()
                s.disconnect()
            except Exception:
                pass
        # Close the shared finger bus once (servos don't own it).
        if self._bus_handles is not None:
            try:
                self._bus_handles[0].closePort()
            except Exception:
                pass
        try:
            self.cell.disconnect()
        except Exception:
            pass
        if self.logger:
            self.logger.close()

    def closeEvent(self, e):
        self._shutdown()
        super().closeEvent(e)


# =====================================================================
# entry point
# =====================================================================
def _build_devices(args):
    mock_servo = args.mock or args.mock_servo
    mock_cell = args.mock or args.mock_loadcell
    # Per-finger default pull direction: finger A winds opposite to B on this
    # build, so positive ΔL takes up slack the right way for each (see config).
    if mock_servo:
        fa = MockServo(dxl_id=args.a_id, spool_radius_m=args.spool_radius,
                       soft_delta_l_cap_mm=args.finger_cap,
                       pull_sign=config.FINGER_A_PULL_SIGN)
        fb = MockServo(dxl_id=args.b_id, spool_radius_m=args.spool_radius,
                       soft_delta_l_cap_mm=args.finger_cap,
                       pull_sign=config.FINGER_B_PULL_SIGN)
        pull = MockServo(dxl_id=args.pull_id, spool_radius_m=args.pull_spool_radius,
                         soft_delta_l_cap_mm=args.pull_cap,
                         pull_sign=config.PULL_PULL_SIGN)
    else:
        # Finger servos are created with the shared finger port; the dashboard
        # opens ONE bus and attaches both at connect time (open_bus/attach_bus).
        fa = Servo(port=args.finger_port, baud=args.baud, dxl_id=args.a_id,
                   spool_radius_m=args.spool_radius,
                   soft_delta_l_cap_mm=args.finger_cap,
                   pull_sign=config.FINGER_A_PULL_SIGN)
        fb = Servo(port=args.finger_port, baud=args.baud, dxl_id=args.b_id,
                   spool_radius_m=args.spool_radius,
                   soft_delta_l_cap_mm=args.finger_cap,
                   pull_sign=config.FINGER_B_PULL_SIGN)
        # The pull servo shares the finger U2D2 bus; it is attached to the
        # already-open bus at connect time (see _ensure_bus), so it is created
        # with the same finger port and never opens a port of its own.
        pull = Servo(port=args.finger_port, baud=args.baud, dxl_id=args.pull_id,
                     spool_radius_m=args.pull_spool_radius,
                     soft_delta_l_cap_mm=args.pull_cap,
                     pull_sign=config.PULL_PULL_SIGN)
    if mock_cell:
        cell = MockLoadCell(baud=args.loadcell_baud)
    else:
        cell = LoadCell(port=args.loadcell_port, baud=args.loadcell_baud)
    return fa, fb, pull, cell


def main():
    p = argparse.ArgumentParser(
        description="Load-carrying (pull-out) hardware test dashboard")
    p.add_argument("--mock", action="store_true",
                   help="no hardware at all (mock servos + load cell)")
    p.add_argument("--mock-servo", action="store_true",
                   help="mock the three servos (real load cell)")
    p.add_argument("--mock-loadcell", action="store_true",
                   help="mock the load cell (real servos)")
    p.add_argument("--finger-port", default="auto",
                   help="U2D2 bus shared by all three servos (finger A+B + "
                        "pull, daisy-chained); 'auto' scans")
    p.add_argument("--loadcell-port", default="auto",
                   help="USB220 serial port; 'auto' probes for a numeric stream")
    p.add_argument("--a-id", type=int, default=config.FINGER_A_DXL_ID)
    p.add_argument("--b-id", type=int, default=config.FINGER_B_DXL_ID)
    p.add_argument("--pull-id", type=int, default=config.PULL_DXL_ID)
    p.add_argument("--baud", type=int, default=57600, help="Dynamixel baud")
    p.add_argument("--loadcell-baud", type=int, default=config.LOADCELL_BAUD)
    p.add_argument("--spool-radius", type=float, default=config.SPOOL_RADIUS,
                   help=f"finger tendon spool radius [m] (default {config.SPOOL_RADIUS})")
    p.add_argument("--pull-spool-radius", type=float,
                   default=config.PULL_SPOOL_RADIUS,
                   help="pull-string spool radius [m] (TODO: set when designed)")
    p.add_argument("--finger-cap", type=float, default=60.0,
                   help="finger soft ΔL cap [mm]")
    p.add_argument("--pull-cap", type=float, default=config.PULL_MAX_DELTA_MM,
                   help="pull-servo soft ΔL cap [mm]")
    p.add_argument("--finger-speed", type=float, default=3.0,
                   help="finger grip ramp speed [mm/s]")
    p.add_argument("--pull-speed", type=float, default=config.PULL_SPEED_MM_S,
                   help="pull winding speed [mm/s]")
    args = p.parse_args()

    fa, fb, pull, cell = _build_devices(args)
    app = QApplication(sys.argv)
    app.setFont(QFont("Consolas", 10))
    win = LoadTestDashboard(
        fa, fb, pull, cell,
        finger_cap_mm=args.finger_cap, pull_cap_mm=args.pull_cap,
        finger_speed_mm_s=args.finger_speed, pull_speed_mm_s=args.pull_speed,
        finger_port=args.finger_port, finger_ids=(args.a_id, args.b_id))
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
