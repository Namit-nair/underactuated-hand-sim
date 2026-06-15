"""Serial interface for a Futek LCM300 axial load cell read via a USB220 module.

The USB220 USB-output kit enumerates as a plain serial port (Linux:
``/dev/ttyUSB*``) and *free-runs* ASCII — one numeric reading per line — so we
just open the port and parse the stream in a background thread. (Futek's SENSIT
configuration software is Windows-only and not used here.)

Force pipeline
--------------
Each line yields the first float it contains (``config.LOADCELL_LINE_REGEX``).
That value is scaled and converted to newtons::

    force_N = parsed_value * scale * unit_factor

where ``unit_factor`` is the pound-force→newton constant when the stream is in
``lb`` (``config.LBF_TO_N``) or ``1.0`` when it is already in newtons / raw
counts. A first-order low-pass (``filter_alpha``) smooths the reading, a
software **tare** subtracts a captured zero, and the running **peak** (by
magnitude, sign preserved) is tracked so the pull-out release force is caught
even between dashboard ticks.

The exact baud / unit / scale must be confirmed against the physical USB220
stream — they are all configurable (and defaulted in ``config.py``). A
:class:`MockLoadCell` with the same interface lets the dashboard and tests run
with no hardware attached.
"""

from __future__ import annotations

import glob
import math
import os
import re
import sys
import threading
import time
from typing import List, Optional

import serial  # pyserial

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402


def _unit_factor(input_unit: str) -> float:
    """Multiplier converting one stream unit to newtons."""
    u = (input_unit or "").strip().lower()
    if u in ("lb", "lbf", "lbs", "pound", "pounds"):
        return config.LBF_TO_N
    # "n" / "newton" / "raw" / anything else: treat the (scaled) value as N.
    return 1.0


def parse_force(
    line: str,
    *,
    regex: str = config.LOADCELL_LINE_REGEX,
    scale: float = config.LOADCELL_SCALE,
    input_unit: str = config.LOADCELL_INPUT_UNIT,
) -> Optional[float]:
    """Parse one ASCII line into a force in newtons, or ``None`` if no number.

    Pure + side-effect-free so it can be unit-tested against captured sample
    lines without any hardware.
    """
    m = re.search(regex, line)
    if not m:
        return None
    try:
        value = float(m.group(0))
    except ValueError:
        return None
    return value * float(scale) * _unit_factor(input_unit)


def list_serial_ports() -> List[str]:
    """Likely USB serial ports (Linux: ttyUSB*/ttyACM*)."""
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


def autodetect_loadcell(
    baud: int = config.LOADCELL_BAUD,
    *,
    regex: str = config.LOADCELL_LINE_REGEX,
    exclude_ports: Optional[List[str]] = None,
    settle_lines: int = 3,
    timeout_s: float = 2.0,
):
    """Find the USB220 port by opening candidates and looking for numeric lines.

    Skips ``exclude_ports`` (e.g. ports already bound to the U2D2 servos), then
    accepts the first port that streams a line matching ``regex`` within
    ``timeout_s``. Returns ``(ok, port, message)``.
    """
    exclude = set(exclude_ports or [])
    ports = [p for p in list_serial_ports() if p not in exclude]
    if not ports:
        return (False, None,
                "no free /dev/ttyUSB*/ttyACM* ports for the load cell "
                "(USB220 plugged in? servos may be using the others)")
    for port in ports:
        try:
            with serial.Serial(port, baud, timeout=0.3) as ser:
                deadline = time.monotonic() + timeout_s
                seen = 0
                while time.monotonic() < deadline:
                    raw = ser.readline()
                    if not raw:
                        continue
                    if re.search(regex, raw.decode("ascii", "ignore")):
                        seen += 1
                        if seen >= settle_lines:
                            return True, port, f"load cell streaming on {port}"
        except (serial.SerialException, OSError):
            continue
    return (False, None,
            "no port produced numeric lines; check the USB220 baud "
            f"({baud}) and that the cell is powered")


class LoadCell:
    """Threaded reader for a free-running ASCII load cell (Futek LCM300/USB220)."""

    def __init__(
        self,
        port: str = "auto",
        baud: int = config.LOADCELL_BAUD,
        *,
        line_regex: str = config.LOADCELL_LINE_REGEX,
        scale: float = config.LOADCELL_SCALE,
        input_unit: str = config.LOADCELL_INPUT_UNIT,
        capacity_n: float = config.LOADCELL_CAPACITY_N,
        filter_alpha: float = config.LOADCELL_FILTER_ALPHA,
        exclude_ports: Optional[List[str]] = None,
    ) -> None:
        self.port = port
        self.baud = int(baud)
        self.line_regex = line_regex
        self.scale = float(scale)
        self.input_unit = input_unit
        self.capacity_n = float(capacity_n)
        self.filter_alpha = min(1.0, max(0.0, float(filter_alpha)))
        self.exclude_ports = list(exclude_ports or [])

        self._ser: Optional[serial.Serial] = None
        self._connected = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._raw_n = 0.0       # filtered force [N], NOT tared
        self._tare_n = 0.0      # captured zero [N]
        self._peak_n = 0.0      # peak tared force by magnitude (sign kept)
        self._have_sample = False
        self._stamps: List[float] = []   # recent read times for the rate estimate

    # -- lifecycle --------------------------------------------------------
    def connect(self):
        """Open the serial port (auto-detecting if requested) and start reading."""
        if self._connected:
            return True, "already connected"
        port = self.port
        if port in (None, "", "auto"):
            ok, port, msg = autodetect_loadcell(
                self.baud, regex=self.line_regex,
                exclude_ports=self.exclude_ports)
            if not ok:
                return False, f"load cell auto-detect failed: {msg}"
        try:
            self._ser = serial.Serial(port, self.baud, timeout=0.3)
        except (serial.SerialException, OSError) as exc:
            return False, f"failed to open load-cell port {port}: {exc}"
        self.port = port
        self._connected = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return True, f"load cell connected on {port} @ {self.baud} baud"

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -- reader thread ----------------------------------------------------
    def _reader_loop(self) -> None:
        while not self._stop.is_set() and self._ser is not None:
            try:
                raw = self._ser.readline()
            except (serial.SerialException, OSError):
                break
            if not raw:
                continue
            f = parse_force(raw.decode("ascii", "ignore"),
                            regex=self.line_regex, scale=self.scale,
                            input_unit=self.input_unit)
            if f is None:
                continue
            self._note_sample(f)

    def _note_sample(self, inst_n: float) -> None:
        now = time.monotonic()
        with self._lock:
            if not self._have_sample or self.filter_alpha >= 1.0:
                self._raw_n = inst_n
                self._have_sample = True
            else:
                a = self.filter_alpha
                self._raw_n = a * inst_n + (1.0 - a) * self._raw_n
            tared = self._raw_n - self._tare_n
            if abs(tared) > abs(self._peak_n):
                self._peak_n = tared
            self._stamps.append(now)
            if len(self._stamps) > 50:
                self._stamps = self._stamps[-50:]

    # -- readings ---------------------------------------------------------
    def read(self) -> float:
        """Latest tared, filtered force [N]."""
        with self._lock:
            return self._raw_n - self._tare_n

    def read_raw(self) -> float:
        """Latest filtered force [N] without the tare offset."""
        with self._lock:
            return self._raw_n

    def tare(self) -> None:
        """Capture the present reading as zero and reset the peak."""
        with self._lock:
            self._tare_n = self._raw_n
            self._peak_n = 0.0

    def reset_peak(self) -> None:
        with self._lock:
            self._peak_n = 0.0

    def _rate_hz(self) -> float:
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        return (len(self._stamps) - 1) / span if span > 1e-6 else 0.0

    def get_state(self) -> dict:
        with self._lock:
            force_n = self._raw_n - self._tare_n
            return {
                "connected": self._connected,
                "force_n": force_n,
                "force_lb": force_n / config.LBF_TO_N,
                "raw_n": self._raw_n,
                "tare_n": self._tare_n,
                "peak_n": self._peak_n,
                "rate_hz": self._rate_hz(),
                "have_sample": self._have_sample,
            }


class MockLoadCell:
    """Hardware-free stand-in for :class:`LoadCell` with the same interface.

    The dashboard's mock pull loop drives :meth:`set_sim_force`; ``read`` returns
    the simulated force (tared, with a touch of noise) so the whole pull-out →
    release → peak-latch flow can be exercised with no cell attached.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.port = "mock"
        self.baud = int(kwargs.get("baud", config.LOADCELL_BAUD))
        self.capacity_n = float(kwargs.get("capacity_n", config.LOADCELL_CAPACITY_N))
        self._connected = False
        self._sim_n = 0.0
        self._tare_n = 0.0
        self._peak_n = 0.0
        self._noise = 0.05
        self._t0 = time.monotonic()

    def connect(self):
        self._connected = True
        return True, "mock load cell connected (no hardware)"

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def set_sim_force(self, force_n: float) -> None:
        """Drive the simulated *raw* (untared) force [N]."""
        self._sim_n = float(force_n)

    def _reading(self) -> float:
        if not self._connected:
            return 0.0
        # Small deterministic-ish ripple so plots/filters look alive.
        ripple = self._noise * math.sin((time.monotonic() - self._t0) * 7.0)
        tared = (self._sim_n + ripple) - self._tare_n
        if abs(tared) > abs(self._peak_n):
            self._peak_n = tared
        return tared

    def read(self) -> float:
        return self._reading()

    def read_raw(self) -> float:
        return self._sim_n

    def tare(self) -> None:
        self._tare_n = self._sim_n
        self._peak_n = 0.0

    def reset_peak(self) -> None:
        self._peak_n = 0.0

    def get_state(self) -> dict:
        force_n = self._reading()
        return {
            "connected": self._connected,
            "force_n": force_n,
            "force_lb": force_n / config.LBF_TO_N,
            "raw_n": self._sim_n,
            "tare_n": self._tare_n,
            "peak_n": self._peak_n,
            "rate_hz": 100.0 if self._connected else 0.0,
            "have_sample": self._connected,
        }
