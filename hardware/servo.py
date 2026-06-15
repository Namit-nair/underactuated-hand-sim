"""Safety-wrapped Dynamixel servo interface for a tendon-driven finger rig.

This module provides a thin, framework-agnostic wrapper around a single
Dynamixel X-series motor (model XM430-W350-T/R) driven over Protocol 2.0
using the ``dynamixel_sdk`` package. It is intentionally free of any GUI,
Qt, or threading dependencies so it can be embedded inside a dashboard,
a script, or an automated sweep loop.

Operating mode
--------------
The motor runs in EXTENDED POSITION CONTROL MODE (operating mode value 4),
which is multi-turn: goal positions are commanded in encoder *ticks* with
``4096 ticks == 1 revolution`` and the position is not wrapped at one turn.

Tendon / spool model (ΔL <-> servo)
-----------------------------------
The finger tendon winds on a spool of radius ``spool_radius_m`` (default
``config.SPOOL_RADIUS`` = 0.011175 m, i.e. the measured Ø22.35 mm spool).
One revolution of the spool pays out / takes in one circumference of tendon.
Defining a captured "zero" revolution reference at :meth:`Servo.set_zero` time::

    delta_L_m = pull_sign * (2*pi*spool_radius_m) * (goal_rev - zero_rev)
    goal_rev  = zero_rev + pull_sign * delta_L_m / (2*pi*spool_radius_m)

``pull_sign`` (+1 / -1) selects which servo direction flexes the finger and
is calibrated at runtime (the dashboard jogs the motor and flips the sign).
Positive ΔL is "pull" (flexion). ΔL is reported / commanded in millimetres
through the public API for convenience.

Safety model
------------
* On :meth:`Servo.connect` the Current Limit register (control-table
  address 38, unit 2.69 mA, hardware range 0..1193 -> ~3.21 A absolute
  max) is written to ``current_limit_units`` (default 1193, the hardware
  ceiling). Because this register lives in EEPROM it can only be written
  while torque is disabled, so the connect sequence is strictly::

      torque OFF -> operating mode = 4 -> current limit -> torque ON

* Every :meth:`Servo.service` tick reads present current and, if the
  magnitude exceeds ``overcurrent_warn_ma`` (default 3000 mA) for a few
  consecutive ticks, triggers an automatic E-STOP (torque disabled) and
  raises the ``over_current`` flag in the state dict.

* :meth:`Servo.e_stop` immediately disables torque and latches an estop
  flag; motion commands are refused until :meth:`Servo.enable` clears it.

* A soft ΔL cap (``soft_delta_l_cap_mm``, default 25 mm) bounds all jog /
  ramp / goto requests to ``[0, soft_delta_l_cap_mm]`` so the tendon is
  never over-pulled and never driven negative past the zero reference.

A :class:`MockServo` with an identical interface is provided so the
dashboard / scripts can run with no hardware attached.
"""

from __future__ import annotations

import glob
import math
import os
import sys
import time
from typing import List, Optional, Tuple

# --- repo single-source-of-truth (spool radius etc.) -------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import config  # noqa: E402

# --- dynamixel_sdk import robustness -----------------------------------------
# Keep the import guarded: the module must still import (for the MockServo
# path) even when the SDK is absent. The hard error is deferred to connect().
try:  # pragma: no cover - exercised by environment, not unit tests
    from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
    _SDK_AVAILABLE = True
    _SDK_IMPORT_ERROR: Optional[Exception] = None
except ImportError:  # pragma: no cover
    try:
        sys.path.insert(0, "/home/namit/iitgn/dynamixel-control")
        from dynamixel_sdk import COMM_SUCCESS, PacketHandler, PortHandler
        _SDK_AVAILABLE = True
        _SDK_IMPORT_ERROR = None
    except ImportError as exc:  # pragma: no cover
        COMM_SUCCESS = 0  # type: ignore[assignment]
        PacketHandler = None  # type: ignore[assignment]
        PortHandler = None  # type: ignore[assignment]
        _SDK_AVAILABLE = False
        _SDK_IMPORT_ERROR = exc


# --- Dynamixel X-series control table (Protocol 2.0) -------------------------
ADDR_OPERATING_MODE = 11    # 1 byte; 4 = Extended Position Control
ADDR_CURRENT_LIMIT = 38     # 2 bytes; EEPROM (write only while torque OFF)
ADDR_TORQUE_ENABLE = 64     # 1 byte; 1 = on, 0 = off
ADDR_GOAL_POSITION = 116    # 4 bytes; ticks
ADDR_PRESENT_CURRENT = 126  # 2 bytes; signed; unit 2.69 mA
ADDR_PRESENT_VELOCITY = 128  # 4 bytes; signed; unit 0.229 rpm
ADDR_PRESENT_POSITION = 132  # 4 bytes; ticks
ADDR_PRESENT_VOLTAGE = 144  # 2 bytes; unit 0.1 V
ADDR_PRESENT_TEMP = 146     # 1 byte; deg C

TICKS_PER_REV = 4096

# Operating-mode / register constants
OPERATING_MODE_EXTENDED_POSITION = 4
TORQUE_ON = 1
TORQUE_OFF = 0

# Current limit register hardware bounds (XM430-W350-T/R datasheet)
CURRENT_LIMIT_UNIT_MA = 2.69
CURRENT_LIMIT_MAX_UNITS = 1193  # ~3.21 A absolute hardware maximum

# Unit conversions
VELOCITY_UNIT_RPM = 0.229
CURRENT_UNIT_MA = 2.69
VOLTAGE_UNIT_V = 0.1

# Overcurrent E-stop debounce: number of consecutive offending ticks.
_OVERCURRENT_TRIP_TICKS = 3


def _circumference(spool_radius_m: float) -> float:
    """Tendon length paid out per servo revolution (metres)."""
    return 2.0 * math.pi * spool_radius_m


# Baud rates tried during auto-detection (common Dynamixel rates, default first).
_AUTODETECT_BAUDS = (57600, 1000000, 115200, 2000000, 3000000, 4000000, 9600)


def list_serial_ports() -> List[str]:
    """Enumerate likely Dynamixel USB serial ports (Linux: ttyUSB*/ttyACM*)."""
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


def autodetect_servo(
    preferred_port: Optional[str] = None,
    protocol: float = 2.0,
    baudrates: Optional[Tuple[int, ...]] = None,
    candidate_ports: Optional[List[str]] = None,
    preferred_id: Optional[int] = None,
) -> Tuple[bool, Optional[str], Optional[int], Optional[int], str]:
    """Scan serial ports + baud rates for a Dynamixel, regardless of USB port.

    Tries ``preferred_port`` first (if given), then every ``/dev/ttyUSB*`` /
    ``/dev/ttyACM*`` port, broadcast-pinging at each baud until a motor answers.
    When ``preferred_id`` is given and that id is among the motors that answer,
    it is selected; otherwise the lowest responding id is used (so on a
    multi-motor daisy-chain we bind the configured servo, not an arbitrary one).

    Returns ``(ok, port, baud, dxl_id, message)``. On success ``port/baud/id``
    are the discovered values; on failure they are ``None`` and ``message``
    explains why.
    """
    if not _SDK_AVAILABLE:
        return False, None, None, None, f"dynamixel_sdk not available: {_SDK_IMPORT_ERROR}"

    bauds = tuple(baudrates) if baudrates else _AUTODETECT_BAUDS

    ports: List[str] = []
    if preferred_port:
        ports.append(preferred_port)
    for p in (candidate_ports if candidate_ports is not None else list_serial_ports()):
        if p not in ports:
            ports.append(p)
    if not ports:
        return (False, None, None, None,
                "No /dev/ttyUSB* or /dev/ttyACM* ports found. Is the U2D2 "
                "interface plugged in and powered?")

    for port in ports:
        port_handler = PortHandler(port)
        packet_handler = PacketHandler(protocol)
        if not port_handler.openPort():
            continue
        try:
            for baud in bauds:
                if not port_handler.setBaudRate(baud):
                    continue
                data_list, _comm = packet_handler.broadcastPing(port_handler)
                if data_list:
                    ids = sorted(int(i) for i in data_list.keys())
                    # Honour the requested id when it actually responded;
                    # otherwise fall back to the lowest id on the bus.
                    found_id = (preferred_id if preferred_id in ids else ids[0])
                    return (True, port, baud, found_id,
                            f"found Dynamixel id {found_id} on {port} @ {baud} baud")
        finally:
            port_handler.closePort()

    return (False, None, None, None,
            "No Dynamixel responded on any port/baud. Check 12 V power, the "
            "U2D2 cable, and the motor daisy-chain.")


def open_bus(
    port: str = "auto",
    baud: int = 57600,
    protocol: float = 2.0,
    expected_ids: Optional[Tuple[int, ...]] = None,
):
    """Open ONE U2D2 serial bus shared by several daisy-chained servos.

    Opens a single ``PortHandler``/``PacketHandler`` (auto-detecting the port +
    baud when ``port`` is ``"auto"``), broadcast-pings the bus, and returns
    handles the caller hands to each :meth:`Servo.attach_bus`. This avoids
    opening the same ``/dev/ttyUSB*`` once per motor (which fails on the 2nd).

    Returns ``(ok, port_handler, packet_handler, port, baud, ids, message)``.
    On failure the handlers are ``None`` and ``message`` explains why. When
    ``expected_ids`` is given, ``ok`` requires every one of them to answer.
    """
    if not _SDK_AVAILABLE:
        return (False, None, None, None, None, [],
                f"dynamixel_sdk not available: {_SDK_IMPORT_ERROR}")

    # Resolve port + baud (and confirm at least one motor answers) first.
    detect_id = expected_ids[0] if expected_ids else None
    ok, found_port, found_baud, _id, msg = autodetect_servo(
        preferred_port=None if port in (None, "", "auto") else port,
        protocol=protocol,
        baudrates=None if port in (None, "", "auto") else (baud,),
        preferred_id=detect_id,
    )
    if not ok:
        return False, None, None, None, None, [], f"bus open failed: {msg}"

    port_handler = PortHandler(found_port)
    packet_handler = PacketHandler(protocol)
    if not port_handler.openPort():
        return (False, None, None, None, None, [],
                f"failed to open shared port {found_port}")
    if not port_handler.setBaudRate(found_baud):
        port_handler.closePort()
        return (False, None, None, None, None, [],
                f"failed to set baud {found_baud} on {found_port}")

    data_list, _comm = packet_handler.broadcastPing(port_handler)
    ids = sorted(int(i) for i in (data_list or {}).keys())
    if expected_ids is not None:
        missing = [i for i in expected_ids if i not in ids]
        if missing:
            port_handler.closePort()
            return (False, None, None, None, None, ids,
                    f"expected servo id(s) {missing} not found on {found_port} "
                    f"@ {found_baud} baud (saw {ids or 'none'})")

    return (True, port_handler, packet_handler, found_port, found_baud, ids,
            f"bus open on {found_port} @ {found_baud} baud, ids {ids}")


class SoftLimitError(ValueError):
    """Raised when a commanded ΔL would exceed the soft cap or go negative."""


class Servo:
    """Safety-wrapped interface to a single Dynamixel XM430-W350 motor.

    All public motion methods honour the soft ΔL cap and refuse to move
    while the estop flag is latched. See the module docstring for the
    spool conversion and safety model.
    """

    def __init__(
        self,
        port: str = "auto",
        baud: int = 57600,
        dxl_id: int = 15,
        protocol: float = 2.0,
        spool_radius_m: float = config.SPOOL_RADIUS,
        soft_delta_l_cap_mm: float = 100.0,
        current_limit_units: int = 1193,
        overcurrent_warn_ma: float = 3000.0,
        pull_sign: int = +1,
    ) -> None:
        self.port = port
        self.baud = baud
        self.dxl_id = dxl_id
        self.protocol = protocol
        self.spool_radius_m = float(spool_radius_m)
        self.soft_delta_l_cap_mm = float(soft_delta_l_cap_mm)
        # Clamp the configured ceiling into the valid hardware range.
        self.current_limit_units = int(
            max(0, min(CURRENT_LIMIT_MAX_UNITS, current_limit_units))
        )
        self.overcurrent_warn_ma = float(overcurrent_warn_ma)
        self.pull_sign = 1 if pull_sign >= 0 else -1

        # SDK handles (created at connect(), or injected via attach_bus()).
        self._port_handler = None
        self._packet_handler = None
        # Whether this Servo owns its serial port (opens/closes it). False when
        # the port is a shared U2D2 bus owned elsewhere (see attach_bus / open_bus).
        self._owns_port = True

        # State.
        self._connected = False
        self._torque_on = False
        self._estop = False
        self._over_current = False
        self._overcurrent_count = 0

        # Reference / goal tracking (in revolutions, absolute ticks frame).
        self._zero_rev = 0.0          # captured ΔL = 0 reference
        self._goal_rev = 0.0          # last commanded goal
        self._pos_rev = 0.0           # last read present position

        # Ramp state.
        self._ramp_active = False
        self._ramp_target_delta_mm = 0.0
        self._ramp_rate_mm_s = 0.0    # signed per-second rate toward target

        # Latest telemetry cache.
        self._current_ma = 0.0
        self._velocity_rpm = 0.0
        self._voltage_v = 0.0
        self._temp_c = 0.0

    # -- conversions ------------------------------------------------------
    def _delta_mm_to_goal_rev(self, delta_l_mm: float) -> float:
        delta_l_m = delta_l_mm / 1000.0
        return self._zero_rev + self.pull_sign * delta_l_m / _circumference(
            self.spool_radius_m
        )

    def _goal_rev_to_delta_mm(self, goal_rev: float) -> float:
        delta_l_m = (
            self.pull_sign
            * _circumference(self.spool_radius_m)
            * (goal_rev - self._zero_rev)
        )
        return delta_l_m * 1000.0

    def _check_soft_cap(self, delta_l_mm: float) -> None:
        if delta_l_mm < -1e-9:
            raise SoftLimitError(
                f"ΔL {delta_l_mm:.3f} mm is below zero reference (no negative pull)."
            )
        if delta_l_mm > self.soft_delta_l_cap_mm + 1e-9:
            raise SoftLimitError(
                f"ΔL {delta_l_mm:.3f} mm exceeds soft cap "
                f"{self.soft_delta_l_cap_mm:.3f} mm."
            )

    # -- low-level SDK helpers -------------------------------------------
    def _write1(self, addr: int, value: int) -> Tuple[bool, str]:
        result, error = self._packet_handler.write1ByteTxRx(
            self._port_handler, self.dxl_id, addr, value & 0xFF
        )
        return self._check_comm(result, error, f"write1 @{addr}")

    def _write2(self, addr: int, value: int) -> Tuple[bool, str]:
        result, error = self._packet_handler.write2ByteTxRx(
            self._port_handler, self.dxl_id, addr, value & 0xFFFF
        )
        return self._check_comm(result, error, f"write2 @{addr}")

    def _write4(self, addr: int, value: int) -> Tuple[bool, str]:
        result, error = self._packet_handler.write4ByteTxRx(
            self._port_handler, self.dxl_id, addr, value & 0xFFFFFFFF
        )
        return self._check_comm(result, error, f"write4 @{addr}")

    def _read1(self, addr: int) -> int:
        value, result, error = self._packet_handler.read1ByteTxRx(
            self._port_handler, self.dxl_id, addr
        )
        self._check_comm(result, error, f"read1 @{addr}")
        return value

    def _read2(self, addr: int) -> int:
        value, result, error = self._packet_handler.read2ByteTxRx(
            self._port_handler, self.dxl_id, addr
        )
        self._check_comm(result, error, f"read2 @{addr}")
        return value

    def _read4(self, addr: int) -> int:
        value, result, error = self._packet_handler.read4ByteTxRx(
            self._port_handler, self.dxl_id, addr
        )
        self._check_comm(result, error, f"read4 @{addr}")
        return value

    def _check_comm(self, result: int, error: int, ctx: str) -> Tuple[bool, str]:
        if result != COMM_SUCCESS:
            msg = self._packet_handler.getTxRxResult(result)
            return False, f"{ctx}: {msg}"
        if error != 0:
            msg = self._packet_handler.getRxPacketError(error)
            return False, f"{ctx}: {msg}"
        return True, "ok"

    # -- connection lifecycle --------------------------------------------
    def connect(self) -> Tuple[bool, str]:
        """Open the port and run the safe init sequence.

        Sequence: torque OFF -> operating mode 4 -> current limit -> torque ON.
        Returns ``(ok, message)``.
        """
        if not _SDK_AVAILABLE:
            return (
                False,
                f"dynamixel_sdk not available: {_SDK_IMPORT_ERROR}",
            )
        if self._connected:
            return True, "already connected"

        if self._owns_port:
            # Auto-detect the port/baud/id when requested (port "auto" / None / "").
            # Robust to the U2D2 enumerating on a different /dev/ttyUSB* each plug-in.
            if self.port in (None, "", "auto"):
                ok, port, baud, dxl_id, msg = autodetect_servo(
                    protocol=self.protocol, preferred_id=self.dxl_id)
                if not ok:
                    return False, f"servo auto-detect failed: {msg}"
                self.port = port
                self.baud = baud
                self.dxl_id = dxl_id

            self._port_handler = PortHandler(self.port)
            self._packet_handler = PacketHandler(self.protocol)

            if not self._port_handler.openPort():
                return False, f"failed to open port {self.port}"
            if not self._port_handler.setBaudRate(self.baud):
                self._port_handler.closePort()
                return False, f"failed to set baud rate {self.baud}"
        else:
            # Shared bus: attach_bus() injected an already-open port + packet
            # handler owned by the bus (e.g. two fingers on one U2D2). Don't open
            # or close it here — just run the per-id init sequence below.
            if self._port_handler is None or self._packet_handler is None:
                return False, ("shared-bus servo has no attached port — "
                               "call attach_bus() before connect()")

        # Safe init: torque OFF before touching EEPROM (operating mode +
        # current limit both require torque disabled).
        ok, msg = self._write1(ADDR_TORQUE_ENABLE, TORQUE_OFF)
        if not ok:
            self._close_owned_port()
            return False, f"torque off failed ({msg})"

        ok, msg = self._write1(
            ADDR_OPERATING_MODE, OPERATING_MODE_EXTENDED_POSITION
        )
        if not ok:
            self._close_owned_port()
            return False, f"set operating mode failed ({msg})"

        ok, msg = self._write2(ADDR_CURRENT_LIMIT, self.current_limit_units)
        if not ok:
            self._close_owned_port()
            return False, f"set current limit failed ({msg})"

        # Torque ON to begin position control.
        ok, msg = self._write1(ADDR_TORQUE_ENABLE, TORQUE_ON)
        if not ok:
            self._close_owned_port()
            return False, f"torque on failed ({msg})"

        self._connected = True
        self._torque_on = True
        self._estop = False
        self._over_current = False
        self._overcurrent_count = 0

        # Initialise references from the present position so ΔL = 0 here.
        try:
            raw = self._read4(ADDR_PRESENT_POSITION)
            self._pos_rev = self._signed32(raw) / TICKS_PER_REV
        except Exception:
            self._pos_rev = 0.0
        self._zero_rev = self._pos_rev
        self._goal_rev = self._pos_rev

        return True, (
            f"connected on {self.port} @ {self.baud} baud, id {self.dxl_id}, "
            f"current limit {self.current_limit_units} units "
            f"(~{self.current_limit_units * CURRENT_LIMIT_UNIT_MA / 1000.0:.2f} A)"
        )

    def disconnect(self) -> None:
        """Disable torque and close the port (guarded if not open).

        A shared-bus servo (``_owns_port == False``) does NOT close the port —
        the bus owner is responsible for closing it once all motors are done.
        """
        if not self._connected:
            return
        try:
            self._write1(ADDR_TORQUE_ENABLE, TORQUE_OFF)
        except Exception:
            pass
        self._torque_on = False
        self._close_owned_port()
        self._connected = False
        self._ramp_active = False

    def attach_bus(self, port_handler, packet_handler, port: str,
                   baud: int) -> None:
        """Bind this servo to a shared, already-open U2D2 bus.

        Use when several daisy-chained motors share one serial port (e.g. the
        two finger servos on one U2D2). The bus owner opens the port once via
        :func:`open_bus`, then calls this for each motor; :meth:`connect` then
        skips opening/closing and only runs the per-id init sequence.
        """
        self._port_handler = port_handler
        self._packet_handler = packet_handler
        self.port = port
        self.baud = baud
        self._owns_port = False

    def _close_owned_port(self) -> None:
        """Close the serial port only if this servo owns it (never a shared bus)."""
        if self._owns_port and self._port_handler is not None:
            try:
                self._port_handler.closePort()
            except Exception:
                pass

    # -- torque control ---------------------------------------------------
    def enable(self) -> Tuple[bool, str]:
        """Enable torque. Clears the estop latch."""
        if not self._connected:
            return False, "not connected"
        ok, msg = self._write1(ADDR_TORQUE_ENABLE, TORQUE_ON)
        if ok:
            self._torque_on = True
            self._estop = False
            self._over_current = False
            self._overcurrent_count = 0
        return ok, msg

    def disable(self) -> Tuple[bool, str]:
        """Disable torque (does not latch estop)."""
        if not self._connected:
            return False, "not connected"
        ok, msg = self._write1(ADDR_TORQUE_ENABLE, TORQUE_OFF)
        if ok:
            self._torque_on = False
        return ok, msg

    def e_stop(self) -> None:
        """Immediate torque disable; latch estop and refuse motion."""
        self._estop = True
        self._ramp_active = False
        if self._connected:
            try:
                self._write1(ADDR_TORQUE_ENABLE, TORQUE_OFF)
            except Exception:
                pass
        self._torque_on = False

    def is_connected(self) -> bool:
        return self._connected

    # -- references / calibration ----------------------------------------
    def set_zero(self) -> None:
        """Capture the present position as the ΔL = 0 reference."""
        if self._connected:
            try:
                raw = self._read4(ADDR_PRESENT_POSITION)
                self._pos_rev = self._signed32(raw) / TICKS_PER_REV
            except Exception:
                pass
        self._zero_rev = self._pos_rev
        self._goal_rev = self._pos_rev
        self._ramp_active = False
        self._ramp_target_delta_mm = 0.0

    def set_pull_direction(self, sign: int) -> None:
        """Set the pull direction (+1 or -1)."""
        self.pull_sign = 1 if sign >= 0 else -1

    # -- motion: jog ------------------------------------------------------
    def jog(self, direction: int, step_rev: float = 0.02) -> Tuple[bool, str]:
        """Nudge the goal by ``step_rev`` revolutions in ``direction`` (+1/-1).

        Respects the soft ΔL cap and the estop latch. Sends the new goal.
        """
        if self._estop:
            return False, "refused: estopped"
        if not self._connected:
            return False, "not connected"

        step = (1.0 if direction >= 0 else -1.0) * abs(step_rev)
        candidate_rev = self._goal_rev + step
        candidate_delta_mm = self._goal_rev_to_delta_mm(candidate_rev)
        try:
            self._check_soft_cap(candidate_delta_mm)
        except SoftLimitError as exc:
            return False, f"refused: {exc}"

        self._goal_rev = candidate_rev
        ok, msg = self._send_goal()
        return ok, msg

    # -- motion: ramp -----------------------------------------------------
    def start_ramp(
        self, target_delta_L_mm: float, speed_mm_s: float = 2.0
    ) -> bool:
        """Arm a slow ramp from the current ΔL to ``target_delta_L_mm``.

        Returns ``False`` (and arms nothing) if the target exceeds the soft
        cap / is negative, or if estopped. Stores target plus the signed
        per-second rate consumed by :meth:`service`.
        """
        if self._estop:
            return False
        try:
            self._check_soft_cap(target_delta_L_mm)
        except SoftLimitError:
            return False

        current_delta = self.current_delta_L_mm()
        direction = 1.0 if target_delta_L_mm >= current_delta else -1.0
        self._ramp_target_delta_mm = float(target_delta_L_mm)
        self._ramp_rate_mm_s = direction * abs(speed_mm_s)
        self._ramp_active = abs(target_delta_L_mm - current_delta) > 1e-9
        return True

    def service(self, dt: float = 0.05) -> dict:
        """Advance an active ramp and refresh telemetry. Call periodically.

        Steps the active ramp's goal toward the target by ``rate*dt``
        (mm -> rev via the spool), sends the goal, then reads telemetry and
        enforces the overcurrent E-stop. Returns :meth:`get_state`.
        """
        if not self._connected:
            return self.get_state()

        # Advance the ramp (only if not estopped).
        if self._ramp_active and not self._estop:
            current_delta = self.current_delta_L_mm()
            remaining = self._ramp_target_delta_mm - current_delta
            step = self._ramp_rate_mm_s * dt
            if abs(step) >= abs(remaining):
                next_delta = self._ramp_target_delta_mm
                self._ramp_active = False
            else:
                next_delta = current_delta + step
            # Defensive clamp into the soft range.
            next_delta = max(0.0, min(self.soft_delta_l_cap_mm, next_delta))
            self._goal_rev = self._delta_mm_to_goal_rev(next_delta)
            self._send_goal()
        elif self._estop:
            self._ramp_active = False

        # Read telemetry and enforce overcurrent E-stop.
        self._read_telemetry()
        self._enforce_overcurrent()
        return self.get_state()

    def goto_delta_L(
        self,
        target_delta_L_mm: float,
        speed_mm_s: float = 2.0,
        settle_s: float = 0.0,
    ) -> dict:
        """Blocking convenience ramp (uses ``time.sleep`` + service loop).

        Honours the soft cap and estop. Returns the final state dict.
        """
        dt = 0.05
        if not self.start_ramp(target_delta_L_mm, speed_mm_s):
            return self.get_state()
        # Drive the ramp to completion.
        while self.is_ramping() and not self._estop:
            self.service(dt)
            time.sleep(dt)
        # Final telemetry refresh.
        self.service(dt)
        if settle_s > 0:
            elapsed = 0.0
            while elapsed < settle_s and not self._estop:
                self.service(dt)
                time.sleep(dt)
                elapsed += dt
        return self.get_state()

    # -- queries ----------------------------------------------------------
    def current_delta_L_mm(self) -> float:
        """Current ΔL (mm) derived from the last commanded goal."""
        return self._goal_rev_to_delta_mm(self._goal_rev)

    def is_ramping(self) -> bool:
        return self._ramp_active

    def get_state(self) -> dict:
        return {
            "connected": self._connected,
            "torque_on": self._torque_on,
            "estop": self._estop,
            "over_current": self._over_current,
            "pos_rev": (self._pos_rev - self._zero_rev),
            "goal_rev": (self._goal_rev - self._zero_rev),
            "delta_L_mm": self.current_delta_L_mm(),
            "target_delta_L_mm": (
                self._ramp_target_delta_mm
                if self._ramp_active
                else self.current_delta_L_mm()
            ),
            "ramping": self._ramp_active,
            "current_ma": self._current_ma,
            "velocity_rpm": self._velocity_rpm,
            "voltage_v": self._voltage_v,
            "temp_c": self._temp_c,
        }

    # -- internals --------------------------------------------------------
    @staticmethod
    def _signed32(raw: int) -> int:
        return raw - 4294967296 if raw > 2147483647 else raw

    @staticmethod
    def _signed16(raw: int) -> int:
        return raw - 65536 if raw > 32767 else raw

    def _send_goal(self) -> Tuple[bool, str]:
        if self._estop:
            return False, "refused: estopped"
        if not self._connected:
            return False, "not connected"
        ticks = int(round(self._goal_rev * TICKS_PER_REV))
        return self._write4(ADDR_GOAL_POSITION, ticks)

    def _read_telemetry(self) -> None:
        try:
            raw_pos = self._read4(ADDR_PRESENT_POSITION)
            self._pos_rev = self._signed32(raw_pos) / TICKS_PER_REV
        except Exception:
            pass
        try:
            raw_vel = self._signed32(self._read4(ADDR_PRESENT_VELOCITY))
            self._velocity_rpm = raw_vel * VELOCITY_UNIT_RPM
        except Exception:
            pass
        try:
            raw_cur = self._read2(ADDR_PRESENT_CURRENT)
            self._current_ma = self._signed16(raw_cur) * CURRENT_UNIT_MA
        except Exception:
            pass
        try:
            raw_vol = self._read2(ADDR_PRESENT_VOLTAGE)
            self._voltage_v = raw_vol * VOLTAGE_UNIT_V
        except Exception:
            pass
        try:
            self._temp_c = float(self._read1(ADDR_PRESENT_TEMP))
        except Exception:
            pass

    def _enforce_overcurrent(self) -> None:
        if abs(self._current_ma) > self.overcurrent_warn_ma:
            self._overcurrent_count += 1
            if self._overcurrent_count >= _OVERCURRENT_TRIP_TICKS:
                self._over_current = True
                self.e_stop()
        else:
            self._overcurrent_count = 0


class MockServo:
    """Hardware-free stand-in for :class:`Servo` with the same interface.

    Simulates ``pos_rev`` / ``delta_L_mm`` tracking the ramp, a plausible
    current that rises with ΔL, and respects the soft cap and e-stop. Lets
    the dashboard and scripts run without a motor attached.
    """

    def __init__(
        self,
        port: str = "auto",
        baud: int = 57600,
        dxl_id: int = 15,
        protocol: float = 2.0,
        spool_radius_m: float = config.SPOOL_RADIUS,
        soft_delta_l_cap_mm: float = 25.0,
        current_limit_units: int = 1193,
        overcurrent_warn_ma: float = 3000.0,
        pull_sign: int = +1,
    ) -> None:
        self.port = port
        self.baud = baud
        self.dxl_id = dxl_id
        self.protocol = protocol
        self.spool_radius_m = float(spool_radius_m)
        self.soft_delta_l_cap_mm = float(soft_delta_l_cap_mm)
        self.current_limit_units = int(
            max(0, min(CURRENT_LIMIT_MAX_UNITS, current_limit_units))
        )
        self.overcurrent_warn_ma = float(overcurrent_warn_ma)
        self.pull_sign = 1 if pull_sign >= 0 else -1

        self._connected = False
        self._torque_on = False
        self._estop = False
        self._over_current = False
        self._overcurrent_count = 0

        self._zero_rev = 0.0
        self._goal_rev = 0.0
        self._pos_rev = 0.0

        self._ramp_active = False
        self._ramp_target_delta_mm = 0.0
        self._ramp_rate_mm_s = 0.0

        # Simulated telemetry.
        self._current_ma = 0.0
        self._velocity_rpm = 0.0
        self._voltage_v = 12.0
        self._temp_c = 30.0

    # -- conversions (mirror Servo) --------------------------------------
    def _delta_mm_to_goal_rev(self, delta_l_mm: float) -> float:
        delta_l_m = delta_l_mm / 1000.0
        return self._zero_rev + self.pull_sign * delta_l_m / _circumference(
            self.spool_radius_m
        )

    def _goal_rev_to_delta_mm(self, goal_rev: float) -> float:
        delta_l_m = (
            self.pull_sign
            * _circumference(self.spool_radius_m)
            * (goal_rev - self._zero_rev)
        )
        return delta_l_m * 1000.0

    def _check_soft_cap(self, delta_l_mm: float) -> None:
        if delta_l_mm < -1e-9:
            raise SoftLimitError(
                f"ΔL {delta_l_mm:.3f} mm is below zero reference (no negative pull)."
            )
        if delta_l_mm > self.soft_delta_l_cap_mm + 1e-9:
            raise SoftLimitError(
                f"ΔL {delta_l_mm:.3f} mm exceeds soft cap "
                f"{self.soft_delta_l_cap_mm:.3f} mm."
            )

    # -- lifecycle --------------------------------------------------------
    def connect(self) -> Tuple[bool, str]:
        self._connected = True
        self._torque_on = True
        self._estop = False
        self._over_current = False
        self._overcurrent_count = 0
        self._zero_rev = self._pos_rev
        self._goal_rev = self._pos_rev
        return True, "mock connected (no hardware)"

    def disconnect(self) -> None:
        if not self._connected:
            return
        self._torque_on = False
        self._connected = False
        self._ramp_active = False

    def enable(self) -> Tuple[bool, str]:
        if not self._connected:
            return False, "not connected"
        self._torque_on = True
        self._estop = False
        self._over_current = False
        self._overcurrent_count = 0
        return True, "ok"

    def disable(self) -> Tuple[bool, str]:
        if not self._connected:
            return False, "not connected"
        self._torque_on = False
        return True, "ok"

    def e_stop(self) -> None:
        self._estop = True
        self._ramp_active = False
        self._torque_on = False

    def is_connected(self) -> bool:
        return self._connected

    # -- references -------------------------------------------------------
    def set_zero(self) -> None:
        self._zero_rev = self._pos_rev
        self._goal_rev = self._pos_rev
        self._ramp_active = False
        self._ramp_target_delta_mm = 0.0

    def set_pull_direction(self, sign: int) -> None:
        self.pull_sign = 1 if sign >= 0 else -1

    # -- motion -----------------------------------------------------------
    def jog(self, direction: int, step_rev: float = 0.02) -> Tuple[bool, str]:
        if self._estop:
            return False, "refused: estopped"
        if not self._connected:
            return False, "not connected"
        step = (1.0 if direction >= 0 else -1.0) * abs(step_rev)
        candidate_rev = self._goal_rev + step
        candidate_delta_mm = self._goal_rev_to_delta_mm(candidate_rev)
        try:
            self._check_soft_cap(candidate_delta_mm)
        except SoftLimitError as exc:
            return False, f"refused: {exc}"
        self._goal_rev = candidate_rev
        return True, "ok"

    def start_ramp(
        self, target_delta_L_mm: float, speed_mm_s: float = 2.0
    ) -> bool:
        if self._estop:
            return False
        try:
            self._check_soft_cap(target_delta_L_mm)
        except SoftLimitError:
            return False
        current_delta = self.current_delta_L_mm()
        direction = 1.0 if target_delta_L_mm >= current_delta else -1.0
        self._ramp_target_delta_mm = float(target_delta_L_mm)
        self._ramp_rate_mm_s = direction * abs(speed_mm_s)
        self._ramp_active = abs(target_delta_L_mm - current_delta) > 1e-9
        return True

    def service(self, dt: float = 0.05) -> dict:
        if not self._connected:
            return self.get_state()

        prev_delta = self.current_delta_L_mm()

        if self._ramp_active and not self._estop:
            current_delta = prev_delta
            remaining = self._ramp_target_delta_mm - current_delta
            step = self._ramp_rate_mm_s * dt
            if abs(step) >= abs(remaining):
                next_delta = self._ramp_target_delta_mm
                self._ramp_active = False
            else:
                next_delta = current_delta + step
            next_delta = max(0.0, min(self.soft_delta_l_cap_mm, next_delta))
            self._goal_rev = self._delta_mm_to_goal_rev(next_delta)
        elif self._estop:
            self._ramp_active = False

        # Simulated plant: present position chases the goal instantly.
        self._pos_rev = self._goal_rev
        new_delta = self.current_delta_L_mm()

        # Simulated telemetry.
        if dt > 0:
            self._velocity_rpm = ((new_delta - prev_delta) / dt) * 0.5
        else:
            self._velocity_rpm = 0.0
        # Plausible current: baseline plus a term rising with ΔL.
        self._current_ma = 120.0 + 90.0 * abs(new_delta)
        self._voltage_v = 12.0
        self._temp_c = 30.0 + 0.2 * abs(new_delta)

        self._enforce_overcurrent()
        return self.get_state()

    def goto_delta_L(
        self,
        target_delta_L_mm: float,
        speed_mm_s: float = 2.0,
        settle_s: float = 0.0,
    ) -> dict:
        dt = 0.05
        if not self.start_ramp(target_delta_L_mm, speed_mm_s):
            return self.get_state()
        while self.is_ramping() and not self._estop:
            self.service(dt)
            time.sleep(dt)
        self.service(dt)
        if settle_s > 0:
            elapsed = 0.0
            while elapsed < settle_s and not self._estop:
                self.service(dt)
                time.sleep(dt)
                elapsed += dt
        return self.get_state()

    # -- queries ----------------------------------------------------------
    def current_delta_L_mm(self) -> float:
        return self._goal_rev_to_delta_mm(self._goal_rev)

    def is_ramping(self) -> bool:
        return self._ramp_active

    def get_state(self) -> dict:
        return {
            "connected": self._connected,
            "torque_on": self._torque_on,
            "estop": self._estop,
            "over_current": self._over_current,
            "pos_rev": (self._pos_rev - self._zero_rev),
            "goal_rev": (self._goal_rev - self._zero_rev),
            "delta_L_mm": self.current_delta_L_mm(),
            "target_delta_L_mm": (
                self._ramp_target_delta_mm
                if self._ramp_active
                else self.current_delta_L_mm()
            ),
            "ramping": self._ramp_active,
            "current_ma": self._current_ma,
            "velocity_rpm": self._velocity_rpm,
            "voltage_v": self._voltage_v,
            "temp_c": self._temp_c,
        }

    # -- internals --------------------------------------------------------
    def _enforce_overcurrent(self) -> None:
        if abs(self._current_ma) > self.overcurrent_warn_ma:
            self._overcurrent_count += 1
            if self._overcurrent_count >= _OVERCURRENT_TRIP_TICKS:
                self._over_current = True
                self.e_stop()
        else:
            self._overcurrent_count = 0


__all__ = [
    "Servo",
    "MockServo",
    "SoftLimitError",
    "autodetect_servo",
    "list_serial_ports",
]
