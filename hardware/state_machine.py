#!/usr/bin/env python3
"""Experiment state machine + settle detector for the finger-validation rig.

Framework-agnostic (no Qt). The dashboard owns the timer/loop and calls into
these helpers to (a) track which phase the operator is in and (b) decide when
the finger has reached quasi-static equilibrium so a capture is meaningful.

State flow (operator-paced, the default):

    IDLE ──connect──▶ JOG ──"Set Zero"──▶ ZEROED
      ▲                                     │ choose ΔL + "Go"
      │                                     ▼
   CAPTURE ◀── "Capture" ── SETTLED ◀── SETTLING ◀── RAMP
      │                                     ▲
      └──────────── choose next ΔL ─────────┘

AUTO_SWEEP walks a fixed ΔL list, ramping + settling + auto-capturing each,
repeating for a configurable number of trials.
"""
from __future__ import annotations

from enum import Enum, auto


class State(Enum):
    IDLE = auto()        # nothing connected / not ready
    JOG = auto()         # connected, manual nudging, not yet zeroed
    ZEROED = auto()      # reference pose captured, ready to choose a ΔL
    RAMP = auto()        # servo slowly pulling toward the target ΔL
    SETTLING = auto()    # ramp done, waiting for joints to go quasi-static
    SETTLED = auto()     # quasi-static — a capture is valid
    CAPTURE = auto()     # a row was just logged
    AUTO_SWEEP = auto()  # automated walk over the ΔL list

    def label(self) -> str:
        return {
            State.IDLE: "IDLE",
            State.JOG: "JOG / manual",
            State.ZEROED: "ZEROED — pick ΔL",
            State.RAMP: "RAMPING",
            State.SETTLING: "SETTLING…",
            State.SETTLED: "SETTLED ✓",
            State.CAPTURE: "CAPTURED",
            State.AUTO_SWEEP: "AUTO SWEEP",
        }[self]


class SettleDetector:
    """Declare quasi-static equilibrium from joint angular velocity.

    The dashboard feeds successive joint-angle dicts with timestamps. We track
    the peak per-joint angular speed; once it stays below ``vel_thresh_deg_s``
    continuously for ``hold_s`` the pose is SETTLED. If that never happens within
    ``timeout_s`` of the ramp finishing we report TIMEOUT (a capture is still
    allowed, but the row is flagged so it can be filtered later).
    """

    SETTLING = "settling"
    SETTLED = "settled"
    TIMEOUT = "timeout"

    def __init__(self, vel_thresh_deg_s: float = 2.0,
                 hold_s: float = 0.5, timeout_s: float = 8.0):
        self.vel_thresh_deg_s = float(vel_thresh_deg_s)
        self.hold_s = float(hold_s)
        self.timeout_s = float(timeout_s)
        self.reset()

    def reset(self, t0: float | None = None) -> None:
        self._prev_theta = None      # dict mcp/pip/dip
        self._prev_t = None
        self._below_since = None     # time we last dropped below threshold
        self._start_t = t0           # ramp-finished time (timeout origin)
        self.last_speed = float("nan")

    def start(self, t: float) -> None:
        """Mark the moment the ramp finished (timeout origin)."""
        self.reset(t0=t)

    @staticmethod
    def _max_speed(prev: dict, cur: dict, dt: float) -> float:
        if dt <= 0:
            return float("nan")
        speeds = []
        for j in ("mcp", "pip", "dip"):
            a, b = prev.get(j), cur.get(j)
            if a is None or b is None:
                continue
            speeds.append(abs(b - a) / dt)
        return max(speeds) if speeds else float("nan")

    def update(self, theta: dict, t: float) -> str:
        """Feed a new joint-angle sample; return SETTLING / SETTLED / TIMEOUT."""
        if self._start_t is None:
            self._start_t = t

        if self._prev_theta is not None and self._prev_t is not None:
            dt = t - self._prev_t
            speed = self._max_speed(self._prev_theta, theta, dt)
            self.last_speed = speed
            if speed == speed:  # not NaN
                if speed < self.vel_thresh_deg_s:
                    if self._below_since is None:
                        self._below_since = t
                    elif (t - self._below_since) >= self.hold_s:
                        self._prev_theta, self._prev_t = theta, t
                        return self.SETTLED
                else:
                    self._below_since = None  # broke the quiet streak

        self._prev_theta, self._prev_t = theta, t

        if (t - self._start_t) >= self.timeout_s:
            return self.TIMEOUT
        return self.SETTLING

    def elapsed(self, t: float) -> float:
        return 0.0 if self._start_t is None else max(0.0, t - self._start_t)


class AutoSweep:
    """Drives an automated ΔL walk: for each trial, step through the ΔL list,
    ramping + settling + capturing each point.

    The dashboard polls :meth:`current_target`, calls :meth:`advance` after each
    capture, and stops when :meth:`done` is True.
    """

    def __init__(self, delta_l_list=(5.0, 10.0, 15.0, 20.0), n_trials: int = 5):
        self.delta_l_list = list(delta_l_list)
        self.n_trials = int(n_trials)
        self.reset()

    def reset(self) -> None:
        self._i = 0           # index into delta_l_list
        self._trial = 0       # 0-based trial counter
        self._finished = False

    @property
    def trial_idx(self) -> int:
        return self._trial

    def current_target(self):
        """Return the current ΔL [mm], or None if the sweep is finished."""
        if self._finished or self._trial >= self.n_trials:
            return None
        return self.delta_l_list[self._i]

    def advance(self) -> None:
        """Call after a point has been captured to move to the next one."""
        if self._finished:
            return
        self._i += 1
        if self._i >= len(self.delta_l_list):
            self._i = 0
            self._trial += 1
            if self._trial >= self.n_trials:
                self._finished = True

    @property
    def done(self) -> bool:
        return self._finished or self._trial >= self.n_trials

    def progress(self) -> str:
        if self.done:
            return "sweep complete"
        return (f"trial {self._trial + 1}/{self.n_trials}  •  "
                f"ΔL {self.delta_l_list[self._i]:.0f} mm "
                f"({self._i + 1}/{len(self.delta_l_list)})")
