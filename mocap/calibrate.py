#!/usr/bin/env python3
"""Standalone flexion-plane calibration for the PhaseSpace finger tracker.

Records a short flex sweep and fits the flexion-plane basis (normal + in-plane
axes) used to turn 3D segment vectors into signed joint angles, then writes
``mocap/mocap_calibration.json``. The GUI dashboard has the same flow on a
button; this CLI is for calibrating without launching the full window.

Run:
    python mocap/calibrate.py --mock              # synthetic, for a dry run
    python mocap/calibrate.py --seconds 8         # real PhaseSpace, 8 s sweep

While it records, slowly flex the finger through its FULL range and back so the
moving segments sweep the plane. All four segments must stay visible.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mocap_config as mcfg  # noqa: E402
from tracker import build_tracker  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="PhaseSpace flexion-plane calibration")
    p.add_argument("--mock", action="store_true", help="synthetic tracker (dry run)")
    p.add_argument("--server", default=mcfg.MOCAP_SERVER_IP)
    p.add_argument("--no-slave", action="store_true")
    p.add_argument("--seconds", type=float, default=8.0,
                   help="how long to record the flex sweep (default 8 s)")
    p.add_argument("--rate", type=float, default=30.0, help="sample rate [Hz]")
    args = p.parse_args()

    tracker = build_tracker(
        mock=args.mock, server=args.server,
        segment_marker_ids=mcfg.MOCAP_SEGMENT_MARKER_IDS,
        calib_path=mcfg.MOCAP_CALIB_PATH,
        timeout_us=mcfg.MOCAP_EVENT_TIMEOUT_US,
        slave=(not args.no_slave) and mcfg.MOCAP_SLAVE,
    )

    print(f"Connecting to {'MOCK' if args.mock else args.server} ...")
    tracker.start()
    time.sleep(0.5)  # let the stream warm up

    print(f"Recording for {args.seconds:.0f} s - FLEX THE FINGER through its full "
          "range and back now.")
    tracker.begin_calibration()
    dt = 1.0 / max(1.0, args.rate)
    t_end = time.monotonic() + args.seconds
    last_print = 0.0
    while time.monotonic() < t_end:
        n = tracker.record_calibration_sample()
        now = time.monotonic()
        if now - last_print >= 1.0:
            remaining = t_end - now
            print(f"  {remaining:4.1f} s left - {n} complete samples")
            last_print = now
        time.sleep(dt)

    ok = tracker.finalize_calibration(save=True)
    if ok:
        print(f"\n[OK] Calibration saved to {mcfg.MOCAP_CALIB_PATH}")
        print(f"  n      = {tracker.n}")
        print(f"  u_axis = {tracker.u_axis}")
        print(f"  u_perp = {tracker.u_perp}")
        rc = 0
    else:
        print("\n[FAIL] Calibration failed - not enough complete samples (need "
              "all 4 segments visible). Re-run with a slower, fuller flex.")
        rc = 1

    tracker.stop()
    return rc


if __name__ == "__main__":
    sys.exit(main())
