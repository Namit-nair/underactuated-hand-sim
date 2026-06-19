#!/usr/bin/env python3
"""Live PhaseSpace diagnostic — dump raw markers + per-segment vectors / phi.

Connects to the mocap stream and prints, for a few frames, every visible marker
id with its (x, y, z), then the four configured segment vectors and the in-plane
angle they project to. Use it to confirm the LED-id mapping, the units / scale,
and which world plane the finger actually flexes in (so calibration makes sense).

Run:
    python mocap/diagnose.py            # real PhaseSpace (slave; GUI can stay open)
    python mocap/diagnose.py --mock     # synthetic, sanity check
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mocap_config as mcfg  # noqa: E402
from tracker import _NSEG, build_tracker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="PhaseSpace live diagnostic")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--server", default=mcfg.MOCAP_SERVER_IP)
    ap.add_argument("--no-slave", action="store_true")
    ap.add_argument("--frames", type=int, default=30)
    a = ap.parse_args()

    tr = build_tracker(
        mock=a.mock, server=a.server,
        segment_marker_ids=mcfg.MOCAP_SEGMENT_MARKER_IDS,
        calib_path=mcfg.MOCAP_CALIB_PATH,
        timeout_us=mcfg.MOCAP_EVENT_TIMEOUT_US,
        slave=(not a.no_slave) and mcfg.MOCAP_SLAVE,
    )
    print(f"Connecting to {'MOCK' if a.mock else a.server} (slave={not a.no_slave}) ...")
    tr.start()
    time.sleep(0.6)
    print(f"calibrated={tr.calibrated}")
    print(f"  n      = {np.round(tr.n, 3)}")
    print(f"  u_axis = {np.round(tr.u_axis, 3)}")
    print(f"  u_perp = {np.round(tr.u_perp, 3)}")
    print(f"  segment map (near,far) = {mcfg.MOCAP_SEGMENT_MARKER_IDS}")

    seen = set()
    for k in range(a.frames):
        snap = tr._snapshot()
        seen |= set(snap.keys())
        det = tr.detect()
        vecs = tr.segment_vectors()
        if k % 6 == 0 or k == a.frames - 1:
            print(f"\n--- frame {k}: ids present = {sorted(snap.keys())}")
            for mid in sorted(snap.keys()):
                m = snap[mid]
                print(f"   id {mid:2d}: x={m.x:9.2f} y={m.y:9.2f} z={m.z:9.2f} cond={m.cond:.1f}")
            for si in range(_NSEG):
                v = vecs.get(si)
                p = det["phi"].get(si)
                vs = "None" if v is None else f"[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}]"
                ps = "None" if p is None else f"{p:+7.2f} deg"
                print(f"   seg{si} {mcfg.SEGMENT_LABELS[si]:4s} "
                      f"ids={mcfg.MOCAP_SEGMENT_MARKER_IDS[si]} vec={vs} phi={ps}")
        time.sleep(0.05)

    print(f"\nall marker ids seen over {a.frames} frames: {sorted(seen)}")
    expected = sorted({i for pair in mcfg.MOCAP_SEGMENT_MARKER_IDS for i in pair})
    missing = [i for i in expected if i not in seen]
    print(f"expected ids: {expected}")
    print(f"missing ids : {missing if missing else 'none'}")
    tr.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
