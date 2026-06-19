#!/usr/bin/env python3
"""Mocap-rig-only configuration (PhaseSpace finger tracking).

Kept SEPARATE from the repo's root ``config.py`` on purpose: the root config is
the single source of truth for the *physical* finger (geometry, joint limits,
spring stiffnesses, spool radius, moment arms) and is consumed by the analytical
model + the hardware servo. This file holds ONLY mocap-specific knobs — the
PhaseSpace server address, which LED ids belong to which finger segment, and
where calibration / results live. Nothing here changes the physics.

Marker layout (decided with the user)
-------------------------------------
Two LED markers per segment, on FOUR segments — base/palm, proximal, middle,
distal (8 LEDs total). Each segment's direction vector is ``p_far - p_near``;
the joint angle is the in-plane angle between consecutive segment vectors, which
two points give directly (no rigid body needed — the OWL SDK requires >=4
markers for a rigid body, and we only need a direction).

``MOCAP_SEGMENT_MARKER_IDS`` is ordered ``[base, proximal, middle, distal]``;
each entry is ``(near_id, far_id)`` where ``near`` is the LED nearer the palm and
``far`` is the LED nearer the fingertip, so ``far - near`` points distally along
the finger. EDIT these to match the LED ids you assigned in the PhaseSpace
Master Client.
"""

import os

# --- PhaseSpace server / streaming ------------------------------------------
MOCAP_SERVER_IP = "192.168.1.230"   # Impulse server address (same as the test scripts)
MOCAP_EVENT_TIMEOUT_US = 1_000_000  # nextEvent() timeout [microseconds]
# Connect as a read-only slave so the PhaseSpace Master Client can stay open on
# the same server at the same time. Set False to be the sole/primary client.
MOCAP_SLAVE = True

# --- LED id -> segment mapping ----------------------------------------------
# Ordered [base, proximal, middle, distal]; (near_id, far_id) per segment.
# Index 0..3 here lines up with hardware/joints.py's marker indices
# (0=base, 1=prox, 2=mid, 3=dist), so the existing MCP/PIP/DIP differencing and
# zeroing work unchanged.
#
# As-built LED placement (reported by the user, given tip -> bottom per link):
#   DIP / distal link : 6 at the tip, 4 at the bottom   -> near=4, far=6
#   PIP / middle link : 3 at the tip, 5 at the bottom   -> near=5, far=3
#   MCP / proximal link: 7 at the tip, 1 at the bottom  -> near=1, far=7
#   base / palm       : 2 at the tip, 0 at the bottom   -> near=0, far=2
# (near = nearer the palm, far = nearer the fingertip; far - near points distally.)
MOCAP_SEGMENT_MARKER_IDS = (
    (0, 2),   # base / palm
    (1, 7),   # proximal phalanx  (MCP link)
    (5, 3),   # middle phalanx    (PIP link)
    (4, 6),   # distal phalanx    (DIP link)
)

SEGMENT_LABELS = ("base", "prox", "mid", "dist")

# --- servo ΔL travel --------------------------------------------------------
# Manual/GO/jog tendon-pull ceiling [mm]. The hardware dashboard hard-caps the
# target field at 25 mm; the mocap rig raises it (and the servo soft cap) so the
# tendon can be driven well past 70 mm for the pull-out / large-flex tests.
MOCAP_MAX_DELTA_MM = 120.0

# --- paths (everything mocap-related lives inside mocap/) -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MOCAP_CALIB_PATH = os.path.join(_HERE, "mocap_calibration.json")
MOCAP_RESULTS_DIR = os.path.join(_HERE, "results")
