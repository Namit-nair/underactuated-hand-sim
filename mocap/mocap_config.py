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
# As-built LED placement (CORRECTED): the finger is mounted end-for-end relative
# to the first guess — the (0,2) pair is the FINGERTIP, not the base. The real
# order runs base->tip from marker 6 down to marker 0, so the whole chain is
# reversed AND each pair's near/far flips (the LED that pointed toward the old
# "tip" actually points toward the real palm). near = nearer the palm, far =
# nearer the fingertip; far - near points distally.
#   base / palm        : near=6, far=4
#   MCP / proximal link: near=3, far=5
#   PIP / middle link  : near=7, far=1
#   DIP / distal link  : near=2, far=0   (the (0,2) pair we first called "base")
MOCAP_SEGMENT_MARKER_IDS = (
    (6, 4),   # base / palm
    (3, 5),   # proximal phalanx  (MCP link)
    (7, 1),   # middle phalanx    (PIP link)
    (2, 0),   # distal phalanx    (DIP link)
)

SEGMENT_LABELS = ("base", "prox", "mid", "dist")

# --- flexion-plane definition -----------------------------------------------
# The finger flexes in a FIXED plane (no abduction), so the flexion plane is the
# horizontal plane and its normal is simply the lab VERTICAL axis. We define the
# plane from that known normal — no calibration flex, no extra marker needed. The
# projection then strips every bit of out-of-plane marker tilt/rotation, and the
# straight-pose Set Zero cancels the residual constant in-plane offsets.
#
# Only the NORMAL affects joint angles (they are differences of consecutive
# segment angles, which are invariant to the in-plane long-axis reference). So
# the single thing to get right is which axis points up.
#
# CHECK THIS LIVE: lift a marker off the table and watch which coordinate grows.
MOCAP_VERTICAL_AXIS = 1      # which PhaseSpace axis is UP: 0=X, 1=Y, 2=Z
MOCAP_VERTICAL_SIGN = +1     # +1 if that axis points up, -1 if it points down

# Sign of the joint angles. Flexion (servo pull) should read POSITIVE; with the
# as-built marker direction the pull comes out negative, so default to -1 to flip
# it (the dashboard's "FLIP θ SIGN" button toggles this live).
MOCAP_FLEXION_SIGN = -1      # +1 or -1

# --- servo ΔL travel --------------------------------------------------------
# Manual/GO/jog tendon-pull ceiling [mm]. The hardware dashboard hard-caps the
# target field at 25 mm; the mocap rig raises it (and the servo soft cap) so the
# tendon can be driven well past 70 mm for the pull-out / large-flex tests.
MOCAP_MAX_DELTA_MM = 120.0

# --- paths (everything mocap-related lives inside mocap/) -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
MOCAP_CALIB_PATH = os.path.join(_HERE, "mocap_calibration.json")
MOCAP_RESULTS_DIR = os.path.join(_HERE, "results")
