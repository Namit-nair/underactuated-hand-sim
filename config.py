#!/usr/bin/env python3
"""Single source of truth for the tendon-driven finger simulation.

EVERY physical / numerical parameter lives here. Change it once and it
propagates everywhere:

    config.py
      │
      ├─ high_fidelity/interactive_viewer.py  → generates finger.xml (geometry,
      │                                          joint ranges, visual tendon)
      ├─ finger_model.py                       → physics-faithful model surgery
      │                                          (constant moment arm, stiff
      │                                          tendon, hard limits)
      ├─ analytical_model.py                   → joint limits for the closed-form
      │                                          morphology law
      └─ high_fidelity/validation.py           → ΔL, hardware springs, tolerances,
                                                 all plots + CSVs

This module is pure data — it must NOT import mujoco/numpy so that the
analytical model stays importable on machines without a simulator.
"""

# =====================================================================
# Joint ordering (used everywhere; keep all per-joint sequences in this order)
# =====================================================================
JOINT_NAMES = ("mcp", "pip", "dip")   # proximal → middle → distal

# =====================================================================
# 1. GEOMETRY — from CAD, in millimetres. Joint axis = +Z for all hinges.
#    These drive finger.xml generation and (via the model) the analytical
#    moment arms / link lengths.
# =====================================================================
MCP_CENTER = (12.5, 0.0, 0.0)
PIP_CENTER = (-43.0, 0.0, 0.0)
DIP_CENTER = (-84.6, 0.0, 0.0)
TIP_POINT = (-120.399, 0.0, 0.0)
JOINT_AXIS = (0, 0, 1)

# Palmar (tendon/flexor) side normal
PALMAR_NORMAL = (0.0, -1.0, 0.0)

# Tendon routing offsets from each link centerline (mm)
MCP_OFFSET = 7.0
PIP_OFFSET = 7.0
DIP_OFFSET = 7.0

# Routing fractions along each link (0 = proximal joint end, 1 = distal end)
MCP_ENTRY_FRAC = 0.20
MCP_EXIT_FRAC = 0.80
PIP_ENTRY_FRAC = 0.20
PIP_EXIT_FRAC = 0.80
DIP_ENTRY_FRAC = 0.20
DIP_ANCHOR_FRAC = 0.80

# =====================================================================
# 2. JOINT MECHANICS — passive springs, dampers, and mechanical stops.
#    Ranges (deg) are the SINGLE definition of the joint limits: they are
#    written into finger.xml AND consumed by the analytical model and the
#    validation stick-figure clipping (so the three can never drift apart).
# =====================================================================
MCP_RANGE = (-5, 90)    # degrees
PIP_RANGE = (0, 110)
DIP_RANGE = (0, 90)

# Per-joint passive stiffness written into finger.xml [N·m/rad].
# (The validation suite overrides these per-cell to sweep stiffness ratios.)
MCP_STIFFNESS = 1.0
PIP_STIFFNESS = 1.0
DIP_STIFFNESS = 1.0

# Stiffness on the <default> hinge — finger_model's hard-limit surgery keys
# off this joint. Kept distinct from the per-joint values above.
DEFAULT_JOINT_STIFFNESS = 2.0

MCP_DAMPING = 0.08      # N·m·s/rad
PIP_DAMPING = 0.08
DIP_DAMPING = 0.08

# Ordered (3, 2) view of the limits for array consumers (analytical/validation).
JOINT_RANGES_DEG = (MCP_RANGE, PIP_RANGE, DIP_RANGE)

# =====================================================================
# 3. VISUAL TENDON + SIM — properties of the spatial tendon in finger.xml
#    as generated for the interactive viewer (NOT the physics-faithful model).
# =====================================================================
VIS_TENDON_STIFFNESS = 1e6   # N/m
VIS_TENDON_DAMPING = 1.0
MAX_DELTA_L = 0.20              # m — viewer slider / actuator ctrl range
TENDON_WIDTH = 0.0006          # m (visual only)

TIMESTEP = 0.002               # s — finger.xml viewer timestep
INTEGRATOR = "implicitfast"
GRAVITY = (0, 0, -9.81)

# =====================================================================
# 4. PHYSICS-FIDELITY MODEL — finger_model.py rewrites the soft viewer XML
#    into a hardware-faithful model (steel-string tendon in a sheath):
#      * constant sheath moment arm (flip-free),
#      * near-inextensible tendon,
#      * near-rigid joint limits + matching timestep.
# =====================================================================
SHEATH_MOMENT_ARM = 0.007      # m — tendon moment arm at full extension (0°)
                               #     (still the constant arm used by the MuJoCo
                               #      sheath geom + the analytical fallback).

# ---- Angle-dependent moment arm (analytical model) -------------------------
# CAD shows the tendon moment arm is NOT constant: it grows with joint flexion
# from SHEATH_MOMENT_ARM at 0° (links straight) to MOMENT_ARM_FULL_FLEXION at
# MOMENT_ARM_FLEXION_REF_DEG of flexion. Assumed LINEAR between the two CAD
# measurements. analytical_model.py adds this per-joint increment on top of each
# joint's extension arm and solves the resulting implicit equilibrium.
# Set MOMENT_ARM_ANGLE_DEPENDENT = False to fall back to the constant arm.
MOMENT_ARM_ANGLE_DEPENDENT = True
MOMENT_ARM_FULL_FLEXION = 0.012        # m — moment arm at full flexion (from CAD)
MOMENT_ARM_FLEXION_REF_DEG = 90.0      # deg — flexion at which the above is measured

TENDON_STIFFNESS = 1.0e5       # N/m — near-inextensible steel string
TENDON_DAMPING = 6.0           # N·s/m
SIM_TIMESTEP = 0.001           # s — small enough for near-rigid limits
LIMIT_SOLREF = "0.002 1"               # timeconst = 2*timestep (stiffest stable)
LIMIT_SOLIMP = "0.99 0.9999 0.0001 0.5 2"

# =====================================================================
# 5. HARDWARE SPRINGS — measured torsional stiffnesses [N·m/rad].
#    Spring 2 is the reference k2 for the ρ ratios in the validation sweep.
# =====================================================================
SPRING_1 = 0.6487   # large
SPRING_2 = 0.1184   # medium — reference k2
SPRING_3 = 0.0286   # small

# =====================================================================
# 6. VALIDATION — actuation magnitude and equilibrium-solver tolerances.
#    Change DELTA_L here and it flows to every angle, plot, and CSV.
# =====================================================================
DELTA_L = 0.010         # m — tendon pull for the validation study
EQUIL_MAX_TIME = 4.0    # s — convergence cap per equilibrium run
VEL_TOL = 1.0e-3        # rad/s — settle threshold (free joints)
SATURATION_TOL = 0.5    # deg from a joint limit that counts as saturated

# =====================================================================
# 7. GRIPPER — two-finger assembly (gripper/build_gripper.py +
#    gripper/interactive_gripper.py). BOTH fingers reuse the shared joint
#    stiffness/damping/ranges above, so a single MCP/PIP/DIP stiffness edit
#    applies to BOTH fingers. The numbers here are gripper-only layout and
#    default graspable-object parameters. The per-finger tendon physics is
#    the SAME tested high-fidelity model (SHEATH_MOMENT_ARM, TENDON_STIFFNESS,
#    TENDON_DAMPING, SIM_TIMESTEP, LIMIT_SOLREF/SOLIMP from sections 2-4).
# =====================================================================
GRIPPER_SEPARATION = 0.110            # m — default centre-to-centre finger gap (aperture)
GRIPPER_SEPARATION_RANGE = (0.020, 0.200)  # m — UI aperture slider bounds
# NOTE: the open finger half-thickness is ~13 mm, so the object must FIT the gap:
# aperture/2 - 0.013 >= object radius, else the fingers start embedded in the
# object at rest and contact forces explode. Keep the aperture wide enough.
GRIPPER_MOUNT_HEIGHT = 0.040          # m — finger base height above the floor
GRIPPER_GRAVITY_DEFAULT = False       # match the tested (gravity-free) fidelity model

# Finger drive: each flexor is driven by ΔL — we shorten the tendon spring's rest
# length (validation.py method: tendon_lengthspring = L_rest - ΔL). The stiff,
# near-inextensible string (TENDON_STIFFNESS) then pulls the joints flexed. The
# small gripper timestep + rigid contacts below keep this stable against the
# static object (the displacement-vs-rigid-wall blow-up only happened at a coarse
# timestep). Grip rises with ΔL; the string is stiff, so creep the slider.
GRIPPER_MAX_PULL_MM = 40.0            # mm — per-finger ΔL slider/entry upper bound

# Joint stiffness is taken from MCP/PIP/DIP_STIFFNESS (section 2) when the gripper
# loads, and is live-overridable from the control panel (shared by both fingers)
# for on-the-go testing of how the stiffness ratio affects grip. The override is
# NOT persisted — relaunching reads the config values again.
GRIPPER_STIFFNESS_MAX = 3.0          # N·m/rad — upper bound of the live stiffness sliders

# Gripper sub-step timestep — SMALLER than the morphology model's SIM_TIMESTEP so
# the contacts can be made near-rigid (a contact's stiffness is capped at ≈ 2×
# timestep, so a coarse step lets the finger punch transiently into the object
# during a hard close — the "cutting in"). The interactive viewer sub-steps to
# stay real-time. With this, peak penetration stays < ~0.35 mm even at 200+ N.
GRIPPER_TIMESTEP = 0.00025            # s

# Near-rigid contacts — the research-backed recipe for stable MuJoCo grasping:
#   * solref ≈ 2×timestep, solimp dmin→1   -> rigid from first contact (no "mush"),
#   * cone="elliptic" + impratio>1         -> stiff friction, objects don't slip,
#   * condim=6 + torsional/rolling friction -> round objects don't spin/roll out.
# (See MuJoCo docs "Contact" + the Menagerie Robotiq/hand grasping threads.)
GRIPPER_CONTACT_SOLREF = "0.0005 1"   # ≈ 2×GRIPPER_TIMESTEP, stiffest stable
GRIPPER_CONTACT_SOLIMP = "0.99 0.9999 0.0001 0.5 2"   # dmin→1: rigid boundary
GRIPPER_CONTACT_CONDIM = 6            # 6 = normal + slide + torsional + rolling
GRIPPER_CONTACT_FRICTION = "1 0.1 0.01"   # slide, torsional, rolling
GRIPPER_FRICTION_CONE = "elliptic"    # elliptic + impratio is the anti-slip combo
GRIPPER_IMPRATIO = 10.0               # raise (→50-200) for firmer free-object load tests

# Probe object (live-editable from the control panel). It is WELDED in space
# (a static body, no joint) so the fingers press against an immovable target —
# the point is to study how the joint stiffness ratio affects grip, not to lift
# a free object. Only its depth between the fingers is adjustable, along world Z:
#   low Z  -> deep in the palm  (enveloping grasp)
#   high Z -> out at the fingertips (pinch grasp)
# The cylinder lies along X so the fingers wrap its circular cross-section.
GRIPPER_OBJECT_ENABLED = True
GRIPPER_OBJECT_SHAPE = "cylinder"     # box | cylinder | sphere
GRIPPER_OBJECT_SIZE_MM = 35.0         # radius / box half-width
GRIPPER_OBJECT_LENGTH_MM = 40.0       # cylinder half-length (along X) / box half-height
GRIPPER_OBJECT_SIZE_RANGE_MM = (4.0, 80.0)     # UI size bounds (radius up to 80 mm)
GRIPPER_OBJECT_LENGTH_RANGE_MM = (4.0, 120.0)  # UI length bounds
GRIPPER_OBJECT_DEPTH_RANGE = (0.01, 0.15)      # m — UI depth bounds
GRIPPER_OBJECT_DEPTH_Z = 0.075        # m — default object depth between the fingers

# =====================================================================
# 8. LOAD TEST — horizontal gripper pull-out test
#    (gripper/interactive_load_test.py — single self-contained file).
#    Two fingers extend along +X (horizontal), gripping an object that
#    can slide freely along X. A motor actuator applies a controllable
#    extraction force T to measure grip retention for a given stiffness
#    ratio.
# =====================================================================
LOAD_TEST_SLIDE_RANGE = (-0.01, 0.15)   # m — object slide joint limits along X
LOAD_TEST_SLIDE_DAMPING = 0.5           # N·s/m — light viscous drag on the slide
LOAD_TEST_OBJECT_MASS_KG = 0.050        # kg — object mass (50 g default)
LOAD_TEST_MAX_TENSION = 300.0           # N — pull-force ceiling (T slider + motor ctrlrange).
# Sized for the ~441 N/finger servo cap below, which holds ~200 N at the default
# grasp; raise this if a deeper/stronger grasp never fails within the slider range.

# Where along +X the object sits relative to the finger bases (set at build time;
# the panel has no live depth slider — depth defines the grasp TYPE, so relaunch
# to change it). This is the load test's OWN depth (separate from the static
# gripper's GRIPPER_OBJECT_DEPTH_Z): a deep/ENVELOPING grasp (~50 mm) wraps the
# object and holds a real pull-out load, whereas a shallow fingertip PINCH (~75+
# mm) resists by friction alone and slips almost immediately. The object must
# still FIT the aperture: aperture/2 - 0.013 >= radius, else the fingers start
# embedded in it and contact forces explode.
LOAD_TEST_OBJECT_DEPTH_X = 0.050        # m — default: an enveloping grasp that holds

# ---- Actuator force limit (the "infinite grip" fix) -------------------------
# A ΔL-displacement tendon is a near-rigid spring (TENDON_STIFFNESS ≈ 1e5 N/m),
# so a finger blocked by the object resists with essentially UNBOUNDED force —
# the grip would be "infinitely strong" and no external load could ever pull the
# object out. A real tendon actuator (servo + string) can only output a finite
# force. We cap the flexor tension at this ceiling (see
# interactive_load_test.capped_lengthspring); once the pull T exceeds what the
# capped grip can hold, the object slips and slides to the LOAD_TEST_SLIDE_RANGE end.
#
# Tendon spool at the servo horn — SINGLE SOURCE OF TRUTH for the winding radius.
# Used both by the hardware rig (ΔL <-> servo revolutions, hardware/servo.py) and
# by the load-test force ceiling below. MEASURED on the as-built spool:
# outer diameter 22.35 mm -> radius 11.175 mm. (Was previously guessed at 10 mm /
# 12.5 mm in different places; this constant unifies them.)
SPOOL_RADIUS = 0.011175                 # m — measured tendon winding radius (Ø22.35 mm)

# The ceiling comes from the servo, which winds the tendon onto a spool of radius
# SPOOL_RADIUS:  tendon tension = servo_torque / spool_radius.
# One servo per finger, so each flexor gets the full stall torque.
#   Dynamixel XM430-W350-T: ~45 kgf·cm ≈ 4.41 N·m stall (max output).
#   r_spool = 11.175 mm  →  F_max ≈ 395 N per finger.
LOAD_TEST_SERVO_STALL_TORQUE = 4.41     # N·m — servo max output (Dynamixel XM430-W350-T, ~45 kgf·cm)
LOAD_TEST_SPOOL_RADIUS = SPOOL_RADIUS   # m — string winding radius at the servo horn (see SPOOL_RADIUS)
LOAD_TEST_MAX_TENDON_FORCE = LOAD_TEST_SERVO_STALL_TORQUE / LOAD_TEST_SPOOL_RADIUS  # N — per-finger flexor force ceiling
LOAD_TEST_MAX_FORCE_UI_MAX = 600.0      # N — upper bound of the live "F max" slider
