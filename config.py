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

# Link masses [kg] — written into finger.xml <inertial mass="...">.
# Change here and re-run interactive_viewer.py to regenerate.
PROXIMAL_MASS = 0.0200
MIDDLE_MASS   = 0.0169
DISTAL_MASS   = 0.0137

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
PIP_RANGE = (0, 90)     # hardware PIP mechanically locks out at ~90° (was 110)
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
# from SHEATH_MOMENT_ARM at 0° (links straight) and SATURATES toward full
# flexion. The growth is SUB-LINEAR (NOT the straight 7→12 mm line previously
# assumed): the table below is MEASURED from CAD at 10° steps. analytical_model.py
# interpolates it on |θ| (holding the 90° value beyond the measured range, e.g.
# PIP > 90°), adds the increment over the 0° arm on top of each joint's extension
# arm, and solves the resulting implicit equilibrium.
# Set MOMENT_ARM_ANGLE_DEPENDENT = False to fall back to the constant arm.
MOMENT_ARM_ANGLE_DEPENDENT = True
# Measured moment-arm calibration: joint flexion [deg] -> tendon moment arm [mm].
# Index-aligned; the 0° entry equals SHEATH_MOMENT_ARM (7 mm) — the extension arm
# the per-joint increment is referenced to.
MOMENT_ARM_CURVE_DEG = (0,   10,   20,  30,   40,   50,   60,    70,    80,    90)
MOMENT_ARM_CURVE_MM = (7.0, 7.63, 8.2, 8.7, 9.14, 9.51, 9.81, 10.04, 10.18, 10.25)
# Back-compat scalars derived from the curve (some readers still expect these);
# they now reflect the measured saturated value, not the old linear 12 mm.
MOMENT_ARM_FULL_FLEXION = MOMENT_ARM_CURVE_MM[-1] / 1000.0   # m — arm at full flexion
MOMENT_ARM_FLEXION_REF_DEG = float(MOMENT_ARM_CURVE_DEG[-1]) # deg — its flexion angle

TENDON_STIFFNESS = 1.0e5       # N/m — near-inextensible steel string
TENDON_DAMPING = 6.0           # N·s/m
SIM_TIMESTEP = 0.001           # s — small enough for near-rigid limits
LIMIT_SOLREF = "0.002 1"               # timeconst = 2*timestep (stiffest stable)
LIMIT_SOLIMP = "0.99 0.9999 0.0001 0.5 2"

# =====================================================================
# 5. HARDWARE SPRINGS — measured torsional stiffnesses [N·m/rad].
#    Spring 2 is the reference k2 for the ρ ratios in the validation sweep.
# =====================================================================
SPRING_1 = 0.789   # large
SPRING_2 = 0.165   # medium — reference k2
SPRING_3 = 0.072   # small

# Sweep grid: log-spaced points BETWEEN adjacent springs, plus the 3 springs
# themselves. Total grid size = 2 * SWEEP_N_PER_INTERVAL + 3.
# E.g. SWEEP_N_PER_INTERVAL = 2 → grid of 7: S3, ·, ·, S2, ·, ·, S1.
SWEEP_N_PER_INTERVAL = 2

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
GRIPPER_APERTURE_MAX_MM = 95.0       # mm — hardware maximum aperture (fully-open fingers)
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
# Sized for the ~306 N/finger servo cap below; raise this if a deeper/stronger
# grasp never fails within the slider range (or the sweep censors Tmax at the cap).

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
# One servo per finger, so each flexor gets the full (de-rated) stall torque.
KGFCM_TO_NM = 0.0980665                  # 1 kgf·cm -> N·m
LOAD_TEST_SERVO_STALL_KGFCM = 41.0       # kgf·cm — measured servo max stall torque
# De-rate the stall torque by a small safety margin (servos overheat and lose
# torque near stall): usable torque = stall * SERVO_SAFETY_FACTOR.
LOAD_TEST_SERVO_SAFETY_FACTOR = 0.85     # 15 % margin below stall ("small safety mark")
LOAD_TEST_SERVO_STALL_TORQUE = LOAD_TEST_SERVO_STALL_KGFCM * KGFCM_TO_NM            # N·m (~4.02)
LOAD_TEST_SERVO_USABLE_TORQUE = LOAD_TEST_SERVO_STALL_TORQUE * LOAD_TEST_SERVO_SAFETY_FACTOR  # N·m (~3.42)
LOAD_TEST_SPOOL_RADIUS = SPOOL_RADIUS   # m — string winding radius at the servo horn (see SPOOL_RADIUS)
#   r_spool = 11.175 mm, usable τ ≈ 3.42 N·m  →  F_max ≈ 306 N per finger.
LOAD_TEST_MAX_TENDON_FORCE = LOAD_TEST_SERVO_USABLE_TORQUE / LOAD_TEST_SPOOL_RADIUS  # N — per-finger flexor force ceiling
# ΔL at which a fully-blocked flexor reaches the force ceiling (tendon is a stiff
# spring: F = TENDON_STIFFNESS * ΔL_block). The close must add this to the slack
# take-up, so the sweep commands a larger ΔL and lets the cap hold force at F_max.
LOAD_TEST_DELTA_L_AT_FMAX = LOAD_TEST_MAX_TENDON_FORCE / TENDON_STIFFNESS           # m (~3.1 mm)
LOAD_TEST_MAX_FORCE_UI_MAX = 600.0      # N — upper bound of the live "F max" slider


# =====================================================================
# Load-carrying (pull-out) HARDWARE test  —  hardware/load_test_dashboard.py
# =====================================================================
# Physical counterpart of the simulated load test (gripper/interactive_load_test.py):
# two tendon-driven fingers (Dynamixel A/B daisy-chained on one U2D2) grip an object;
# a third servo (its own U2D2) winds a stainless string that pulls the object out via a
# Futek LCM300 axial load cell (read over a USB220 module that free-runs ASCII on a
# /dev/ttyUSB* serial port). We ramp the pull, watch the measured force rise, and record
# the PEAK force at which the grip releases = the load-carrying capacity.

LBF_TO_N = 4.4482216153                 # exact pound-force -> newton conversion
KGF_TO_N = 9.80665                       # standard gravity: 1 kgf -> N (N / this = kg)

# -- Futek LCM300 load cell (via USB220 serial module) ----------------
LOADCELL_CAPACITY_LB = 250.0            # rated capacity [lbf] (LCM300 as ordered)
LOADCELL_CAPACITY_N = LOADCELL_CAPACITY_LB * LBF_TO_N   # ~1112.06 N
LOADCELL_SENSITIVITY_MV_V = 2.0         # rated output [mV/V] (cal cert) — for reference
LOADCELL_BAUD = 9600                    # USB220 serial baud — confirm against the device
# First float on each free-running ASCII line (handles "+0012.34", "12.34 lb", etc.).
LOADCELL_LINE_REGEX = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
LOADCELL_INPUT_UNIT = "n"               # falls through _unit_factor() to 1.0: LOADCELL_SCALE already does lbf->N
# Combined factor: lbf->N conversion AND an empirically-fitted ~5x gain
# correction (NOT a pure unit conversion). Derived from 5 known-mass points
# (0.5/0.75/1.0/1.25/1.2 kg) via an origin-forced least-squares fit of tared
# raw lbf vs. expected N. Re-derive if the cable, connector, or USB220 module
# is ever replaced — the gain error is specific to this hardware chain.
LOADCELL_SCALE = 22.20                  # N per raw tared lbf unit (see comment above)
LOADCELL_FILTER_ALPHA = 0.3             # low-pass on force, 0..1 (1 = raw, no filtering)

# -- Servo ids / pull-out winding -------------------------------------
FINGER_A_DXL_ID = 15                    # finger A (U2D2 #1, shared bus)
FINGER_B_DXL_ID = 16                    # finger B (U2D2 #1, shared bus)
PULL_DXL_ID = 17                        # pull servo  (U2D2 #2, its own port)

# Per-servo default pull (tensioning / flex) direction (+1 or -1). Positive ΔL
# must wind in the slack-takeup direction; this depends on how each horn/spool
# was wound. Finger A's horn winds opposite to B on this build, so its default
# is flipped — the ⇄ A button still toggles it live if a rebuild changes that.
FINGER_A_PULL_SIGN = -1
FINGER_B_PULL_SIGN = +1
PULL_PULL_SIGN = +1

PULL_SPOOL_RADIUS = SPOOL_RADIUS        # m — TODO: set once the pull horn/spool is designed
PULL_SPEED_MM_S = 2.0                   # default string take-up (winding) speed
PULL_MAX_DELTA_MM = 120.0               # pull-servo soft ΔL cap (string take-up range)

# -- Stiffness configurations (which hardware springs are installed) --
# A load-test trial set is run with a chosen set of torsional springs at the
# three finger joints. Each named configuration below maps a dropdown label to
# the (MCP, PIP, DIP) spring stiffnesses [N·m/rad] physically installed for that
# set — the name is just shorthand for the SPRING_1/2/3 values above. The
# dashboard's selector reads this dict, and the chosen config's values are
# logged per trial (k_mcp/k_pip/k_dip) so every CSV records which springs
# produced it. Add or rename entries here as new spring combinations are tested.
LOAD_TEST_STIFFNESS_CONFIGS = {
    "Uniform — large (S1·S1·S1)":   (SPRING_1, SPRING_1, SPRING_1),
    "Uniform — medium (S2·S2·S2)":  (SPRING_2, SPRING_2, SPRING_2),
    "Uniform — small (S3·S3·S3)":   (SPRING_3, SPRING_3, SPRING_3),
    "Graded stiff→soft (S1·S2·S3)": (SPRING_1, SPRING_2, SPRING_3),
    "Graded soft→stiff (S3·S2·S1)": (SPRING_3, SPRING_2, SPRING_1),
}
# Selection shown on dashboard startup (must be a key of the dict above).
LOAD_TEST_STIFFNESS_DEFAULT = "Uniform — medium (S2·S2·S2)"

# -- Release detection ------------------------------------------------
RELEASE_DROP_FRAC = 0.30                # force drop from the running peak that flags release
RELEASE_MIN_FORCE_N = 2.0               # noise floor — ignore drops below this force


# =====================================================================
# 9. CHAOS / NON-DETERMINISM STUDY  —  high_fidelity/chaos_study.py
# =====================================================================
# Tests whether a ZERO-joint-stiffness underactuated finger is chaotic /
# non-deterministic. The fixed tendon (L = Σ coef_i·θ_i) is ONE constraint on
# THREE joints, so at k=0 (gravity & contact off) the equilibrium is a 2-D
# manifold — the final pose is selected only by the transient, hence acutely
# sensitive to the initial joint angles. We inject small initial-angle noise and
# measure how far the SETTLED pose scatters, contrasting k=0 against a non-zero
# baseline stiffness (the scatter collapsing under stiffness is the "why
# stiffness is needed" result). The MuJoCo run itself is deterministic, so we use
# a fixed grid (to expose sensitivity / branch-jumps) AND a seeded Monte-Carlo
# (to quantify the real-world outcome distribution).
CHAOS_DELTA_L = 0.020              # m — fixed tendon pull for the study (20 mm)
CHAOS_IC_RANGE_DEG = (-1.0, 1.0)  # initial-angle perturbation bounds [deg]
CHAOS_IC_GRID_STEP_DEG = 0.1      # grid step [deg] -> 21 points over the range
CHAOS_RANDOM_N = 300              # Monte-Carlo samples for the all-joints case
CHAOS_RANDOM_SEED = 0             # RNG seed — reproducible random sweep
CHAOS_BASELINE_K = SPRING_2       # N·m/rad — uniform non-zero baseline stiffness
CHAOS_RAMP_TIME = 1.0             # s — quasi-static ΔL ramp (servo-like)
CHAOS_HOLD_TIME = 2.0             # s — settle hold after the ramp
CHAOS_GRAVITY = False             # match the validated gravity-free fidelity model

# =====================================================================
# 10. TENDON-SHEATH & PIN FRICTION  —  high_fidelity/chaos_study.py
# =====================================================================
# The ideal fidelity model is frictionless: every joint feels the SAME tendon
# tension, so a zero-stiffness closure is a near-simultaneous dead heat. Real
# hardware is not — and two GROUNDED, MEASURABLE friction sources break that
# symmetry. They are the leading candidate for the non-repeatable hardware close.
#
#   (a) CAPSTAN (tendon-in-sheath) friction.  Where the tendon wraps each joint
#       it loses tension by the capstan law  T_out = T_in · e^(-mu·Phi), with
#       Phi = wrap angle at that joint (≈ the joint's own flexion + a small
#       resting routing wrap). Tension is therefore HIGHEST at the actuator
#       (proximal/MCP) and WEAKEST at the tip (distal/DIP): distal joints are
#       under-driven, and the deficit grows with curl. Modelled as an opposing
#       joint torque  ARM · T · (1 - e^(-mu·Phi_upstream))  folded into
#       dof_frictionloss (so it is strictly dissipative).
#       PROVENANCE of mu (the only genuinely new number):
#         * direct two-end test — clamp the joints rigid, drive the tendon with
#           the servo, read the far-end tension on the Futek load cell; regress
#           ln(T_in/T_out) against total wrap ΣPhi → slope = mu (offset-free).
#         * sanity band by material pair: PTFE/Teflon-lined sheath + steel cable
#           mu≈0.04-0.10; bare nylon/Delrin mu≈0.15-0.25; bare metal mu≈0.2-0.4.
#       The wrap angles Phi are GEOMETRY (joint angles + routing), NOT free knobs.
#
#   (b) PIN (hinge) Coulomb friction.  Even with pins, each hinge has a dry
#       breakaway torque tau_c. Modelled directly as MuJoCo dof_frictionloss.
#       PROVENANCE: per-joint breakaway test (tendon detached) — hang a known
#       mass at a known moment arm, raise until the joint just rotates,
#       tau_c = m·g·L. Equivalently the half-width of the torque-angle hysteresis
#       loop when driving the bare joint slowly back and forth.
#
# RIGOROUS TIE-DOWN (so these are not tuned in a vacuum): fit (mu, tau_c) so the
# SIMULATED ΔL→tendon-tension hysteresis loop matches the MEASURED one (load then
# unload the tendon over the full ΔL, logging servo-side tension vs ΔL). The loop
# WIDTH is set by total friction; the distal CURL SKEW by the capstan gradient —
# fitting both uses the friction's entire observable footprint, not one point.
# Then report the chaos result as a BAND over FRICTION_MU_SWEEP, carrying the
# measurement uncertainty into the conclusion instead of trusting a single value.
FRICTION_ENABLED = False              # opt-in; the primary chaos study stays frictionless
FRICTION_MU_TENDON = 0.10             # capstan tendon-sheath coeff [-] — REPLACE with the
                                      #   measured ln(T_in/T_out)/ΣPhi slope. Default = mid
                                      #   PTFE-lined-sheath prior pending that measurement.
FRICTION_REST_WRAP_DEG = 0.0          # resting wrap per joint at θ=0 [deg] from the palmar
                                      #   routing offset (refine from CAD; 0 = ignore).
FRICTION_PIN_TORQUE = (0.002, 0.002, 0.002)  # per-joint pin Coulomb torque [N·m] — REPLACE
                                      #   with the per-joint breakaway measurements.
FRICTION_MU_SWEEP = (0.05, 0.10, 0.20)       # capstan-coeff band (low/nominal/high) for the
                                      #   sensitivity report — propagate measured ±uncertainty.
