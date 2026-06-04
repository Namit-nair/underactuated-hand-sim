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
SHEATH_MOMENT_ARM = 0.007      # m — constant tendon moment arm = sheath offset
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
