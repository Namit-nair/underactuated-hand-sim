# MuJoCo simulations for a tendon-driven underactuated finger

CAD-accurate MuJoCo simulations of an anthropomorphic 3-joint (MCP/PIP/DIP)
tendon-driven underactuated finger, used to study the geometric and mechanical
relationship

$$\Delta L \;\longrightarrow\; (\theta_{\text{MCP}},\, \theta_{\text{PIP}},\, \theta_{\text{DIP}})$$

where $\Delta L$ is the active flexor-tendon displacement and the joint angles
emerge passively from quasi-static equilibrium. The high-fidelity finger model
is the foundation for ongoing gripper development.

---

## Single source of truth: `config.py`

**Every physical and numerical parameter lives in `config.py`** — geometry,
joint limits, tendon properties, hardware spring stiffnesses, the validation
tendon pull `DELTA_L`, and solver tolerances. Change a value once there and it
propagates everywhere:

```
config.py
  ├─ high_fidelity/interactive_viewer.py  → generates finger.xml
  ├─ finger_model.py                       → physics-faithful model surgery
  ├─ analytical_model.py                   → analytical joint limits
  └─ high_fidelity/validation.py           → ΔL, springs, all plots + CSVs
```

There is no need to edit any other file to change finger length, ΔL, stiffness
ratios, or joint stops.

---

## Repository layout

```text
underactuated_finger/
├── config.py                 # SINGLE SOURCE OF TRUTH for all parameters
├── analytical_model.py       # Closed-form morphology law (energy minimisation)
├── finger_model.py           # Turns the soft viewer XML into a physics-faithful model
├── requirements.txt
│
├── high_fidelity/
│   ├── interactive_viewer.py # Regenerates finger.xml from config + launches viewer
│   ├── static_viewer.py      # Loads finger.xml and opens the viewer (optional ramp)
│   ├── validation.py         # Analytical-vs-MuJoCo stiffness-ratio validation suite
│   ├── finger.xml            # Generated CAD-geometry model (do not hand-edit)
│   ├── params.json           # CAD inertial properties (mass, COM, inertia)
│   ├── meshes/               # proximal/middle/distal STL meshes
│   └── validation_results/   # Plots + CSVs written by validation.py
│
├── hardware/                 # Physical validation rig: PySide6 dashboard,
│   │                         # Dynamixel servo, RealSense + ArUco joint angles,
│   │                         # load-carrying (pull-out) test dashboard
│   └── README.md
│
├── mocap/                    # PhaseSpace (OWL2) optical-tracking validation rig:
│   │                         # alternative joint-angle source, reuses the hardware
│   │                         # dashboard/servo/logger (see mocap/README.md)
│   └── README.md
│
├── gripper/                  # Gripper development: build_gripper.py + headless
│                             # stiffness-ratio holding-capacity (Tmax) sweep
│
└── legacy/                   # Initial low-fidelity model + early research outputs
                              # (kept for reference; see legacy/README.md)
```

---

## Setup

```bash
python3 -m venv mujoco_env && source mujoco_env/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)        # so scripts can `import config`
```

(An `.envrc` is included that activates the env and sets `PYTHONPATH`.)

---

## How to run

**Interactive viewer** — edit `config.py`, then regenerate `finger.xml` and open
the MuJoCo viewer with a tendon-displacement slider:

```bash
python3 high_fidelity/interactive_viewer.py
```

**Validation suite** — sweeps the stiffness ratios $\rho_1=k_1/k_2$,
$\rho_3=k_3/k_2$ (anchored on the three measured hardware springs) and compares
the MuJoCo equilibrium against the analytical morphology law. Writes all plots
and CSVs to `high_fidelity/validation_results/`:

```bash
python3 high_fidelity/validation.py
```

Outputs include trend lines, morphology stick figures, sim-vs-analytical
scatter, and per-cell heatmaps for the joint angles, the morphology metrics
$M_{12}=\theta_1/\theta_2$ and $M_{32}=\theta_3/\theta_2$, and their error —
reported **both** as absolute error and as signed percentage error
$(M^{sim}-M^{ana})/M^{ana}\times100$.

---

## The two models

1. **Analytical (`analytical_model.py`)** — quasi-static joint angles from
   minimum elastic energy under the inextensible-tendon constraint,
   $\theta_i = (r_i/k_i)\,\Delta L / \sum_j (r_j^2/k_j)$, with cascaded
   redistribution when a joint hits its mechanical stop.

2. **High-fidelity MuJoCo (`finger_model.py` + `finger.xml`)** — the CAD geometry
   from `finger.xml` with a hardware-faithful flexor: a constant sheath moment
   arm, a near-inextensible steel-string tendon, and near-rigid joint limits.
   Moment arms and link lengths are extracted from the loaded model so both
   halves share one geometric source of truth.
