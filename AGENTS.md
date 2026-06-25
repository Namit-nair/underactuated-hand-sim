# Project Context — Shared Across All AI Agents

> **Canonical instructions file.** Claude, Gemini, Codex, and Copilot all read this.
> Tool-specific files (`CLAUDE.md`, `CODEX.md`) reference this file and add only
> their own mechanics on top. Edit shared rules HERE — not in the per-tool files.

This project contains MuJoCo simulations for a tendon-driven underactuated finger:
an anthropomorphic 3-joint (MCP / PIP / DIP) model, plus a hardware-validation rig.

---

## 👤 User Preferences (MANDATORY)

- **Attribution — NO AI WATERMARKS.** All code, comments, and commit messages are
  attributed to the user (Namit Nair). Never add `Co-Authored-By` lines, "Generated
  with AI", or any mention of AI involvement anywhere — commits, code, or docs.
- **Stack:** Python (MuJoCo, NumPy, SciPy); XML/YAML for models.
- **Style:** Clean, documented, performance-oriented.

---

## 🏗️ Architecture & Source of Truth

- **Single Source of Truth:** `config.py` — all physical & numerical parameters
  (geometry, joint limits, tendon properties, spool radius, etc.) live here.
  Notably `config.SPOOL_RADIUS = 0.011175 m` (measured Ø22.35 mm) is the one
  winding-radius constant used by both the hardware rig and the load-test ceiling.
- **Physics model:** `finger_model.py` — builds the MuJoCo model from `config.py`.
- **Analytical model:** `analytical_model.py` — closed-form morphology laws via
  energy minimization (incl. the angle-dependent tendon moment arm).
- **High-fidelity sim:** `high_fidelity/` — CAD-accurate geometry, interactive
  viewer, and validation suite. Results auto-write to `high_fidelity/validation_results/`.
- **Paper table generation:** `high_fidelity/generate_tables.py` — generates Tables II/III/IV
  for the RAL 2026 publication from validation results.
- **Gripper load-test sim:** `gripper/` — the *simulated* two-finger pull-out rig
  (digital twin of the hardware load test). `interactive_load_test.py` is a single
  self-contained file that **builds** `load_test.xml` (git-ignored artifact) and
  opens a Tk dashboard; it is the source-of-truth XML builder + step physics.
  `stiffness_sweep.py` reuses that builder headlessly to (a) sweep a ρ1×ρ3
  stiffness-ratio grid → `sweep_results/heatmap_Tmax.png` + `stiffness_sweep.csv`,
  and (b) `--combos` mode → the 3 hardware spring sets as a Tmax bar chart
  (`load_test_combos_Tmax.png`). All geometry/friction/servo numbers come from
  `config.py` (`LOAD_TEST_*`, `FRICTION_MU_*`), so the sim tracks the real fixture.
- **Hardware rig:** `hardware/` — PySide6 dashboard, Dynamixel servo, RealSense
  + ArUco joint-angle measurement. All three Dynamixels (finger A, B, pull)
  share one U2D2 serial bus (daisy-chained); `_ensure_bus()` opens it once.
- **Mocap rig:** `mocap/` — PhaseSpace (OWL2) optical tracking as an alternative
  joint-angle source. Self-contained: vendored `owl.py`, `tracker.py`
  (`PhaseSpaceTracker`/`MockTracker`), `dashboard.py` (subclasses the hardware
  `Dashboard`, reusing servo/logger/predictor/joints/auto-sweep verbatim),
  `calibrate.py`, and `mocap_config.py` (mocap-only knobs; physical params still
  come from root `config.py`). Results write to `mocap/results/`.
- **Archived mocap prototypes:** `motion capture test files/` — standalone test scripts
  predating the integrated `mocap/` directory. Not actively used; kept for reference.

When finger geometry or mechanics change, update **both** the high-fidelity model
**and** the analytical validation so they stay consistent.

---

## 📌 Key Technical Decisions (durable — read before touching the model)

- **Spool radius is unified:** `config.SPOOL_RADIUS = 0.011175 m` (measured Ø22.35 mm)
  is the single source for the servo ΔL↔revolutions mapping *and* the load-test force
  ceiling. The old guessed 10 mm / 12.5 mm values are gone — never reintroduce them.
- **Angle-dependent tendon moment arm:** the arm is NOT constant. It grows sub-linearly
  from **7 mm @ 0°** to **~10.25 mm @ 90°** (CAD-measured at 10° steps; saturates
  toward full flexion — not a straight line). `analytical_model.py` solves the
  resulting implicit equilibrium via a **Picard fixed-point** iteration over a
  PCHIP-interpolated curve. Toggle with `config.MOMENT_ARM_ANGLE_DEPENDENT`; the
  constant-arm path is the documented fallback.
- **HW-validation analysis tool** (`high_fidelity/analyze_hw_validation.py`): picks a
  dataset by substring / path / latest; cleans data (drops ΔL=0, marker+PIP guards,
  per-ΔL MAD outlier rejection); reports M12/M32 agreement, angle error, repeatability,
  a per-ΔL table, and optional angle-dependent-arm recompute.
- **ArUco angle capture is averaged:** never trust a single frame — zeroing and capture
  average `N_AVG_SAMPLES` detections with a wrap-safe circular mean per marker.
- **RE-ZERO-theta feature was removed** from the dashboard; auto-sweep advances directly.
  Don't re-add it.
- **Load cell calibration is empirical, not a pure unit conversion.**
  `LOADCELL_SCALE = 22.20` (N per raw tared lbf unit) was derived from an
  origin-forced least-squares fit over 5 known-mass points. Re-derive it if the
  cable, connector, or USB220 module is replaced — the gain error is specific to
  the hardware chain. Force readouts are in kg (via `KGF_TO_N`), not lb.
- **Pull-out uses multi-peak detection with hysteresis**, not halt-on-first-drop.
  Pulling continues after a force drop; an arm → valley → re-arm cycle prevents
  one long slip from registering as many peaks. The capacity latches the largest
  peak across the entire pull; the operator halts manually.
- **Emergency shutdown on any exit.** `install_emergency_shutdown()` wires
  `atexit` + `SIGINT`/`SIGTERM` handlers so Dynamixels are de-energised on
  Ctrl-C or kill, not just on GUI window close. Both dashboards use it.
- **PhaseSpace = labeled POINT markers, not rigid bodies.** Two LEDs per segment
  on four segments (base/prox/mid/dist). A per-segment rigid body is impossible
  here — the OWL SDK needs ≥4 markers for a 6-DOF body (`owl.py`), and we only
  need each segment's direction vector. `tracker.py` projects each 3D segment
  vector onto the flexion plane and reports a per-segment in-plane angle, so the
  existing `joints.py` differencing/zeroing is reused unchanged. The finger
  flexes in a FIXED horizontal plane (no abduction), so the plane normal is the
  lab vertical axis — set once via `MOCAP_VERTICAL_AXIS`/`MOCAP_VERTICAL_SIGN`,
  NO calibration flex. Only the normal affects joint angles (they difference
  consecutive segment angles → invariant to the in-plane long-axis reference,
  which is derived live from the stationary base markers). Residual marker
  pitch/yaw is removed by the projection + the straight-pose Set Zero.

> Live/evolving decisions beyond this list live in the dual-graph MCP store and
> `CONTEXT.md` (see Shared Memory below). This list is for the stable, load-bearing ones.

---

## 🔧 Key Workflows

```bash
# Environment
python3 -m venv mujoco_env && source mujoco_env/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$(pwd)

# Interactive viewer (visualize finger, test tendon displacement)
python3 high_fidelity/interactive_viewer.py

# Validation (MuJoCo vs analytical model)
python3 high_fidelity/validation.py

# Hardware dashboard (real RealSense + Dynamixel)
python3 hardware/dashboard.py          # add --mock to run with no hardware

# Load-test dashboard (pull-out rig: all servos on one U2D2 + Futek LCM300)
python3 hardware/load_test_dashboard.py              # auto-detect ports
python3 hardware/load_test_dashboard.py --mock       # no hardware at all

# Mocap dashboard (PhaseSpace optical tracking + Dynamixel)
python3 mocap/dashboard.py             # add --mock for synthetic mocap + servo
python3 mocap/calibrate.py --seconds 5 # CHECK plane / confirm which axis is up

# Analyze a hardware-validation sweep CSV (M12/M32 agreement, errors, repeatability)
python3 high_fidelity/analyze_hw_validation.py   # picks latest dataset by default
```

---

## 🧠 Shared Memory — how to NOT forget across sessions

Memory is split into two layers. Use both.

1. **Live evolving memory → the dual-graph MCP store.** Whenever you make a
   decision, identify a task/next step, or hit a blocker, record it via
   `graph_add_memory(type=..., content="<15 words", tags=[...], files=[...])`.
   This is the real cross-session, cross-tool memory — any agent connected to the
   MCP reads it back. **Do NOT edit `context-store.json` by hand.**

2. **Rolling resume note → `CONTEXT.md`.** At session end (user says "bye/done/
   wrap up"), (re)write `CONTEXT.md` in the project root, under 20 lines:
   - **Current Task** (one sentence)
   - **Key Decisions** (≤3 bullets)
   - **Next Steps** (≤3 bullets)
   This is the fallback any tool can read even without the MCP. Read it FIRST.

> Tool-specific retrieval mechanics (dual-graph `graph_continue`, confidence caps,
> context layering) live in `CLAUDE.md` / `CODEX.md`. Gemini & Copilot can ignore
> those and rely on `config.py`, this file, and `CONTEXT.md`.
