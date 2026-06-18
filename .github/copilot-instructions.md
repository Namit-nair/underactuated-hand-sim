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
- **Hardware rig:** `hardware/` — PySide6 dashboard, Dynamixel servo, RealSense
  + ArUco joint-angle measurement.

When finger geometry or mechanics change, update **both** the high-fidelity model
**and** the analytical validation so they stay consistent.

---

## 📌 Key Technical Decisions (durable — read before touching the model)

- **Spool radius is unified:** `config.SPOOL_RADIUS = 0.011175 m` (measured Ø22.35 mm)
  is the single source for the servo ΔL↔revolutions mapping *and* the load-test force
  ceiling. The old guessed 10 mm / 12.5 mm values are gone — never reintroduce them.
- **Angle-dependent tendon moment arm:** the arm is NOT constant. It grows linearly
  from **7 mm @ 0°** to **12 mm @ 90°** (CAD-measured). `analytical_model.py` solves the
  resulting implicit equilibrium via a **Picard fixed-point** iteration. Toggle with
  `config.MOMENT_ARM_ANGLE_DEPENDENT`; the constant-arm path is the documented fallback.
- **HW-validation analysis tool** (`high_fidelity/analyze_hw_validation.py`): picks a
  dataset by substring / path / latest; cleans data (drops ΔL=0, marker+PIP guards,
  per-ΔL MAD outlier rejection); reports M12/M32 agreement, angle error, repeatability,
  a per-ΔL table, and optional angle-dependent-arm recompute.
- **ArUco angle capture is averaged:** never trust a single frame — zeroing and capture
  average `N_AVG_SAMPLES` detections with a wrap-safe circular mean per marker.
- **RE-ZERO-theta feature was removed** from the dashboard; auto-sweep advances directly.
  Don't re-add it.

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
