#!/usr/bin/env python3
"""
Zero-Stiffness Chaos / Non-Determinism Study
============================================
Tests whether a ZERO-joint-stiffness underactuated finger is chaotic /
non-deterministic, and whether adding joint stiffness restores deterministic
morphology — the core "why stiffness is needed" argument.

Why the physics predicts it (the claim under test)
--------------------------------------------------
The physics-faithful model (finger_model.py) drives the finger with a *fixed
tendon*  L = Σ coef_i·θ_i  — ONE holonomic constraint on THREE joints. With zero
joint stiffness, gravity off and contacts off, a non-zero tendon pull λ exerts
the SAME torque c·λ on every joint with no restoring spring to balance it; the
only static equilibria are configurations where joints are pinned against their
mechanical limits. WHICH joints pin (and in what order) is decided purely by the
transient, so the settled pose is acutely sensitive to the initial joint angles.
Add joint stiffness k_i and each joint gets a unique interior equilibrium
θ_i = c·λ/k_i — deterministic, smooth, repeatable.

The MuJoCo run itself is deterministic (same IC → same output), so we probe the
IC→outcome MAP two ways:
  * fixed 0.1° grid on one joint at a time  → reproducible sensitivity curves
    (smooth = merely sensitive; jagged/branch-jumps = chaotic-looking),
  * seeded random Uniform(±1°)^3 on all joints → the real-world outcome scatter.

Every case is run at k=0 (chaotic candidate) AND at a uniform non-zero baseline
(control). Output (./chaos_results/):
  runs.csv                  - one row per equilibrium run
  sensitivity_curves.png    - single-joint grid: final angles vs initial angle
  fingertip_scatter.png     - all-joints case: fingertip cloud, k=0 vs baseline
  final_pose_spaghetti.png  - overlaid settled stick figures, all cases
  nondeterminism_index.png  - fingertip scatter + angle spread, k=0 vs baseline

All physical/numerical knobs live in config.py (Section 9, CHAOS_*).
"""
import csv
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")            # headless — write PNGs, never open a window
import matplotlib.pyplot as plt
import mujoco
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config                                              # noqa: E402
import finger_model                                        # noqa: E402
from analytical_model import extract_kinematics_from_model  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "finger.xml")
OUT_DIR = os.path.join(HERE, "chaos_results")
os.makedirs(OUT_DIR, exist_ok=True)

# --- study parameters (single source of truth: config.py, Section 9) ---------
JOINT_NAMES = config.JOINT_NAMES                 # ("mcp", "pip", "dip")
JOINT_RANGES_DEG = config.JOINT_RANGES_DEG
LIMITS_LO = np.array([lo for lo, _ in JOINT_RANGES_DEG], dtype=float)
LIMITS_HI = np.array([hi for _, hi in JOINT_RANGES_DEG], dtype=float)

DELTA_L = config.CHAOS_DELTA_L
IC_LO, IC_HI = config.CHAOS_IC_RANGE_DEG
GRID_STEP = config.CHAOS_IC_GRID_STEP_DEG
RANDOM_N = config.CHAOS_RANDOM_N
RANDOM_SEED = config.CHAOS_RANDOM_SEED
BASELINE_K = config.CHAOS_BASELINE_K
RAMP_TIME = config.CHAOS_RAMP_TIME
HOLD_TIME = config.CHAOS_HOLD_TIME
GRAVITY = config.CHAOS_GRAVITY
VEL_TOL = config.VEL_TOL
SAT_TOL = config.SATURATION_TOL

# --- friction (config Section 10) — capstan + pin; opt-in via FRICTION_ENABLED ---
FRICTION_ON = config.FRICTION_ENABLED
MU_TENDON = config.FRICTION_MU_TENDON
REST_WRAP_RAD = np.radians(config.FRICTION_REST_WRAP_DEG)
PIN_TAU = np.asarray(config.FRICTION_PIN_TORQUE, dtype=float)
ARM = config.SHEATH_MOMENT_ARM                    # constant sheath moment arm [m]

# Two stiffness conditions compared in every case.
K_ZERO = np.zeros(3)
K_BASE = np.full(3, BASELINE_K)
COND_ZERO = "k=0"
COND_BASE = f"k={BASELINE_K:.4f}"

ZERO_COLOR = "#C0392B"   # chaotic candidate
BASE_COLOR = "#1F4E79"   # deterministic baseline
JOINT_COLORS = ("#1B9E77", "#D95F02", "#7570B3")  # mcp, pip, dip

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.labelsize": 10.5, "axes.linewidth": 1.0,
    "legend.fontsize": 8.5, "legend.framealpha": 0.95,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
    "grid.linewidth": 0.5, "figure.dpi": 110, "savefig.dpi": 300,
    "savefig.bbox": "tight", "mathtext.fontset": "cm",
})


# =====================================================================
# Model — built ONCE, reused across every run (only jnt_stiffness/qpos change)
# =====================================================================
def _ensure_xml():
    """Regenerate finger.xml from the interactive viewer if it is missing."""
    if os.path.exists(XML_PATH):
        return
    print("  finger.xml not found — regenerating...")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "interactive_viewer", os.path.join(HERE, "interactive_viewer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._build_xml()


_ensure_xml()
MODEL = finger_model.load_fidelity_model(XML_PATH, disable_gravity=not GRAVITY)
JIDS = [mujoco.mj_name2id(MODEL, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JOINT_NAMES]
DT = MODEL.opt.timestep

# Link lengths (m) — shared single source of truth for fingertip kinematics.
_R_EXT, LINK_LENGTHS = extract_kinematics_from_model(MODEL)

# Rest tendon length at the NOMINAL (zero) pose. Referencing L_rest here keeps the
# ΔL actuation identical across every initial condition (only the start pose
# differs run-to-run), so the scatter is attributable to the IC alone.
_d0 = mujoco.MjData(MODEL)
mujoco.mj_forward(MODEL, _d0)
L_REST = float(_d0.ten_length[0])

# DOF addresses of the three hinges — friction is written to dof_frictionloss.
DOFADR = np.array([MODEL.jnt_dofadr[j] for j in JIDS])


def _apply_friction(data, mu_t, pin_tau):
    """Set dof_frictionloss = pin Coulomb + capstan tendon-sheath loss.

    Capstan: the tendon wraps each joint by  Phi ≈ rest_wrap + |theta|,  losing
    tension along the path by  T(joint) = T_act · exp(-mu · Phi_upstream).  The
    UNtransmitted tension  T_act·(1 - exp(-mu·Phi_upstream))  times the local
    moment arm is the drag torque resisting that joint, so distal joints (more
    upstream wrap) are dragged harder. Folded into Coulomb dof_frictionloss
    (opposes motion either way) → strictly dissipative, no energy injection.
    """
    tlen = float(data.ten_length[0])
    ls = float(MODEL.tendon_lengthspring[0, 0])
    T = float(MODEL.tendon_stiffness[0]) * max(0.0, tlen - ls)     # actuator tension [N]
    th = np.abs(np.array([data.qpos[j] for j in JIDS]))           # wrap ≈ |joint angle|
    wrap = REST_WRAP_RAD + th
    cum_upstream = np.array([0.0, wrap[0], wrap[0] + wrap[1]])     # wrap between actuator & joint
    transmitted = np.exp(-mu_t * cum_upstream)                    # capstan fraction reaching joint
    cap_tau = ARM * T * (1.0 - transmitted)                       # lost tension → opposing torque
    MODEL.dof_frictionloss[DOFADR] = pin_tau + cap_tau


def _chain_points(angles_deg):
    """2-D planar finger chain from joint angles; pts[0]=base, pts[-1]=tip [m]."""
    ang = np.radians(angles_deg)
    pts = np.zeros((4, 2))
    cum = 0.0
    for j in range(3):
        cum += ang[j]
        pts[j + 1] = pts[j] + (LINK_LENGTHS[j] * np.sin(cum),
                               LINK_LENGTHS[j] * np.cos(cum))
    return pts


def _tip_xy_mm(angles_deg):
    return _chain_points(angles_deg)[-1] * 1000.0


# =====================================================================
# Core episode runner — settle the finger from a perturbed initial condition
# =====================================================================
def settle_from_ic(k_vec, delta_L, q0_deg, *, ramp_time=RAMP_TIME,
                   hold_time=HOLD_TIME, record_traj=False,
                   friction=FRICTION_ON, mu_t=MU_TENDON, pin_tau=None):
    """Pull the tendon by delta_L (quasi-static ramp + hold) starting from the
    initial joint angles q0_deg [deg], and return the settled state.

    The tendon target is L_REST - delta_L for every IC (identical actuation), so
    only the start pose differs between runs.
    """
    MODEL.jnt_stiffness[JIDS] = k_vec
    MODEL.dof_frictionloss[DOFADR] = 0.0          # cleared; re-set per-step if friction on
    _pin = PIN_TAU if pin_tau is None else np.asarray(pin_tau, dtype=float)
    data = mujoco.MjData(MODEL)
    for jid, a in zip(JIDS, q0_deg):
        data.qpos[jid] = np.radians(a)
    mujoco.mj_forward(MODEL, data)

    ramp_steps = max(1, int(ramp_time / DT))
    n_total = ramp_steps + int(hold_time / DT)

    POS_WIN = 200
    POS_TOL = np.radians(0.02)       # 0.02° drift over the window = settled
    q_hist = np.zeros((POS_WIN, 3))
    conv_time = None
    traj = [] if record_traj else None

    for step in range(n_total):
        frac = min(1.0, (step + 1) / ramp_steps)
        target = L_REST - delta_L * frac
        MODEL.tendon_lengthspring[0] = [target, target]
        if friction:
            _apply_friction(data, mu_t, _pin)
        mujoco.mj_step(MODEL, data)

        q_now = np.array([data.qpos[j] for j in JIDS])
        q_hist[step % POS_WIN] = q_now
        if record_traj:
            traj.append((step * DT, *np.degrees(q_now)))

        # Only test convergence once the ramp is done and the window is filled.
        if step > ramp_steps + POS_WIN:
            vels = np.array([data.qvel[j] for j in JIDS])
            drift = float(np.max(q_hist.max(axis=0) - q_hist.min(axis=0)))
            if np.linalg.norm(vels) < VEL_TOL or drift < POS_TOL:
                conv_time = step * DT
                break

    angles = np.degrees(np.array([data.qpos[j] for j in JIDS]))
    sat = [n for n, a, lo, hi in zip(JOINT_NAMES, angles, LIMITS_LO, LIMITS_HI)
           if (a >= hi - SAT_TOL) or (a <= lo + SAT_TOL)]
    angles = np.clip(angles, LIMITS_LO, LIMITS_HI)

    tlen = float(data.ten_length[0])
    ls = float(MODEL.tendon_lengthspring[0, 0])
    tension = float(MODEL.tendon_stiffness[0]) * max(0.0, tlen - ls)

    tip = _tip_xy_mm(angles)
    return {"angles": angles, "tip": tip, "tension": tension,
            "conv_time": conv_time, "saturated": sat,
            "q0": np.asarray(q0_deg, dtype=float), "traj": traj}


# =====================================================================
# Sweep drivers
# =====================================================================
def grid_points():
    n = int(round((IC_HI - IC_LO) / GRID_STEP)) + 1
    return np.linspace(IC_LO, IC_HI, n)


def run_single_joint_case(joint_idx, k_vec):
    """Fixed 0.1° grid on one joint (others 0) — returns the settled results."""
    results = []
    for v in grid_points():
        q0 = [0.0, 0.0, 0.0]
        q0[joint_idx] = float(v)
        results.append(settle_from_ic(k_vec, DELTA_L, q0))
    return results


def run_all_joints_case(k_vec, rng):
    """Seeded Monte-Carlo: Uniform(±1°) on all three joints simultaneously."""
    results = []
    for _ in range(RANDOM_N):
        q0 = rng.uniform(IC_LO, IC_HI, size=3)
        results.append(settle_from_ic(k_vec, DELTA_L, q0))
    return results


# =====================================================================
# Metrics
# =====================================================================
def scatter_metrics(results):
    """Outcome spread for a list of run results."""
    tips = np.array([r["tip"] for r in results])              # (N, 2) mm
    angs = np.array([r["angles"] for r in results])           # (N, 3) deg
    centroid = tips.mean(axis=0)
    dists = np.linalg.norm(tips - centroid, axis=1)
    return {
        "tip_rms_mm": float(np.sqrt(np.mean(dists ** 2))),
        "tip_max_mm": float(dists.max()),
        "angle_std_deg": angs.std(axis=0),
        "angle_std_max_deg": float(angs.std(axis=0).max()),
        "n_converged": sum(r["conv_time"] is not None for r in results),
        "n": len(results),
    }


# =====================================================================
# Plotting
# =====================================================================
def plot_sensitivity_curves(single_cases, filename):
    """single_cases: dict label -> {COND: results} for the 3 single-joint cases.
    Rows = stiffness condition, cols = perturbed joint."""
    labels = list(single_cases.keys())
    grid = grid_points()
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True,
                             constrained_layout=True)
    for row, (cond, ctitle) in enumerate(
            [(COND_ZERO, "Zero stiffness  (k = 0)"),
             (COND_BASE, f"Baseline stiffness  ({COND_BASE})")]):
        for col, label in enumerate(labels):
            ax = axes[row, col]
            res = single_cases[label][cond]
            angs = np.array([r["angles"] for r in res])
            for j in range(3):
                ax.plot(grid, angs[:, j], "-o", ms=3, lw=1.6,
                        color=JOINT_COLORS[j],
                        label=fr"$\theta_{{{j+1}}}$ ({JOINT_NAMES[j].upper()})")
            if row == 0:
                ax.set_title(f"perturb {label.upper()}")
            if col == 0:
                ax.set_ylabel(f"{ctitle}\nsettled angle [deg]")
            if row == 1:
                ax.set_xlabel(f"initial {label.upper()} angle [deg]")
            if row == 0 and col == 2:
                ax.legend(loc="best", ncol=1)
    fig.suptitle("Initial-condition sensitivity — settled joint angles vs IC "
                 f"(ΔL = {DELTA_L*1000:.0f} mm)\n"
                 "jagged / branch-jumps at k=0 = chaotic;  smooth baseline = deterministic",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(filename)
    plt.close(fig)


def plot_fingertip_scatter(all_zero, all_base, filename):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), constrained_layout=True)
    data = [(all_zero, ZERO_COLOR, f"Zero stiffness  ({COND_ZERO})"),
            (all_base, BASE_COLOR, f"Baseline  ({COND_BASE})")]
    tips_all = np.vstack([[r["tip"] for r in d[0]] for d in data])
    pad = 5.0
    xlim = (tips_all[:, 0].min() - pad, tips_all[:, 0].max() + pad)
    ylim = (tips_all[:, 1].min() - pad, tips_all[:, 1].max() + pad)
    for ax, (res, color, title) in zip(axes, data):
        tips = np.array([r["tip"] for r in res])
        m = scatter_metrics(res)
        ax.scatter(tips[:, 0], tips[:, 1], s=18, c=color, alpha=0.55,
                   edgecolors="none")
        ax.scatter(*tips.mean(axis=0), s=120, marker="+", c="black", lw=2,
                   zorder=5, label="centroid")
        ax.set_title(f"{title}\nfingertip RMS scatter = {m['tip_rms_mm']:.2f} mm  "
                     f"(max {m['tip_max_mm']:.2f} mm)")
        ax.set_xlabel("fingertip x [mm]")
        ax.set_ylabel("fingertip y [mm]")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.legend(loc="best")
    fig.suptitle(f"All-joints IC noise (Uniform ±1°, N={RANDOM_N}) — settled "
                 "fingertip cloud", fontsize=12.5, fontweight="bold")
    fig.savefig(filename)
    plt.close(fig)


def plot_spaghetti(cases_by_cond, filename):
    """cases_by_cond: dict case_label -> {COND: results}.  Overlaid settled
    stick figures: rows = stiffness, cols = case."""
    labels = list(cases_by_cond.keys())
    fig, axes = plt.subplots(2, len(labels), figsize=(3.2 * len(labels), 7.0),
                             constrained_layout=True)
    conds = [(COND_ZERO, ZERO_COLOR), (COND_BASE, BASE_COLOR)]
    for row, (cond, color) in enumerate(conds):
        for col, label in enumerate(labels):
            ax = axes[row, col]
            res = cases_by_cond[label][cond]
            for r in res:
                pts = _chain_points(r["angles"]) * 1000.0
                ax.plot(pts[:, 0], pts[:, 1], "-", color=color, lw=0.8,
                        alpha=0.25)
            ax.plot(0, 0, "o", color="black", ms=5, zorder=5)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(f"{label.upper()}")
            if col == 0:
                ax.set_ylabel(f"{cond}\ny [mm]")
            if row == 1:
                ax.set_xlabel("x [mm]")
    fig.suptitle("Settled finger poses overlaid (all initial conditions)\n"
                 "fan-out at k=0 = non-deterministic;  collapsed baseline = deterministic",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(filename)
    plt.close(fig)


def plot_nondeterminism_index(summary, filename):
    """summary: dict case_label -> {COND: scatter_metrics}."""
    labels = list(summary.keys())
    x = np.arange(len(labels))
    w = 0.38
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.0),
                                   constrained_layout=True)
    for ax, key, ylab, title in [
        (ax1, "tip_rms_mm", "fingertip RMS scatter [mm]",
         "Fingertip outcome scatter"),
        (ax2, "angle_std_max_deg", "max joint-angle std [deg]",
         "Joint-angle spread")]:
        z = [summary[l][COND_ZERO][key] for l in labels]
        b = [summary[l][COND_BASE][key] for l in labels]
        ax.bar(x - w / 2, z, w, color=ZERO_COLOR, label=COND_ZERO)
        ax.bar(x + w / 2, b, w, color=BASE_COLOR, label=COND_BASE)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([l.upper() for l in labels])
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, which="both", axis="y", alpha=0.3)
    fig.suptitle("Non-determinism index — zero vs baseline stiffness "
                 "(log scale; higher = less predictable)",
                 fontsize=12.5, fontweight="bold")
    fig.savefig(filename)
    plt.close(fig)


# =====================================================================
# CSV
# =====================================================================
def write_csv(rows, filename):
    fields = ["case", "stiffness", "run_idx",
              "q0_mcp_deg", "q0_pip_deg", "q0_dip_deg",
              "theta_mcp_deg", "theta_pip_deg", "theta_dip_deg",
              "tip_x_mm", "tip_y_mm", "tension_N", "conv_time_s", "saturated"]
    with open(filename, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _rows_for(case, cond, results):
    out = []
    for i, r in enumerate(results):
        out.append({
            "case": case, "stiffness": cond, "run_idx": i,
            "q0_mcp_deg": round(r["q0"][0], 4),
            "q0_pip_deg": round(r["q0"][1], 4),
            "q0_dip_deg": round(r["q0"][2], 4),
            "theta_mcp_deg": round(r["angles"][0], 4),
            "theta_pip_deg": round(r["angles"][1], 4),
            "theta_dip_deg": round(r["angles"][2], 4),
            "tip_x_mm": round(r["tip"][0], 4),
            "tip_y_mm": round(r["tip"][1], 4),
            "tension_N": round(r["tension"], 4),
            "conv_time_s": r["conv_time"] if r["conv_time"] is not None else "",
            "saturated": ",".join(r["saturated"]),
        })
    return out


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 76)
    print("  ZERO-STIFFNESS CHAOS / NON-DETERMINISM STUDY")
    print("  Does an underactuated finger need joint stiffness to be deterministic?")
    print("=" * 76)
    print(f"  ΔL = {DELTA_L*1000:.0f} mm   gravity = {'on' if GRAVITY else 'off'}   "
          f"baseline k = {BASELINE_K:.4f} N·m/rad")
    print(f"  grid: {IC_LO:+.1f}..{IC_HI:+.1f}° step {GRID_STEP}° "
          f"({len(grid_points())} pts)   random N = {RANDOM_N} (seed {RANDOM_SEED})")
    print(f"  link lengths [mm]: "
          + ", ".join(f"{L*1000:.1f}" for L in LINK_LENGTHS))

    t0 = time.time()
    rng = np.random.default_rng(RANDOM_SEED)
    all_rows = []
    single_cases = {}                # label -> {COND: results}
    cases_by_cond = {}               # label -> {COND: results}  (incl. all-joints)
    summary = {}                     # label -> {COND: scatter_metrics}

    # ---- single-joint grid cases (MCP / PIP / DIP) ----
    for joint_idx, label in enumerate(JOINT_NAMES):
        single_cases[label] = {}
        cases_by_cond[label] = {}
        summary[label] = {}
        for cond, k in [(COND_ZERO, K_ZERO), (COND_BASE, K_BASE)]:
            res = run_single_joint_case(joint_idx, k)
            single_cases[label][cond] = res
            cases_by_cond[label][cond] = res
            summary[label][cond] = scatter_metrics(res)
            all_rows += _rows_for(label, cond, res)
        m0 = summary[label][COND_ZERO]
        mb = summary[label][COND_BASE]
        print(f"\n  [{label.upper()} grid]  "
              f"tip RMS: k=0 {m0['tip_rms_mm']:6.2f} mm  vs  "
              f"base {mb['tip_rms_mm']:6.3f} mm   "
              f"(ratio {m0['tip_rms_mm']/max(mb['tip_rms_mm'],1e-6):7.1f}×)")

    # ---- all-joints Monte-Carlo case ----
    label = "all"
    cases_by_cond[label] = {}
    summary[label] = {}
    all_results = {}
    for cond, k in [(COND_ZERO, K_ZERO), (COND_BASE, K_BASE)]:
        res = run_all_joints_case(k, np.random.default_rng(RANDOM_SEED))
        all_results[cond] = res
        cases_by_cond[label][cond] = res
        summary[label][cond] = scatter_metrics(res)
        all_rows += _rows_for(label, cond, res)
    m0 = summary[label][COND_ZERO]
    mb = summary[label][COND_BASE]
    print(f"\n  [ALL-joints random]  "
          f"tip RMS: k=0 {m0['tip_rms_mm']:6.2f} mm  vs  "
          f"base {mb['tip_rms_mm']:6.3f} mm   "
          f"(ratio {m0['tip_rms_mm']/max(mb['tip_rms_mm'],1e-6):7.1f}×)")

    # ---- write CSV + figures ----
    csv_path = os.path.join(OUT_DIR, "runs.csv")
    write_csv(all_rows, csv_path)
    print(f"\n  [SAVED] {csv_path}  ({len(all_rows)} rows)")

    plot_sensitivity_curves(single_cases,
                            os.path.join(OUT_DIR, "sensitivity_curves.png"))
    plot_fingertip_scatter(all_results[COND_ZERO], all_results[COND_BASE],
                           os.path.join(OUT_DIR, "fingertip_scatter.png"))
    plot_spaghetti(cases_by_cond,
                   os.path.join(OUT_DIR, "final_pose_spaghetti.png"))
    plot_nondeterminism_index(summary,
                              os.path.join(OUT_DIR, "nondeterminism_index.png"))
    for name in ("sensitivity_curves", "fingertip_scatter",
                 "final_pose_spaghetti", "nondeterminism_index"):
        print(f"  [SAVED] {name}.png")

    # ---- verdict ----
    n_conv = sum(r["conv_time"] is not None
                 for rows in cases_by_cond.values()
                 for res in rows.values() for r in res)
    elapsed = time.time() - t0
    print("\n" + "=" * 76)
    print(f"  SUMMARY   (total {elapsed:.1f} s,  {len(all_rows)} runs,  "
          f"{n_conv} converged)")
    print("=" * 76)
    print(f"  {'case':<6} {'tipRMS k=0':>12} {'tipRMS base':>12} {'ratio':>9}  "
          f"{'angStd k=0':>11} {'angStd base':>12}")
    ratios = []
    for label in summary:
        m0 = summary[label][COND_ZERO]
        mb = summary[label][COND_BASE]
        ratio = m0["tip_rms_mm"] / max(mb["tip_rms_mm"], 1e-6)
        ratios.append(ratio)
        print(f"  {label.upper():<6} {m0['tip_rms_mm']:>10.2f}mm "
              f"{mb['tip_rms_mm']:>10.3f}mm {ratio:>8.1f}×  "
              f"{m0['angle_std_max_deg']:>9.2f}° {mb['angle_std_max_deg']:>10.3f}°")
    med = float(np.median(ratios))
    # Absolute scatter matters as much as the ratio: a flat (neutral) equilibrium
    # can be far MORE IC-sensitive than baseline yet still settle within a fraction
    # of a mm under a clean, symmetric, quasi-static pull. "Chaotic" requires the
    # absolute scatter to be large relative to the finger, not just the ratio.
    finger_len_mm = float(np.sum(LINK_LENGTHS) * 1000.0)
    max_abs = max(summary[l][COND_ZERO]["tip_rms_mm"] for l in summary)
    abs_frac = max_abs / finger_len_mm
    print("-" * 76)
    print(f"  zero-stiffness IC sensitivity: {med:.0f}× baseline (median ratio)")
    print(f"  zero-stiffness absolute scatter: {max_abs:.2f} mm = "
          f"{abs_frac*100:.2f}% of the {finger_len_mm:.0f} mm finger (max over cases)")
    if abs_frac >= 0.05 and med >= 5.0:
        verdict = (f"CHAOTIC: large absolute scatter ({abs_frac*100:.1f}% of finger) "
                   f"and {med:.0f}× baseline → non-deterministic without stiffness.")
    elif med >= 3.0:
        verdict = (
            "NEUTRAL-but-TAME: the zero-stiffness equilibrium is a flat manifold "
            f"({med:.0f}× more IC-sensitive than baseline), but a clean symmetric "
            f"quasi-static pull still lands within {abs_frac*100:.2f}% of the finger. "
            "This config does NOT reproduce dramatic hardware chaos — the flat "
            "manifold has no restoring force, so SUSTAINED disturbances (gravity, "
            "friction, fast/jerky pulls) excluded here are what make hardware wander. "
            "Next: enable gravity / use a fast or stepped pull / add per-step noise / "
            "sweep ΔL (sensitivity should peak mid-closure) / measure trajectory divergence.")
    else:
        verdict = (f"NOT CONFIRMED: scatter ratio {med:.2f}× — zero stiffness is "
                   "about as repeatable as baseline. Back to the drawing board.")
    print("  " + verdict)
    print(f"  Output dir: {OUT_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()
