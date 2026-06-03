#!/home/namit/iitgn/mujoco_env/bin/python3
"""
Stiffness-Ratio Trend Validation for 3R Underactuated Tendon-Driven Finger
===========================================================================
Standalone non-interactive script. Runs four sequential validation stages:

  1. Numerical moment arm extraction from MuJoCo spatial routing geometry.
  2. Ratio-based trend plots: θ1/θ2 vs ρ1=k1/k2 and θ3/θ2 vs ρ3=k3/k2.
  3. Closure morphology comparison: planar stick figures for three stiffness
     profiles, analytical vs MuJoCo side by side.
  4. Spearman rank correlation over a 7×7 (ρ1, ρ3) grid with heatmap overlay.

All MuJoCo sweeps run with gravity disabled and 2-second equilibration.
Figures are saved as PNGs in the same directory as this script.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from low_fidelity.finger_delta_l_control import (
    xml_content,
    MCP_STIFFNESS, PIP_STIFFNESS, DIP_STIFFNESS,
    L_PROX, L_MID, L_DIST,
    TENDON_STIFFNESS,
)

import mujoco

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------
DELTA_L = 0.020          # 20 mm tendon displacement
K2_BASE = 2.0            # base stiffness for the PIP joint throughout
EQUIL_TIME = 2.0         # seconds to settle to equilibrium
LINK_LENGTHS = np.array([L_PROX, L_MID, L_DIST])

# ============================================================================
# 1. MOMENT ARM EXTRACTION
# ============================================================================

def extract_moment_arms():
    """Numerically extract effective moment arms ri = ΔL/Δθi at the straight
    posture by perturbing each joint by dθ = 0.001 rad."""
    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    L0 = data.ten_length[0]

    joint_names = ["mcp", "pip", "dip"]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                 for n in joint_names]

    dtheta = 0.001  # rad
    r = np.zeros(3)
    for i, jid in enumerate(joint_ids):
        mujoco.mj_resetData(model, data)
        data.qpos[jid] = dtheta
        mujoco.mj_forward(model, data)
        L1 = data.ten_length[0]
        r[i] = (L0 - L1) / dtheta

    return r

# ============================================================================
# Analytical model
# ============================================================================

def analytical_angles_deg(delta_L, r, k):
    """Quasi-static analytical prediction for joint angles (degrees).

    θ_i = (r_i / k_i) · ΔL / Σ(r_j² / k_j)

    Parameters
    ----------
    delta_L : float  – tendon displacement [m]
    r : array (3,)   – moment arms [m]
    k : array (3,)   – joint stiffnesses [Nm/rad]

    Returns
    -------
    theta_deg : array (3,)
    """
    r = np.asarray(r, dtype=float)
    k = np.asarray(k, dtype=float)
    denom = np.sum(r**2 / k)
    theta_rad = (r / k) * (delta_L / denom)
    return np.degrees(theta_rad)

# ============================================================================
# MuJoCo single-point equilibrium evaluation
# ============================================================================

def mujoco_equilibrium(k_vals, delta_L):
    """Run MuJoCo to equilibrium with the given stiffnesses and displacement.

    Returns joint angles in degrees [MCP, PIP, DIP].
    """
    model = mujoco.MjModel.from_xml_string(xml_content)
    model.opt.gravity[:] = 0.0

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ("mcp", "pip", "dip")]

    for idx, jid in enumerate(jids):
        model.jnt_stiffness[jid] = k_vals[idx]

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    L_resting = data.ten_length[0]

    target_L = L_resting - delta_L
    model.tendon_lengthspring[0] = [target_L, target_L]

    n_steps = int(EQUIL_TIME / model.opt.timestep)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    angles_deg = np.array([np.degrees(data.qpos[jid]) for jid in jids])
    return angles_deg

# ============================================================================
# 2. RATIO-BASED TREND PLOT
# ============================================================================

def ratio_trend_plot(r_ext):
    """Sweep ρ1 = k1/k2 and ρ3 = k3/k2, plot θ1/θ2 and θ3/θ2."""
    n_pts = 15
    rho_vals = np.linspace(0.25, 4.0, n_pts)
    total_pts = 2 * n_pts
    t_start = time.time()

    # --- Sweep ρ1 (k3 = k2) ---
    ana_ratio_12 = np.zeros(n_pts)
    muj_ratio_12 = np.zeros(n_pts)
    print("\n--- Sweep ρ1 = k1/k2 (k3 = k2) ---")
    for i, rho1 in enumerate(rho_vals):
        k1 = rho1 * K2_BASE
        k_vec = np.array([k1, K2_BASE, K2_BASE])

        th_ana = analytical_angles_deg(DELTA_L, r_ext, k_vec)
        ana_ratio_12[i] = th_ana[0] / th_ana[1] if abs(th_ana[1]) > 1e-6 else np.nan

        th_muj = mujoco_equilibrium(k_vec, DELTA_L)
        muj_ratio_12[i] = th_muj[0] / th_muj[1] if abs(th_muj[1]) > 1e-6 else np.nan

        t_now = time.time()
        elapsed = t_now - t_start
        pt_idx = i + 1
        avg_time = elapsed / pt_idx
        eta = avg_time * (total_pts - pt_idx)

        print(f"  [{pt_idx:2d}/{total_pts}] ρ1={rho1:.2f} | "
              f"Ana θ1/θ2={ana_ratio_12[i]:.4f}  MuJ θ1/θ2={muj_ratio_12[i]:.4f} | "
              f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    # --- Sweep ρ3 (k1 = k2) ---
    ana_ratio_32 = np.zeros(n_pts)
    muj_ratio_32 = np.zeros(n_pts)
    print("\n--- Sweep ρ3 = k3/k2 (k1 = k2) ---")
    for i, rho3 in enumerate(rho_vals):
        k3 = rho3 * K2_BASE
        k_vec = np.array([K2_BASE, K2_BASE, k3])

        th_ana = analytical_angles_deg(DELTA_L, r_ext, k_vec)
        ana_ratio_32[i] = th_ana[2] / th_ana[1] if abs(th_ana[1]) > 1e-6 else np.nan

        th_muj = mujoco_equilibrium(k_vec, DELTA_L)
        muj_ratio_32[i] = th_muj[2] / th_muj[1] if abs(th_muj[1]) > 1e-6 else np.nan

        t_now = time.time()
        elapsed = t_now - t_start
        pt_idx = n_pts + i + 1
        avg_time = elapsed / pt_idx
        eta = avg_time * (total_pts - pt_idx)

        print(f"  [{pt_idx:2d}/{total_pts}] ρ3={rho3:.2f} | "
              f"Ana θ3/θ2={ana_ratio_32[i]:.4f}  MuJ θ3/θ2={muj_ratio_32[i]:.4f} | "
              f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(rho_vals, ana_ratio_12, 'b-o', markersize=4, label='Analytical')
    ax1.plot(rho_vals, muj_ratio_12, 'r--s', markersize=4, label='MuJoCo')
    ax1.set_xlabel('ρ₁ = k₁/k₂')
    ax1.set_ylabel('θ₁ / θ₂')
    ax1.set_title('θ₁/θ₂  vs  ρ₁ = k₁/k₂   (k₃ = k₂)')
    ax1.legend()
    ax1.grid(True, ls=':', alpha=0.5)

    ax2.plot(rho_vals, ana_ratio_32, 'b-o', markersize=4, label='Analytical')
    ax2.plot(rho_vals, muj_ratio_32, 'r--s', markersize=4, label='MuJoCo')
    ax2.set_xlabel('ρ₃ = k₃/k₂')
    ax2.set_ylabel('θ₃ / θ₂')
    ax2.set_title('θ₃/θ₂  vs  ρ₃ = k₃/k₂   (k₁ = k₂)')
    ax2.legend()
    ax2.grid(True, ls=':', alpha=0.5)

    fig.suptitle(f'Ratio-Based Trend Validation   (ΔL = {DELTA_L*1000:.0f} mm,  k₂ = {K2_BASE} Nm/rad)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'validation_trend.png')
    fig.savefig(out, dpi=150)
    print(f"\n[SAVED] {out}")
    return fig

# ============================================================================
# 3. CLOSURE MORPHOLOGY STICK FIGURES
# ============================================================================

def draw_stick_figure(ax, angles_deg, title, color='k'):
    """Draw a planar 3-link stick figure from joint angles (degrees).

    Convention: finger starts vertical at (0,0) and curls in the X-Z plane.
    Each joint flexes positively (curling the finger down/forward).
    """
    angles_rad = np.radians(angles_deg)
    pts = np.zeros((4, 2))  # [base, MCP, PIP, DIP tip]
    cumulative_angle = 0.0
    for j in range(3):
        cumulative_angle += angles_rad[j]
        # finger starts pointing up (+Z), flexion rotates toward +X
        dx = LINK_LENGTHS[j] * np.sin(cumulative_angle)
        dz = LINK_LENGTHS[j] * np.cos(cumulative_angle)
        pts[j+1] = pts[j] + np.array([dx, dz])

    ax.plot(pts[:, 0], pts[:, 1], '-o', color=color, linewidth=2.5,
            markersize=7, markerfacecolor='white', markeredgecolor=color,
            markeredgewidth=2, solid_capstyle='round')

    # mark fingertip
    ax.plot(pts[-1, 0], pts[-1, 1], 's', color=color, markersize=6)

    # equal aspect, light grid
    ax.set_aspect('equal')
    ax.grid(True, ls=':', alpha=0.3)
    ax.set_title(title, fontsize=9)
    lim = sum(LINK_LENGTHS) * 1.15
    ax.set_xlim(-lim * 0.3, lim)
    ax.set_ylim(-lim * 0.3, lim)


def morphology_comparison(r_ext):
    """3 stiffness profiles × 2 models = 6 stick figures in a 2×3 grid."""
    profiles = [
        ("Proximal-heavy\nρ₁=3.0, ρ₃=1.0", 3.0, 1.0),
        ("Uniform\nρ₁=1.0, ρ₃=1.0",         1.0, 1.0),
        ("Distal-heavy\nρ₁=1.0, ρ₃=3.0",     1.0, 3.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    t_start = time.time()

    for col, (label, rho1, rho3) in enumerate(profiles):
        clean_label = label.replace('\n', ', ')
        print(f"  Profile {col+1}/3: Evaluating {clean_label}...")
        k_vec = np.array([rho1 * K2_BASE, K2_BASE, rho3 * K2_BASE])

        th_ana = analytical_angles_deg(DELTA_L, r_ext, k_vec)
        th_muj = mujoco_equilibrium(k_vec, DELTA_L)

        draw_stick_figure(axes[0, col], th_ana,
                          f"Analytical\n{label}\nθ=[{th_ana[0]:.1f}, {th_ana[1]:.1f}, {th_ana[2]:.1f}]°",
                          color='#2060C0')
        draw_stick_figure(axes[1, col], th_muj,
                          f"MuJoCo\n{label}\nθ=[{th_muj[0]:.1f}, {th_muj[1]:.1f}, {th_muj[2]:.1f}]°",
                          color='#C03020')

        print(f"    Analytical → θ = [{th_ana[0]:.2f}, {th_ana[1]:.2f}, {th_ana[2]:.2f}]°")
        print(f"    MuJoCo     → θ = [{th_muj[0]:.2f}, {th_muj[1]:.2f}, {th_muj[2]:.2f}]°")

    axes[0, 0].set_ylabel('Analytical', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('MuJoCo', fontsize=12, fontweight='bold')

    fig.suptitle(f'Closure Morphology Comparison   (ΔL = {DELTA_L*1000:.0f} mm,  k₂ = {K2_BASE} Nm/rad)',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'validation_morphology.png')
    fig.savefig(out, dpi=150)
    print(f"\n[SAVED] {out}")
    return fig

# ============================================================================
# 4. SPEARMAN RANK CORRELATION + HEATMAP
# ============================================================================

def spearman_grid(r_ext):
    """7×7 grid sweep of (ρ1, ρ3), compute Spearman rank correlation."""
    n_grid = 7
    rho1_vals = np.geomspace(0.25, 4.0, n_grid)
    rho3_vals = np.geomspace(0.25, 4.0, n_grid)

    n_total = n_grid * n_grid
    ana_r12 = np.zeros(n_total)
    ana_r32 = np.zeros(n_total)
    muj_r12 = np.zeros(n_total)
    muj_r32 = np.zeros(n_total)

    # Also store MuJoCo θ1/θ2 on a 2D grid for the heatmap
    muj_grid_12 = np.zeros((n_grid, n_grid))
    ana_grid_12 = np.zeros((n_grid, n_grid))

    print(f"\n--- 7×7 Grid Sweep ({n_total} points) ---")
    idx = 0
    t_start = time.time()
    for i, rho1 in enumerate(rho1_vals):
        for j, rho3 in enumerate(rho3_vals):
            k_vec = np.array([rho1 * K2_BASE, K2_BASE, rho3 * K2_BASE])

            th_ana = analytical_angles_deg(DELTA_L, r_ext, k_vec)
            th_muj = mujoco_equilibrium(k_vec, DELTA_L)

            a12 = th_ana[0] / th_ana[1] if abs(th_ana[1]) > 1e-6 else np.nan
            a32 = th_ana[2] / th_ana[1] if abs(th_ana[1]) > 1e-6 else np.nan
            m12 = th_muj[0] / th_muj[1] if abs(th_muj[1]) > 1e-6 else np.nan
            m32 = th_muj[2] / th_muj[1] if abs(th_muj[1]) > 1e-6 else np.nan

            ana_r12[idx] = a12
            ana_r32[idx] = a32
            muj_r12[idx] = m12
            muj_r32[idx] = m32

            muj_grid_12[i, j] = m12
            ana_grid_12[i, j] = a12

            elapsed = time.time() - t_start
            pt_idx = idx + 1
            avg_time = elapsed / pt_idx
            eta = avg_time * (n_total - pt_idx)

            print(f"  [{pt_idx:2d}/{n_total}] ρ1={rho1:.3f} ρ3={rho3:.3f} | "
                  f"Ana θ1/θ2={a12:.3f}  MuJ θ1/θ2={m12:.3f} | "
                  f"Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
            idx += 1

    # --- Spearman ---
    mask = np.isfinite(ana_r12) & np.isfinite(muj_r12)
    sp_12 = stats.spearmanr(ana_r12[mask], muj_r12[mask])
    mask = np.isfinite(ana_r32) & np.isfinite(muj_r32)
    sp_32 = stats.spearmanr(ana_r32[mask], muj_r32[mask])

    print("\n" + "=" * 55)
    print("=== Spearman Rank Correlation (Trend Preservation) ===")
    print(f"  θ1/θ2 correlation:  {sp_12.statistic:.3f}  (p = {sp_12.pvalue:.3e})")
    print(f"  θ3/θ2 correlation:  {sp_32.statistic:.3f}  (p = {sp_32.pvalue:.3e})")
    print("=" * 55)

    # --- Heatmap ---
    fig, ax = plt.subplots(figsize=(8, 7))

    # MuJoCo heatmap
    im = ax.pcolormesh(rho3_vals, rho1_vals, muj_grid_12,
                       shading='nearest', cmap='viridis')
    cbar = fig.colorbar(im, ax=ax, label='MuJoCo θ₁/θ₂')

    # Analytical contour overlay
    R3, R1 = np.meshgrid(rho3_vals, rho1_vals)
    cs = ax.contour(R3, R1, ana_grid_12, levels=8,
                    colors='white', linewidths=1.2, linestyles='--')
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')

    ax.set_xlabel('ρ₃ = k₃/k₂')
    ax.set_ylabel('ρ₁ = k₁/k₂')
    ax.set_title(f'MuJoCo θ₁/θ₂ heatmap  +  Analytical contours (dashed)\n'
                 f'Spearman ρ(θ₁/θ₂) = {sp_12.statistic:.3f},  '
                 f'ρ(θ₃/θ₂) = {sp_32.statistic:.3f}',
                 fontsize=11, fontweight='bold')

    fig.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'validation_heatmap.png')
    fig.savefig(out, dpi=150)
    print(f"\n[SAVED] {out}")

    return fig, sp_12, sp_32

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("  STIFFNESS-RATIO TREND VALIDATION")
    print("  3R Underactuated Tendon-Driven Finger")
    print("=" * 70)

    t_start_global = time.time()

    # ------------------------------------------------------------------
    # Stage 1: Moment arm extraction
    # ------------------------------------------------------------------
    print("\n[STAGE 1] Extracting moment arms from MuJoCo geometry...")
    t_stage1 = time.time()
    r_ext = extract_moment_arms()
    t_stage1_elap = time.time() - t_stage1
    print(f"\n  Extracted moment arms (straight posture, dθ = 0.001 rad) in {t_stage1_elap:.3f}s:")
    print(f"    r1 (MCP) = {r_ext[0]:.6f} m")
    print(f"    r2 (PIP) = {r_ext[1]:.6f} m")
    print(f"    r3 (DIP) = {r_ext[2]:.6f} m")
    print(f"\n  Imported joint stiffnesses:")
    print(f"    k1 (MCP) = {MCP_STIFFNESS:.4f} Nm/rad")
    print(f"    k2 (PIP) = {PIP_STIFFNESS:.4f} Nm/rad")
    print(f"    k3 (DIP) = {DIP_STIFFNESS:.4f} Nm/rad")
    print(f"\n  Fixed parameters:")
    print(f"    ΔL       = {DELTA_L*1000:.1f} mm")
    print(f"    k2 base  = {K2_BASE} Nm/rad")
    print(f"    Tendon K = {TENDON_STIFFNESS} N/m")
    print(f"    Equil. T = {EQUIL_TIME} s")

    # ------------------------------------------------------------------
    # Stage 2: Ratio-based trend plot
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[STAGE 2] Ratio-based trend sweep...")
    t_stage2 = time.time()
    ratio_trend_plot(r_ext)
    t_stage2_elap = time.time() - t_stage2
    print(f"\n[STAGE 2] Finished in {t_stage2_elap:.1f}s")

    # ------------------------------------------------------------------
    # Stage 3: Closure morphology
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[STAGE 3] Closure morphology comparison...")
    t_stage3 = time.time()
    morphology_comparison(r_ext)
    t_stage3_elap = time.time() - t_stage3
    print(f"\n[STAGE 3] Finished in {t_stage3_elap:.1f}s")

    # ------------------------------------------------------------------
    # Stage 4: Spearman rank correlation + heatmap
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("[STAGE 4] Spearman rank correlation grid sweep...")
    t_stage4 = time.time()
    _, sp_12, sp_32 = spearman_grid(r_ext)
    t_stage4_elap = time.time() - t_stage4
    print(f"\n[STAGE 4] Finished in {t_stage4_elap:.1f}s")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    t_global_elap = time.time() - t_start_global
    print("\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    print(f"  Total execution time: {t_global_elap:.1f}s ({t_global_elap/60:.2f} mins)")
    print(f"    Stage 1: {t_stage1_elap:.3f}s")
    print(f"    Stage 2: {t_stage2_elap:.1f}s")
    print(f"    Stage 3: {t_stage3_elap:.1f}s")
    print(f"    Stage 4: {t_stage4_elap:.1f}s")
    print(f"\n  Extracted moment arms:")
    print(f"    r1 = {r_ext[0]:.6f} m")
    print(f"    r2 = {r_ext[1]:.6f} m")
    print(f"    r3 = {r_ext[2]:.6f} m")
    print(f"\n  Spearman rank correlations (analytical vs MuJoCo):")
    print(f"    θ1/θ2:  ρ = {sp_12.statistic:.4f}  (p = {sp_12.pvalue:.3e})")
    print(f"    θ3/θ2:  ρ = {sp_32.statistic:.4f}  (p = {sp_32.pvalue:.3e})")
    print(f"\n  Output files:")
    print(f"    {os.path.join(SCRIPT_DIR, 'validation_trend.png')}")
    print(f"    {os.path.join(SCRIPT_DIR, 'validation_morphology.png')}")
    print(f"    {os.path.join(SCRIPT_DIR, 'validation_heatmap.png')}")
    print("=" * 70)
    print("Done.")

    plt.show()


if __name__ == "__main__":
    main()
