#!/usr/bin/env python3
"""
High-Fidelity Stiffness-Ratio Validation
=========================================
Validates the closed-form morphology law from RAL 2026 against the
CAD-accurate MuJoCo high-fidelity finger model.

Convention (matches paper and analytical_model.py):
  k1 = MCP (proximal), k2 = PIP (middle, reference), k3 = DIP (distal)
  rho1 = k1/k2, rho3 = k3/k2

The 7-value rho grid is anchored on the three physical spring ratios
measured on the real hardware (Spring 3 / Spring 2 / Spring 1 → 0.242,
1.000, 5.479) with four log-spaced interpolated points between them.

Outputs (all in ./validation_results/):
  trend.png            - M12 vs rho1 and M32 vs rho3 line plots
  morphology.png       - Stick figures, three named regimes, ana vs sim
  scatter_M12.png      - Global trend scatter for M12 (sim vs ana)
  scatter_M32.png      - Global trend scatter for M32 (sim vs ana)
  heatmap_<metric>.png - Per-cell annotated heatmaps for each quantity
  summary.csv          - All 49 grid cells, raw numbers
  metric_table.csv     - The three named regimes with errors
"""
import csv
import os
import sys
import time

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mujoco
import numpy as np
from matplotlib.colors import LogNorm
from scipy import stats

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import config  # noqa: E402  — single source of truth
import finger_model  # noqa: E402
from analytical_model import (  # noqa: E402
    analytical_angles_deg,
    extract_kinematics_from_model,
    morphology_metrics,
    tendon_tension,
)

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "finger.xml")
OUT_DIR = os.path.join(HERE, "validation_results")
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# Hardware springs + validation parameters — ALL sourced from config.py
# (single source of truth). Change ΔL, springs, or tolerances there.
# =====================================================================
SPRING_1 = config.SPRING_1  # large
SPRING_2 = config.SPRING_2  # medium - used as k2 reference
SPRING_3 = config.SPRING_3  # small

K2_BASE = SPRING_2
RHO_LOW = SPRING_3 / K2_BASE   # ≈ 0.2416
RHO_MID = SPRING_2 / K2_BASE   # = 1.0000
RHO_HIGH = SPRING_1 / K2_BASE  # ≈ 5.4789

# Physics-fidelity model parameters live in finger_model (itself sourced from
# config). Override per-call via finger_model.load_fidelity_model(moment_arm=…).
SHEATH_MOMENT_ARM = finger_model.SHEATH_MOMENT_ARM

DELTA_L = config.DELTA_L                # m — tendon pull (edit in config.py)
EQUIL_MAX_TIME = config.EQUIL_MAX_TIME  # s; convergence cap
VEL_TOL = config.VEL_TOL                # rad/s
SATURATION_TOL = config.SATURATION_TOL  # deg from limit that counts as saturated

LINK_LENGTHS = None  # set at runtime by extract_kinematics_from_model


def pct_err(sim, ana):
    """Absolute percentage error of a morphology metric:  |sim − ana|/ana × 100.

    Returns NaN where the analytical reference is undefined or ~0 (so the
    heatmap shows a blank cell rather than a blow-up)."""
    if not (np.isfinite(sim) and np.isfinite(ana)) or abs(ana) < 1e-9:
        return np.nan
    return abs(sim - ana) / ana * 100.0

ANA_COLOR = '#1F4E79'
SIM_COLOR = '#C0392B'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'axes.labelweight': 'normal',
    'axes.titleweight': 'bold',
    'axes.linewidth': 1.0,
    'legend.fontsize': 9.5,
    'legend.framealpha': 0.95,
    'legend.edgecolor': '0.5',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'mathtext.fontset': 'cm',
})


# =====================================================================
# Model setup helpers
# =====================================================================
def ensure_xml():
    if not os.path.exists(XML_PATH):
        print(f"  finger.xml not found — regenerating...")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "interactive_viewer", os.path.join(HERE, "interactive_viewer.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod._build_xml()


def _fidelity_model():
    """Build (once) and load the physics-faithful finger model from the
    auto-generated finger.xml, via the shared finger_model module."""
    ensure_xml()
    return finger_model.load_fidelity_model(XML_PATH)


def _load_model(k_vec):
    model = _fidelity_model()
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in ("mcp", "pip", "dip")]
    for idx, jid in enumerate(jids):
        model.jnt_stiffness[jid] = k_vec[idx]
    data = mujoco.MjData(model)
    return model, data, jids


def extract_geometry():
    """Pull moment arms AND link lengths from the faithful MuJoCo model so
    the analytical and simulation halves share one source of truth.
    Delegates to `analytical_model.extract_kinematics_from_model`.
    """
    return extract_kinematics_from_model(_fidelity_model())


def mujoco_equilibrium(k_vec, delta_L):
    model, data, jids = _load_model(k_vec)
    mujoco.mj_forward(model, data)
    L_rest = data.ten_length[0]
    model.tendon_lengthspring[0] = [L_rest - delta_L, L_rest - delta_L]
    dt = model.opt.timestep
    n_max = int(EQUIL_MAX_TIME / dt)
    conv_time = None
    # Convergence is declared when EITHER the joint velocities fall below VEL_TOL
    # (free joints settling) OR the joint positions stop moving (a joint pinned
    # against a near-rigid limit micro-oscillates, so |qvel| plateaus above
    # VEL_TOL even though the angle is stationary). The position test compares the
    # angle now against POS_WIN steps ago.
    POS_WIN = 200
    POS_TOL = np.radians(0.02)   # 0.02 deg drift over the window = settled
    q_hist = np.zeros((POS_WIN, 3))
    for step in range(n_max):
        mujoco.mj_step(model, data)
        q_now = np.array([data.qpos[jid] for jid in jids])
        q_hist[step % POS_WIN] = q_now
        if step > 200:
            vels = np.array([data.qvel[jid] for jid in jids])
            pos_drift = np.max(np.abs(q_hist.max(axis=0) - q_hist.min(axis=0)))
            if np.linalg.norm(vels) < VEL_TOL or pos_drift < POS_TOL:
                conv_time = step * dt
                break
    angles = np.array([np.degrees(data.qpos[jid]) for jid in jids])
    lo = np.degrees(np.array([model.jnt_range[jid, 0] for jid in jids]))
    hi = np.degrees(np.array([model.jnt_range[jid, 1] for jid in jids]))
    # Saturation is detected on the raw (pre-clamp) angle so we still flag a
    # joint that reached its stop.
    sat = [n for n, a, l, h in zip(("mcp", "pip", "dip"), angles, lo, hi)
           if (a >= h - SATURATION_TOL) or (a <= l + SATURATION_TOL)]
    # Hard backstop: the near-rigid limit constraint already holds the joint at
    # its mechanical stop; clamp here so the reported angle is GUARANTEED to never
    # exceed the travel range, regardless of stiffness or residual numerical drift.
    angles = np.clip(angles, lo, hi)
    return angles, {'conv_time': conv_time, 'saturated': sat}


def build_rho_grid():
    n = config.SWEEP_N_PER_INTERVAL
    log_low, log_mid, log_high = np.log10([RHO_LOW, RHO_MID, RHO_HIGH])
    lower = np.linspace(log_low, log_mid, n + 2)
    upper = np.linspace(log_mid, log_high, n + 2)[1:]
    return 10 ** np.concatenate([lower, upper])


def rho_tick_label(rho):
    """Format a rho value, marking spring anchors with (S1/S2/S3)."""
    if abs(rho - RHO_LOW) < 1e-3:
        return f"{rho:.2f}\n(S3)"
    if abs(rho - RHO_MID) < 1e-3:
        return f"{rho:.2f}\n(S2)"
    if abs(rho - RHO_HIGH) < 1e-3:
        return f"{rho:.2f}\n(S1)"
    return f"{rho:.2f}"


# =====================================================================
# Generic annotated-heatmap plotter
# =====================================================================
def plot_heatmap(grid, rho_grid, title, cbar_label, filename,
                 cmap='magma', fmt='{:.2f}', sat_mask=None,
                 center=None, vmin=None, vmax=None, log_norm=False,
                 dark_text_at_high=False):
    """ρ1 on y (increasing up = MCP softer→stiffer),
       ρ3 on x (increasing right = DIP softer→stiffer).
       Cells annotated; saturated cells marked with a small triangle in corner.

       log_norm=True applies logarithmic color scaling (useful for error maps
       spanning multiple orders of magnitude).
       dark_text_at_high=True inverts the text-color choice for colormaps
       where the high end is dark (e.g. YlOrRd, Reds, Blues)."""
    n = len(rho_grid)
    fig, ax = plt.subplots(figsize=(7.4, 6.4), constrained_layout=True)

    finite = grid[np.isfinite(grid)]
    if vmin is None: vmin = float(np.min(finite)) if finite.size else 0.0
    if vmax is None: vmax = float(np.max(finite)) if finite.size else 1.0

    if log_norm:
        positive = finite[finite > 0]
        lo = max(float(np.min(positive)), vmax * 1e-4) if positive.size else 1e-4
        norm = LogNorm(vmin=lo, vmax=vmax)
        im = ax.imshow(grid, origin='lower', cmap=cmap, aspect='equal',
                       norm=norm)
    else:
        if center is not None:
            rng = max(abs(vmin - center), abs(vmax - center))
            vmin, vmax = center - rng, center + rng
        im = ax.imshow(grid, origin='lower', cmap=cmap, aspect='equal',
                       vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, label=cbar_label,
                        fraction=0.046, pad=0.04)
    cbar.outline.set_linewidth(0.5)

    for i in range(n):
        for j in range(n):
            val = grid[i, j]
            if not np.isfinite(val):
                txt = '—'
                color = 'gray'
            else:
                txt = fmt.format(val)
                if log_norm:
                    norm_val = im.norm(val) if val > 0 else 0.0
                else:
                    norm_val = (val - vmin) / (vmax - vmin + 1e-12)
                if center is not None:
                    # Diverging map: light bg near the center, dark at the ends.
                    color = 'white' if abs(norm_val - 0.5) > 0.32 else 'black'
                elif dark_text_at_high:
                    color = 'white' if norm_val > 0.55 else '#2A0E0E'
                else:
                    color = 'white' if norm_val < 0.55 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    color=color, fontsize=9.5, fontweight='bold')

    if sat_mask is not None:
        for i in range(n):
            for j in range(n):
                if sat_mask[i, j]:
                    ax.add_patch(plt.Polygon(
                        [[j+0.32, i+0.5], [j+0.5, i+0.32], [j+0.5, i+0.5]],
                        facecolor='red', edgecolor='white', linewidth=0.5,
                        zorder=3))

    labels = [rho_tick_label(v) for v in rho_grid]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel(r'$\rho_3 = k_3/k_2$  (DIP / PIP)')
    ax.set_ylabel(r'$\rho_1 = k_1/k_2$  (MCP / PIP)')
    ax.set_title(title)
    ax.grid(False)
    ax.tick_params(length=0)

    if sat_mask is not None and sat_mask.any():
        ax.text(0.02, -0.10, '▲ red corner = joint saturation in MuJoCo',
                transform=ax.transAxes, fontsize=8.5, color='dimgray',
                va='top', ha='left')

    fig.savefig(filename)
    plt.close(fig)


# =====================================================================
# Trend plot — M12 vs rho1 and M32 vs rho3, linear axes
# =====================================================================
def plot_trend(rho_grid, M12_ana, M12_sim, M32_ana, M32_sim,
               sat_12, sat_32, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6),
                                    constrained_layout=True)

    for ax, x, ya, ys, sat_list, xlab, ylab, title in [
        (ax1, rho_grid, M12_ana, M12_sim, sat_12,
         r'$\rho_1 = k_1/k_2$',
         r'$M_{12} = \theta_1 / \theta_2$',
         r'$M_{12}$  vs  $\rho_1$    (with $\rho_3 = 1$)'),
        (ax2, rho_grid, M32_ana, M32_sim, sat_32,
         r'$\rho_3 = k_3/k_2$',
         r'$M_{32} = \theta_3 / \theta_2$',
         r'$M_{32}$  vs  $\rho_3$    (with $\rho_1 = 1$)'),
    ]:
        ax.plot(x, ya, '-o', color=ANA_COLOR, lw=2.0, ms=7,
                mfc='white', mew=2.0, label='Analytical (Eq. 5)', zorder=3)
        ax.plot(x, ys, '--s', color=SIM_COLOR, lw=1.8, ms=6,
                mfc=SIM_COLOR, mew=0, label='MuJoCo (high-fidelity)', zorder=4)

        for xi, sat in zip(x, sat_list):
            if sat:
                ax.axvspan(xi * 0.95, xi * 1.05, color='red', alpha=0.06,
                           zorder=0)

        for anchor, label in [(RHO_LOW, 'S3'), (RHO_MID, 'S2'), (RHO_HIGH, 'S1')]:
            ax.axvline(anchor, color='black', lw=0.5, alpha=0.25, zorder=1)
            ax.text(anchor, 1.02, label, transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=9, color='dimgray',
                    fontweight='bold')

        ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(loc='upper right' if 'M_{12}' in ylab else 'upper right')
        ax.set_xlim(0, max(rho_grid) * 1.05)
        ax.set_ylim(bottom=0)

    fig.suptitle(f'High-Fidelity Trend Validation   '
                 f'(ΔL = {DELTA_L*1000:.0f} mm,   $k_2$ = {K2_BASE:.4f} N·m/rad)',
                 fontsize=12.5, fontweight='bold')

    fig.savefig(filename)
    plt.close(fig)


# =====================================================================
# Global scatter plot — sim vs analytical with stats
# =====================================================================
def plot_scatter(ana_grid, sim_grid, metric_label, filename):
    fa = ana_grid.flatten()
    fs = sim_grid.flatten()
    mask = np.isfinite(fa) & np.isfinite(fs)
    fa, fs = fa[mask], fs[mask]

    sp = stats.spearmanr(fa, fs)
    pr, _ = stats.pearsonr(fa, fs)
    slope, intercept, r_value, _, _ = stats.linregress(fa, fs)

    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    ax.scatter(fa, fs, s=55, c=ANA_COLOR, edgecolors='black',
               linewidths=0.6, alpha=0.75, zorder=3,
               label=f'Grid cells (n = {len(fa)})')

    lim_lo = min(fa.min(), fs.min())
    lim_hi = max(fa.max(), fs.max())
    pad = 0.08 * (lim_hi - lim_lo)
    lo, hi = lim_lo - pad, lim_hi + pad

    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.55,
            label='Perfect agreement (y = x)', zorder=2)

    xfit = np.linspace(lo, hi, 100)
    ax.plot(xfit, slope * xfit + intercept, color=SIM_COLOR, lw=2.2,
            label=f'Best fit (y = {slope:.3f}x + {intercept:.3f})', zorder=2)

    stats_text = (f"Spearman $\\rho$ = {sp.statistic:.3f}\n"
                  f"(p = {sp.pvalue:.2e})\n"
                  f"Pearson $r$ = {pr:.3f}\n"
                  f"$R^2$ = {r_value**2:.3f}")
    ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      alpha=0.92, edgecolor='gray', linewidth=0.7))

    ax.set_xlabel(f'Analytical {metric_label}')
    ax.set_ylabel(f'MuJoCo {metric_label}')
    ax.set_title(f'Global Trend Validation — {metric_label}')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)

    fig.savefig(filename)
    plt.close(fig)
    return sp, pr, r_value**2, slope, intercept


# =====================================================================
# Morphology stick-figure comparison
# =====================================================================
JOINT_LIMITS_DEG = dict(zip(config.JOINT_NAMES, config.JOINT_RANGES_DEG))
JOINT_NAMES = tuple(n.upper() for n in config.JOINT_NAMES)


def _chain_points(angles_deg, clip=True):
    if clip:
        angles = np.array([
            np.clip(angles_deg[0], *JOINT_LIMITS_DEG['mcp']),
            np.clip(angles_deg[1], *JOINT_LIMITS_DEG['pip']),
            np.clip(angles_deg[2], *JOINT_LIMITS_DEG['dip']),
        ])
    else:
        angles = np.asarray(angles_deg)
    angles_rad = np.radians(angles)
    pts = np.zeros((4, 2))
    cum_dirs = np.zeros(3)
    cum = 0.0
    for j in range(3):
        cum += angles_rad[j]
        cum_dirs[j] = cum
        pts[j+1] = pts[j] + (LINK_LENGTHS[j] * np.sin(cum),
                             LINK_LENGTHS[j] * np.cos(cum))
    return pts, cum_dirs


def _draw_finger(ax, angles_deg, color, label, linestyle='-', alpha=1.0,
                 lw=2.8, marker_size=9, zorder=3, clip=True):
    pts, _ = _chain_points(angles_deg, clip=clip)
    ax.plot(pts[:, 0], pts[:, 1], linestyle=linestyle, color=color, lw=lw,
            solid_capstyle='round', solid_joinstyle='round',
            alpha=alpha, zorder=zorder, label=label)
    ax.plot(pts[:-1, 0], pts[:-1, 1], 'o', color=color, ms=marker_size,
            mfc='white', mew=1.8, alpha=alpha, zorder=zorder + 1)
    ax.plot(pts[-1, 0], pts[-1, 1], 'D', color=color, ms=marker_size - 1.5,
            alpha=alpha, zorder=zorder + 1)
    return pts


REGIMES = [
    # (label_short, label_full, k1, k2, k3)
    ("proximal", "Proximal-dominant\n$\\rho_1\\!\\approx\\!0.24,\\ \\rho_3\\!=\\!1$",
     SPRING_3, SPRING_2, SPRING_2),
    ("uniform",  "Uniform\n$\\rho_1\\!=\\!1,\\ \\rho_3\\!=\\!1$",
     SPRING_2, SPRING_2, SPRING_2),
    ("distal",   "Distal-dominant\n$\\rho_1\\!=\\!1,\\ \\rho_3\\!\\approx\\!0.24$",
     SPRING_2, SPRING_2, SPRING_3),
]


def plot_morphology(regime_data, filename):
    fig = plt.figure(figsize=(13.5, 8.4), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[3.4, 1.0],
                          hspace=0.36, wspace=0.20,
                          left=0.05, right=0.98, top=0.84, bottom=0.05)

    # Auto-scale the frame to the drawn chains so any ΔL fits (origin and the
    # base block at ~-0.012 are always included).
    all_pts = np.vstack([
        pts for d in regime_data
        for pts in (_chain_points(d['theta_ana'])[0],
                    _chain_points(d['theta_sim'])[0])
    ])
    pad = 0.012
    xlim = (min(all_pts[:, 0].min(), -0.015) - pad, all_pts[:, 0].max() + pad)
    ylim = (min(all_pts[:, 1].min(), -0.005) - pad, all_pts[:, 1].max() + pad)

    handles_legend = None
    for col, d in enumerate(regime_data):
        ax = fig.add_subplot(gs[0, col])
        ax_info = fig.add_subplot(gs[1, col]); ax_info.axis('off')

        th_a = d['theta_ana']; th_s = d['theta_sim']
        sat = d['saturated']

        ax.axhline(0, color='0.88', lw=0.8, zorder=0)
        ax.axvline(0, color='0.88', lw=0.8, zorder=0)
        ax.add_patch(mpatches.Rectangle((-0.012, -0.005), 0.010, 0.010,
                                        facecolor='#444', edgecolor='black',
                                        lw=0.6, zorder=2))
        ax.add_patch(mpatches.Circle((0, 0), 0.0035, facecolor='#222',
                                     edgecolor='black', lw=0.5, zorder=3))

        pts_a = _draw_finger(ax, th_a, ANA_COLOR, 'Analytical (Eq. 5)',
                             linestyle='-', alpha=1.0, lw=3.2,
                             marker_size=10, zorder=4)
        pts_s = _draw_finger(ax, th_s, SIM_COLOR, 'MuJoCo (high-fidelity)',
                             linestyle='--', alpha=0.92, lw=2.4,
                             marker_size=7.5, zorder=5)

        for pt_a, pt_s in zip(pts_a[1:], pts_s[1:]):
            ax.plot([pt_a[0], pt_s[0]], [pt_a[1], pt_s[1]],
                    color='gray', lw=0.7, ls=':', alpha=0.55, zorder=2)

        for jname, p in zip(JOINT_NAMES, pts_a[:3]):
            ax.annotate(jname, xy=p, xytext=(8, 8),
                        textcoords='offset points', fontsize=7.5,
                        color='dimgray', fontweight='bold', alpha=0.75)

        ax.set_title(d['regime_label'], fontsize=11)

        if sat:
            ax.text(0.97, 0.03,
                    f"⚠ joint saturated: {', '.join(sat)}",
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8.5, color='#A0140A',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF4F0',
                              edgecolor='#C0392B', linewidth=0.7))

        if col == 1:
            handles_legend = ax.get_legend_handles_labels()

        ax.set_aspect('equal')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_xlabel('x  [m]', fontsize=9.5)
        ax.set_ylabel('y  [m]', fontsize=9.5)
        ax.tick_params(axis='both', labelsize=8.5)
        ax.grid(True, alpha=0.22, ls='--', lw=0.4)

        e12 = pct_err(d['M12_sim'], d['M12_ana'])
        e32 = pct_err(d['M32_sim'], d['M32_ana'])
        info_text = (
            f"$\\theta_{{\\mathrm{{ana}}}}$ = "
            f"[{th_a[0]:6.1f}°, {th_a[1]:6.1f}°, {th_a[2]:6.1f}°]\n"
            f"$\\theta_{{\\mathrm{{sim}}}}$ = "
            f"[{th_s[0]:6.1f}°, {th_s[1]:6.1f}°, {th_s[2]:6.1f}°]\n"
            f"$M_{{12}}$: ana = {d['M12_ana']:.3f},  "
            f"sim = {d['M12_sim']:.3f}    "
            f"($e_{{12}}$ = {e12:+.1f}%)\n"
            f"$M_{{32}}$: ana = {d['M32_ana']:.3f},  "
            f"sim = {d['M32_sim']:.3f}    "
            f"($e_{{32}}$ = {e32:+.1f}%)\n"
            f"tendon tension  $\\lambda$ = {d['tension_ana']:.2f} N"
        )
        ax_info.text(0.5, 1.0, info_text, transform=ax_info.transAxes,
                     ha='center', va='top', fontsize=9.5, family='monospace',
                     bbox=dict(boxstyle='round,pad=0.45', facecolor='#FAFAF7',
                               edgecolor='0.6', linewidth=0.6))

    if handles_legend is not None:
        fig.legend(*handles_legend, loc='upper center',
                   bbox_to_anchor=(0.5, 0.91), ncol=2,
                   fontsize=10.5, frameon=True,
                   facecolor='white', edgecolor='0.5')

    fig.suptitle('High-Fidelity Closure Morphology — Analytical vs MuJoCo  '
                 f'(ΔL = {DELTA_L*1000:.0f} mm)',
                 fontsize=13, fontweight='bold', y=0.965)

    fig.savefig(filename)
    plt.close(fig)


# =====================================================================
# Stage runners
# =====================================================================
def stage_trend_sweep(r_ext, rho_grid):
    n = len(rho_grid)
    print(f"\n[STAGE 2] Trend sweep — {2*n} equilibrium runs")

    M12_ana = np.zeros(n); M12_sim = np.zeros(n)
    M32_ana = np.zeros(n); M32_sim = np.zeros(n)
    sat_12 = []; sat_32 = []

    print("  ρ1 sweep (ρ3 = 1)")
    for i, rho1 in enumerate(rho_grid):
        k = np.array([rho1 * K2_BASE, K2_BASE, K2_BASE])
        th_a = analytical_angles_deg(DELTA_L, r_ext, k)
        th_s, info = mujoco_equilibrium(k, DELTA_L)
        M12_ana[i], _ = morphology_metrics(th_a)
        M12_sim[i], _ = morphology_metrics(th_s)
        sat_12.append(info['saturated'])
        s = 'SAT:' + ','.join(info['saturated']) if info['saturated'] else 'free'
        print(f"    [{i+1:2d}/{n}] ρ1={rho1:6.3f} | "
              f"ana M12={M12_ana[i]:+6.3f}  sim M12={M12_sim[i]:+6.3f} | {s}")

    print("  ρ3 sweep (ρ1 = 1)")
    for i, rho3 in enumerate(rho_grid):
        k = np.array([K2_BASE, K2_BASE, rho3 * K2_BASE])
        th_a = analytical_angles_deg(DELTA_L, r_ext, k)
        th_s, info = mujoco_equilibrium(k, DELTA_L)
        _, M32_ana[i] = morphology_metrics(th_a)
        _, M32_sim[i] = morphology_metrics(th_s)
        sat_32.append(info['saturated'])
        s = 'SAT:' + ','.join(info['saturated']) if info['saturated'] else 'free'
        print(f"    [{i+1:2d}/{n}] ρ3={rho3:6.3f} | "
              f"ana M32={M32_ana[i]:+6.3f}  sim M32={M32_sim[i]:+6.3f} | {s}")

    out = os.path.join(OUT_DIR, 'trend.png')
    plot_trend(rho_grid, M12_ana, M12_sim, M32_ana, M32_sim,
               sat_12, sat_32, out)
    print(f"  [SAVED] {out}")


def stage_morphology(r_ext):
    print("\n[STAGE 3] Morphology stick figures — 3 regimes")
    regime_data = []
    for short, label, k1, k2, k3 in REGIMES:
        k = np.array([k1, k2, k3])
        th_a = analytical_angles_deg(DELTA_L, r_ext, k)
        th_s, info = mujoco_equilibrium(k, DELTA_L)
        T = tendon_tension(DELTA_L, r_ext, k)
        M12_a, M32_a = morphology_metrics(th_a)
        M12_s, M32_s = morphology_metrics(th_s)
        print(f"  {short}: ρ1={k1/k2:.3f}, ρ3={k3/k2:.3f}")
        print(f"    ana θ = [{th_a[0]:+6.2f}, {th_a[1]:+6.2f}, {th_a[2]:+6.2f}]°  "
              f"M12 = {M12_a:+.3f}  M32 = {M32_a:+.3f}  T = {T:.3f} N")
        print(f"    sim θ = [{th_s[0]:+6.2f}, {th_s[1]:+6.2f}, {th_s[2]:+6.2f}]°  "
              f"M12 = {M12_s:+.3f}  M32 = {M32_s:+.3f}  "
              f"sat = {info['saturated'] or '—'}")
        regime_data.append({
            'short': short, 'regime_label': label,
            'k': k, 'rho1': k1/k2, 'rho3': k3/k2,
            'theta_ana': th_a, 'theta_sim': th_s,
            'M12_ana': M12_a, 'M32_ana': M32_a,
            'M12_sim': M12_s, 'M32_sim': M32_s,
            'tension_ana': T, 'saturated': info['saturated'],
        })

    out = os.path.join(OUT_DIR, 'morphology.png')
    plot_morphology(regime_data, out)
    print(f"  [SAVED] {out}")
    return regime_data


def stage_grid_sweep(r_ext, rho_grid):
    n = len(rho_grid)
    n_total = n * n
    print(f"\n[STAGE 4] {n}×{n} grid sweep — {n_total} equilibrium runs")

    grids = {key: np.zeros((n, n)) for key in [
        'theta1_ana', 'theta2_ana', 'theta3_ana',
        'theta1_sim', 'theta2_sim', 'theta3_sim',
        'M12_ana', 'M12_sim', 'M12_abs_err', 'M12_pct_err',
        'M32_ana', 'M32_sim', 'M32_abs_err', 'M32_pct_err',
        'tension_ana',
    ]}
    sat_mask = np.zeros((n, n), dtype=bool)
    raw_rows = []

    t_start = time.time()
    idx = 0
    for i, rho1 in enumerate(rho_grid):
        for j, rho3 in enumerate(rho_grid):
            k = np.array([rho1 * K2_BASE, K2_BASE, rho3 * K2_BASE])
            th_a = analytical_angles_deg(DELTA_L, r_ext, k)
            th_s, info = mujoco_equilibrium(k, DELTA_L)
            M12_a, M32_a = morphology_metrics(th_a)
            M12_s, M32_s = morphology_metrics(th_s)
            T = tendon_tension(DELTA_L, r_ext, k)

            grids['theta1_ana'][i, j] = th_a[0]
            grids['theta2_ana'][i, j] = th_a[1]
            grids['theta3_ana'][i, j] = th_a[2]
            grids['theta1_sim'][i, j] = th_s[0]
            grids['theta2_sim'][i, j] = th_s[1]
            grids['theta3_sim'][i, j] = th_s[2]
            grids['M12_ana'][i, j] = M12_a
            grids['M12_sim'][i, j] = M12_s
            grids['M12_abs_err'][i, j] = abs(M12_s - M12_a)
            grids['M12_pct_err'][i, j] = pct_err(M12_s, M12_a)
            grids['M32_ana'][i, j] = M32_a
            grids['M32_sim'][i, j] = M32_s
            grids['M32_abs_err'][i, j] = abs(M32_s - M32_a)
            grids['M32_pct_err'][i, j] = pct_err(M32_s, M32_a)
            grids['tension_ana'][i, j] = T
            sat_mask[i, j] = bool(info['saturated'])

            raw_rows.append({
                'rho1': rho1, 'rho3': rho3,
                'k1': k[0], 'k2': k[1], 'k3': k[2],
                'theta1_ana_deg': th_a[0], 'theta2_ana_deg': th_a[1],
                'theta3_ana_deg': th_a[2],
                'theta1_sim_deg': th_s[0], 'theta2_sim_deg': th_s[1],
                'theta3_sim_deg': th_s[2],
                'M12_ana': M12_a, 'M12_sim': M12_s,
                'M12_abs_err': abs(M12_s - M12_a),
                'M12_pct_err': pct_err(M12_s, M12_a),
                'M32_ana': M32_a, 'M32_sim': M32_s,
                'M32_abs_err': abs(M32_s - M32_a),
                'M32_pct_err': pct_err(M32_s, M32_a),
                'tension_ana_N': T,
                'saturated': ','.join(info['saturated']),
                'conv_time_s': info['conv_time'] if info['conv_time'] else '',
            })
            idx += 1
            if idx % 7 == 0:
                el = time.time() - t_start
                print(f"  [{idx:3d}/{n_total}]  elapsed {el:.1f}s  "
                      f"eta {el/idx*(n_total-idx):.1f}s")

    csv_path = os.path.join(OUT_DIR, 'summary.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        w.writeheader(); w.writerows(raw_rows)
    print(f"  [SAVED] {csv_path}")

    print("  generating heatmaps...")
    HEATMAP_SPECS = [
        # (key, title, cbar_label, cmap, fmt, log_norm, dark_text_at_high)
        ('theta1_ana', 'Analytical $\\theta_1$ (MCP)', 'angle [deg]', 'viridis', '{:.1f}', False, False),
        ('theta1_sim', 'MuJoCo $\\theta_1$ (MCP)',    'angle [deg]', 'viridis', '{:.1f}', False, False),
        ('theta2_ana', 'Analytical $\\theta_2$ (PIP)', 'angle [deg]', 'viridis', '{:.1f}', False, False),
        ('theta2_sim', 'MuJoCo $\\theta_2$ (PIP)',    'angle [deg]', 'viridis', '{:.1f}', False, False),
        ('theta3_ana', 'Analytical $\\theta_3$ (DIP)', 'angle [deg]', 'viridis', '{:.1f}', False, False),
        ('theta3_sim', 'MuJoCo $\\theta_3$ (DIP)',    'angle [deg]', 'viridis', '{:.1f}', False, False),
        ('M12_ana',  'Analytical $M_{12}=\\theta_1/\\theta_2$', '$M_{12}$', 'magma', '{:.2f}', False, False),
        ('M12_sim',  'MuJoCo $M_{12}=\\theta_1/\\theta_2$',     '$M_{12}$', 'magma', '{:.2f}', False, False),
        ('M12_abs_err', 'Absolute error  $|M_{12}^{sim} - M_{12}^{ana}|$',
         'error (log scale)', 'YlOrRd', '{:.3f}', True, True),
        ('M32_ana',  'Analytical $M_{32}=\\theta_3/\\theta_2$', '$M_{32}$', 'magma', '{:.2f}', False, False),
        ('M32_sim',  'MuJoCo $M_{32}=\\theta_3/\\theta_2$',     '$M_{32}$', 'magma', '{:.2f}', False, False),
        ('M32_abs_err', 'Absolute error  $|M_{32}^{sim} - M_{32}^{ana}|$',
         'error (log scale)', 'YlOrRd', '{:.3f}', True, True),
        ('tension_ana', 'Analytical tendon tension $\\lambda$ (Eq. 4)',
         'tension [N]', 'cividis', '{:.2f}', False, False),
    ]

    # Signed percentage-error maps: (M_sim − M_ana)/M_ana × 100, diverging,
    # centered at 0 (blue = sim under-predicts, red = over-predicts).
    PCT_ERR_SPECS = [
        ('M12_pct_err',
         'Percentage error  $(M_{12}^{sim}\\!-\\!M_{12}^{ana})/M_{12}^{ana}\\times100$'),
        ('M32_pct_err',
         'Percentage error  $(M_{32}^{sim}\\!-\\!M_{32}^{ana})/M_{32}^{ana}\\times100$'),
    ]

    for key, title, cbar, cmap, fmt, log_norm, dark_high in HEATMAP_SPECS:
        out = os.path.join(OUT_DIR, f'heatmap_{key}.png')
        plot_heatmap(grids[key], rho_grid, title, cbar, out,
                     cmap=cmap, fmt=fmt,
                     sat_mask=sat_mask if key.endswith('_sim') else None,
                     log_norm=log_norm, dark_text_at_high=dark_high)
        print(f"  [SAVED] heatmap_{key}.png")

    for key, title in PCT_ERR_SPECS:
        out = os.path.join(OUT_DIR, f'heatmap_{key}.png')
        plot_heatmap(grids[key], rho_grid, title, '% error', out,
                     cmap='RdBu_r', fmt='{:+.1f}', center=0.0)
        print(f"  [SAVED] heatmap_{key}.png")

    print("  generating scatter plots...")
    s_path_12 = os.path.join(OUT_DIR, 'scatter_M12.png')
    s_path_32 = os.path.join(OUT_DIR, 'scatter_M32.png')
    sp12_res = plot_scatter(grids['M12_ana'], grids['M12_sim'],
                            r'$M_{12} = \theta_1/\theta_2$', s_path_12)
    sp32_res = plot_scatter(grids['M32_ana'], grids['M32_sim'],
                            r'$M_{32} = \theta_3/\theta_2$', s_path_32)
    print(f"  [SAVED] {s_path_12}")
    print(f"  [SAVED] {s_path_32}")

    sp12, pr12, r2_12, sl12, int12 = sp12_res
    sp32, pr32, r2_32, sl32, int32 = sp32_res
    print(f"\n  M12: Spearman ρ={sp12.statistic:+.4f}  Pearson r={pr12:.4f}  "
          f"R²={r2_12:.4f}  fit: y = {sl12:.3f}x + {int12:.3f}")
    print(f"  M32: Spearman ρ={sp32.statistic:+.4f}  Pearson r={pr32:.4f}  "
          f"R²={r2_32:.4f}  fit: y = {sl32:.3f}x + {int32:.3f}")

    return {'grids': grids, 'sat_mask': sat_mask,
            'sp12': sp12, 'sp32': sp32, 'pr12': pr12, 'pr32': pr32,
            'r2_12': r2_12, 'r2_32': r2_32}


def stage_metric_table(regime_data):
    print("\n[STAGE 5] Morphology metric table")
    rows = []
    for d in regime_data:
        e12 = pct_err(d['M12_sim'], d['M12_ana'])
        e32 = pct_err(d['M32_sim'], d['M32_ana'])
        rows.append({
            'regime': d['short'],
            'rho1': round(d['rho1'], 4),
            'rho3': round(d['rho3'], 4),
            'k1_Nm_rad': d['k'][0], 'k2_Nm_rad': d['k'][1], 'k3_Nm_rad': d['k'][2],
            'theta1_ana_deg': round(d['theta_ana'][0], 2),
            'theta2_ana_deg': round(d['theta_ana'][1], 2),
            'theta3_ana_deg': round(d['theta_ana'][2], 2),
            'theta1_sim_deg': round(d['theta_sim'][0], 2),
            'theta2_sim_deg': round(d['theta_sim'][1], 2),
            'theta3_sim_deg': round(d['theta_sim'][2], 2),
            'M12_ana': round(d['M12_ana'], 4),
            'M12_sim': round(d['M12_sim'], 4),
            'M12_err_pct': round(e12, 2),
            'M32_ana': round(d['M32_ana'], 4),
            'M32_sim': round(d['M32_sim'], 4),
            'M32_err_pct': round(e32, 2),
            'tendon_tension_ana_N': round(d['tension_ana'], 4),
            'saturated': ','.join(d['saturated']) if d['saturated'] else '',
        })

    csv_path = os.path.join(OUT_DIR, 'metric_table.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"  [SAVED] {csv_path}")

    print()
    print(f"  {'Regime':<10} {'M12_ana':>9} {'M12_sim':>9} {'e12%':>6}  "
          f"{'M32_ana':>9} {'M32_sim':>9} {'e32%':>6}  "
          f"{'T[N]':>7}  sat")
    print(f"  {'-'*10} {'-'*9} {'-'*9} {'-'*6}  "
          f"{'-'*9} {'-'*9} {'-'*6}  {'-'*7}  ---")
    for r in rows:
        print(f"  {r['regime']:<10} "
              f"{r['M12_ana']:>9.3f} {r['M12_sim']:>9.3f} {r['M12_err_pct']:>6.1f}  "
              f"{r['M32_ana']:>9.3f} {r['M32_sim']:>9.3f} {r['M32_err_pct']:>6.1f}  "
              f"{r['tendon_tension_ana_N']:>7.3f}  {r['saturated'] or '—'}")


def main():
    print("=" * 76)
    print("  HIGH-FIDELITY STIFFNESS-RATIO VALIDATION")
    print("  CAD-accurate 3R tendon-driven finger vs analytical morphology law")
    print("=" * 76)
    print(f"  Hardware springs:  S1 = {SPRING_1:.4f}   "
          f"S2 = {SPRING_2:.4f}   S3 = {SPRING_3:.4f}   [N·m/rad]")
    print(f"  Reference k2 = Spring 2 = {K2_BASE:.4f} N·m/rad")
    print(f"  ρ anchors: low = {RHO_LOW:.4f} (S3)   "
          f"mid = {RHO_MID:.4f} (S2)   high = {RHO_HIGH:.4f} (S1)")
    print(f"  ΔL = {DELTA_L*1000:.0f} mm   "
          f"equilibrium cap = {EQUIL_MAX_TIME:.1f} s   vel tol = {VEL_TOL} rad/s")

    rho_grid = build_rho_grid()
    print(f"  ρ grid ({len(rho_grid)} values): "
          + ", ".join(f"{v:.3f}" for v in rho_grid))

    t0 = time.time()

    print("\n[STAGE 1] Geometry extraction (single source of truth)")
    global LINK_LENGTHS
    r_ext, LINK_LENGTHS = extract_geometry()
    print(f"  r1 (MCP) = {r_ext[0]*1000:.3f} mm")
    print(f"  r2 (PIP) = {r_ext[1]*1000:.3f} mm")
    print(f"  r3 (DIP) = {r_ext[2]*1000:.3f} mm")
    print(f"  L1 proximal = {LINK_LENGTHS[0]*1000:.2f} mm")
    print(f"  L2 middle   = {LINK_LENGTHS[1]*1000:.2f} mm")
    print(f"  L3 distal   = {LINK_LENGTHS[2]*1000:.2f} mm")

    stage_trend_sweep(r_ext, rho_grid)
    regime_data = stage_morphology(r_ext)
    grid_stats = stage_grid_sweep(r_ext, rho_grid)
    stage_metric_table(regime_data)

    elapsed = time.time() - t0
    n_sat = int(grid_stats['sat_mask'].sum())
    print("\n" + "=" * 76)
    print(f"  FINAL SUMMARY   (total {elapsed:.1f} s)")
    print("=" * 76)
    print(f"  Spearman ρ(M12) = {grid_stats['sp12'].statistic:+.4f}  "
          f"(p = {grid_stats['sp12'].pvalue:.2e})")
    print(f"  Spearman ρ(M32) = {grid_stats['sp32'].statistic:+.4f}  "
          f"(p = {grid_stats['sp32'].pvalue:.2e})")
    print(f"  Pearson r(M12)  = {grid_stats['pr12']:+.4f}   "
          f"R²(M12) = {grid_stats['r2_12']:.4f}")
    print(f"  Pearson r(M32)  = {grid_stats['pr32']:+.4f}   "
          f"R²(M32) = {grid_stats['r2_32']:.4f}")
    print(f"  Saturated cells: {n_sat}/{grid_stats['sat_mask'].size} "
          f"({100*n_sat/grid_stats['sat_mask'].size:.1f}%)")
    print(f"  Output dir: {OUT_DIR}")
    print("=" * 76)


if __name__ == "__main__":
    main()
