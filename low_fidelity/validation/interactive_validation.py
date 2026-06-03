#!/usr/bin/env python3
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
import mujoco

from low_fidelity.core.finger_delta_l_control import (
    xml_content, MCP_STIFFNESS, PIP_STIFFNESS, DIP_STIFFNESS,
)

# ============================================================================
# Parameters (initial values)
# ============================================================================
L = np.array([0.05, 0.038, 0.03])      # link lengths [m]
r = np.array([0.008, 0.008, 0.008])    # moment-arm radii [m]

# Synchronized stiffness parameters (single source of truth from MuJoCo model)
k = np.array([
    MCP_STIFFNESS,
    PIP_STIFFNESS,
    DIP_STIFFNESS
])

Delta_max = 0.04          # 15 mm max displacement for validation range
Delta_init = 0.02         # 8 mm initial displacement

# ============================================================================
# Analytical model functions
# ============================================================================
def joint_angles_from_delta(Delta, r, k):
    """Delta: scalar or array [m]; returns theta of shape (3,) or (3,n)"""
    r = np.asarray(r)
    k = np.asarray(k)
    denom = np.sum(r**2 / k)
    # The output is directly converted to degrees for plotting
    theta_rad = (r / k).reshape(-1, 1) * (np.asarray(Delta) / denom)
    return np.degrees(theta_rad)

# ============================================================================
# MuJoCo evaluation
# ============================================================================
def compute_mujoco_sweep(k_vals, delta_vals):
    model = mujoco.MjModel.from_xml_string(xml_content)
    
    # Disable gravity for pure spring physics comparison
    model.opt.gravity[:] = 0
    
    # Set the stiffness dynamically
    mcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mcp")
    pip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pip")
    dip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dip")
    
    model.jnt_stiffness[mcp_id] = k_vals[0]
    model.jnt_stiffness[pip_id] = k_vals[1]
    model.jnt_stiffness[dip_id] = k_vals[2]
    
    mujoco_angles = np.zeros((3, len(delta_vals)))
    
    for i, delta in enumerate(delta_vals):
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        L_resting = data.ten_length[0]
        
        target_L = L_resting - delta
        model.tendon_lengthspring[0] = [target_L, target_L]
        
        # Step simulation to equilibrium (1.5 seconds is plenty with high damping)
        for _ in range(int(1.5 / model.opt.timestep)):
            mujoco.mj_step(model, data)
            
        mujoco_angles[0, i] = np.degrees(data.qpos[mcp_id])
        mujoco_angles[1, i] = np.degrees(data.qpos[pip_id])
        mujoco_angles[2, i] = np.degrees(data.qpos[dip_id])
        
    return mujoco_angles

# ============================================================================
# Moment Arm Extraction
# ============================================================================
def extract_moment_arms(model_xml):
    """Numerically extracts the effective moment arms from the MuJoCo spatial tendon routing."""
    model = mujoco.MjModel.from_xml_string(model_xml)
    data = mujoco.MjData(model)
    
    # Evaluate at straight posture
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    L0 = data.ten_length[0]
    
    mcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mcp")
    pip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pip")
    dip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dip")
    
    dtheta = 0.001
    r_extracted = np.zeros(3)
    
    for i, jid in enumerate([mcp_id, pip_id, dip_id]):
        mujoco.mj_resetData(model, data)
        data.qpos[jid] = dtheta
        mujoco.mj_forward(model, data)
        L1 = data.ten_length[0]
        # Tension shortens tendon: r = (L0 - L1) / dtheta
        r_extracted[i] = (L0 - L1) / dtheta
        
    return r_extracted

# ============================================================================
# Main UI setup
# ============================================================================
def main():
    r_extracted = extract_moment_arms(xml_content)

    print("Initializing Analytical Validation UI...")
    print("=" * 60)
    print("Imported Single Source-of-Truth Joint Stiffnesses:")
    print(f"  MCP Joint Stiffness (k1): {MCP_STIFFNESS:.4f} Nm/rad")
    print(f"  PIP Joint Stiffness (k2): {PIP_STIFFNESS:.4f} Nm/rad")
    print(f"  DIP Joint Stiffness (k3): {DIP_STIFFNESS:.4f} Nm/rad")
    print("-" * 60)
    print("Numerically Extracted Moment Arms (straight posture):")
    print(f"  MCP Moment Arm (r1): {r_extracted[0]:.6f} m")
    print(f"  PIP Moment Arm (r2): {r_extracted[1]:.6f} m")
    print(f"  DIP Moment Arm (r3): {r_extracted[2]:.6f} m")
    print("=" * 60)
    
    # Pre-generate delta points for the sweep curve
    delta_vals = np.linspace(0, Delta_max, 25)
    
    # Setup Figure and Axes
    fig = plt.figure(figsize=(12, 8))
    # Re-adjusted bottom margin to fit stiffness and displacement sliders
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.75, top=0.9)
    ax = fig.add_subplot(111)
    
    # Initial data
    theta_ana_const = joint_angles_from_delta(delta_vals, r, k)
    theta_ana_exact = joint_angles_from_delta(delta_vals, r_extracted, k)
    theta_muj = compute_mujoco_sweep(k, delta_vals)
    
    # Plot Analytical (Constant r)
    line_ana_mcp_c, = ax.plot(delta_vals * 1000, theta_ana_const[0], 'b:', linewidth=1.5, alpha=0.6, label='Ana (const r) MCP')
    line_ana_pip_c, = ax.plot(delta_vals * 1000, theta_ana_const[1], 'g:', linewidth=1.5, alpha=0.6, label='Ana (const r) PIP')
    line_ana_dip_c, = ax.plot(delta_vals * 1000, theta_ana_const[2], 'r:', linewidth=1.5, alpha=0.6, label='Ana (const r) DIP')
    
    # Plot Analytical (Extracted r)
    line_ana_mcp, = ax.plot(delta_vals * 1000, theta_ana_exact[0], 'b-', linewidth=2, label='Ana (exact r) MCP')
    line_ana_pip, = ax.plot(delta_vals * 1000, theta_ana_exact[1], 'g-', linewidth=2, label='Ana (exact r) PIP')
    line_ana_dip, = ax.plot(delta_vals * 1000, theta_ana_exact[2], 'r-', linewidth=2, label='Ana (exact r) DIP')
    
    # Plot MuJoCo (Dashed lines with markers)
    line_muj_mcp, = ax.plot(delta_vals * 1000, theta_muj[0], 'b--', marker='o', markersize=5, label='MuJoCo MCP')
    line_muj_pip, = ax.plot(delta_vals * 1000, theta_muj[1], 'g--', marker='o', markersize=5, label='MuJoCo PIP')
    line_muj_dip, = ax.plot(delta_vals * 1000, theta_muj[2], 'r--', marker='o', markersize=5, label='MuJoCo DIP')
    
    # Current Delta vertical line
    vline = ax.axvline(Delta_init * 1000, color='k', linestyle=':', label='Current Delta Selection')
    
    ax.set_title("Analytical Model vs. MuJoCo Tendon-Driven Physics", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tendon Displacement (DeltaL) [mm]", fontsize=12)
    ax.set_ylabel("Joint Angle [deg]", fontsize=12)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.grid(True, linestyle=":", alpha=0.6)
    
    # Text box for exact values at current Delta
    text_ax = plt.axes([0.78, 0.40, 0.20, 0.35])
    text_ax.axis('off')
    info_text = text_ax.text(0, 1, "", verticalalignment='top', fontfamily='monospace', fontsize=11)
    
    # Sliders
    axcolor = 'lightgoldenrodyellow'
    ax_k1 = plt.axes([0.15, 0.23, 0.60, 0.03], facecolor=axcolor)
    ax_k2 = plt.axes([0.15, 0.17, 0.60, 0.03], facecolor=axcolor)
    ax_k3 = plt.axes([0.15, 0.11, 0.60, 0.03], facecolor=axcolor)
    ax_delta = plt.axes([0.15, 0.05, 0.60, 0.03], facecolor=axcolor)
    
    s_k1 = Slider(ax_k1, 'k1 (MCP)', 0.01, 4.0, valinit=MCP_STIFFNESS, valfmt='%.3f Nm/rad')
    s_k2 = Slider(ax_k2, 'k2 (PIP)', 0.01, 4.0, valinit=PIP_STIFFNESS, valfmt='%.3f Nm/rad')
    s_k3 = Slider(ax_k3, 'k3 (DIP)', 0.01, 4.0, valinit=DIP_STIFFNESS, valfmt='%.3f Nm/rad')
    s_delta = Slider(ax_delta, 'DeltaL (mm)', 0.0, Delta_max * 1000, valinit=Delta_init * 1000, valfmt='%.1f mm')
    
    # Update function
    def update(val):
        k_curr = np.array([s_k1.val, s_k2.val, s_k3.val])
        d_val = s_delta.val / 1000.0
        
        # Recompute sweeps for plots
        th_ana_const = joint_angles_from_delta(delta_vals, r, k_curr)
        th_ana_exact = joint_angles_from_delta(delta_vals, r_extracted, k_curr)
        th_muj = compute_mujoco_sweep(k_curr, delta_vals)
        
        line_ana_mcp_c.set_ydata(th_ana_const[0])
        line_ana_pip_c.set_ydata(th_ana_const[1])
        line_ana_dip_c.set_ydata(th_ana_const[2])
        
        line_ana_mcp.set_ydata(th_ana_exact[0])
        line_ana_pip.set_ydata(th_ana_exact[1])
        line_ana_dip.set_ydata(th_ana_exact[2])
        
        line_muj_mcp.set_ydata(th_muj[0])
        line_muj_pip.set_ydata(th_muj[1])
        line_muj_dip.set_ydata(th_muj[2])
        
        vline.set_xdata([d_val * 1000, d_val * 1000])
        
        # Get specific values at exact slider Delta
        th_ana_current = joint_angles_from_delta(d_val, r_extracted, k_curr).flatten()
        th_muj_current = compute_mujoco_sweep(k_curr, [d_val]).flatten()
        
        # Safe percentage error calculation (using analytical as reference denominator)
        def get_pct_error(ana, muj):
            if abs(ana) < 0.5:
                return "N/A"
            pct = (abs(ana - muj) / abs(ana)) * 100.0
            return f"{pct:.1f}%"
            
        pct_mcp = get_pct_error(th_ana_current[0], th_muj_current[0])
        pct_pip = get_pct_error(th_ana_current[1], th_muj_current[1])
        pct_dip = get_pct_error(th_ana_current[2], th_muj_current[2])
        
        info = (f"=== Model Stiffnesses ===\n"
                f"k1 (MCP): {k_curr[0]:.2f} Nm/rad\n"
                f"k2 (PIP): {k_curr[1]:.2f} Nm/rad\n"
                f"k3 (DIP): {k_curr[2]:.2f} Nm/rad\n\n"
                f"=== Current Setpoint ===\n"
                f"Delta: {s_delta.val:5.1f} mm\n\n"
                f"--- Analytical ---\n"
                f"MCP: {th_ana_current[0]:6.1f}°\n"
                f"PIP: {th_ana_current[1]:6.1f}°\n"
                f"DIP: {th_ana_current[2]:6.1f}°\n\n"
                f"--- MuJoCo ---\n"
                f"MCP: {th_muj_current[0]:6.1f}°\n"
                f"PIP: {th_muj_current[1]:6.1f}°\n"
                f"DIP: {th_muj_current[2]:6.1f}°\n\n"
                f"--- Error (Abs / %) ---\n"
                f"MCP: {abs(th_ana_current[0]-th_muj_current[0]):5.1f}° / {pct_mcp}\n"
                f"PIP: {abs(th_ana_current[1]-th_muj_current[1]):5.1f}° / {pct_pip}\n"
                f"DIP: {abs(th_ana_current[2]-th_muj_current[2]):5.1f}° / {pct_dip}\n")
        info_text.set_text(info)
        
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()
    
    # Initialize the text box info once
    update(None)
    
    s_k1.on_changed(update)
    s_k2.on_changed(update)
    s_k3.on_changed(update)
    s_delta.on_changed(update)
    
    print("UI Ready. Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()
