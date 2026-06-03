#!/home/namit/iitgn/mujoco_env/bin/python3
import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from low_fidelity.utils.sim_utils import setup_simulation, run_finger_trajectory, extract_moment_arms
from low_fidelity.utils.math_utils import analytical_angles_deg, convex_hull_2d, polygon_area_2d

SYS_SWEEP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single joint stiffness ratio configuration.")
    parser.add_argument("--rho1", type=float, default=1.0, help="MCP to PIP stiffness ratio: k1 / k2")
    parser.add_argument("--rho3", type=float, default=1.0, help="DIP to PIP stiffness ratio: k3 / k2")
    parser.add_argument("--plot", action="store_true", default=False, help="Generate and save trajectory/validation plot")
    return parser.parse_args()

def run_case(rho1, rho3, generate_plot=False):
    # 1. Load config
    config_path = os.path.join(SYS_SWEEP_DIR, 'configs', 'sweep_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    k2 = config['k2_reference']
    k1 = rho1 * k2
    k3 = rho3 * k2
    
    print("=" * 60)
    print(f"RUNNING STIFFNESS CONFIGURATION: rho1={rho1:.2f}, rho3={rho3:.2f}")
    print(f"Stiffness values: k1 (MCP)={k1:.3f}, k2 (PIP)={k2:.3f}, k3 (DIP)={k3:.3f} Nm/rad")
    print("=" * 60)
    
    # 2. Extract moment arms
    r = extract_moment_arms()
    print("Extracted Moment Arms (straight posture):")
    print(f"  r1 (MCP) = {r[0]:.6f} m")
    print(f"  r2 (PIP) = {r[1]:.6f} m")
    print(f"  r3 (DIP) = {r[2]:.6f} m\n")
    
    # 3. Setup and run simulation
    model, data = setup_simulation(k1, k2, k3)
    history = run_finger_trajectory(model, data, config)
    
    # 4. Extract metrics
    # Joint Angles
    theta1_final = history['theta1'][-1]
    theta2_final = history['theta2'][-1]
    theta3_final = history['theta3'][-1]
    
    theta12_ratio = theta1_final / theta2_final if abs(theta2_final) > 1e-5 else 0.0
    theta32_ratio = theta3_final / theta2_final if abs(theta2_final) > 1e-5 else 0.0
    
    # Fingertip path length
    x = np.array(history['x_tip'])
    z = np.array(history['z_tip'])
    dx = np.diff(x)
    dz = np.diff(z)
    path_length = np.sum(np.sqrt(dx**2 + dz**2))
    
    # Convex Hull Workspace Area
    points = list(zip(x, z))
    hull = convex_hull_2d(points)
    workspace_area = polygon_area_2d(hull)
    
    # Tendon tension
    tension_final = history['tendon_tension'][-1]
    
    # 5. Analytical Validation
    delta_L_arr = np.array(history['delta_L'])
    # Compute analytical angles for every DeltaL in simulation trajectory
    # Returns array of shape (3, N)
    theta_analytical = analytical_angles_deg(delta_L_arr, r, [k1, k2, k3])
    
    theta_sim = np.vstack([history['theta1'], history['theta2'], history['theta3']])
    errors = theta_sim - theta_analytical
    rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(np.abs(errors))
    
    # Analytical ratios at final state
    theta1_ana_final = theta_analytical[0, -1]
    theta2_ana_final = theta_analytical[1, -1]
    theta3_ana_final = theta_analytical[2, -1]
    
    theta12_ratio_ana = theta1_ana_final / theta2_ana_final if abs(theta2_ana_final) > 1e-5 else 0.0
    theta32_ratio_ana = theta3_ana_final / theta2_ana_final if abs(theta2_ana_final) > 1e-5 else 0.0
    
    # Morphology errors (absolute difference)
    theta12_error = abs(theta12_ratio - theta12_ratio_ana)
    theta32_error = abs(theta32_ratio - theta32_ratio_ana)
        
    # 6. Print Results
    print("--- RESULTS SUMMARY ---")
    print(f"Final Joint Angles:  MCP={theta1_final:6.2f}° | PIP={theta2_final:6.2f}° | DIP={theta3_final:6.2f}°")
    print(f"Joint Angle Ratios:  theta1/theta2={theta12_ratio:.4f} (Ana: {theta12_ratio_ana:.4f}, Err: {theta12_error:.4f})")
    print(f"                     theta3/theta2={theta32_ratio:.4f} (Ana: {theta32_ratio_ana:.4f}, Err: {theta32_error:.4f})")
    print(f"Workspace Metrics:   Path Length={path_length*1000:6.2f} mm | Area={workspace_area*1e6:8.2f} mm^2")
    print(f"Tendon Force:        Final Tension={tension_final:6.2f} N")
    print(f"Analytical Error:    RMSE={rmse:6.3f}° | Max Error={max_error:6.3f}°")
    print("-" * 60)
    
    # 7. Generate Plot if requested
    if generate_plot:
        plots_dir = os.path.join(SYS_SWEEP_DIR, 'results', 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        fig, (ax_path, ax_angles) = plt.subplots(1, 2, figsize=(13, 6))
        
        # Left Panel: Fingertip Trajectory and Workspace Area
        ax_path.plot(x * 1000, z * 1000, 'b-', linewidth=2.5, label='Fingertip Path')
        ax_path.plot(x[0]*1000, z[0]*1000, 'go', markersize=8, label='Start (DeltaL=0)')
        ax_path.plot(x[-1]*1000, z[-1]*1000, 'ro', markersize=8, label='End (DeltaL=40mm)')
        
        # Draw convex hull polygon
        if len(hull) >= 3:
            hull_x = [p[0] * 1000 for p in hull] + [hull[0][0] * 1000]
            hull_z = [p[1] * 1000 for p in hull] + [hull[0][1] * 1000]
            ax_path.fill(hull_x, hull_z, color='skyblue', alpha=0.3, linestyle='--', edgecolor='deepskyblue', label='Convex Hull Workspace')
            
        ax_path.set_title(f"Fingertip 2D Path & Workspace\nArea={workspace_area*1e6:.1f} mm² | Length={path_length*1000:.1f} mm", fontweight='bold')
        ax_path.set_xlabel("X Position [mm]")
        ax_path.set_ylabel("Z Position [mm]")
        ax_path.axis('equal')
        ax_path.grid(True, linestyle=':', alpha=0.6)
        ax_path.legend()
        
        # Right Panel: Joint Angles vs Tendon Displacement
        delta_L_mm = delta_L_arr * 1000.0
        # Sim paths (Solid)
        ax_angles.plot(delta_L_mm, history['theta1'], 'r-', linewidth=2, label='MuJoCo MCP')
        ax_angles.plot(delta_L_mm, history['theta2'], 'g-', linewidth=2, label='MuJoCo PIP')
        ax_angles.plot(delta_L_mm, history['theta3'], 'b-', linewidth=2, label='MuJoCo DIP')
        
        # Analytical paths (Dashed)
        ax_angles.plot(delta_L_mm, theta_analytical[0], 'r--', linewidth=1.5, alpha=0.7, label='Analytical MCP')
        ax_angles.plot(delta_L_mm, theta_analytical[1], 'g--', linewidth=1.5, alpha=0.7, label='Analytical PIP')
        ax_angles.plot(delta_L_mm, theta_analytical[2], 'b--', linewidth=1.5, alpha=0.7, label='Analytical DIP')
        
        ax_angles.set_title(f"Joint Angles vs. displacement\nRMSE = {rmse:.3f}° | Max Error = {max_error:.3f}°", fontweight='bold')
        ax_angles.set_xlabel("Tendon Displacement (DeltaL) [mm]")
        ax_angles.set_ylabel("Joint Angle [degrees]")
        ax_angles.grid(True, linestyle=':', alpha=0.6)
        ax_angles.legend()
        
        fig.suptitle(f"Single Stiffness Case: $\\rho_1$={rho1:.2f}, $\\rho_3$={rho3:.2f}", fontsize=14, fontweight='bold')
        fig.tight_layout()
        
        fig_path = os.path.join(plots_dir, f'single_case_rho1_{rho1}_rho3_{rho3}.png')
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[PLOT SAVED] {fig_path}\n")
        
    return {
        'rho1': rho1,
        'rho3': rho3,
        'theta1_final': theta1_final,
        'theta2_final': theta2_final,
        'theta3_final': theta3_final,
        'theta12_ratio': theta12_ratio,
        'theta32_ratio': theta32_ratio,
        'workspace_area': workspace_area,
        'path_length': path_length,
        'tension_final': tension_final,
        'rmse': rmse,
        'max_error': max_error
    }

if __name__ == "__main__":
    args = parse_args()
    run_case(args.rho1, args.rho3, args.plot)
