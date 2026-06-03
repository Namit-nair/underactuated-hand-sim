#!/usr/bin/env python3
import os
import sys
import json
import csv
import time
import numpy as np

# Ensure the mujoco_simulations package root is on the path regardless of CWD
_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)

from low_fidelity.sim_utils import setup_simulation, run_finger_trajectory, extract_moment_arms
from low_fidelity.math_utils import analytical_angles_deg, convex_hull_2d, polygon_area_2d

SYS_SWEEP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def main():
    start_time = time.time()
    
    # 1. Load config
    config_path = os.path.join(SYS_SWEEP_DIR, 'sweep_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    ratios = config['ratios']
    k2 = config['k2_reference']
    
    # Create results folder if it doesn't exist
    results_dir = os.path.join(SYS_SWEEP_DIR, '..', '..', 'research_outputs', 'stiffness_sweep_latest')
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, 'summary.csv')
    
    print("=" * 70)
    print("  STARTING 49-CASE STIFFNESS SWEEP  ")
    print("=" * 70)
    print(f"Ratios: {ratios}")
    print(f"Total simulations to run: {len(ratios) * len(ratios)}")
    print(f"Output CSV path: {csv_path}")
    print("-" * 70)
    
    # 2. Extract moment arms for analytical comparison
    r = extract_moment_arms()
    print("Numerically Extracted Moment Arms (straight posture):")
    print(f"  r1 (MCP) = {r[0]:.6f} m")
    print(f"  r2 (PIP) = {r[1]:.6f} m")
    print(f"  r3 (DIP) = {r[2]:.6f} m\n")
    
    # 3. Open CSV file and write header
    csv_header = [
        'rho1', 'rho3',
        'theta1_final', 'theta2_final', 'theta3_final',
        'theta12_ratio_analytical', 'theta12_ratio_simulation', 'theta12_ratio_error',
        'theta32_ratio_analytical', 'theta32_ratio_simulation', 'theta32_ratio_error',
        'workspace_area', 'path_length',
        'rmse', 'max_error',
        'tension_final'
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
        
    # 4. Sweep loop
    run_idx = 0
    total_runs = len(ratios) * len(ratios)
    
    for rho1 in ratios:
        for rho3 in ratios:
            run_idx += 1
            k1 = rho1 * k2
            k3 = rho3 * k2
            
            # Setup and run simulation
            model, data = setup_simulation(k1, k2, k3)
            history = run_finger_trajectory(model, data, config)
            
            # Extract joint angles
            theta1_final = history['theta1'][-1]
            theta2_final = history['theta2'][-1]
            theta3_final = history['theta3'][-1]
            
            theta12_ratio_simulation = theta1_final / theta2_final if abs(theta2_final) > 1e-5 else 0.0
            theta32_ratio_simulation = theta3_final / theta2_final if abs(theta2_final) > 1e-5 else 0.0
            
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
            
            # Analytical Validation
            delta_L_arr = np.array(history['delta_L'])
            theta_analytical = analytical_angles_deg(delta_L_arr, r, [k1, k2, k3])
            theta_sim = np.vstack([history['theta1'], history['theta2'], history['theta3']])
            
            errors = theta_sim - theta_analytical
            rmse = np.sqrt(np.mean(errors ** 2))
            max_error = np.max(np.abs(errors))
            
            # Analytical ratios at final state
            theta1_ana_final = theta_analytical[0, -1]
            theta2_ana_final = theta_analytical[1, -1]
            theta3_ana_final = theta_analytical[2, -1]
            
            theta12_ratio_analytical = theta1_ana_final / theta2_ana_final if abs(theta2_ana_final) > 1e-5 else 0.0
            theta32_ratio_analytical = theta3_ana_final / theta2_ana_final if abs(theta2_ana_final) > 1e-5 else 0.0
            
            # Morphology errors (absolute difference)
            theta12_ratio_error = abs(theta12_ratio_simulation - theta12_ratio_analytical)
            theta32_ratio_error = abs(theta32_ratio_simulation - theta32_ratio_analytical)
            
            # Save row to CSV
            row = [
                rho1, rho3,
                theta1_final, theta2_final, theta3_final,
                theta12_ratio_analytical, theta12_ratio_simulation, theta12_ratio_error,
                theta32_ratio_analytical, theta32_ratio_simulation, theta32_ratio_error,
                workspace_area, path_length,
                rmse, max_error,
                tension_final
            ]
            
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
                
            elapsed = time.time() - start_time
            avg_time = elapsed / run_idx
            eta = avg_time * (total_runs - run_idx)
            
            print(f"  [{run_idx:2d}/{total_runs}] rho1={rho1:4.2f} rho3={rho3:4.2f} | "
                  f"Angles: [{theta1_final:5.1f}°, {theta2_final:5.1f}°, {theta3_final:5.1f}°] | "
                  f"RMSE: {rmse:6.3f}° | Elapsed: {elapsed:5.1f}s | ETA: {eta:5.1f}s")
            
    total_elapsed = time.time() - start_time
    print("-" * 70)
    print(f"SWEEP COMPLETED! Total time: {total_elapsed:.2f} seconds.")
    print(f"Results saved to: {csv_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
