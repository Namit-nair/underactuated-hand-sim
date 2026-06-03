#!/home/namit/iitgn/mujoco_env/bin/python3
import os
import csv
import json
import numpy as np
import matplotlib.pyplot as plt

# Set plotting style parameters for publication-quality figures
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 15,
    'figure.dpi': 150
})

SYS_SWEEP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    # 1. Load config to get ratios and references
    config_path = os.path.join(SYS_SWEEP_DIR, 'configs', 'sweep_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    ratios = config['ratios']
    n_ratios = len(ratios)
    
    # Map ratios to index
    ratio_to_idx = {val: idx for idx, val in enumerate(ratios)}
    
    # 2. Initialize grids
    theta1_grid = np.zeros((n_ratios, n_ratios))
    theta2_grid = np.zeros((n_ratios, n_ratios))
    theta3_grid = np.zeros((n_ratios, n_ratios))
    ratio12_grid = np.zeros((n_ratios, n_ratios))
    ratio32_grid = np.zeros((n_ratios, n_ratios))
    workspace_grid = np.zeros((n_ratios, n_ratios))
    path_grid = np.zeros((n_ratios, n_ratios))
    rmse_grid = np.zeros((n_ratios, n_ratios))
    max_err_grid = np.zeros((n_ratios, n_ratios))
    tension_grid = np.zeros((n_ratios, n_ratios))
    err12_grid = np.zeros((n_ratios, n_ratios))
    err32_grid = np.zeros((n_ratios, n_ratios))
    
    csv_path = os.path.join(SYS_SWEEP_DIR, 'results', 'summary.csv')
    if not os.path.exists(csv_path):
        print(f"Error: summary file not found at {csv_path}. Run the sweep first!")
        return
        
    # 3. Read and pivot CSV data
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rho1 = float(row['rho1'])
            rho3 = float(row['rho3'])
            i = ratio_to_idx[rho1]
            j = ratio_to_idx[rho3]
            
            theta1_grid[i, j] = float(row['theta1_final'])
            theta2_grid[i, j] = float(row['theta2_final'])
            theta3_grid[i, j] = float(row['theta3_final'])
            ratio12_grid[i, j] = float(row['theta12_ratio_simulation'])
            ratio32_grid[i, j] = float(row['theta32_ratio_simulation'])
            workspace_grid[i, j] = float(row['workspace_area'])
            path_grid[i, j] = float(row['path_length'])
            rmse_grid[i, j] = float(row['rmse'])
            max_err_grid[i, j] = float(row['max_error'])
            tension_grid[i, j] = float(row['tension_final'])
            err12_grid[i, j] = float(row['theta12_ratio_error'])
            err32_grid[i, j] = float(row['theta32_ratio_error'])
            
    # Create plots directory
    plots_dir = os.path.join(SYS_SWEEP_DIR, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Helper to plot a standard publication-quality heatmap
    def save_heatmap(grid, title, filename, cbar_label, cmap='viridis', scale=1.0, fmt='%.1f'):
        fig, ax = plt.subplots(figsize=(8, 7))
        
        scaled_grid = grid * scale
        
        # Display heatmap using origin='lower' to keep rho1 increasing along the Y-axis
        im = ax.imshow(scaled_grid, origin='lower', cmap=cmap, aspect='auto')
        
        # Set ticks
        ax.set_xticks(np.arange(n_ratios))
        ax.set_yticks(np.arange(n_ratios))
        ax.set_xticklabels([str(r) for r in ratios])
        ax.set_yticklabels([str(r) for r in ratios])
        
        # Axis labels
        ax.set_xlabel(r'DIP Joint Stiffness Ratio $\rho_3 = k_3 / k_2$', fontweight='bold')
        ax.set_ylabel(r'MCP Joint Stiffness Ratio $\rho_1 = k_1 / k_2$', fontweight='bold')
        ax.set_title(title, pad=15, fontweight='bold')
        
        # Annotate each cell with its value
        # Choose text color based on cell brightness
        threshold = (scaled_grid.max() + scaled_grid.min()) / 2.0
        for i in range(n_ratios):
            for j in range(n_ratios):
                val = scaled_grid[i, j]
                color = 'white' if val > threshold and cmap in ['viridis', 'plasma', 'magma'] or (val < threshold and cmap == 'coolwarm') else 'black'
                if cmap in ['magma', 'inferno', 'plasma', 'viridis'] and val < threshold:
                    color = 'white'
                else:
                    color = 'black' if val > threshold else 'white'
                # Custom overrides for visibility on standard colormaps
                if cmap == 'viridis':
                    color = 'black' if val > threshold else 'white'
                elif cmap == 'magma':
                    color = 'white' if val < threshold else 'black'
                    
                ax.text(j, i, fmt % val, ha='center', va='center', color=color, fontweight='semibold')
                
        # Colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.03)
        cbar.set_label(cbar_label, fontweight='bold')
        
        plt.tight_layout()
        out_path = os.path.join(plots_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[PLOT SAVED] {out_path}")

    # 4. Generate the required 6 Heatmaps
    
    # 1. Workspace Area (Convert to mm^2 for neat visualization)
    save_heatmap(
        grid=workspace_grid,
        title="Fingertip Workspace Convex Hull Area",
        filename="heatmap_workspace_area.png",
        cbar_label="Workspace Area [mm²]",
        cmap="viridis",
        scale=1e6, # m^2 to mm^2
        fmt='%.0f'
    )
    
    # 2. Path Length (Convert to mm)
    save_heatmap(
        grid=path_grid,
        title="Fingertip Trajectory Path Length",
        filename="heatmap_path_length.png",
        cbar_label="Path Length [mm]",
        cmap="viridis",
        scale=1e3, # m to mm
        fmt='%.1f'
    )
    
    # 3. Theta1 Final (MCP)
    save_heatmap(
        grid=theta1_grid,
        title="Final MCP Joint Angle (theta1_final)",
        filename="heatmap_theta1_final.png",
        cbar_label="MCP Angle [degrees]",
        cmap="plasma",
        scale=1.0,
        fmt='%.1f'
    )
    
    # 4. Theta2 Final (PIP)
    save_heatmap(
        grid=theta2_grid,
        title="Final PIP Joint Angle (theta2_final)",
        filename="heatmap_theta2_final.png",
        cbar_label="PIP Angle [degrees]",
        cmap="plasma",
        scale=1.0,
        fmt='%.1f'
    )
    
    # 5. Theta3 Final (DIP)
    save_heatmap(
        grid=theta3_grid,
        title="Final DIP Joint Angle (theta3_final)",
        filename="heatmap_theta3_final.png",
        cbar_label="DIP Angle [degrees]",
        cmap="plasma",
        scale=1.0,
        fmt='%.1f'
    )
    
    # 6. RMSE
    save_heatmap(
        grid=rmse_grid,
        title="Analytical vs Simulation Joint Angles RMSE",
        filename="heatmap_rmse.png",
        cbar_label="RMSE [degrees]",
        cmap="magma",
        scale=1.0,
        fmt='%.2f'
    )
    
    # Optional: Max Error heatmap for completeness
    save_heatmap(
        grid=max_err_grid,
        title="Analytical vs Simulation Joint Angles Max Error",
        filename="heatmap_max_error.png",
        cbar_label="Max Error [degrees]",
        cmap="magma",
        scale=1.0,
        fmt='%.1f'
    )
    
    # 7. Morphology Error (theta1/theta2)
    save_heatmap(
        grid=err12_grid,
        title="Morphology Preservation Error\n(theta1/theta2)",
        filename="heatmap_theta12_ratio_error.png",
        cbar_label="Error",
        cmap="magma",
        scale=1.0,
        fmt='%.3f'
    )
    
    # 8. Morphology Error (theta3/theta2)
    save_heatmap(
        grid=err32_grid,
        title="Morphology Preservation Error\n(theta3/theta2)",
        filename="heatmap_theta32_ratio_error.png",
        cbar_label="Error",
        cmap="magma",
        scale=1.0,
        fmt='%.3f'
    )
    
    # 5. Print a concise analytical insight report based on grids
    print("\n" + "=" * 60)
    print("  STIFFNESS SWEEP FRAMEWORK - ANALYTICAL FINDINGS  ")
    print("=" * 60)
    
    # Locate maximum workspace area stiffness configuration
    max_idx = np.unravel_index(np.argmax(workspace_grid), workspace_grid.shape)
    best_rho1 = ratios[max_idx[0]]
    best_rho3 = ratios[max_idx[1]]
    max_area_mm2 = workspace_grid[max_idx] * 1e6
    
    # Locate minimum workspace area stiffness configuration
    min_idx = np.unravel_index(np.argmin(workspace_grid), workspace_grid.shape)
    worst_rho1 = ratios[min_idx[0]]
    worst_rho3 = ratios[min_idx[1]]
    min_area_mm2 = workspace_grid[min_idx] * 1e6
    
    # Locate minimum analytical RMSE configuration
    min_rmse_idx = np.unravel_index(np.argmin(rmse_grid), rmse_grid.shape)
    best_rmse_rho1 = ratios[min_rmse_idx[0]]
    best_rmse_rho3 = ratios[min_rmse_idx[1]]
    min_rmse_val = rmse_grid[min_rmse_idx]
    
    print(f"1. Workspace Optimization:")
    print(f"   - Max Workspace Area: {max_area_mm2:.1f} mm² at (rho1={best_rho1}, rho3={best_rho3})")
    print(f"     * Analysis: Soft MCP (rho1={best_rho1}) and stiff DIP (rho3={best_rho3}) allows")
    print(f"       the proximal joint to flex early and deeply while keeping the DIP joint extended,")
    print(f"       sweeping a much larger physical envelope.")
    print(f"   - Min Workspace Area: {min_area_mm2:.1f} mm² at (rho1={worst_rho1}, rho3={worst_rho3})")
    print(f"     * Analysis: Stiff MCP (rho1={worst_rho1}) and soft DIP (rho3={worst_rho3}) creates")
    print(f"       immediate curling of the distal phalanx while the MCP remains upright,")
    print(f"       restricting the fingertip to a tight, localized path.")
    print("")
    print(f"2. Fingertip Path Length:")
    max_path_idx = np.unravel_index(np.argmax(path_grid), path_grid.shape)
    print(f"   - Max Path Length: {path_grid[max_path_idx]*1000:.1f} mm at (rho1={ratios[max_path_idx[0]]}, rho3={ratios[max_path_idx[1]]})")
    print("")
    print(f"3. Analytical Validation Accuracy:")
    print(f"   - Minimum Model Error (RMSE): {min_rmse_val:.3f}° at (rho1={best_rmse_rho1}, rho3={best_rmse_rho3})")
    print(f"     * Analysis: The linear analytical model predicts joint angles with high accuracy")
    print(f"       at moderate joint ratios, but errors grow substantially under extreme ratios")
    print(f"       due to non-linear physical interactions like joint limits and spatial tendon slack.")
    print("")
    
    print(f"4. Morphology Preservation Validation:")
    min_err12_idx = np.unravel_index(np.argmin(err12_grid), err12_grid.shape)
    max_err12_idx = np.unravel_index(np.argmax(err12_grid), err12_grid.shape)
    
    min_err32_idx = np.unravel_index(np.argmin(err32_grid), err32_grid.shape)
    max_err32_idx = np.unravel_index(np.argmax(err32_grid), err32_grid.shape)
    
    print(f"   theta1/theta2 Error:")
    print(f"   - Min Error:  {err12_grid[min_err12_idx]:.3f} at (rho1={ratios[min_err12_idx[0]]}, rho3={ratios[min_err12_idx[1]]})")
    print(f"   - Max Error:  {err12_grid[max_err12_idx]:.3f} at (rho1={ratios[max_err12_idx[0]]}, rho3={ratios[max_err12_idx[1]]})")
    print(f"   - Mean Error: {np.mean(err12_grid):.3f}")
    print(f"")
    print(f"   theta3/theta2 Error:")
    print(f"   - Min Error:  {err32_grid[min_err32_idx]:.3f} at (rho1={ratios[min_err32_idx[0]]}, rho3={ratios[min_err32_idx[1]]})")
    print(f"   - Max Error:  {err32_grid[max_err32_idx]:.3f} at (rho1={ratios[max_err32_idx[0]]}, rho3={ratios[max_err32_idx[1]]})")
    print(f"   - Mean Error: {np.mean(err32_grid):.3f}")
    print(f"")
    
    # Calculate overall best/worst by average of the two percentage errors
    avg_err_grid = (err12_grid + err32_grid) / 2.0
    best_morph_idx = np.unravel_index(np.argmin(avg_err_grid), avg_err_grid.shape)
    worst_morph_idx = np.unravel_index(np.argmax(avg_err_grid), avg_err_grid.shape)
    
    print(f"   - Best configuration for morphology preservation: (rho1={ratios[best_morph_idx[0]]}, rho3={ratios[best_morph_idx[1]]}) with {avg_err_grid[best_morph_idx]:.3f} avg ratio error")
    print(f"   - Worst configuration for morphology preservation: (rho1={ratios[worst_morph_idx[0]]}, rho3={ratios[worst_morph_idx[1]]}) with {avg_err_grid[worst_morph_idx]:.3f} avg ratio error")
    print("=" * 60)
    print("Analysis complete.")

if __name__ == "__main__":
    main()
