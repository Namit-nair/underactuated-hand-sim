#!/usr/bin/env python3
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, linregress

# Set plotting style parameters
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

def save_heatmap(grid, ratios, title, filename, cbar_label, cmap='viridis', fmt='%.3f'):
    plots_dir = os.path.join(SYS_SWEEP_DIR, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    n_ratios = len(ratios)
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(grid, origin='lower', cmap=cmap, aspect='auto')
    
    ax.set_xticks(np.arange(n_ratios))
    ax.set_yticks(np.arange(n_ratios))
    ax.set_xticklabels([str(r) for r in ratios])
    ax.set_yticklabels([str(r) for r in ratios])
    
    ax.set_xlabel(r'DIP Joint Stiffness Ratio $\rho_3 = k_3 / k_2$', fontweight='bold')
    ax.set_ylabel(r'MCP Joint Stiffness Ratio $\rho_1 = k_1 / k_2$', fontweight='bold')
    ax.set_title(title, pad=15, fontweight='bold')
    
    threshold = (grid.max() + grid.min()) / 2.0
    for i in range(n_ratios):
        for j in range(n_ratios):
            val = grid[i, j]
            if cmap == 'magma':
                color = 'white' if val < threshold else 'black'
            else:
                color = 'black' if val > threshold else 'white'
            ax.text(j, i, fmt % val, ha='center', va='center', color=color, fontweight='semibold')
            
    cbar = fig.colorbar(im, ax=ax, pad=0.03)
    cbar.set_label(cbar_label, fontweight='bold')
    
    plt.tight_layout()
    out_path = os.path.join(plots_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT SAVED] {out_path}")

def save_global_scatter(x, y, title, filename, xlabel, ylabel, stats):
    plots_dir = os.path.join(SYS_SWEEP_DIR, 'results', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Scatter points
    ax.scatter(x, y, color='blue', alpha=0.7, edgecolors='k', label='Sim vs Ana (n=49)')
    
    # 45-degree perfect-agreement line
    min_val = min(np.min(x), np.min(y))
    max_val = max(np.max(x), np.max(y))
    range_pad = (max_val - min_val) * 0.05
    line_min = min_val - range_pad
    line_max = max_val + range_pad
    
    ax.plot([line_min, line_max], [line_min, line_max], 'k--', alpha=0.5, label='Perfect Agreement (y=x)')
    
    # Best-fit regression line
    fit_x = np.linspace(line_min, line_max, 100)
    fit_y = stats['slope'] * fit_x + stats['intercept']
    ax.plot(fit_x, fit_y, 'r-', linewidth=2, label=f'Best Fit (y={stats["slope"]:.3f}x + {stats["intercept"]:.3f})')
    
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_title(title, pad=15, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Annotation box for statistics
    textstr = '\n'.join((
        f"Spearman $\\rho$: {stats['spearman_rho']:.3f} (p={stats['spearman_p']:.2e})",
        f"Pearson $r$: {stats['pearson_r']:.3f}",
        f"$R^2$: {stats['r_squared']:.3f}"
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    out_path = os.path.join(plots_dir, filename)
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[PLOT SAVED] {out_path}")

def main():
    summary_path = os.path.join(SYS_SWEEP_DIR, 'results', 'summary.csv')
    if not os.path.exists(summary_path):
        print(f"Error: {summary_path} not found. Run the sweep first.")
        return
        
    data = []
    ratios_set = set()
    
    with open(summary_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = {
                'rho1': float(row['rho1']),
                'rho3': float(row['rho3']),
                'R12_ana': float(row['theta12_ratio_analytical']),
                'R12_sim': float(row['theta12_ratio_simulation']),
                'R12_err': float(row['theta12_ratio_error']),
                'R32_ana': float(row['theta32_ratio_analytical']),
                'R32_sim': float(row['theta32_ratio_simulation']),
                'R32_err': float(row['theta32_ratio_error']),
            }
            data.append(d)
            ratios_set.add(d['rho1'])
            
    ratios = sorted(list(ratios_set))
    n_ratios = len(ratios)
    ratio_to_idx = {val: idx for idx, val in enumerate(ratios)}
    
    # Create output CSV
    output_csv = os.path.join(SYS_SWEEP_DIR, 'results', 'morphology_trend_validation.csv')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rho1', 'rho3', 'R12_analytical', 'R12_simulation', 'R12_error', 'R32_analytical', 'R32_simulation', 'R32_error'])
        for d in data:
            writer.writerow([d['rho1'], d['rho3'], d['R12_ana'], d['R12_sim'], d['R12_err'], d['R32_ana'], d['R32_sim'], d['R32_err']])
            
    print(f"Saved local trend data to {output_csv}")
    
    # 1. Local Error Heatmaps
    err12_grid = np.zeros((n_ratios, n_ratios))
    err32_grid = np.zeros((n_ratios, n_ratios))
    
    for d in data:
        i = ratio_to_idx[d['rho1']]
        j = ratio_to_idx[d['rho3']]
        err12_grid[i, j] = d['R12_err']
        err32_grid[i, j] = d['R32_err']
        
    save_heatmap(err12_grid, ratios, "Absolute Error in θ1/θ2 Ratio", "heatmap_theta12_ratio_error.png", "Error", "magma")
    save_heatmap(err32_grid, ratios, "Absolute Error in θ3/θ2 Ratio", "heatmap_theta32_ratio_error.png", "Error", "magma")
    
    # 2. Global Trend Validation
    R12_ana_all = np.array([d['R12_ana'] for d in data])
    R12_sim_all = np.array([d['R12_sim'] for d in data])
    R32_ana_all = np.array([d['R32_ana'] for d in data])
    R32_sim_all = np.array([d['R32_sim'] for d in data])
    
    def compute_stats(ana, sim):
        spear_corr, spear_p = spearmanr(ana, sim)
        pears_corr, pears_p = pearsonr(ana, sim)
        slope, intercept, r_val, p_val, std_err = linregress(ana, sim)
        r_squared = r_val ** 2
        return {
            'spearman_rho': spear_corr,
            'spearman_p': spear_p,
            'pearson_r': pears_corr,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'mean_err': np.mean(np.abs(sim - ana)),
            'median_err': np.median(np.abs(sim - ana)),
            'max_err': np.max(np.abs(sim - ana))
        }
        
    stats12 = compute_stats(R12_ana_all, R12_sim_all)
    stats32 = compute_stats(R32_ana_all, R32_sim_all)
    
    save_global_scatter(R12_ana_all, R12_sim_all, "Global Trend Validation (θ1/θ2)", "global_theta12_trend_validation.png", "Analytical θ1/θ2", "Simulation θ1/θ2", stats12)
    save_global_scatter(R32_ana_all, R32_sim_all, "Global Trend Validation (θ3/θ2)", "global_theta32_trend_validation.png", "Analytical θ3/θ2", "Simulation θ3/θ2", stats32)
    
    # 3. Report
    print("\n" + "="*60)
    print("THETA1/THETA2 VALIDATION")
    print("-" * 60)
    print(f"- Mean Error:   {stats12['mean_err']:.3f}")
    print(f"- Median Error: {stats12['median_err']:.3f}")
    print(f"- Maximum Error:{stats12['max_err']:.3f}\n")
    print(f"- Spearman rho: {stats12['spearman_rho']:.3f}")
    print(f"- Spearman p-value: {stats12['spearman_p']:.3e}\n")
    print(f"- Pearson r:    {stats12['pearson_r']:.3f}")
    print(f"- R²:           {stats12['r_squared']:.3f}")
    print("="*60)
    
    print("THETA3/THETA2 VALIDATION")
    print("-" * 60)
    print(f"- Mean Error:   {stats32['mean_err']:.3f}")
    print(f"- Median Error: {stats32['median_err']:.3f}")
    print(f"- Maximum Error:{stats32['max_err']:.3f}\n")
    print(f"- Spearman rho: {stats32['spearman_rho']:.3f}")
    print(f"- Spearman p-value: {stats32['spearman_p']:.3e}\n")
    print(f"- Pearson r:    {stats32['pearson_r']:.3f}")
    print(f"- R²:           {stats32['r_squared']:.3f}")
    print("="*60)

if __name__ == "__main__":
    main()
