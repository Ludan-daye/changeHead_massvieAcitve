#!/usr/bin/env python3
"""
Generate Exp2 3D comparison plots: Layer-wise suppression vs Baseline
Reference style: 3D scatter plot + surface
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from pathlib import Path


def setup_style():
    """Setup academic style"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.2,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


def load_exp2_data(model_name, base_dir='results/models'):
    """Load Exp2 data: disabled layer values + baseline MA values for each layer"""
    summary_path = Path(base_dir) / model_name / 'exp2b_mlp_layer_ablation' / 'summary.json'
    baseline_path = Path(base_dir) / model_name / 'exp2b_mlp_layer_ablation' / 'baseline.json'

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary not found: {summary_path}")
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline not found: {baseline_path}")

    with open(summary_path, 'r') as f:
        summary_data = json.load(f)

    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)

    # Parse ablation data (final MA value after disabling each layer)
    ablation = summary_data.get('ablation', {})
    layers = sorted([int(k) for k in ablation.keys()])
    disabled_values = np.array([ablation[str(l)] for l in layers])

    # Parse baseline data (MA value for each layer during normal operation, showing layer-by-layer accumulation)
    baseline_values = []
    if 'results' in baseline_data:
        results = baseline_data['results']
        for layer in layers:
            layer_data = results.get(str(layer), {})
            if isinstance(layer_data, dict) and 'mean' in layer_data:
                value = layer_data['mean']
                # Filter out nan values
                if np.isfinite(value):
                    baseline_values.append(value)
                else:
                    baseline_values.append(0)
                    print(f"Warning: Layer {layer} baseline is non-finite, using 0")
            else:
                baseline_values.append(0)
    else:
        raise ValueError("baseline.json format not recognized")

    baseline_values = np.array(baseline_values)

    return {
        'model': model_name,
        'layers': np.array(layers),
        'disabled_values': disabled_values,
        'baseline_values': baseline_values
    }


def plot_3d_comparison_surface(data, outdir, style='surface'):
    """
    Generate 3D comparison plot: Layer-wise suppression vs Baseline

    Args:
        data: Dictionary containing layers, disabled_values, baseline_values
        outdir: Output directory
        style: 'surface' or 'scatter' or 'both'
    """
    model = data['model']
    layers = data['layers']
    disabled_values = data['disabled_values']
    baseline_values = data['baseline_values']

    # Calculate percentage change relative to baseline
    baseline_mean = baseline_values.mean()
    drop_percentage = ((disabled_values - baseline_mean) / baseline_mean) * 100

    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax = fig.add_subplot(111, projection='3d')

    # Create grid data for surface
    x = layers  # Layer Index
    y = disabled_values  # MA Value (Disabled)
    z = drop_percentage  # Drop from Baseline (%)

    if style in ['scatter', 'both']:
        # Plot scatter points
        scatter = ax.scatter(x, y, z, c=z, cmap='coolwarm',
                           s=120, alpha=0.9, edgecolors='black', linewidth=0.8)
        plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=15,
                    label='Drop from Baseline (%)', pad=0.1)

    if style in ['surface', 'both']:
        # Create surface grid
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        Xi, Yi = np.meshgrid(xi, yi)

        # Use griddata interpolation
        Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

        # Plot surface
        surf = ax.plot_surface(Xi, Yi, Zi, cmap='viridis',
                              alpha=0.5, edgecolor='none',
                              linewidth=0, antialiased=True)

    # Add baseline reference plane
    baseline_x = np.array([layers.min(), layers.max()])
    baseline_y = np.array([baseline_mean, baseline_mean])
    baseline_z = np.array([0, 0])
    ax.plot(baseline_x, baseline_y, baseline_z,
           color='red', linewidth=3.5, linestyle='--',
           label=f'Baseline Mean (MA={baseline_mean:.1f})', alpha=0.9)

    # Set labels and title
    ax.set_xlabel('Layer Index', fontsize=13, labelpad=12)
    ax.set_ylabel('MA Value (Disabled)', fontsize=13, labelpad=12)
    ax.set_zlabel('Drop from Baseline (%)', fontsize=13, labelpad=12)

    # Adjust axis scale
    # Calculate data ranges
    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    z_range = z.max() - z.min()

    # Set reasonable axis aspect ratio (avoid extremely flat or elongated axes)
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([x_range/max_range * 1.2,
                       y_range/max_range * 1.0,
                       z_range/max_range * 0.8])

    # Optimize viewing angle: increase elevation, adjust azimuth
    # elev: elevation angle (0-90 degrees, higher = more top-down view)
    # azim: azimuth angle (0-360 degrees, rotation around z-axis)
    ax.view_init(elev=25, azim=135)

    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    outfile_png = outdir / f'{model}_exp2_3d_comparison_{style}.png'
    outfile_pdf = outdir / f'{model}_exp2_3d_comparison_{style}.pdf'
    fig.savefig(outfile_png, dpi=400, bbox_inches='tight')
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Generated 3D {style}: {outfile_png.name}")


def plot_cumulative_contribution(data, outdir):
    """
    Generate cumulative contribution curve plots (split into 2 independent plots)
    - Plot 1: Cumulative curve sorted by contribution
    - Plot 2: Cumulative curve in layer sequence order
    """
    model = data['model']
    layers = data['layers']
    disabled_values = data['disabled_values']
    baseline_values = data['baseline_values']

    # Determine baseline final value
    valid_baseline_values = baseline_values[:-1] if baseline_values[-1] < baseline_values[-2] * 0.5 else baseline_values
    baseline_final = valid_baseline_values.max()

    # Calculate absolute contribution of each layer
    absolute_contribution = baseline_final - disabled_values

    # Sort by contribution (descending)
    sorted_indices = np.argsort(absolute_contribution)[::-1]
    sorted_layers = layers[sorted_indices]
    sorted_contribution = absolute_contribution[sorted_indices]

    # Calculate cumulative contribution
    cumulative_contribution = np.cumsum(sorted_contribution)
    cumulative_percentage = (cumulative_contribution / baseline_final) * 100

    # Also calculate cumulative contribution in layer index order
    sequential_cumulative = np.cumsum(absolute_contribution)
    sequential_percentage = (sequential_cumulative / baseline_final) * 100

    # Early layer statistics
    early_layers_end = min(3, len(layers) - 1)
    early_contrib_pct = sequential_percentage[early_layers_end]

    # ========== Plot 1: Cumulative curve sorted by contribution ==========
    fig1, ax1 = plt.subplots(figsize=(10, 7), dpi=300)

    ax1.plot(range(len(sorted_layers)), cumulative_percentage, '-o',
            linewidth=3.5, markersize=9, color='#e74c3c',
            markerfacecolor='white', markeredgewidth=2.5,
            label='Cumulative Contribution', zorder=3)

    # Fill area
    ax1.fill_between(range(len(sorted_layers)), 0, cumulative_percentage,
                    alpha=0.25, color='#e74c3c', hatch='///',
                    edgecolor='#e74c3c', linewidth=1.0, zorder=1)

    # Add key threshold lines
    thresholds = [50, 70, 90]
    threshold_colors = ['#f39c12', '#3498db', '#2ecc71']
    threshold_layers = []

    for threshold, color in zip(thresholds, threshold_colors):
        ax1.axhline(threshold, color=color, linestyle='--', linewidth=2.0,
                   alpha=0.7, label=f'{threshold}% threshold', zorder=2)

        # Find layer reaching this threshold
        idx = np.where(cumulative_percentage >= threshold)[0]
        if len(idx) > 0:
            first_idx = idx[0]
            ax1.axvline(first_idx, color=color, linestyle=':', linewidth=1.5, alpha=0.5)
            ax1.scatter([first_idx], [threshold], s=200, color=color,
                       edgecolors='white', linewidth=2, zorder=4, marker='*')
            threshold_layers.append((threshold, first_idx + 1, sorted_layers[first_idx]))

            # Add text annotation
            ax1.text(first_idx, threshold + 3,
                    f'{first_idx + 1} layers\n(Layer {sorted_layers[first_idx]})',
                    fontsize=10, color=color, fontweight='bold',
                    ha='center', bbox=dict(boxstyle='round,pad=0.5',
                                          facecolor='white', alpha=0.8, edgecolor=color))

    # Add zero-axis reference line (if there are negative values)
    if cumulative_percentage.min() < 0:
        ax1.axhline(0, color='black', linestyle='-', linewidth=2.0, alpha=0.8, zorder=2)

    ax1.set_xlabel('Number of Layers (Sorted by Contribution)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Contribution (%)', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.95, fontsize=11,
              edgecolor='gray', fancybox=True, shadow=True)
    ax1.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax1.set_xlim(-0.5, len(sorted_layers) - 0.5)

    # Dynamically set y-axis range, supporting negative values
    y_min_sorted = min(cumulative_percentage.min(), 0)
    y_max_sorted = max(cumulative_percentage.max(), 105)
    # Add margin for both positive and negative values
    y_min_margin = y_min_sorted * 1.1 if y_min_sorted < 0 else y_min_sorted - abs(y_max_sorted) * 0.05
    y_max_margin = y_max_sorted * 1.05
    ax1.set_ylim(y_min_margin, y_max_margin)

    # Beautify borders
    for spine in ax1.spines.values():
        spine.set_linewidth(1.3)

    # Add statistics info box (placed in right-center to avoid curve overlap)
    if threshold_layers:
        textstr = 'Key Milestones:\n'
        for threshold, n_layers, critical_layer in threshold_layers:
            textstr += f'• {threshold}%: Top {n_layers} layers\n'
        textstr += f'\nEarly layers (0-{early_layers_end}): {early_contrib_pct:.1f}%'
    else:
        textstr = f'Early layers (0-{early_layers_end}): {early_contrib_pct:.1f}%'

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='gray', linewidth=1.5)
    # Dynamically adjust text box position based on number of layers (bottom-right for many layers, center-right for fewer)
    text_y_pos = 0.25 if len(sorted_layers) > 30 else 0.45
    ax1.text(0.98, text_y_pos, textstr, transform=ax1.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='right', bbox=props, fontweight='bold')

    plt.tight_layout()

    # Save Plot 1
    outfile1_png = outdir / f'{model}_cumulative_contribution_sorted.png'
    outfile1_pdf = outdir / f'{model}_cumulative_contribution_sorted.pdf'
    fig1.savefig(outfile1_png, dpi=400, bbox_inches='tight')
    fig1.savefig(outfile1_pdf, bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ Generated: {outfile1_png.name}")

    # ========== Plot 2: Cumulative curve in layer sequence order ==========
    fig2, ax2 = plt.subplots(figsize=(10, 7), dpi=300)

    ax2.plot(layers, sequential_percentage, '-o',
            linewidth=3.5, markersize=9, color='#3498db',
            markerfacecolor='white', markeredgewidth=2.5,
            label='Sequential Cumulative', zorder=3)

    # Fill area
    ax2.fill_between(layers, 0, sequential_percentage,
                    alpha=0.25, color='#3498db', hatch='\\\\\\',
                    edgecolor='#3498db', linewidth=1.0, zorder=1)

    # Mark early layers (Layer 0-3) cumulative contribution
    ax2.axvline(early_layers_end, color='#e74c3c', linestyle='--',
               linewidth=2.0, alpha=0.7, label=f'Layer 0-{early_layers_end}', zorder=2)
    ax2.axhline(early_contrib_pct, color='#e74c3c', linestyle=':',
               linewidth=1.5, alpha=0.5, zorder=2)
    ax2.scatter([early_layers_end], [early_contrib_pct], s=200, color='#e74c3c',
               edgecolors='white', linewidth=2, zorder=4, marker='D')

    # Dynamically adjust annotation position based on number of layers and cumulative contribution value
    # If early_contrib_pct is near or exceeds y-axis upper limit, shift downward (place below point)
    # Otherwise shift upward (place above point)
    text_fontsize = 10 if len(layers) > 30 else 11
    if early_contrib_pct > 180:  # Very high contribution, place annotation below
        text_offset_y = -15
        va_align = 'top'
    else:  # Normal case, place annotation above
        text_offset_y = 8 if len(layers) > 30 else 3
        va_align = 'bottom'

    ax2.text(early_layers_end, early_contrib_pct + text_offset_y,
            f'Early Layers\n{early_contrib_pct:.1f}%',
            fontsize=text_fontsize, color='#e74c3c', fontweight='bold',
            ha='center', va=va_align, bbox=dict(boxstyle='round,pad=0.5',
                                  facecolor='yellow', alpha=0.8, edgecolor='#e74c3c'))

    # Add zero-axis reference line
    ax2.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

    ax2.set_xlabel('Layer Index (Sequential)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative Contribution (%)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', framealpha=0.95, fontsize=11,
              edgecolor='gray', fancybox=True, shadow=True)
    ax2.grid(alpha=0.3, linestyle='--', linewidth=0.8)

    # Adjust x-axis range
    x_min = layers.min()
    x_max = layers.max()
    x_range = x_max - x_min
    ax2.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    # Beautify borders
    for spine in ax2.spines.values():
        spine.set_linewidth(1.3)

    plt.tight_layout()

    # Save Plot 2
    outfile2_png = outdir / f'{model}_cumulative_contribution_sequential.png'
    outfile2_pdf = outdir / f'{model}_cumulative_contribution_sequential.pdf'
    fig2.savefig(outfile2_png, dpi=400, bbox_inches='tight')
    fig2.savefig(outfile2_pdf, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Generated: {outfile2_png.name}")

    # Return key statistics
    return {
        'threshold_50_layers': threshold_layers[0][1] if len(threshold_layers) > 0 else None,
        'threshold_70_layers': threshold_layers[1][1] if len(threshold_layers) > 1 else None,
        'threshold_90_layers': threshold_layers[2][1] if len(threshold_layers) > 2 else None,
        'early_layers_contribution': float(early_contrib_pct)
    }


def plot_suppression_effect(data, outdir):
    """
    Generate suppression/promotion effect comparison plot
    - X-axis: Layer index
    - Y-axis: Suppression/promotion effect (change relative to baseline final value)
    - Zero-axis centered, positive=promotion, negative=suppression
    """
    from scipy.interpolate import make_interp_spline

    model = data['model']
    layers = data['layers']
    disabled_values = data['disabled_values']
    baseline_values = data['baseline_values']

    # Determine baseline final value (use second-to-last layer as last layer might be output layer)
    # Filter out outliers (don't use last layer if it's too small)
    valid_baseline_values = baseline_values[:-1] if baseline_values[-1] < baseline_values[-2] * 0.5 else baseline_values
    baseline_final = valid_baseline_values.max()

    # Calculate suppression/promotion effect of each layer
    # effect = baseline_final - disabled_value
    # Positive: MA decreases after disabling layer, indicating layer has promotion effect
    # Negative: MA increases after disabling layer, indicating layer has suppression effect
    effects = baseline_final - disabled_values

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

    # Create smooth curve
    if len(layers) > 3:
        layers_smooth = np.linspace(layers.min(), layers.max(), 300)
        try:
            # Check for invalid values
            if np.any(~np.isfinite(effects)):
                print(f"Warning: effects contains non-finite values, skipping interpolation")
                layers_smooth = layers
                effects_smooth = effects
            else:
                spl_effects = make_interp_spline(layers, effects, k=3)
                effects_smooth = spl_effects(layers_smooth)
                # Verify interpolation result
                if np.any(~np.isfinite(effects_smooth)):
                    print(f"Warning: interpolation produced non-finite values, using original data")
                    layers_smooth = layers
                    effects_smooth = effects
        except Exception as e:
            print(f"Warning: interpolation failed ({e}), using original data")
            layers_smooth = layers
            effects_smooth = effects
    else:
        layers_smooth = layers
        effects_smooth = effects

    # Plot main curve
    ax.plot(layers_smooth, effects_smooth, '-', color='#3498db',
           linewidth=3.5, label='Suppression/Promotion Effect', alpha=0.95, zorder=3)

    # Add markers on original data points
    colors = ['#2ecc71' if e > 0 else '#e74c3c' for e in effects]
    ax.scatter(layers, effects, s=120, c=colors,
              edgecolors='white', linewidth=2.0, zorder=4, alpha=0.95)

    # Plot zero-axis reference line
    ax.axhline(0, color='black', linestyle='-', linewidth=2.0, alpha=0.8, zorder=2)

    # Fill positive/negative regions (with diagonal hatching)
    # Promotion region (positive, green)
    ax.fill_between(layers_smooth, 0, effects_smooth,
                    where=(effects_smooth > 0),
                    color='#2ecc71', alpha=0.25, hatch='///',
                    edgecolor='#2ecc71', linewidth=1.0,
                    label='Promotion (Layer contributes to MA)', zorder=1)

    # Suppression region (negative, red)
    ax.fill_between(layers_smooth, 0, effects_smooth,
                    where=(effects_smooth <= 0),
                    color='#e74c3c', alpha=0.25, hatch='\\\\\\',
                    edgecolor='#e74c3c', linewidth=1.0,
                    label='Suppression (Layer inhibits MA)', zorder=1)

    # Adjust y-axis range based on actual data range (not enforcing symmetry)
    y_min_data = effects.min()
    y_max_data = effects.max()
    y_range = y_max_data - y_min_data
    # Add 30% margin to ensure all data points are fully within plot with ample space
    y_margin = y_range * 0.30
    ax.set_ylim(y_min_data - y_margin, y_max_data + y_margin)

    # Adjust x-axis range
    x_min = layers.min()
    x_max = layers.max()
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    # Add grid, emphasize zero-axis
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.grid(axis='x', alpha=0.2, linestyle=':', linewidth=0.6)

    # Labels
    ax.set_xlabel('Layer Index', fontsize=15, fontweight='bold')
    ax.set_ylabel('Effect on MA (Baseline - Disabled)', fontsize=15, fontweight='bold')

    # Legend
    ax.legend(loc='best', framealpha=0.95, fontsize=12,
             edgecolor='gray', fancybox=True, shadow=True)

    # Beautify borders
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)

    # Statistics info box removed (per user request)
    # n_promotion = (effects > 0).sum()
    # n_suppression = (effects <= 0).sum()
    # avg_promotion = effects[effects > 0].mean() if n_promotion > 0 else 0
    # avg_suppression = effects[effects <= 0].mean() if n_suppression > 0 else 0
    #
    # textstr = f'Promotion Layers: {n_promotion}\n'
    # textstr += f'Suppression Layers: {n_suppression}\n'
    # textstr += f'Avg Promotion: {avg_promotion:.1f}\n'
    # textstr += f'Avg Suppression: {avg_suppression:.1f}'
    #
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='gray', linewidth=1.5)
    # ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=11,
    #        verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # Save
    outfile_png = outdir / f'{model}_exp2_suppression_effect.png'
    outfile_pdf = outdir / f'{model}_exp2_suppression_effect.pdf'
    fig.savefig(outfile_png, dpi=400, bbox_inches='tight')
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Generated suppression effect plot: {outfile_png.name}")


def plot_2d_comparison(data, outdir):
    """Generate traditional 2D comparison plot (optimized version)"""
    from scipy.interpolate import make_interp_spline

    model = data['model']
    layers = data['layers']
    disabled_values = data['disabled_values']
    baseline_values = data['baseline_values']

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    # Create smooth curves (using spline interpolation)
    if len(layers) > 3:
        # Generate more points for smoothing
        layers_smooth = np.linspace(layers.min(), layers.max(), 300)

        try:
            # Check data validity
            if np.any(~np.isfinite(disabled_values)) or np.any(~np.isfinite(baseline_values)):
                print(f"Warning: data contains non-finite values, skipping interpolation")
                layers_smooth = layers
                disabled_smooth = disabled_values
                baseline_smooth = baseline_values
            else:
                # Smooth disabled values curve
                spl_disabled = make_interp_spline(layers, disabled_values, k=3)
                disabled_smooth = spl_disabled(layers_smooth)

                # Smooth baseline curve
                spl_baseline = make_interp_spline(layers, baseline_values, k=3)
                baseline_smooth = spl_baseline(layers_smooth)

                # Verify results
                if np.any(~np.isfinite(disabled_smooth)) or np.any(~np.isfinite(baseline_smooth)):
                    print(f"Warning: interpolation produced non-finite values, using original data")
                    layers_smooth = layers
                    disabled_smooth = disabled_values
                    baseline_smooth = baseline_values
        except Exception as e:
            print(f"Warning: interpolation failed ({e}), using original data")
            layers_smooth = layers
            disabled_smooth = disabled_values
            baseline_smooth = baseline_values
    else:
        layers_smooth = layers
        disabled_smooth = disabled_values
        baseline_smooth = baseline_values

    # Plot baseline curve (MA value for each layer during normal operation, showing layer-by-layer accumulation)
    ax.plot(layers_smooth, baseline_smooth, '-', color='#2ecc71',
           linewidth=3.0, label='Baseline (All MLP Active)', alpha=0.95, zorder=2)

    # Add markers on baseline original data points
    ax.scatter(layers, baseline_values, s=80, color='#2ecc71',
              edgecolors='white', linewidth=1.5, zorder=3, alpha=0.9)

    # Plot disabled values curve (final MA value after disabling a layer)
    ax.plot(layers_smooth, disabled_smooth, '-', color='#e74c3c',
           linewidth=3.0, label='Layer Disabled (Final MA)', alpha=0.95, zorder=2)

    # Add markers on disabled values original data points
    ax.scatter(layers, disabled_values, s=80, color='#e74c3c',
              edgecolors='white', linewidth=1.5, zorder=3, alpha=0.9)

    # Fill area to show difference (add diagonal hatching to enhance visibility)
    ax.fill_between(layers_smooth, baseline_smooth, disabled_smooth,
                    where=(disabled_smooth > baseline_smooth),
                    color='#e74c3c', alpha=0.25, hatch='///',
                    edgecolor='#e74c3c', linewidth=1.0,
                    label='MA Increase when Layer Disabled', zorder=0)

    ax.fill_between(layers_smooth, baseline_smooth, disabled_smooth,
                    where=(disabled_smooth <= baseline_smooth),
                    color='#2ecc71', alpha=0.25, hatch='\\\\\\',
                    edgecolor='#2ecc71', linewidth=1.0, zorder=0)

    # Adjust y-axis range to ensure all data points and curves are fully displayed
    all_values = np.concatenate([disabled_values, baseline_values, disabled_smooth, baseline_smooth])
    y_min = all_values.min()
    y_max = all_values.max()
    y_range = y_max - y_min

    # Increase top and bottom margins to ensure curves are completely within frame
    ax.set_ylim(y_min - y_range * 0.15, y_max + y_range * 0.15)

    # Adjust x-axis range to ensure all points are fully displayed
    x_min = layers.min()
    x_max = layers.max()
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.05, x_max + x_range * 0.05)

    ax.set_xlabel('Layer Index', fontsize=14, fontweight='bold')
    ax.set_ylabel('MA Value (Top1)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', framealpha=0.95, fontsize=12,
             edgecolor='gray', fancybox=True, shadow=True)
    ax.grid(alpha=0.25, linestyle='-', linewidth=0.5)

    # Beautify borders
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    plt.tight_layout()

    outfile_png = outdir / f'{model}_exp2_2d_comparison.png'
    outfile_pdf = outdir / f'{model}_exp2_2d_comparison.pdf'
    fig.savefig(outfile_png, dpi=400, bbox_inches='tight')
    fig.savefig(outfile_pdf, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Generated 2D comparison: {outfile_png.name}")


def process_model(model_name, base_results_dir, base_output_dir, enable_3d=True, style_3d='both'):
    """Process a single model and generate comparison plots"""
    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"{'='*60}")

    # Create output directory
    outdir = Path(base_output_dir) / 'exp2_figures' / model_name
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        # Load data
        data = load_exp2_data(model_name, base_results_dir)

        # Generate cumulative contribution curves
        cumulative_stats = plot_cumulative_contribution(data, outdir)

        # Generate suppression/promotion effect plot
        plot_suppression_effect(data, outdir)

        # Generate 2D comparison plot
        plot_2d_comparison(data, outdir)

        # Generate 3D plot (optional)
        if enable_3d:
            plot_3d_comparison_surface(data, outdir, style=style_3d)

        # Print cumulative contribution statistics
        if cumulative_stats:
            print(f"   📊 50% contribution: Top {cumulative_stats['threshold_50_layers']} layers")
            print(f"   📊 Early layers (0-3): {cumulative_stats['early_layers_contribution']:.1f}%")

        print(f"✅ {model_name}: All figures generated successfully!")
        return True

    except Exception as e:
        print(f"❌ {model_name}: Error - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Exp2 comparison visualizations')
    parser.add_argument('--model', type=str, help='Generate for a single model (e.g., gpt2)')
    parser.add_argument('--all', action='store_true', help='Generate for all models')
    parser.add_argument('--no-3d', action='store_true', help='Disable 3D plots')
    parser.add_argument('--style-3d', type=str, default='both', dest='style_3d',
                       choices=['surface', 'scatter', 'both'],
                       help='3D plot style: surface, scatter, or both')
    args = parser.parse_args()

    # Configuration
    BASE_RESULTS_DIR = 'results/models'
    BASE_OUTPUT_DIR = 'results/plot_results'

    ALL_MODELS = [
        'gpt2',
        'gptj_6b',
        'bloom_7b1',
        'falcon_7b',
        'opt_7b',
        'mistral_7b_v03',
        'qwen2.5_7b',
        'llama2_13b',
    ]

    # Determine models to process
    if args.model:
        MODELS = [args.model]
        print(f"\n🎯 Mode: Single model ({args.model})")
    elif args.all:
        MODELS = ALL_MODELS
        print(f"\n🎯 Mode: All models")
    else:
        MODELS = ['gpt2']
        print(f"\n🎯 Mode: Sample (gpt2 only)")
        print("💡 Use --all to generate all models")

    setup_style()

    enable_3d = not args.no_3d

    print("\n" + "="*60)
    print("🎨 Generating Exp2 Comparison Visualizations")
    print("="*60)
    print(f"Models to process: {len(MODELS)}")
    print(f"2D plots: ✅ Enabled")
    print(f"3D plots: {'✅ Enabled' if enable_3d else '❌ Disabled'}")
    if enable_3d:
        print(f"3D style: {args.style_3d}")
    print("="*60)

    success_count = 0
    for model in MODELS:
        if process_model(model, BASE_RESULTS_DIR, BASE_OUTPUT_DIR,
                        enable_3d=enable_3d, style_3d=args.style_3d):
            success_count += 1

    print("\n" + "="*60)
    print(f"✅ Summary: {success_count}/{len(MODELS)} models processed successfully")
    print(f"📁 Output directory: {BASE_OUTPUT_DIR}/exp2_figures/")
    print("="*60)


if __name__ == '__main__':
    main()
