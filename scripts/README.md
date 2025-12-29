# Scripts Directory

This directory contains utility scripts for running experiments, generating visualizations, and project management.

## Directory Structure

```
scripts/
├── visualization/          # Visualization and plotting scripts (36 files)
├── runners/               # Experiment execution scripts
├── analysis/              # Analysis utilities
├── monitoring/            # Monitoring scripts
└── *.sh                   # Shell utility scripts
```

## Visualization Scripts (`visualization/`)

Contains all plotting and figure generation scripts:

### Experiment-Specific Plots
- `plot_exp1_*.py` - Experiment 1 visualizations
- `plot_exp2_*.py` - Experiment 2 visualizations
- `plot_exp3_*.py` - Experiment 3 visualizations
- etc.

### Combined Figures
- `merge_exp2_pdfs*.py` - Merge multiple PDF figures
- `exp*_combine_*.py` - Generate combined analysis plots
- `redraw_*.py` - Redraw figures from data

### Visualization Types
- **2D Comparisons**: `*_2d_comparison.py`
- **3D Visualizations**: `*_3d_*.py`
- **Heatmaps**: `*_heatmap*.py`
- **Energy Plots**: `*_energy*.py`

### Usage Example

```bash
# Generate visualizations for a specific experiment
python scripts/visualization/plot_exp2_all_models.py

# Merge PDF figures
python scripts/visualization/merge_exp2_pdfs_final.py

# Generate combined analysis
python scripts/visualization/redraw_exp2_combined.py
```

## Shell Scripts

### Experiment Runners
- `run_all_experiments.sh` - Run all experiments sequentially
- `run_exp2b_all_models.sh` - Run Experiment 2b for all models
- `run_attribution_experiments.sh` - Run attribution experiments

### Privacy and Cleanup
- `privacy_cleanup.sh` - Remove private information before submission
- `test_batch_system.sh` - Test batch processing system

### Usage

```bash
# Make script executable
chmod +x scripts/SCRIPT_NAME.sh

# Run script
./scripts/SCRIPT_NAME.sh
```

## Output Locations

### Figures
Generated figures are saved to:
- `results/plot_results/` - Main figure directory
- `results/plot_results/exp*_figures/` - Experiment-specific figures
- `results/plot_results/combined_figures_*/` - Combined figure sets

### Logs
Script logs are saved to:
- `logs/` - Main log directory
- `logs/root/` - Root-level script logs

## Best Practices

### For Visualization Scripts
1. Use consistent color schemes across experiments
2. Save both PNG (for preview) and PDF (for publication) formats
3. Use descriptive filenames: `{model}_{experiment}_{plot_type}.{ext}`
4. Set appropriate DPI (300+ for publication quality)

### For Runner Scripts
1. Add progress indicators
2. Log errors to files
3. Provide clear output messages
4. Handle CUDA out-of-memory gracefully

## Dependencies

Key libraries used in scripts:
- `matplotlib` - Plotting
- `numpy` - Numerical operations
- `pandas` - Data manipulation
- `PyMuPDF (fitz)` - PDF manipulation
- `PIL` - Image processing

See `requirements.txt` in project root for full dependency list.

## Adding New Scripts

When adding new scripts:
1. Place in appropriate subdirectory
2. Use clear, descriptive names
3. Add docstrings explaining purpose
4. Update this README with script description
5. Test with multiple models before committing

---

For experiment execution details, see `experiments/README.md`.
