# Documentation Index

This directory contains project documentation for the Massive Activation research.

## Directory Structure

### `/reports/` - Research Reports

- **`model_analysis/`** - Model-specific analysis
  - `OPT_6.7B_MECHANISM_ANALYSIS.md` - Detailed analysis of OPT-6.7B architecture

- **`README_STRUCTURED.md`** - Structured overview of all reports

## Quick Links

### For New Users
1. Start with the main project **[README.md](../README.md)** in the repository root
2. Review **[Experiment Guide](../experiments/README.md)** for experimental protocols
3. Check **[Script Documentation](../scripts/README.md)** for visualization tools

### For Researchers
- **Experimental Results**: See `results/experiments/` directory in repository root
- **Model Analysis**: `reports/model_analysis/OPT_6.7B_MECHANISM_ANALYSIS.md`
- **Methodology**: Detailed in main README and paper

### For Developers
- **Installation**: See main README.md for setup instructions
- **Code Structure**: Repository follows standard Python package structure
  - `lib/` - Core library (model loading, data processing, utilities)
  - `experiments/` - Experiment implementations (exp1-exp8)
  - `scripts/` - Visualization and analysis scripts
  - `results/` - Experimental data and figures

## Key Findings

All experimental findings are summarized in the main **[README.md](../README.md)**, organized around 5 research questions (RQ1-RQ5):

1. **RQ1**: Attention mechanisms as regulators (not generators)
2. **RQ2**: MLP layers as physical source of MA
3. **RQ3**: Function words as primary triggers
4. **RQ4**: SVD geometric amplification mechanism
5. **RQ5**: V-matrix causal necessity

## Data Organization

Experimental data is stored in `results/` directory:

```
results/
├── experiments/          # Organized by experiment (exp1-exp8)
│   ├── exp1/ exp2/ exp3/ exp4/ exp4b/
│   ├── exp6/ exp7/ exp8/
│   └── Each contains JSON results for all models
├── plot_results/         # Generated figures
└── archive/              # Archived data
```

---

**Last Updated**: 2026-01-03
**Project**: Investigating Massive Activations in Large Language Models
**Repository**: [Anonymous Repository - Will be made public upon acceptance]
