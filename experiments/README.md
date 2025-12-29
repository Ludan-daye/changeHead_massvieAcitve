# Experiments Directory

This directory contains all experimental code organized by experiment series.

## Directory Structure

```
experiments/
├── exp1_attention_heads/        # Experiment 1: Attention Head Ablation
├── exp2_mlp_layers/            # Experiment 2: MLP Layer Ablation
├── exp3_svd_alignment/         # Experiment 3: SVD Alignment Analysis
├── exp4_attention_svd/         # Experiment 4: Attention Matrix SVD
├── exp5_uv_interaction/        # Experiment 5: U-V Interaction Analysis
├── exp6_v_ablation/            # Experiment 6: V Matrix Ablation
├── exp7_direction_superposition/ # Experiment 7: Direction vs Magnitude
├── exp8_decomposed_attribution/ # Experiment 8: Decomposed Attribution
├── shared/                     # Shared utility scripts
├── research_questions/         # Research question experiments
├── llama/                      # LLaMA-specific experiments
└── opt/                        # OPT-specific experiments
```

## Experiment Series Overview

### Exp1: Attention Head Ablation
- **Goal**: Test if MA originates from attention mechanism
- **Method**: Disable all attention heads, measure MA changes
- **Scripts**: `exp1_feasibility_test.py`, `exp1_feasibility_test_optimized.py`

### Exp2: MLP Layer Ablation
- **Goal**: Identify which MLP layers contribute to MA
- **Method**: Disable each MLP layer individually, track MA
- **Scripts**: `exp2b_mlp_layer_ablation.py`, `run_exp2_with_memory_check.py`

### Exp3: SVD Alignment Analysis
- **Goal**: Analyze SVD decomposition of MLP weight matrices
- **Method**: Compare U/V singular vector directions with MA
- **Scripts**: `exp3_svd_alignment_analysis.py`, `exp3_u_ablation.py`

### Exp4: Attention Matrix SVD
- **Goal**: Analyze attention weight matrices via SVD
- **Method**: Decompose attention weights, measure alignment
- **Scripts**: `exp4_attention_svd.py`, `exp4_mlp_svd_analysis.py`

### Exp5: U-V Interaction Analysis
- **Goal**: Study interaction between U and V matrices in MLP
- **Method**: Ablation studies on U/V matrices separately
- **Scripts**: `exp5_uv_interaction.py`, `exp5_multi_model_mlp_svd.py`

### Exp6: V Matrix Ablation
- **Goal**: Test V matrix contribution to MA generation
- **Method**: Keep/remove top-k V singular vectors
- **Scripts**: `exp6_v_ablation.py`, `exp6_v_ablation_simple.py`

### Exp7: Direction vs Magnitude Superposition
- **Goal**: Decompose MA into direction and magnitude components
- **Method**: Ablate direction/magnitude separately
- **Scripts**: `exp7_direction_superposition.py`

### Exp8: Decomposed Attribution
- **Goal**: Fine-grained attribution of MA sources
- **Method**: Systematic component-wise ablation
- **Scripts**: `exp8_decomposed_attribution.py`

## Running Experiments

### Basic Usage

```bash
# Run a specific experiment
python experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py --model MODEL_NAME

# Run with memory optimization
python experiments/exp2_mlp_layers/run_exp2_with_memory_check.py --model MODEL_NAME
```

### Supported Models
- GPT-2
- GPT-J-6B
- BLOOM-7B1
- Falcon-7B
- OPT-6.7B
- Mistral-7B
- Qwen2.5-7B
- LLaMA2-13B

### Common Parameters
- `--model`: Model name (required)
- `--layer`: Specific layer to test (optional)
- `--output_dir`: Output directory for results (default: `results/experiments/`)

## Shared Utilities

The `shared/` directory contains common utilities:
- `analyze_heads_simple.py` - Attention head analysis
- `prune_attention_heads.py` - Head pruning utilities
- `main_llm.py`, `main_vit.py` - Main entry points for different models

## Model-Specific Code

### LLaMA (`llama/`)
Contains LLaMA-specific optimization and experiment variants.

### OPT (`opt/`)
Contains OPT-specific optimization and experiment variants.

## Results

Experimental results are stored in:
```
results/experiments/
├── exp1/MODEL_NAME/
├── exp2/MODEL_NAME/
└── ...
```

See `results/experiments/` for detailed result structure.

## Contributing

When adding new experiments:
1. Create a new directory following the naming pattern `expN_description/`
2. Add experiment scripts with descriptive names
3. Update this README with experiment overview
4. Document parameters and expected outputs

---

For more details on specific experiments, see the documentation in `docs/reports/`.
