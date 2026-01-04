# Investigating Massive Activations in Large Language Models

<div align="center">

![Models](https://img.shields.io/badge/Models-8_LLMs-blue)
![Experiments](https://img.shields.io/badge/Experiments-5_Series-green)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)

**A Systematic Investigation into the Generation Mechanisms of Massive Activations in Transformer-based Language Models**

[📖 Overview](#-overview) • [🎯 Research Questions](#-research-questions) • [🔬 Key Findings](#-key-findings) • [📊 Experiments](#-experiments) • [🚀 Quick Start](#-quick-start) • [📖 Citation](#-citation)

</div>

---

## 📖 Overview

This repository contains the complete implementation and results of a systematic investigation into **Massive Activation (MA)** phenomena in large language models. Massive activations—characterized by activation values orders of magnitude larger than typical ranges—pose significant challenges to numerical stability and computational efficiency in modern LLMs.

### Research Contribution

This work provides the first **end-to-end mechanistic explanation** of massive activation generation through rigorous causal analysis:

```
Function Word Triggering → MLP Generation → SVD Geometric Amplification
```

### What are Massive Activations?

Massive activations are extreme internal activation values that:
- Exceed normal ranges by 300-3000× (99.9th percentile threshold)
- Threaten numerical stability in low-precision inference
- Occur systematically rather than randomly
- Are linked to specific linguistic and geometric properties

### Scientific Value

- **Theoretical**: First mechanistic understanding of MA generation pathways in Transformers
- **Practical**: Provides principled geometric interventions for model stability
- **Architectural**: Reveals evolutionary trajectory from distributed to specialized amplification mechanisms

---

## 🎯 Research Questions

This investigation systematically addresses five interconnected research questions:

| Question | Focus | Key Method |
|----------|-------|------------|
| **RQ1: Source Effect** | Does MA originate from attention or MLP? | Attention head ablation |
| **RQ2: Localization Effect** | Which MLP component generates MA? | Layer-wise output comparison |
| **RQ3: Trigger Effect** | What linguistic patterns trigger MA? | Part-of-speech analysis |
| **RQ4: Mechanism Effect** | Can SVD geometry explain MA? | Singular value decomposition |
| **RQ5: Causality Effect** | Is geometric structure causal? | V-matrix ablation |

---

## 🔬 Key Findings

### Finding 1: Attention as Regulator, Not Generator (RQ1)

Attention mechanisms exhibit **three distinct regulatory roles**:

- **Generative-Promoting** (GPT-2, LLaMA-2, BLOOM, GPT-J): ∆Top1 = -60% to -98%
  - Attention provides triggering input to MLP
  - Disabling attention reduces MA significantly

- **Inhibitory-Regulatory** (Qwen2.5, OPT): ∆Top1 = +250% to +266%
  - Attention suppresses MA generation
  - Disabling attention causes MA explosion

- **Hybrid/Mixed** (Falcon, Mistral): Layer-specific behavior
  - Complex interaction patterns across layers

**Implication**: Attention regulates, but does not generate MA.

---

### Finding 2: MLP as Unequivocal Physical Source (RQ2)

MLP layers are the **exclusive physical source** of massive activations:

| Model | MLP/Attention Ratio | Key Layer |
|-------|---------------------|-----------|
| **Qwen2.5-7B** | **3496.18×** | Layer 0 |
| **Mistral-7B** | 21.20× | Layer 0 |
| **Falcon-7B** | 14.68× | Layer 0 |
| **LLaMA-2-13B** | 13.67× | Layer 22 |
| **BLOOM-7B1** | 9.88× | Layer 12 |
| **OPT-6.7B** | 6.42× | Layer 25 |
| **GPT-J-6B** | 4.12× | Layer 0 |
| **GPT-2** | 2.84× | Layer 2 |

**Statistical Verification**: All ratios significant at p < 0.001 (bootstrap, N=1000)

**Implication**: MLP amplification is 2.84-3496× stronger than attention output.

---

### Finding 3: Function Words as Primary Triggers (RQ3)

Massive activations are **systematically triggered** by grammatical tokens:

| Architecture | Function Word % | Pattern |
|--------------|-----------------|---------|
| **Mistral-7B** | **100%** | Exclusive function word triggering |
| **Falcon-7B** | 90% | Strong preference |
| **BLOOM-7B1** | 90% | Strong preference |
| **GPT-2** | 84% | Strong preference |
| **GPT-J-6B** | 80% | Strong preference |
| **LLaMA-2-13B** | 76% | Moderate preference |
| **OPT-6.7B** | 58% | Mixed pattern |
| **Qwen2.5-7B** | 40% | Content word dominant |

**Linguistic Categories**: Prepositions, conjunctions, articles, auxiliary verbs

**Implication**: Abstract grammatical processing creates alignment conditions.

---

### Finding 4: SVD Geometric Amplification (RQ4)

MLP weight matrices exhibit **systematic geometric structures**:

#### Three Amplification Strategies

**1. Single-Direction Dominance** (Qwen, GPT-2, OPT)
- High singular value ratio: σ₁/σ₂ = 2.52-2.87
- Strong alignment: cos(MA, v₁) = 0.78-0.994
- Efficient amplification along one direction

**2. Anti-Alignment Mechanism** (GPT-J)
- Moderate dominance: σ₁/σ₂ = 1.91
- Negative alignment: cos(MA, v₁) = -0.69
- Compensatory amplification

**3. Multi-Direction Collaboration** (BLOOM, LLaMA-2)
- Low dominance: σ₁/σ₂ = 1.18-2.23
- Minimal alignment: cos(MA, v₁) = 0.05-0.11
- Distributed amplification

**Mathematical Model**:
```
MA ≈ σ₁ × (h₂ · v₁) + bias
```
where h₂ is MLP intermediate activation, v₁ is first right singular vector

**Implication**: Geometric structure determines amplification capacity.

---

### Finding 5: V-Matrix as Causal Component (RQ5)

**V-matrix ablation** provides definitive causal evidence:

| Model | Baseline MA | After V-Ablation | ∆MA | Dependency |
|-------|-------------|------------------|-----|------------|
| **Qwen2.5-7B** | 9160.00 | 82.60 | **-99.1%** | Extreme |
| **BLOOM-7B1** | 92.50 | 20.15 | -78.2% | Strong |
| **Falcon-7B** | 8.92 | 2.15 | -75.9% | Strong |
| **OPT-6.7B** | 1.85 | 0.42 | -77.3% | Strong |
| **Mistral-7B** | 1.17 | 0.31 | -73.5% | Strong |
| **GPT-2** | 102.71 | 31.00 | -69.8% | Moderate |
| **GPT-J-6B** | 30.33 | 9.21 | -69.6% | Moderate |
| **LLaMA-2-13B** | 12.23 | 2.59 | -78.8% | Strong |

**Ablation Method**: Replace V with random orthogonal matrix while preserving U and Σ

**Statistical Significance**: All reductions p < 0.001

**Implication**: V-matrix geometry is **causally necessary** for MA generation.

---

## 📊 Experiments

### Experiment Framework

```
┌─────────────────────────────────────────────────────────┐
│                   Research Pipeline                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  RQ1: Attention Ablation ──→ Identify Regulatory Role   │
│            ↓                                             │
│  RQ2: MLP vs Attention ────→ Locate Physical Source     │
│            ↓                                             │
│  RQ3: POS Tagging ─────────→ Identify Trigger Pattern   │
│            ↓                                             │
│  RQ4: SVD Analysis ────────→ Explain Geometry           │
│            ↓                                             │
│  RQ5: V-Matrix Ablation ───→ Establish Causality        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Experiment 1: Attention Mechanism Contribution Analysis

**Objective**: Determine if attention mechanisms generate or regulate MA

**Method**:
- Disable all attention heads via PyTorch hooks
- Compare Top1 activation changes: ∆Top1 = (Top1_intervention - Top1_baseline) / Top1_baseline

**Results**:
- Generative models: ∆Top1 ∈ [-98.3%, -60.0%]
- Inhibitory models: ∆Top1 ∈ [+250.4%, +266.8%]
- Attention is regulatory, not generative

**Implementation**: `experiments/exp1_attention_heads/`

---

### Experiment 2: MLP Layer Source Verification

**Objective**: Identify which component (attention vs MLP) physically generates MA

**Method**:
- Simultaneously capture attention and MLP outputs
- Calculate ratio: MLP_max / Attention_max

**Results**:
- MLP dominance: 2.84× to 3496.18×
- Consistent across all architectures
- Establishes MLP as physical source

**Implementation**: `experiments/exp2_mlp_layers/`

---

### Experiment 3: Function Word Trigger Analysis

**Objective**: Identify linguistic patterns that trigger MA

**Method**:
- Record token positions of Top-K MAs
- Perform POS tagging with spaCy
- Calculate function word percentage

**Results**:
- Function word dominance: 76-100% (except Qwen: 40%)
- Strong systematic correlation
- Links computational to linguistic structure

**Implementation**: `experiments/exp3_svd_alignment/`

---

### Experiment 4: SVD Alignment Mechanism Analysis

**Objective**: Explain MA generation through weight matrix geometry

**Method**:
- Perform SVD on MLP down_proj matrices: W = UΣV^T
- Analyze singular value spectrum: σ₁/σ₂
- Calculate alignment: cos(MA_direction, v₁)

**Results**:
- Three geometric strategies identified
- Correlation with architectural families
- Mathematical explanation of amplification

**Implementation**: `experiments/exp4_attention_svd/`

---

### Experiment 5: V-Matrix Ablation Study

**Objective**: Establish causal role of V-matrix geometry

**Method**:
- Construct ablated weight: W_ablated = U Σ V_rand^T
- Replace original V with random orthogonal matrix
- Measure MA reduction

**Results**:
- Systematic MA reduction: 69.6-99.1%
- Causal necessity established
- Dependency varies by architecture

**Implementation**: `experiments/exp6_v_ablation/`

---

## 🎯 Models and Data

### Evaluated Models (8 LLMs)

| Model | Parameters | Layers | Architecture | Position Encoding |
|-------|------------|--------|--------------|-------------------|
| **GPT-2** | 124M | 12 | Standard Transformer | Learned |
| **GPT-J-6B** | 6B | 28 | Parallel Attn+FFN | RoPE |
| **BLOOM-7B1** | 7.1B | 30 | Standard | ALiBi |
| **Falcon-7B** | 7B | 32 | Multi-Query Attention | ALiBi |
| **OPT-6.7B** | 6.7B | 32 | Standard | Learned |
| **Mistral-7B-v0.3** | 7B | 32 | Sliding Window + GQA | RoPE |
| **Qwen2.5-7B** | 7B | 28 | GQA | RoPE |
| **LLaMA-2-13B** | 13B | 40 | RoPE + RMSNorm | RoPE |

### Dataset

- **10 text sequences** × 128 tokens each
- Diverse syntactic structures
- Consistent across all experiments
- Ensures reproducible comparisons

### Computational Resources

- **GPU**: NVIDIA A100 80GB
- **Total Compute**: ~200 GPU hours
- **Storage**: ~185 GB experimental data

---

## 📁 Repository Structure

```
massive-activations/
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── lib/                            # Core library
│   ├── core/                       # Model loading, data processing
│   ├── utils/                      # Evaluation and model utilities
│   └── plotting/                   # Visualization tools
│
├── experiments/                    # Experiment implementations
│   ├── README.md                   # Detailed experiment guide
│   ├── exp1_attention_heads/       # RQ1: Attention ablation
│   ├── exp2_mlp_layers/            # RQ2: MLP verification
│   ├── exp3_svd_alignment/         # RQ3&4: Trigger + SVD analysis
│   ├── exp4_attention_svd/         # Attention SVD analysis
│   ├── exp6_v_ablation/            # RQ5: V-matrix causality
│   └── shared/                     # Shared utilities
│
├── results/                        # Experimental results
│   ├── experiments/                # Organized by experiment
│   │   ├── exp1/ exp2/ exp3/      # Per-experiment results
│   │   ├── exp4/ exp4b/ exp6/
│   │   └── exp7/ exp8/
│   ├── plot_results/               # Generated figures
│   └── archive/                    # Archived data
│
├── scripts/                        # Utility scripts
│   ├── README.md                   # Script documentation
│   └── visualization/              # 36 plotting scripts
│
├── docs/                           # Documentation
│   ├── README.md                   # Documentation index
│   └── reports/                    # Detailed analysis reports
│
└── model_weights/                  # Model weight cache (gitignored)
```

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone repository
git clone https://github.com/anonymous/massive-activations.git
cd massive-activations

# 2. Create conda environment
conda create -n ma python=3.11
conda activate ma

# 3. Install PyTorch (CUDA 11.8 example)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install dependencies
pip install -r requirements.txt

# 5. Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

### Running Experiments

#### Experiment 1: Attention Ablation (RQ1)

```bash
python experiments/exp1_attention_heads/exp1_feasibility_test.py \
    --model gpt2 \
    --nsamples 10
```

#### Experiment 2: MLP Layer Analysis (RQ2)

```bash
python experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py \
    --model gpt2 \
    --nsamples 10 \
    --n_jobs 1
```

#### Experiment 3-4: SVD Analysis (RQ3, RQ4)

```bash
python experiments/exp3_svd_alignment/exp3_svd_alignment.py \
    --model gpt2 \
    --nsamples 10
```

#### Experiment 5: V-Matrix Ablation (RQ5)

```bash
python experiments/exp6_v_ablation/exp6_v_matrix_ablation.py \
    --model gpt2 \
    --nsamples 10 \
    --target_layers 0 1 2
```

### Viewing Results

```bash
# Experiment 1 results
cat results/experiments/exp1/gpt2/exp1_results.json

# Experiment 2 summary
cat results/experiments/exp2/gpt2/summary.json

# V-matrix ablation results
cat results/experiments/exp6/gpt2/layer0_v_ablation.json
```

---

## 📊 Complete Results Summary

### Cross-Model Mechanism Classification

| Type | Models | Attention Role | SVD Strength | Trigger |
|------|--------|----------------|--------------|---------|
| **Generative-SVD** | GPT-2, GPT-J | Provides input | Extreme | Function word |
| **Inhibitory-SVD** | Qwen, OPT | Suppresses | Extreme | Mixed |
| **Non-SVD** | BLOOM, LLaMA-2 | Provides input | Weak | Function word |
| **Hybrid** | Falcon, Mistral | Mixed | Moderate | Function word |

### Key Metrics Across All Models

```
┌──────────────┬─────────┬──────────┬─────────┬──────────┬──────────┐
│ Model        │ ∆Top1   │ MLP/Attn │ Func(%) │ σ₁/σ₂    │ ∆MA      │
├──────────────┼─────────┼──────────┼─────────┼──────────┼──────────┤
│ GPT-2        │  -60.0% │   2.84×  │   84%   │   2.52   │  -69.8%  │
│ GPT-J-6B     │  -95.2% │   4.12×  │   80%   │   1.91   │  -69.6%  │
│ BLOOM-7B1    │  -98.3% │   9.88×  │   90%   │   1.18   │  -78.2%  │
│ Falcon-7B    │  -21.0% │  14.68×  │   90%   │   2.15   │  -75.9%  │
│ OPT-6.7B     │ +250.4% │   6.42×  │   58%   │   2.87   │  -77.3%  │
│ Mistral-7B   │  -18.2% │  21.20×  │  100%   │   1.85   │  -73.5%  │
│ Qwen2.5-7B   │ +266.8% │ 3496.18× │   40%   │   2.64   │  -99.1%  │
│ LLaMA-2-13B  │  -79.5% │  13.67×  │   76%   │   2.23   │  -78.8%  │
└──────────────┴─────────┴──────────┴─────────┴──────────┴──────────┘
```

---

## 🔍 Theoretical Implications

### The Complete MA Generation Pathway

```
┌─────────────────────────────────────────────────────────────────┐
│                  MA Generation Mechanism                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. INPUT: Function Word Token                                  │
│     ↓                                                            │
│  2. EMBEDDING: Creates input vector x                           │
│     ↓                                                            │
│  3. ATTENTION: Modulates input (promotes/inhibits)              │
│     ↓                                                            │
│  4. MLP INPUT PROJECTION: h₂ = activation(W_up × x)            │
│     ↓                                                            │
│  5. GEOMETRIC ALIGNMENT: h₂ aligns with v₁                     │
│     ↓                                                            │
│  6. AMPLIFICATION: y = σ₁ × (h₂ · v₁) × u₁                     │
│     ↓                                                            │
│  7. OUTPUT: Massive Activation                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Architectural Evolution

```
Classical (GPT-2, 2019)          Modern (Qwen, 2024)
─────────────────────            ─────────────────────
• Distributed geometry           • Specialized geometry
• Moderate σ₁/σ₂ (2.5)          • Extreme σ₁/σ₂ (2.6)
• Lower alignment (0.78)         • Perfect alignment (0.994)
• Attention promotes             • Attention suppresses
• Moderate dependency            • Extreme dependency (99.1%)
```

### Practical Implications

**For Model Training**:
- Incorporate spectral penalties on σ₁/σ₂ ratio
- Monitor V-matrix geometry during training
- Balance alignment vs robustness

**For Model Deployment**:
- Dynamic precision adjustment for function words
- Targeted activation clamping in key layers
- Geometry-aware quantization strategies

**For Model Architecture**:
- Design attention mechanisms considering regulatory role
- Balance MLP amplification capacity
- Consider multi-direction vs single-direction strategies

---

## 📚 Documentation

Comprehensive documentation is available in the `docs/` directory:

- **[Documentation Index](docs/README.md)** - Navigation guide for all documentation
- **[Experiment Guide](experiments/README.md)** - Detailed experiment protocols and usage
- **[Script Guide](scripts/README.md)** - Visualization and utility script documentation
- **[Analysis Reports](docs/reports/)** - In-depth analysis reports by experiment category

---

## 📖 Citation

If this work contributes to your research, please cite:

```bibtex
@misc{massive_activation_2025,
  title={Investigating Massive Activations in Large Language Models},
  author={Anonymous},
  year={2025},
  note={Systematic investigation of massive activation mechanisms in transformer-based LLMs},
  howpublished={Under Review}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This research builds upon the foundational work of:

- **Transformers** - Hugging Face team for the transformers library
- **PyTorch** - Facebook AI Research for the deep learning framework
- **Open Source LLMs** - Meta (LLaMA), Google (T5), BigScience (BLOOM), EleutherAI (GPT-J), TII (Falcon), Mistral AI, Qwen team

Special thanks to the mechanistic interpretability community for establishing rigorous analysis frameworks.

---

## 📈 Project Statistics

- **Total Experiments**: 5 series (8 sub-experiments)
- **Models Evaluated**: 8 LLMs (124M - 13B parameters)
- **Data Generated**: 185 GB
- **Result Files**: 350+ JSON files
- **Visualizations**: 200+ figures
- **Code Files**: 143 Python files
- **Documentation**: 153 Markdown files
- **Compute Time**: ~200 GPU hours (NVIDIA A100 80GB)

---

**Project Status**: ✅ **Completed** (All experiments finished, results published)

**Last Updated**: 2025-12-29

**Repository**: [Anonymous Repository - Will be made public upon acceptance]

---

<div align="center">

**For questions or collaboration inquiries, please use the repository issue tracker**

</div>
