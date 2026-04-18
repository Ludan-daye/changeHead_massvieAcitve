# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating **Massive Activations (MAs)** — extreme activation values (300–3000× median) in Large Language Models and Vision Transformers. The central finding is that MAs originate from MLP down-projection matrices (not attention heads) and are triggered by function words (the, and, of, etc.).

## Repository Structure

- `changeHead_massvieAcitve/` — Core research codebase (git repo, experiments, models, analysis)
- `figures/` — All experiment output figures (PNG/JPG) and figure directories (exp1–7, combined)
- `archives/` — ZIP archives of figure collections
- `paper/` — Paper PDFs, rebuttal drafts, method.md, Chinese writing notes (初稿, 公式, 论文)
- `scripts/` — Top-level visualization scripts (`generate_*.py`)

### Core Codebase (`changeHead_massvieAcitve/`)

**Entry points:**
- `main_llm.py` — LLM analysis with `--exp1` through `--exp4` flags
- `main_vit.py` — ViT analysis with `--exp1` through `--exp3` flags

**Experiment scripts:**
- `exp1_feasibility_test.py` — Attention heads disabled → proves heads don't generate MAs
- `exp2a_mlp_feasibility_test.py` — MLP disabled → proves MLPs are the source
- `exp2c_mlp_internal_analysis.py` — Up vs down projection analysis
- `exp3_svd_alignment_analysis.py` — SVD decomposition proving geometric causality (R²=0.89–0.97)
- `exp5_function_words_svd_mapping.py` — Function word → singular vector alignment analysis

**Library (`lib/`):**
- `load_model.py` — Unified loader for 30+ LLM/ViT variants; returns model, tokenizer, device, layers
- `model_dict.py` — Model registry with HuggingFace IDs and cache dirs (update `CACHE_DIR_BASE` for local paths)
- `hook.py` — Hook registration for capturing intermediate activations (h₁, h₂, attention weights)
- `load_data.py` — Data loading (WikiText, C4, PG19)
- `eval_utils.py` — Evaluation: `test_imagenet()`, `eval_ppl()`
- `plot_utils_llm.py` / `plot_utils_vit.py` — Visualization utilities

**Monkey-patching (`monkey_patch/`):**
Replaces layer forward methods to capture intermediate states without altering computation. Supports GPT-2, LLaMA, Mistral, Phi-2, ViT.

## Environment Setup

```bash
conda create -n massive-activations python=3.9
conda activate massive-activations
pip install torch>=2.0.0  # with CUDA
pip install transformers==4.36.0 timm==0.9.12 accelerate==0.23.0 datasets==2.14.5
pip install matplotlib==3.8.0 seaborn sentencepiece protobuf
```

## Running Experiments

```bash
cd changeHead_massvieAcitve

# LLM experiments (e.g., GPT-2, exp1 = 3D feature vis)
python main_llm.py --model gpt2 --exp1

# ViT experiments
python main_vit.py --model dinov2_vitb14 --exp1

# Standalone experiment scripts
python exp1_feasibility_test.py
python exp3_svd_alignment_analysis.py
```

## Data Flow

```
main_llm.py / main_vit.py
  → lib.load_llm/load_vit()       # loads model + tokenizer
  → monkey_patch.enable_*()        # injects hooks for intermediate capture
  → experiment script              # forward pass with feature capture
  → lib.plot_*()                   # generates visualizations
  → results/                       # output JSON + figures
```

## Supported Models

- **LLMs**: GPT-2, LLaMA-2 (7B/13B/70B), Mistral-7B, Phi-2, Falcon-7B/40B, MPT-7B/30B, OPT (125M–66B), Pythia
- **ViTs**: MAE (base/large/huge), CLIP, DINOv2, DINOv2-reg

## Key Mathematical Concepts

- **MA magnitude**: `Top₁ ≈ σ₁ · |h₂ᵀ·v₁| · max_j|(u₁)_j| + bias`
- **Spectral dominance ratio**: η = σ₁/σ₂ (how concentrated the spectrum is)
- **Geometric alignment**: cosine similarity between intermediate representation h₂ and principal right singular vector v₁
- Function words produce concentrated, stable projections onto v₁ → amplified by σ₁ → massive output

## Language

The codebase mixes English and Chinese. Some documentation files (公式.txt, 未命名.md, method.md annotations) are in Chinese.
