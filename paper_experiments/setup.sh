#!/bin/bash
# Setup script for paper experiments
# Usage: bash setup.sh

set -e

echo "=== Setting up environment for MA experiments ==="

# 1. Create conda environment
echo "[1/4] Creating conda environment..."
conda create -n massive-activations python=3.9 -y
eval "$(conda shell.bash hook)"
conda activate massive-activations

# 2. Install dependencies
echo "[2/4] Installing dependencies..."
pip install -r requirements.txt

# 3. Install spaCy English model (for RQ3 function word analysis)
echo "[3/4] Installing spaCy English model..."
python -m spacy download en_core_web_sm

# 4. Configure model cache
echo "[4/4] Configuring model cache..."
if [ ! -f config.py ]; then
    echo "Please edit config.py to set your model cache directory."
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "Usage examples:"
echo "  # RQ1: Attention contribution analysis"
echo "  python RQ1_attention_contribution/exp1_feasibility_test.py --model gpt2 --savedir results/RQ1"
echo ""
echo "  # RQ2: MLP source verification"
echo "  python RQ2_mlp_source/exp2a_mlp_feasibility_test.py --model gpt2 --savedir results/RQ2"
echo ""
echo "  # RQ4: SVD alignment analysis"
echo "  python RQ4_svd_alignment/exp3_svd_alignment_analysis.py --model gpt2 --savedir results/RQ4"
echo ""
echo "  # Via main entry point"
echo "  python main_llm.py --model gpt2 --exp1"
