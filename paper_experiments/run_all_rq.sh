#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch
export HF_ENDPOINT=https://hf-mirror.com
cd ~/ludan/massActive/paper_experiments

MODELS="qwen2.5_7b mistral_7b falcon_7b bloom_7b gptj_6b llama2_7b_chat"

echo "============================================"
echo "$(date): Starting RQ2-RQ5 for all models (using peak MA layer)"
echo "============================================"

for model in $MODELS; do

    echo ""
    echo "########################################"
    echo "$(date): MODEL = $model"
    echo "########################################"

    # Get peak MA layer from RQ1 baseline results
    KEY_LAYER=$(python3 -c "
import json
d = json.load(open('results/wikitext_run/RQ1/${model}/table1_rq1.json'))
print(d['key_layer'])
" 2>/dev/null)

    if [ -z "$KEY_LAYER" ]; then
        echo "ERROR: Could not determine key_layer for $model from RQ1 results!"
        echo "Skipping $model..."
        continue
    fi
    echo "$(date): [$model] Using peak MA layer = $KEY_LAYER"

    # RQ2a: MLP feasibility test (layer-independent, tests all layers)
    echo "$(date): [$model] RQ2a starting..."
    python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
        --model $model --dataset wikitext --nsamples 30 \
        --savedir results/wikitext_run/RQ2a/$model 2>&1 | tail -5
    echo "$(date): [$model] RQ2a done"

    # RQ4: SVD alignment analysis (at peak MA layer)
    echo "$(date): [$model] RQ4 starting (layer=$KEY_LAYER)..."
    python RQ4_svd_alignment/exp3_svd_alignment_analysis.py \
        --model $model --layer_id $KEY_LAYER --nsamples 30 \
        --savedir results/wikitext_run/RQ4/$model 2>&1 | tail -5
    echo "$(date): [$model] RQ4 done"

    # RQ5: V-matrix ablation (at peak MA layer)
    echo "$(date): [$model] RQ5 starting (layer=$KEY_LAYER)..."
    python RQ5_v_matrix_ablation/exp5_v_ablation.py \
        --model $model --layer_id $KEY_LAYER --nsamples 30 \
        --savedir results/wikitext_run/RQ5/$model 2>&1 | tail -5
    echo "$(date): [$model] RQ5 done"

    # RQ3: Function words SVD mapping (at peak MA layer)
    echo "$(date): [$model] RQ3 starting (layer=$KEY_LAYER)..."
    python RQ3_function_words/exp5_function_words_svd_mapping.py \
        --model $model --layer_id $KEY_LAYER --nsamples 30 \
        --savedir results/wikitext_run/RQ3/$model 2>&1 | tail -5
    echo "$(date): [$model] RQ3 done"

    echo "$(date): [$model] ALL RQs COMPLETE"
done

echo ""
echo "============================================"
echo "$(date): ALL MODELS ALL RQs DONE"
echo "============================================"
