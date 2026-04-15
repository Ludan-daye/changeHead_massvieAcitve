#!/bin/bash
# Rerun RQ3/RQ4/RQ5 using each model's peak MA layer (from RQ1 results)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate torch
export HF_ENDPOINT=https://hf-mirror.com
cd ~/ludan/massActive/paper_experiments

MODELS="qwen2.5_7b mistral_7b falcon_7b bloom_7b gptj_6b llama2_7b_chat"
LOG=run_rq345_peak.log
echo "$(date): RQ3/4/5 PEAK LAYER RUN START" > $LOG

# Model -> peak MA layer mapping (from RQ1 baseline analysis)
declare -A PEAK_LAYERS=(
    [qwen2.5_7b]=16
    [mistral_7b]=19
    [falcon_7b]=23
    [bloom_7b]=13
    [gptj_6b]=16
    [llama2_7b_chat]=26
)

for model in $MODELS; do
    KEY_LAYER=${PEAK_LAYERS[$model]}
    echo "" | tee -a $LOG
    echo "$(date): [$model] peak MA layer = $KEY_LAYER" | tee -a $LOG

    # RQ4: SVD alignment
    echo "$(date): [$model] RQ4 starting (layer=$KEY_LAYER)..." | tee -a $LOG
    python RQ4_svd_alignment/exp3_svd_alignment_analysis.py \
        --model $model --layer_id $KEY_LAYER --nsamples 30 \
        --savedir results/wikitext_run/RQ4/$model 2>&1 | tail -5 | tee -a $LOG
    echo "$(date): [$model] RQ4 done" | tee -a $LOG

    # RQ5: V-matrix ablation
    echo "$(date): [$model] RQ5 starting (layer=$KEY_LAYER)..." | tee -a $LOG
    python RQ5_v_matrix_ablation/exp5_v_ablation.py \
        --model $model --layer_id $KEY_LAYER --nsamples 30 \
        --savedir results/wikitext_run/RQ5/$model 2>&1 | tail -5 | tee -a $LOG
    echo "$(date): [$model] RQ5 done" | tee -a $LOG

    # RQ3: Function words
    echo "$(date): [$model] RQ3 starting (layer=$KEY_LAYER)..." | tee -a $LOG
    python RQ3_function_words/exp5_function_words_svd_mapping.py \
        --model $model --layer_id $KEY_LAYER --nsamples 30 \
        --savedir results/wikitext_run/RQ3/$model 2>&1 | tail -5 | tee -a $LOG
    echo "$(date): [$model] RQ3 done" | tee -a $LOG

    echo "$(date): [$model] ALL DONE" | tee -a $LOG
done

echo "$(date): ALL MODELS COMPLETE" | tee -a $LOG
