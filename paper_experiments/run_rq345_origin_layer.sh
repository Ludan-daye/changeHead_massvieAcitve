#!/bin/bash
# run_rq345_origin_layer.sh
# 在 **起源层** (exp2.critical_layer) 跑 RQ3 / RQ4 / RQ5，替代老的 run_rq345_peak_layer.sh。
#
# 用法:
#   bash run_rq345_origin_layer.sh <MODELS> <RQ>
# 其中:
#   MODELS: 空格分隔的模型名；若省略，跑全 23 个已知起源层的模型
#   RQ:     "rq3" | "rq4" | "rq5" | "rq5_macro" | "all"（默认 all，all 不含 rq5_macro）
#
# 示例:
#   bash run_rq345_origin_layer.sh "gptj_6b" all          # Pilot 单模型全套
#   bash run_rq345_origin_layer.sh "" rq3                 # 全 23 模型只跑 RQ3
#   bash run_rq345_origin_layer.sh "gpt2 opt_6.7b qwen3_32b qwen3.5_27b" rq5_macro

set -e
set -u
set -o pipefail

# ====== 环境 ======
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate torch 2>/dev/null || true
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}

# 切到脚本所在目录（即 paper_experiments/）
cd "$(dirname "$0")"

# ====== 25 模型起源层查找表 ======
# 来源: ALL_EXPERIMENTS_SUMMARY.json 的 exp2.critical_layer 字段
declare -A L_ORIGIN=(
    [bloom_7b1]=3      [falcon_7b]=3      [gptj_6b]=2
    [llama3.1_8b]=1    [qwen2.5_0.5b]=0   [qwen2.5_7b]=3
    [qwen2_7b]=3       [qwen3_0.6b]=2     [qwen3_1.7b]=2
    [yi_9b]=1          [mistral_7b_v03]=1 [qwen1.5_14b]=2
    [qwen3_4b]=5       [qwen3_8b]=6       [qwen3_14b]=6
    [qwen3.5_9b]=26    [glm4_9b]=17       [qwen3_30b_a3b]=2
    [gpt2]=3           [opt_6.7b]=1       [qwen3_32b]=43
    [qwen3.5_27b]=54   [qwen3.5_35b_a3b]=39
    # llama2_13b: 待 RQ2b 跑完后填入
    # glm4_32b:   待 RQ1/RQ2a fp32 修复后填入
)

# 模式 B 的模型列表（这些模型 RQ5 需要额外跑 macro 版本）
MODEL_B_FOR_MACRO="gpt2 opt_6.7b qwen3_32b qwen3.5_27b"
# glm4_9b (macro η=4.33) 和 qwen3.5_35b_a3b 若最终判为真 B 也要加入；先不默认加

# 默认跑全部已知起源层模型
ALL_READY="bloom_7b1 falcon_7b gptj_6b llama3.1_8b qwen2.5_0.5b qwen2.5_7b \
    qwen2_7b qwen3_0.6b qwen3_1.7b yi_9b mistral_7b_v03 qwen1.5_14b \
    qwen3_4b qwen3_8b qwen3_14b qwen3.5_9b glm4_9b qwen3_30b_a3b \
    gpt2 opt_6.7b qwen3_32b qwen3.5_27b qwen3.5_35b_a3b"

# ====== 参数解析 ======
MODELS_INPUT=${1:-}
RQ=${2:-all}

if [ -z "$MODELS_INPUT" ]; then
    MODELS="$ALL_READY"
else
    MODELS="$MODELS_INPUT"
fi

LOG="run_rq345_origin_${RQ}_$(date +%Y%m%d_%H%M).log"
echo "$(date): START RQ=$RQ on models=[$MODELS]" | tee "$LOG"

# ====== 执行每个模型 ======
for model in $MODELS; do
    L=${L_ORIGIN[$model]:-}
    if [ -z "$L" ]; then
        echo "$(date): [$model] L_origin unknown, SKIP (add to L_ORIGIN table first)" | tee -a "$LOG"
        continue
    fi
    echo "" | tee -a "$LOG"
    echo "$(date): ======== [$model] L_origin=$L ========" | tee -a "$LOG"

    # glm4 系列默认走 fp32（load_model.py 内部已判断；这里也显式写清楚）
    EXTRA=""
    if [[ "$model" == glm4* ]]; then
        EXTRA="--force_fp32"
    fi

    # ---- RQ3: 功能词 SVD 映射 ----
    if [ "$RQ" = "rq3" ] || [ "$RQ" = "all" ]; then
        echo "$(date): [$model] RQ3 start layer=$L" | tee -a "$LOG"
        python RQ3_function_words/exp5_function_words_svd_mapping.py \
            --model "$model" --layer_id "$L" --nsamples 30 \
            --savedir "results/wikitext_run/RQ3_origin/$model" \
            $EXTRA 2>&1 | tail -10 | tee -a "$LOG" || true
        echo "$(date): [$model] RQ3 done" | tee -a "$LOG"
    fi

    # ---- RQ4: SVD 对齐分析（单层起源层，num_layers=1） ----
    if [ "$RQ" = "rq4" ] || [ "$RQ" = "all" ]; then
        echo "$(date): [$model] RQ4 start layer=$L" | tee -a "$LOG"
        python RQ4_svd_alignment/exp3_svd_alignment_analysis.py \
            --model "$model" --layer_id "$L" --nsamples 30 \
            --savedir "results/wikitext_run/RQ4_origin/$model" \
            $EXTRA 2>&1 | tail -10 | tee -a "$LOG" || true
        echo "$(date): [$model] RQ4 done" | tee -a "$LOG"
    fi

    # ---- RQ5: 单层 V 矩阵消融 ----
    if [ "$RQ" = "rq5" ] || [ "$RQ" = "all" ]; then
        echo "$(date): [$model] RQ5 (single layer) start layer=$L" | tee -a "$LOG"
        python RQ5_v_matrix_ablation/exp5_v_ablation.py \
            --model "$model" --layer_id "$L" --nsamples 30 \
            --savedir "results/wikitext_run/RQ5_origin/$model" \
            $EXTRA 2>&1 | tail -10 | tee -a "$LOG" || true
        echo "$(date): [$model] RQ5 single done" | tee -a "$LOG"
    fi

    # ---- RQ5 Macro: 多层 v₁ 消融（仅模式 B 模型）----
    if [ "$RQ" = "rq5_macro" ]; then
        # 判断是否模式 B
        if echo " $MODEL_B_FOR_MACRO " | grep -q " $model "; then
            # origin_layers 设为 0 到 L_origin+2（覆盖起源层及其邻近层）
            L_END=$((L+2))
            ORIGIN_LAYERS=$(seq -s, 0 $L_END)
            echo "$(date): [$model] RQ5 macro start origin_layers=$ORIGIN_LAYERS" | tee -a "$LOG"
            python RQ5_v_matrix_ablation/exp5_macro_v_ablation.py \
                --model "$model" --origin_layers "$ORIGIN_LAYERS" --nsamples 30 \
                --savedir "results/wikitext_run/RQ5_macro/$model" \
                $EXTRA 2>&1 | tail -10 | tee -a "$LOG" || true
            echo "$(date): [$model] RQ5 macro done" | tee -a "$LOG"
        else
            echo "$(date): [$model] not in MODEL_B list, skip macro RQ5" | tee -a "$LOG"
        fi
    fi
done

echo "" | tee -a "$LOG"
echo "$(date): ALL DONE. Log: $LOG" | tee -a "$LOG"
