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

# ====== 起源层查找表（自动 source 自 origin_layer/output/）======
#
# 单层: L_ORIGIN[model]            ← origin_layer/output/L_ORIGIN.sh
# 多层: ORIGIN_LAYERS_MACRO[model] ← origin_layer/output/ORIGIN_LAYERS_MACRO.sh
#
# 这两个文件由 determine_origin_layer.py 从 exp2c 自动产出。
# 更新层号只需: cd origin_layer && bash run.sh

L_ORIGIN_FILE="origin_layer/output/L_ORIGIN.sh"
MACRO_FILE="origin_layer/output/ORIGIN_LAYERS_MACRO.sh"

if [ ! -f "$L_ORIGIN_FILE" ] || [ ! -f "$MACRO_FILE" ]; then
    echo "❌ 找不到层号文件: $L_ORIGIN_FILE / $MACRO_FILE"
    echo "   运行: cd origin_layer && bash run.sh 生成这些文件"
    exit 1
fi

# 需要 bash 4+ (declare -A)
if ! declare -A test_assoc 2>/dev/null; then
    echo "❌ 当前 bash 不支持关联数组 (需要 4+)"
    echo "   macOS 默认 bash 3.2 → 用 /opt/homebrew/bin/bash 重跑"
    exit 1
fi
unset test_assoc

source "$L_ORIGIN_FILE"
source "$MACRO_FILE"

echo "✓ 已 source $L_ORIGIN_FILE （${#L_ORIGIN[@]} 模型）"
echo "✓ 已 source $MACRO_FILE （${#ORIGIN_LAYERS_MACRO[@]} 模型）"

# 模式 B 的模型（RQ5 macro 默认跑这几个）
MODEL_B_FOR_MACRO="gpt2 opt_6.7b qwen3_32b qwen3.5_27b"

# 默认跑全部已知起源层模型（自动从 L_ORIGIN 数组取 keys）
ALL_READY="${!L_ORIGIN[*]}"

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

    # ---- RQ5 Macro: 多层 v₁ 消融 ----
    # 从 ORIGIN_LAYERS_MACRO 数组直接读（auto-sourced 自 origin_layer/output/）
    if [ "$RQ" = "rq5_macro" ]; then
        MACRO="${ORIGIN_LAYERS_MACRO[$model]:-}"
        if [ -z "$MACRO" ]; then
            echo "$(date): [$model] no macro layers (无 exp2c 和 fallback), skip" | tee -a "$LOG"
        else
            echo "$(date): [$model] RQ5 macro start origin_layers=$MACRO" | tee -a "$LOG"
            python RQ5_v_matrix_ablation/exp5_macro_v_ablation.py \
                --model "$model" --origin_layers "$MACRO" --nsamples 30 \
                --savedir "results/wikitext_run/RQ5_macro/$model" \
                $EXTRA 2>&1 | tail -10 | tee -a "$LOG" || true
            echo "$(date): [$model] RQ5 macro done" | tee -a "$LOG"
        fi
    fi
done

echo "" | tee -a "$LOG"
echo "$(date): ALL DONE. Log: $LOG" | tee -a "$LOG"
