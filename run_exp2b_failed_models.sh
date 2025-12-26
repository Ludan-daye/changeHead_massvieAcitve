#!/bin/bash
# 运行之前失败的4个模型

set -e

echo "========================================="
echo "🔄 重新运行失败的4个模型"
echo "========================================="

PROJECT_ROOT="/home/vicuna/ludan/massActive/changeHead_massvieAcitve"
SCRIPT="$PROJECT_ROOT/experiments/common/exp2b_mlp_layer_ablation.py"
NSAMPLES=10

# 4个失败的模型 (opt_6.7b已改为opt_7b)
MODELS=(
    "opt_7b"
    "mistral_7b_v03"
    "qwen2.5_7b"
    "llama2_13b"
)

START_TIME=$(date +%s)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "🔬 运行模型: $MODEL"
    echo "========================================"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

    python3 $SCRIPT \
        --model $MODEL \
        --nsamples $NSAMPLES \
        --n_jobs 1

    if [ $? -eq 0 ]; then
        echo "✅ $MODEL 完成"
    else
        echo "❌ $MODEL 失败"
    fi

    # 清理显存
    sleep 5

    echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')"
done

# 计算总用时
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
echo "✅ 全部完成！"
echo "========================================"
echo "总用时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "========================================"
