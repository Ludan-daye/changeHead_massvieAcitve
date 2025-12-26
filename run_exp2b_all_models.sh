#!/bin/bash
# 批量运行MLP逐层抑制实验 - 8个模型
# 充分利用A100 80G显存

set -e  # 遇到错误立即退出

echo "================================="
echo "🚀 批量运行 MLP逐层抑制实验"
echo "================================="
echo "GPU: A100 80G"
echo "模型数: 8"
echo "================================="

# 实验参数
NSAMPLES=10
PROJECT_ROOT="/home/vicuna/ludan/massActive/changeHead_massvieAcitve"
SCRIPT="$PROJECT_ROOT/experiments/common/exp2b_mlp_layer_ablation.py"

# 8个模型列表
MODELS=(
    "bloom_7b1"
    "falcon_7b"
    "gpt2"
    "gptj_6b"
    "mistral_7b_v03"
    "opt_6.7b"
    "qwen2.5_7b"
    "llama2_13b"
)

# 记录开始时间
START_TIME=$(date +%s)

# 逐个运行模型实验
for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "🔬 运行模型: $MODEL"
    echo "========================================"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"

    # 运行实验（单进程，避免显存溢出）
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
echo "✅ 所有实验完成！"
echo "========================================"
echo "总用时: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "结果保存在: results/models/{model}/exp2b_mlp_layer_ablation/"
echo "========================================"
