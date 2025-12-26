#!/bin/bash

# 批量生成所有模型的贡献度分析图

MODELS=(gpt2 gptj_6b bloom_7b1 falcon_7b opt_7b mistral_7b_v03 qwen2.5_7b llama2_13b)

echo "=========================================="
echo "批量生成贡献度分析图"
echo "总计: ${#MODELS[@]} 个模型"
echo "=========================================="

SUCCESS=0
FAILED=0

for model in "${MODELS[@]}"; do
    echo ""
    echo ">>> Processing: $model"

    if python3 scripts/analyze_layer_contribution.py --model "$model" > /dev/null 2>&1; then
        echo "✅ $model - 成功"
        ((SUCCESS++))
    else
        echo "❌ $model - 失败"
        ((FAILED++))
    fi
done

echo ""
echo "=========================================="
echo "完成统计:"
echo "  ✅ 成功: $SUCCESS 个模型"
echo "  ❌ 失败: $FAILED 个模型"
echo "=========================================="
