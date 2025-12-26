#!/bin/bash
# 归因实验一键运行脚本
# 运行Exp5/7/8归因分析

set -e

echo "=========================================="
echo "🔬 归因实验批量运行脚本"
echo "=========================================="
echo ""

# 项目根目录
PROJECT_ROOT="/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve"
cd "$PROJECT_ROOT"

# 默认配置
MODELS=("gpt2" "gptj_6b" "falcon_7b" "bloom_7b1" "opt_6.7b" "mistral_7b_v03" "qwen2.5_7b")
EXPERIMENTS=("exp5" "exp7" "exp8")
NSAMPLES=5
MAX_MEMORY=75

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            IFS=',' read -ra MODELS <<< "$2"
            shift 2
            ;;
        --experiments)
            IFS=',' read -ra EXPERIMENTS <<< "$2"
            shift 2
            ;;
        --nsamples)
            NSAMPLES="$2"
            shift 2
            ;;
        --max-memory)
            MAX_MEMORY="$2"
            shift 2
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --models MODELS         模型列表（逗号分隔）"
            echo "                          默认: gpt2,gptj_6b,falcon_7b,bloom_7b1,opt_6.7b,mistral_7b_v03,qwen2.5_7b"
            echo "  --experiments EXPS      实验列表（逗号分隔）"
            echo "                          默认: exp5,exp7,exp8"
            echo "  --nsamples N            样本数（默认: 5）"
            echo "  --max-memory GB         最大显存限制（默认: 75GB）"
            echo "  --help                  显示此帮助信息"
            echo ""
            echo "归因实验说明:"
            echo "  Exp5: U×V交互归因 - 测试U和V是否独立贡献"
            echo "  Exp7: 方向叠加归因 - 找到最小k值恢复90% MA"
            echo "  Exp8: 分解归因 - 分离方向(U×V)和放大(Σ)的贡献"
            echo ""
            echo "示例:"
            echo "  $0 --models gpt2,qwen2.5_7b --experiments exp5,exp7 --nsamples 10"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 显示配置
echo "配置信息:"
echo "  模型: ${MODELS[*]}"
echo "  实验: ${EXPERIMENTS[*]}"
echo "  样本数: $NSAMPLES"
echo "  显存限制: ${MAX_MEMORY}GB"
echo ""

# 构建参数
MODEL_ARGS="--models ${MODELS[@]}"
EXP_ARGS="--experiments ${EXPERIMENTS[@]}"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

# 检查实验代码
echo "检查实验代码..."
for exp in "${EXPERIMENTS[@]}"; do
    if [ "$exp" == "exp5" ]; then
        FILE="experiments/common/exp5_uv_interaction.py"
    elif [ "$exp" == "exp7" ]; then
        FILE="experiments/common/exp7_direction_superposition.py"
    elif [ "$exp" == "exp8" ]; then
        FILE="experiments/common/exp8_decomposed_attribution.py"
    else
        echo "❌ 未知实验类型: $exp"
        exit 1
    fi

    if [ ! -f "$FILE" ]; then
        echo "❌ 错误: 实验代码不存在: $FILE"
        exit 1
    fi
    echo "  ✓ $FILE"
done

# 运行归因实验
echo ""
echo "=========================================="
echo "开始执行归因实验..."
echo "=========================================="
echo ""

python scripts/run_attribution_experiments.py \
    $MODEL_ARGS \
    $EXP_ARGS \
    --nsamples $NSAMPLES \
    --max-memory $MAX_MEMORY

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 所有归因实验已完成！"
    echo "=========================================="
    echo ""
    echo "查看结果:"
    echo "  - JSON汇总: results/experiments/attribution_summary.json"
    echo "  - Markdown报告: results/experiments/ATTRIBUTION_SUMMARY.md"
    echo ""
    echo "各实验结果目录:"
    echo "  - Exp5 (U×V交互): results/experiments/exp5/{model}/"
    echo "  - Exp7 (方向叠加): results/experiments/exp7/{model}/"
    echo "  - Exp8 (分解归因): results/experiments/exp8/{model}/"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 归因实验执行出错"
    echo "=========================================="
    exit 1
fi
