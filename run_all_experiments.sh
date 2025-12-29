#!/bin/bash
# 批量运行Exp3和Exp6实验
# 自动管理显存，每30秒汇报进度

set -e

echo "=========================================="
echo "🚀 批量实验启动脚本"
echo "=========================================="
echo ""

# 项目根目录
PROJECT_ROOT="PROJECT_ROOT"
cd "$PROJECT_ROOT"

# 默认配置
MODELS=("gpt2" "gptj_6b" "falcon_7b" "bloom_7b1" "opt_6.7b" "mistral_7b_v03" "qwen2.5_7b")
EXPERIMENTS=("exp3" "exp6")
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
            echo "                          默认: exp3,exp6"
            echo "  --nsamples N            样本数（默认: 5）"
            echo "  --max-memory GB         最大显存限制（默认: 75GB）"
            echo "  --help                  显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --models gpt2,bloom_7b1 --experiments exp3 --nsamples 10"
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

# 构建参数（所有模型在一个--models后面）
MODEL_ARGS="--models ${MODELS[@]}"
EXP_ARGS="--experiments ${EXPERIMENTS[@]}"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "❌ 错误: 未找到Python"
    exit 1
fi

# 检查NVIDIA-SMI
if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️  警告: 未找到nvidia-smi，无法监控GPU"
fi

# 运行调度器
echo "=========================================="
echo "开始执行实验..."
echo "=========================================="
echo ""

python scripts/experiment_scheduler.py \
    $MODEL_ARGS \
    $EXP_ARGS \
    --nsamples $NSAMPLES \
    --max-memory $MAX_MEMORY

# 检查退出状态
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ 所有实验已完成！"
    echo "=========================================="
    echo ""
    echo "查看结果:"
    echo "  - JSON汇总: results/experiments/experiment_summary.json"
    echo "  - Markdown报告: results/experiments/EXPERIMENT_SUMMARY.md"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ 实验执行出错"
    echo "=========================================="
    exit 1
fi
