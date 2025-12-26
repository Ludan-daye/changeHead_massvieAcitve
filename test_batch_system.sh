#!/bin/bash
# 快速测试批量实验系统

echo "=========================================="
echo "🧪 批量实验系统测试"
echo "=========================================="
echo ""

PROJECT_ROOT="/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve"
cd "$PROJECT_ROOT"

# 测试1: GPU监控
echo "测试1: GPU显存监控..."
python scripts/gpu_monitor.py
if [ $? -eq 0 ]; then
    echo "✅ GPU监控测试通过"
else
    echo "❌ GPU监控测试失败"
    exit 1
fi
echo ""

# 测试2: 进度汇报
echo "测试2: 进度汇报器..."
python scripts/progress_reporter.py
if [ $? -eq 0 ]; then
    echo "✅ 进度汇报测试通过"
else
    echo "❌ 进度汇报测试失败"
    exit 1
fi
echo ""

# 测试3: 快速实验（只用GPT-2，1个样本）
echo "测试3: 运行快速实验（GPT-2, Exp3, 1样本）..."
echo "这将需要约3-5分钟..."
echo ""

./run_all_experiments.sh \
    --models gpt2 \
    --experiments exp3 \
    --nsamples 1 \
    --max-memory 75

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 实验执行测试通过"
    echo ""
    echo "检查生成的文件:"
    ls -lh results/experiments/exp3/gpt2/
    echo ""
else
    echo ""
    echo "❌ 实验执行测试失败"
    exit 1
fi

echo "=========================================="
echo "✅ 所有测试通过！"
echo "=========================================="
echo ""
echo "系统已就绪，可以开始批量运行实验："
echo "  ./run_all_experiments.sh"
echo ""
