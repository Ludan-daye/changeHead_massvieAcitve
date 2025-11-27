#!/bin/bash
# 检查实验4B的运行状态和结果

echo "========================================"
echo "实验4B (L3 Attention SVD) 状态检查"
echo "========================================"
echo ""

# 检查进程
if ps aux | grep "exp4b_attention_svd" | grep -v grep > /dev/null; then
    echo "✓ 实验4B正在运行"
    ps aux | grep "exp4b_attention_svd" | grep -v grep | awk '{print "  PID:", $2, "  运行时间:", $10}'
else
    echo "✗ 实验4B未运行"
fi

echo ""
echo "========================================"
echo "GPU状态"
echo "========================================"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

echo ""
echo "========================================"
echo "日志最新内容 (最后30行)"
echo "========================================"
tail -30 exp4b_layer3_attention.log

echo ""
echo "========================================"
echo "结果文件"
echo "========================================"
if [ -d "results/exp4b_layer3_attention" ]; then
    ls -lh results/exp4b_layer3_attention/
    
    if [ -f "results/exp4b_layer3_attention/LAYER3_ATTENTION_SVD_SUMMARY.txt" ]; then
        echo ""
        echo "========================================"
        echo "✅ 实验完成！总结报告："
        echo "========================================"
        cat results/exp4b_layer3_attention/LAYER3_ATTENTION_SVD_SUMMARY.txt
    fi
else
    echo "结果目录尚未创建"
fi
