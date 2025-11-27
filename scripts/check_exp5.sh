#!/bin/bash
# 检查实验5的运行状态

echo "========================================"
echo "实验5 (Multi-Model MLP SVD) 状态检查"
echo "========================================"
echo ""

# 检查进程
if ps aux | grep "exp5_multi_model" | grep -v grep > /dev/null; then
    echo "✓ 实验5正在运行"
    ps aux | grep "exp5_multi_model" | grep -v grep | awk '{print "  PID:", $2, "  模型:", $14, "  运行时间:", $10}'
else
    echo "✗ 实验5未运行"
fi

echo ""
echo "========================================"
echo "GPU状态"
echo "========================================"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

echo ""
echo "========================================"
echo "日志最新内容"
echo "========================================"
for log in exp5_*.log; do
    if [ -f "$log" ]; then
        echo ""
        echo "--- $log ---"
        tail -15 "$log"
    fi
done

echo ""
echo "========================================"
echo "结果文件"
echo "========================================"
if [ -d "results/exp5_multi_model" ]; then
    ls -lh results/exp5_multi_model/*.json 2>/dev/null
    
    echo ""
    echo "已完成的模型："
    for result in results/exp5_multi_model/*_results.json; do
        if [ -f "$result" ]; then
            model=$(basename "$result" | sed 's/_layer.*//g')
            r2=$(grep -o '"r_squared": [0-9.]*' "$result" | cut -d' ' -f2)
            layer=$(grep -o '"layer": [0-9]*' "$result" | cut -d' ' -f2)
            echo "  ✅ $model (Layer $layer): R²=$r2"
        fi
    done
else
    echo "结果目录尚未创建"
fi
