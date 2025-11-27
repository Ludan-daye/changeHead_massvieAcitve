#!/bin/bash
# 持续监控OPT实验进度

while true; do
    clear
    echo "========================================"
    echo "OPT-6.7B 实验一 - 实时监控"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    # 检查进程
    if ps aux | grep "exp1_opt" | grep python | grep -v grep > /dev/null; then
        echo "✅ 实验正在运行"
        ps aux | grep "exp1_opt" | grep python | grep -v grep | awk '{print "   PID: " $2 "  CPU: " $3 "%  内存: " $4 "%  运行时间: " $10}'
    else
        echo "❌ 实验未运行"
        break
    fi
    
    echo ""
    echo "========================================"
    echo "GPU使用情况"
    echo "========================================"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        awk -F',' '{printf "   GPU %s: 显存 %s/%s MB  利用率 %s%%\n", $1, $2, $3, $4}'
    
    echo ""
    echo "========================================"
    echo "日志最新内容 (最后20行)"
    echo "========================================"
    tail -20 exp1_opt_6.7b.log | sed 's/^/   /'
    
    echo ""
    echo "========================================"
    echo "结果文件"
    echo "========================================"
    if [ -f "results/exp1_opt_6.7b/baseline/results.json" ]; then
        echo "   ✅ Baseline完成"
    else
        echo "   ⏳ Baseline进行中..."
    fi
    
    if [ -f "results/exp1_opt_6.7b/all_heads_disabled/results.json" ]; then
        echo "   ✅ All Heads Disabled完成"
    else
        echo "   ⏳ All Heads Disabled待运行..."
    fi
    
    if [ -f "results/exp1_opt_6.7b/comparison/EXPERIMENT_1_SUMMARY.txt" ]; then
        echo "   ✅ 实验完成！"
        echo ""
        echo "========================================"
        echo "实验结果摘要"
        echo "========================================"
        head -30 results/exp1_opt_6.7b/comparison/EXPERIMENT_1_SUMMARY.txt | sed 's/^/   /'
        break
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    echo "========================================"
    
    sleep 10
done
