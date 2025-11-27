#!/bin/bash
# 监控实验2进度

while true; do
    clear
    echo "========================================"
    echo "OPT-6.7B 实验二 - 单层恢复"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    # 检查进程
    if ps aux | grep "exp2_opt" | grep python | grep -v grep > /dev/null; then
        echo "✅ 实验正在运行"
        ps aux | grep "exp2_opt" | grep python | grep -v grep | awk '{print "   PID: " $2 "  CPU: " $3 "%  运行时间: " $10}'
    else
        echo "❌ 实验未运行"
        break
    fi
    
    echo ""
    echo "========================================"
    echo "当前进度"
    echo "========================================"
    tail -5 exp2_opt_6.7b.log | grep -E "Layer [0-9]+ restored:" | tail -1
    
    echo ""
    echo "========================================"
    echo "已完成的层"
    echo "========================================"
    ls -1 results/exp2_opt_6.7b/layer_*/results.json 2>/dev/null | wc -l | awk '{print "   完成层数: " $1 "/35"}'
    
    echo ""
    echo "========================================"
    echo "最新日志 (最后15行)"
    echo "========================================"
    tail -15 exp2_opt_6.7b.log | sed 's/^/   /'
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    echo "========================================"
    
    sleep 30
done
