#!/bin/bash
# 实时监控并行实验2的进度

while true; do
    clear
    echo "========================================"
    echo "OPT-6.7B 实验二 - 并行版本实时监控"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================"
    echo ""
    
    # 检查进程
    if ps aux | grep "exp2_parallel" | grep python | grep -v grep > /dev/null; then
        echo "✅ 并行实验正在运行"
        echo ""
        echo "进程信息："
        ps aux | grep "exp2_parallel" | grep python | grep -v grep | head -10 | awk '{printf "   PID: %-8s CPU: %-6s%% 内存: %-6s%% 运行时间: %s\n", $2, $3, $4, $10}'
    else
        echo "❌ 实验未运行"
    fi
    
    echo ""
    echo "========================================"
    echo "GPU使用情况"
    echo "========================================"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,utilization.memory --format=csv,noheader,nounits | \
        awk -F',' '{printf "   GPU %s: 显存 %s/%s MB (%s%%)  GPU利用率 %s%%\n", $1, $2, $3, int($2*100/$3), $4}'
    
    echo ""
    echo "========================================"
    echo "层完成进度"
    echo "========================================"
    
    # 统计已完成的层
    total_layers=35  # Layer 3-37
    completed=$(ls -1 results/exp2_opt_6.7b/layer_*_results.json 2>/dev/null | wc -l)
    remaining=$((total_layers - completed))
    progress=$((completed * 100 / total_layers))
    
    echo "   总层数: $total_layers"
    echo "   已完成: $completed 层"
    echo "   剩余: $remaining 层"
    echo "   进度: $progress%"
    echo ""
    
    # 显示进度条
    bar_length=50
    filled=$((progress * bar_length / 100))
    empty=$((bar_length - filled))
    printf "   ["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' '-'
    printf "] $progress%%\n"
    
    echo ""
    echo "========================================"
    echo "已完成的层"
    echo "========================================"
    ls -1t results/exp2_opt_6.7b/layer_*_results.json 2>/dev/null | head -10 | while read file; do
        layer=$(basename "$file" | sed 's/layer_//g' | sed 's/_results.json//g')
        time=$(stat -c %y "$file" | cut -d'.' -f1)
        echo "   ✓ Layer $layer - 完成于 $time"
    done
    
    echo ""
    echo "========================================"
    echo "最新日志 (最后20行)"
    echo "========================================"
    tail -20 exp2_parallel.log 2>/dev/null | sed 's/^/   /'
    
    echo ""
    echo "========================================"
    echo "预计剩余时间"
    echo "========================================"
    if [ $completed -gt 0 ]; then
        # 计算平均每层耗时（假设并行5个进程）
        elapsed_minutes=$(ps -p $(pgrep -f exp2_parallel | head -1) -o etime= 2>/dev/null | awk -F: '{if (NF==3) print ($1*60)+$2+($3/60); else if (NF==2) print $1+($2/60); else print $1/60}')
        if [ ! -z "$elapsed_minutes" ]; then
            avg_time_per_batch=$(echo "scale=2; $elapsed_minutes / ($completed / 5)" | bc 2>/dev/null)
            remaining_batches=$(echo "scale=0; ($remaining + 4) / 5" | bc 2>/dev/null)
            remaining_minutes=$(echo "scale=0; $avg_time_per_batch * $remaining_batches" | bc 2>/dev/null)
            remaining_hours=$(echo "scale=1; $remaining_minutes / 60" | bc 2>/dev/null)
            
            if [ ! -z "$remaining_hours" ]; then
                echo "   预计剩余时间: ${remaining_hours}小时 (${remaining_minutes}分钟)"
                finish_time=$(date -d "+${remaining_minutes} minutes" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)
                if [ ! -z "$finish_time" ]; then
                    echo "   预计完成时间: $finish_time"
                fi
            fi
        fi
    else
        echo "   正在启动..."
    fi
    
    echo ""
    echo "按 Ctrl+C 退出监控"
    echo "========================================"
    
    sleep 10
done
