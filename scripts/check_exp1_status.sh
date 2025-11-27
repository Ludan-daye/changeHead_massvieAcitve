#!/bin/bash
# 检查实验一的运行状态
# 使用方法: ./check_exp1_status.sh [model_name]

MODEL=${1:-"llama2_7b"}
PIDFILE="exp1_${MODEL}.pid"
LOGFILE=$(ls -t exp1_${MODEL}_*.log 2>/dev/null | head -1)

echo "=================================="
echo "实验一状态检查 - ${MODEL}"
echo "=================================="

# 检查PID文件是否存在
if [ -f "${PIDFILE}" ]; then
    PID=$(cat "${PIDFILE}")
    echo "进程ID: ${PID}"
    
    # 检查进程是否还在运行
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "状态: ✅ 运行中"
        echo ""
        echo "进程信息:"
        ps aux | grep ${PID} | grep -v grep
        echo ""
        
        # 显示CPU和内存使用
        echo "资源使用:"
        ps -p ${PID} -o %cpu,%mem,etime,cmd --no-headers
    else
        echo "状态: ❌ 已停止"
    fi
else
    echo "状态: ⚠️  未找到PID文件"
    echo "尝试查找运行中的实验进程..."
    ps aux | grep "exp1_feasibility_test.py.*${MODEL}" | grep -v grep
fi

echo ""
echo "=================================="

# 显示最新日志
if [ -n "${LOGFILE}" ] && [ -f "${LOGFILE}" ]; then
    echo "最新日志文件: ${LOGFILE}"
    echo "最后30行日志:"
    echo "----------------------------------"
    tail -n 30 "${LOGFILE}"
else
    echo "⚠️  未找到日志文件"
fi

echo ""
echo "=================================="
echo "可用命令:"
echo "  实时查看日志: tail -f ${LOGFILE}"
echo "  停止实验: kill \$(cat ${PIDFILE})"
echo "  查看结果: ls -lh results/exp1_${MODEL}/"
echo "=================================="
