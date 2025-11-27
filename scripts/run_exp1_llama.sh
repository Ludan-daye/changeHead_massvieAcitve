#!/bin/bash
# 后台运行实验一 - LLaMA 模型
# 使用方法: ./run_exp1_llama.sh [model_name] [nsamples]
# 示例: ./run_exp1_llama.sh llama2_7b 30

# 默认参数
MODEL=${1:-"llama2_7b"}
NSAMPLES=${2:-30}
SAVEDIR="results/exp1_${MODEL}/"
LOGFILE="exp1_${MODEL}_$(date +%Y%m%d_%H%M%S).log"

echo "=================================="
echo "启动实验一 - LLaMA 模型后台运行"
echo "=================================="
echo "模型: ${MODEL}"
echo "样本数: ${NSAMPLES}"
echo "结果目录: ${SAVEDIR}"
echo "日志文件: ${LOGFILE}"
echo "=================================="

# 创建结果目录
mkdir -p "${SAVEDIR}"

# 使用 nohup 在后台运行，输出重定向到日志文件
nohup python exp1_feasibility_test.py \
    --model "${MODEL}" \
    --dataset wikitext \
    --nsamples "${NSAMPLES}" \
    --savedir "${SAVEDIR}" \
    > "${LOGFILE}" 2>&1 &

# 获取进程ID
PID=$!

echo ""
echo "✅ 实验已在后台启动！"
echo "进程ID: ${PID}"
echo ""
echo "监控命令："
echo "  查看进程状态: ps aux | grep ${PID}"
echo "  实时查看日志: tail -f ${LOGFILE}"
echo "  查看最新日志: tail -n 50 ${LOGFILE}"
echo "  停止实验: kill ${PID}"
echo ""
echo "结果将保存到: ${SAVEDIR}"
echo "=================================="

# 将进程ID保存到文件
echo "${PID}" > "exp1_${MODEL}.pid"
echo "进程ID已保存到: exp1_${MODEL}.pid"
