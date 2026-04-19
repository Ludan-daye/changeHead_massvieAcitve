#!/bin/bash
# run.sh — 一键产出所有模型的起源层到 output/
#
# 用法:  bash run.sh
#
# 产出 output/ 下:
#   - SUMMARY.md                   人类可读表
#   - L_ORIGIN.json                单层起源（程序用）
#   - L_ORIGIN.sh                  单层起源 bash 数组
#   - ORIGIN_LAYERS_MACRO.json     macro 起源集合
#   - ORIGIN_LAYERS_MACRO.sh       macro 起源 bash 数组
#   - compare_v1_vs_v2.txt         和旧 L_ORIGIN 的 diff

set -e
cd "$(dirname "$0")"

echo "=== 跑 determine_origin_layer.py --dump-all ==="
python3 determine_origin_layer.py --dump-all

echo ""
echo "=== 产出摘要 ==="
ls -la output/

echo ""
echo "=== SUMMARY 前 40 行 ==="
head -40 output/SUMMARY.md

echo ""
echo "✓ 完成。详细见 output/SUMMARY.md 和 README.md"
