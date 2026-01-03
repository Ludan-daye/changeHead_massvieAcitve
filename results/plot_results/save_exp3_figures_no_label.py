#!/usr/bin/env python3
"""
Exp3: 生成UV归因堆叠柱状图（不带模型名称、无图例、大字体、稀疏坐标）
为每个模型生成单独的小图
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 配置
DATA_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/experiments/exp3')
OUTPUT_DIR = Path('/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve/results/plot_results/combined_figures_no_label')

# 模型配置
MODEL_CONFIGS = [
    {'key': 'gpt2', 'display': 'GPT-2'},
    {'key': 'gptj_6b', 'display': 'GPT-J-6B'},
    {'key': 'bloom_7b1', 'display': 'BLOOM-7B1'},
    {'key': 'falcon_7b', 'display': 'Falcon-7B'},
    {'key': 'opt_6.7b', 'display': 'OPT-6.7B'},
    {'key': 'mistral_7b_v03', 'display': 'Mistral-7B'},
    {'key': 'qwen2.5_7b', 'display': 'Qwen2.5-7B'},
    {'key': 'llama2_13b', 'display': 'LLaMA2-13B'},
]


def load_model_data(model_key):
    """加载模型数据"""
    data_file = DATA_DIR / model_key / 'summary.json'
    if not data_file.exists():
        return None
    with open(data_file, 'r') as f:
        return json.load(f)


def save_exp3_single(model_config):
    """保存单个exp3模型的UV归因堆叠柱状图（不带模型名称、无图例）"""
    model_key = model_config['key']
    model_display = model_config['display']
    
    data = load_model_data(model_key)
    if data is None:
        print(f"  ⚠ {model_display}: Data not found")
        return False

    attribution = data.get('attribution', {})
    
    # 检查数据有效性
    baseline = attribution.get('baseline')
    if baseline is None or (isinstance(baseline, float) and np.isnan(baseline)):
        print(f"  ⚠ {model_display}: Invalid data")
        return False

    # 计算归因占比
    u_abs = abs(attribution.get('u_attribution_pct', 0))
    v_abs = abs(attribution.get('v_attribution_pct', 0))
    inter_abs = abs(attribution.get('interaction_pct', 0))

    total_attribution = u_abs + v_abs + inter_abs

    if total_attribution > 0:
        u_prop = (u_abs / total_attribution) * 100
        v_prop = (v_abs / total_attribution) * 100
        inter_prop = (inter_abs / total_attribution) * 100
    else:
        u_prop = v_prop = inter_prop = 33.33

    fig, ax = plt.subplots(figsize=(3, 4), dpi=300)

    # 单个堆叠柱状图
    width = 0.6
    x = [0]

    # 绘制堆叠柱状图
    p1 = ax.bar(x, [u_prop], width, color='#3498db', alpha=0.85)
    p2 = ax.bar(x, [v_prop], width, bottom=[u_prop], color='#e74c3c', alpha=0.85)
    p3 = ax.bar(x, [inter_prop], width, bottom=[u_prop + v_prop], color='#f39c12', alpha=0.85)

    # 添加百分比标签
    if u_prop > 10:
        ax.text(0, u_prop/2, f'{u_prop:.1f}%', ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')
    if v_prop > 10:
        ax.text(0, u_prop + v_prop/2, f'{v_prop:.1f}%', ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')
    if inter_prop > 10:
        ax.text(0, u_prop + v_prop + inter_prop/2, f'{inter_prop:.1f}%', ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')

    # 设置坐标轴
    ax.set_xticks([])
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=11)
    
    # 稀疏Y轴刻度
    ax.set_yticks([0, 25, 50, 75, 100])

    # 添加100%参考线
    ax.axhline(y=100, color='black', linestyle='--', linewidth=1, alpha=0.3)

    # 边框
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    # 不添加图例和模型名称

    plt.tight_layout()
    output_file = OUTPUT_DIR / f'exp3_{model_display}.png'
    plt.savefig(output_file, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ {model_display}: exp3_{model_display}.png")
    return True


if __name__ == '__main__':
    print("="*60)
    print("Exp3: 保存UV归因堆叠柱状图（不带模型名称）")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n--- Exp3 (8张) ---")
    for model in MODEL_CONFIGS:
        save_exp3_single(model)

    print("\n✅ All done!")
    print(f"保存位置: {OUTPUT_DIR}")
