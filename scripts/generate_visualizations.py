#!/usr/bin/env python3
"""
可视化图表生成脚本 - P0核心结论图
生成7张核心图表用于支撑论文结论
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path

# 设置中文字体（如果有的话）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

# 全局配置
FIGURE_SIZE_CROSS = (12, 6)
FIGURE_SIZE_SINGLE = (10, 6)
DPI = 300
sns.set_style("whitegrid")

# 颜色方案
COLORS = {
    'attention': '#3498db',
    'mlp': '#e74c3c',
    'baseline': '#2ecc71',
    'ablated': '#95a5a6',
    'positive': '#27ae60',
    'negative': '#e67e22',
    'strong': '#c0392b',
    'medium': '#e67e22',
    'weak': '#3498db'
}

# 模型配置
MODELS = ['gptj_6b', 'bloom_7b1', 'qwen2.5_7b', 'falcon_7b', 'mistral_7b_v03']
MODELS_7 = MODELS + ['gpt2', 'opt_6.7b']

MODEL_NAMES = {
    'gptj_6b': 'GPT-J-6B',
    'bloom_7b1': 'BLOOM-7B1',
    'qwen2.5_7b': 'Qwen-2.5-7B',
    'falcon_7b': 'Falcon-7B',
    'mistral_7b_v03': 'Mistral-7B',
    'gpt2': 'GPT-2',
    'opt_6.7b': 'OPT-6.7B'
}

BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / 'results'
VIS_DIR = BASE_DIR / 'visualizations'


def load_rq2_data():
    """加载RQ2数据 - MLP vs Attention"""
    data = {}
    for model in MODELS:
        json_path = RESULTS_DIR / 'models' / model / 'RQ2_mlp_source' / 'verification.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                model_data = json.load(f)
                data[model] = {
                    'attn_max': model_data.get('attention_output_max', 0),
                    'mlp_max': model_data.get('mlp_output_max', 0),
                    'ratio': model_data.get('ratio', 0)
                }
    return data


def load_rq1_data():
    """加载RQ1数据 - Attention贡献"""
    data = {}
    for model in MODELS:
        readme_path = RESULTS_DIR / 'models' / model / 'exp1' / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 简单解析：找变化率
                if '+266%' in content or '+273%' in content:
                    change = 266
                elif '-98%' in content:
                    change = -98
                elif '-96%' in content:
                    change = -96
                elif '-60%' in content:
                    change = -60
                elif '-21%' in content:
                    change = -21
                elif '-18%' in content:
                    change = -18
                else:
                    change = 0
                data[model] = {'change_pct': change}
    return data


def load_rq3_data():
    """加载RQ3数据 - 功能词触发"""
    json_path = RESULTS_DIR / 'MA_POSITION_TOKEN_ANALYSIS.json'
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def load_rq5_data(include_7=False):
    """加载RQ5数据 - V矩阵消融"""
    models = MODELS_7 if include_7 else MODELS
    data = {}
    for model in models:
        json_path = RESULTS_DIR / 'models' / model / 'exp6' / 'v_ablation_simple.json'
        if json_path.exists():
            with open(json_path, 'r') as f:
                model_data = json.load(f)
                # 处理不同的JSON格式
                if 'change_percentage' in model_data:
                    change_pct = model_data['change_percentage']
                    baseline = model_data.get('baseline_ma', 0)
                    ablated = model_data.get('ablated_ma', 0)
                elif 'v_ablated' in model_data:
                    change_pct = model_data['v_ablated'].get('change_percent', 0)
                    baseline = model_data['baseline'].get('ma_avg', 0)
                    ablated = model_data['v_ablated'].get('ma_avg', 0)
                else:
                    continue
                
                data[model] = {
                    'baseline_ma': baseline,
                    'ablated_ma': ablated,
                    'change_pct': change_pct
                }
    return data


def plot_figure_1_ma_source():
    """图1: MA来源证据 - MLP vs Attention"""
    print("生成图1: MA来源证据...")
    
    data = load_rq2_data()
    if not data:
        print("  ✗ 数据缺失")
        return
    
    models = [m for m in MODELS if m in data]
    model_labels = [MODEL_NAMES[m] for m in models]
    attn_values = [data[m]['attn_max'] for m in models]
    mlp_values = [data[m]['mlp_max'] for m in models]
    ratios = [data[m]['ratio'] for m in models]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_CROSS, dpi=DPI)
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, attn_values, width, label='Attention Output', 
                   color=COLORS['attention'], alpha=0.8)
    bars2 = ax.bar(x + width/2, mlp_values, width, label='MLP Output', 
                   color=COLORS['mlp'], alpha=0.8)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Max Activation Value', fontsize=12, fontweight='bold')
    ax.set_title('Evidence: MA Originates from MLP, Not Attention', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # 添加比值标注
    for i, (ratio, mlp_val) in enumerate(zip(ratios, mlp_values)):
        ax.text(i, mlp_val + max(mlp_values)*0.02, f'{ratio:.1f}x', 
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color=COLORS['mlp'])
    
    # 结论标注
    ax.text(0.5, 0.95, 'Conclusion: MA comes from MLP (3-3496x higher than Attention)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '01_ma_source_evidence.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_figure_2_attention_role():
    """图2: Attention的真实作用 - 触发而非产生"""
    print("生成图2: Attention真实作用...")
    
    data = load_rq1_data()
    if not data:
        print("  ✗ 数据缺失")
        return
    
    models = [m for m in MODELS if m in data]
    model_labels = [MODEL_NAMES[m] for m in models]
    changes = [data[m]['change_pct'] for m in models]
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_CROSS, dpi=DPI)
    
    colors = [COLORS['negative'] if c < 0 else COLORS['positive'] for c in changes]
    bars = ax.bar(model_labels, changes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('MA Change After Disabling Attention (%)', fontsize=12, fontweight='bold')
    ax.set_title('Attention Role: Trigger Input (Not Generating MA)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='y', alpha=0.3)
    
    # 标注数值
    for bar, val in zip(bars, changes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.0f}%',
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=10, fontweight='bold')
    
    # 结论标注
    conclusion_text = 'Attention provides trigger signal → MLP generates MA\n'
    conclusion_text += '(Negative: Attn-triggered型, Positive: MLP主导型)'
    ax.text(0.5, 0.02, conclusion_text,
            transform=ax.transAxes, ha='center', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '02_attention_role.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_figure_3_function_word():
    """图3: MA出现位置 - 功能词触发"""
    print("生成图3: 功能词触发...")
    
    data = load_rq3_data()
    if not data:
        print("  ✗ 数据缺失")
        return
    
    models = [m for m in MODELS if m in data]
    model_labels = [MODEL_NAMES[m] for m in models]
    
    # 提取token类型数据
    type_names = ['标点符号', '功能词', '空白/换行', '实义词']
    type_keys_map = {
        '标点符号': '标点符号',
        '功能词': '功能词',
        '空白/换行': '空白/换行',
        '实义词': '实义词'
    }
    
    # 构建堆叠数据
    data_matrix = []
    semantic_free_pcts = []
    
    for model in models:
        model_data = data[model]
        type_stats = model_data.get('type_statistics', {})
        row = []
        for chinese_name, key in type_keys_map.items():
            count = type_stats.get(key, {}).get('count', 0)
            row.append(count)
        data_matrix.append(row)
        semantic_free_pcts.append(model_data.get('semantic_free_percentage', 0))
    
    data_matrix = np.array(data_matrix).T
    
    fig, ax = plt.subplots(figsize=FIGURE_SIZE_CROSS, dpi=DPI)
    
    colors_stack = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60']
    bottom = np.zeros(len(models))
    
    for i, (type_name, color) in enumerate(zip(type_names, colors_stack)):
        ax.bar(model_labels, data_matrix[i], bottom=bottom, label=type_name,
               color=color, alpha=0.8, edgecolor='white', linewidth=1)
        bottom += data_matrix[i]
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count (Top50 MA Positions)', fontsize=12, fontweight='bold')
    ax.set_title('MA Trigger Position: Function Words (Non-Semantic)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # 顶部标注无语义词占比
    for i, (pct, total) in enumerate(zip(semantic_free_pcts, bottom)):
        ax.text(i, total + 2, f'{pct:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='darkred')
    
    # 结论标注
    avg_pct = np.mean(semantic_free_pcts)
    ax.text(0.5, 0.95, f'Average: {avg_pct:.1f}% MA appear at non-semantic words',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')
    
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '03_function_word_trigger.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_figure_4_v_dependency():
    """图4: V矩阵依赖强度 - 7模型全景"""
    print("生成图4: V矩阵依赖（7模型）...")
    
    data = load_rq5_data(include_7=True)
    if not data:
        print("  ✗ 数据缺失")
        return
    
    # 按|变化率|降序排序
    sorted_models = sorted(data.keys(), key=lambda m: abs(data[m]['change_pct']), reverse=True)
    model_labels = [MODEL_NAMES[m] for m in sorted_models]
    changes = [data[m]['change_pct'] for m in sorted_models]
    
    # 颜色分级
    colors = []
    for change in changes:
        abs_change = abs(change)
        if abs_change > 80:
            colors.append(COLORS['strong'])
        elif abs_change > 50:
            colors.append(COLORS['medium'])
        else:
            colors.append(COLORS['weak'])
    
    fig, ax = plt.subplots(figsize=(10, 7), dpi=DPI)
    
    bars = ax.barh(model_labels, changes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('MA Change After V-Matrix Ablation (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Models', fontsize=12, fontweight='bold')
    ax.set_title('V-Matrix Dependency Strength (7 Models)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=-50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3)
    
    # 标注数值
    for bar, val in zip(bars, changes):
        width = bar.get_width()
        ax.text(width - 2, bar.get_y() + bar.get_height()/2.,
                f'{val:.1f}%',
                ha='right' if val < 0 else 'left', va='center',
                fontsize=10, fontweight='bold', color='white' if abs(val) > 60 else 'black')
    
    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['strong'], label='Strong (>80%)'),
        Patch(facecolor=COLORS['medium'], label='Medium (50-80%)'),
        Patch(facecolor=COLORS['weak'], label='Weak (<50%)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # 结论标注
    strong_count = sum(1 for c in changes if abs(c) > 50)
    ax.text(0.5, 0.98, f'Conclusion: {strong_count}/7 models strongly depend on V-matrix (>50%)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '04_v_matrix_dependency.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_figure_5_heatmap():
    """图5: 综合热力图 - 跨RQ全景"""
    print("生成图5: 综合热力图...")
    
    rq1_data = load_rq1_data()
    rq2_data = load_rq2_data()
    rq3_data = load_rq3_data()
    rq5_data = load_rq5_data()
    
    if not all([rq1_data, rq2_data, rq3_data, rq5_data]):
        print("  ✗ 数据不完整")
        return
    
    models = [m for m in MODELS if m in rq1_data and m in rq2_data and m in rq3_data and m in rq5_data]
    model_labels = [MODEL_NAMES[m] for m in models]
    
    # 构建数据矩阵
    matrix = []
    for model in models:
        row = [
            abs(rq1_data[model]['change_pct']),  # RQ1: |Attn变化率|
            rq2_data[model]['ratio'],            # RQ2: MLP/Attn比值
            rq3_data[model]['semantic_free_percentage'],  # RQ3: 无语义词占比
            abs(rq5_data[model]['change_pct'])   # RQ5: |V消融变化率|
        ]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # 归一化 (按列)
    matrix_norm = np.zeros_like(matrix)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        if col.max() > col.min():
            matrix_norm[:, j] = (col - col.min()) / (col.max() - col.min())
        else:
            matrix_norm[:, j] = 0.5
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=DPI)
    
    im = ax.imshow(matrix_norm, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(['RQ1:\n|Attn\nChange|', 'RQ2:\nMLP/Attn\nRatio', 
                        'RQ3:\nFunction\nWord %', 'RQ5:\n|V-Ablation\nChange|'],
                       fontsize=10)
    ax.set_yticklabels(model_labels, fontsize=10)
    
    # 标注原始数值
    for i in range(len(models)):
        for j in range(4):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black" if matrix_norm[i, j] < 0.5 else "white",
                          fontsize=9, fontweight='bold')
    
    ax.set_title('Comprehensive Heatmap: Cross-RQ Metrics', 
                 fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Value', rotation=270, labelpad=20, fontsize=10)
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '05_comprehensive_heatmap.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_figure_6_classification():
    """图6: 模型机制分类树"""
    print("生成图6: 机制分类树...")
    
    fig, ax = plt.subplots(figsize=(12, 8), dpi=DPI)
    ax.axis('off')
    
    # 手动绘制树状图
    tree_text = """
MA Generation Mechanism Classification

├─ Attention-Triggered (MA↓ >50% when disabled)
│  ├─ Strong V-Dep: GPT-J (-96%, V-71%)
│  └─ Weak V-Dep:   BLOOM (-98%, V-19%)
│
├─ MLP-Dominant (MA↑ when Attn disabled)
│  └─ Strong V-Dep: Qwen (+266%, V-99%)
│
└─ Hybrid (|MA change| <50%)
   ├─ Falcon (-21%, V-79%)
   └─ Mistral (-18%, V-83%)
"""
    
    ax.text(0.5, 0.5, tree_text, 
            transform=ax.transAxes,
            fontsize=14,
            fontfamily='monospace',
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3, pad=20))
    
    ax.set_title('Three MA Generation Mechanisms', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_path = VIS_DIR / 'conclusion' / '06_mechanism_classification.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def plot_figure_7_bloom_special():
    """图7: BLOOM特例分析"""
    print("生成图7: BLOOM特例...")
    
    # 读取BLOOM的L0和L28数据
    bloom_json = RESULTS_DIR / 'models' / 'bloom_7b1' / 'exp6' / 'v_ablation_simple.json'
    if not bloom_json.exists():
        print("  ✗ BLOOM数据缺失")
        return
    
    with open(bloom_json, 'r') as f:
        bloom_data = json.load(f)
    
    fig = plt.figure(figsize=(14, 5), dpi=DPI)
    
    # 子图1: L0 vs L28 V消融对比
    ax1 = plt.subplot(1, 3, 1)
    layers = ['Layer 0', 'Layer 28']
    
    if 'layer_comparison' in bloom_data:
        layer_comp = bloom_data['layer_comparison']
        l0_baseline = layer_comp['layer_0']['baseline_ma']
        l0_ablated = layer_comp['layer_0']['ablated_ma']
        l0_change = layer_comp['layer_0']['change_percent']
        
        l28_baseline = bloom_data['baseline']['ma_avg']
        l28_ablated = bloom_data['v_ablated']['ma_avg']
        l28_change = bloom_data['v_ablated']['change_percent']
        
        baselines = [l0_baseline, l28_baseline]
        ablateds = [l0_ablated, l28_ablated]
        
        x = np.arange(2)
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baselines, width, label='Baseline', color=COLORS['baseline'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, ablateds, width, label='V-Ablated', color=COLORS['ablated'], alpha=0.8)
        
        ax1.set_ylabel('MA Value', fontweight='bold')
        ax1.set_title('V-Ablation Effect\nAcross Layers', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 标注变化率
        ax1.text(0, max(baselines[0], ablateds[0]) * 1.1, f'{l0_change:.1f}%', 
                ha='center', fontsize=10, fontweight='bold', color='red')
        ax1.text(1, max(baselines[1], ablateds[1]) * 1.1, f'{l28_change:.1f}%', 
                ha='center', fontsize=10, fontweight='bold', color='blue')
    
    # 子图2: 标点符号相关性
    ax2 = plt.subplot(1, 3, 2)
    punctuations = [',', '.', '\\n']
    similarities = [0.44, 0.42, 0.38]
    
    bars = ax2.bar(punctuations, similarities, color=COLORS['mlp'], alpha=0.8)
    ax2.set_ylabel('Cosine Similarity', fontweight='bold')
    ax2.set_title('MA Direction\nvs Punctuation', fontweight='bold')
    ax2.set_ylim(0, 0.5)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, similarities):
        ax2.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                f'{val:.2f}', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    
    # 子图3: 机制示意
    ax3 = plt.subplot(1, 3, 3)
    ax3.axis('off')
    
    mechanism_text = """
BLOOM特例机制:

1. 早期生成 (L0)
   MLP产生MA
   V-依赖强 (-71%)

2. 残差传递 (L28)
   通过残差连接
   传递累积MA
   V-依赖弱 (-19%)

3. 语义对齐
   MA方向 ≈ 标点符号
   用于边界标记
"""
    
    ax3.text(0.5, 0.5, mechanism_text,
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5, pad=15))
    
    fig.suptitle('BLOOM Special Case: Early Generation + Residual Propagation', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = VIS_DIR / 'conclusion' / '07_bloom_special_case.png'
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 已保存: {output_path}")


def main():
    """主函数 - 生成P0核心结论图"""
    print("\n" + "="*60)
    print("开始生成P0核心结论图（7张）")
    print("="*60 + "\n")
    
    # 确保输出目录存在
    (VIS_DIR / 'conclusion').mkdir(parents=True, exist_ok=True)
    
    # 生成7张图
    plot_figure_1_ma_source()
    plot_figure_2_attention_role()
    plot_figure_3_function_word()
    plot_figure_4_v_dependency()
    plot_figure_5_heatmap()
    plot_figure_6_classification()
    plot_figure_7_bloom_special()
    
    print("\n" + "="*60)
    print("✓ P0核心结论图生成完成！")
    print(f"输出目录: {VIS_DIR / 'conclusion'}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
