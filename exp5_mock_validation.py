#!/usr/bin/env python3
"""
Exp 5 Logic Validation & Mock Test
演示Exp 5的核心逻辑，使用合成数据验证分析方法
"""

import numpy as np
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# MOCK DATA GENERATION
# ============================================================================

def generate_mock_data():
    """生成模拟数据来演示Exp 5的分析逻辑"""

    np.random.seed(42)

    # SVD参数 (模拟W₂的SVD分解)
    n_sv = 50  # 保留前50个奇异向量
    n_intermediate = 3072

    # 创建模拟的U矩阵和奇异值
    U = np.random.randn(n_intermediate, n_sv)
    U, _ = np.linalg.qr(U)  # 正交化

    S = np.array([10.0 - i*0.15 for i in range(n_sv)])  # 衰减的奇异值

    print("="*80)
    print("EXP 5 LOGIC VALIDATION - MOCK TEST")
    print("="*80)
    print(f"\nMock Data Parameters:")
    print(f"  - Singular vectors (U): {U.shape}")
    print(f"  - Singular values (S): {S.shape}")
    print(f"  - σ₁/σ₂ ratio: {S[0]/S[1]:.3f}")

    # 模拟数据：函数词 vs 内容词
    function_words = {
        'the': {'occurrences': 50, 'type': 'function'},
        'and': {'occurrences': 40, 'type': 'function'},
        'is': {'occurrences': 35, 'type': 'function'},
        'of': {'occurrences': 45, 'type': 'function'},
        'in': {'occurrences': 38, 'type': 'function'},
    }

    content_words = {
        'dog': {'occurrences': 15, 'type': 'content'},
        'tree': {'occurrences': 12, 'type': 'content'},
        'run': {'occurrences': 18, 'type': 'content'},
        'sky': {'occurrences': 14, 'type': 'content'},
    }

    all_words = {**function_words, **content_words}

    # 生成h2向量 (中间激活)
    h2_data = {}

    for word, info in all_words.items():
        n_occur = info['occurrences']
        word_type = info['type']

        if word_type == 'function':
            # 函数词：高度集中在前几个奇异向量方向上
            # 创建h2向量使其在U的前2-3个列上有大投影
            projections = np.zeros((n_occur, n_sv))

            # 集中在前3个奇异向量
            for i in range(n_occur):
                projections[i, 0] = np.random.normal(5.0, 0.5)  # 强v₁对齐
                projections[i, 1] = np.random.normal(2.0, 0.3)
                projections[i, 2] = np.random.normal(1.0, 0.2)
                # 其他向量很小
                projections[i, 3:] = np.random.normal(0, 0.1, n_sv-3)

            # 重构h2
            h2_vecs = projections @ U.T  # [n_occur, 3072]

        else:
            # 内容词：分散在多个奇异向量方向上
            projections = np.zeros((n_occur, n_sv))

            # 均匀分布在所有向量上
            for i in range(n_occur):
                projections[i, :] = np.random.normal(0.5, 0.3, n_sv)

            h2_vecs = projections @ U.T

        h2_data[word] = {
            'h2': h2_vecs,  # [n_occur, 3072]
            'projections': projections,  # [n_occur, n_sv]
            'type': word_type,
        }

    return U, S, h2_data


# ============================================================================
# ANALYSIS 1: CONCENTRATION
# ============================================================================

def analyze_concentration(h2_data, U):
    """分析投影在奇异向量上的集中性"""
    print("\n" + "="*80)
    print("ANALYSIS 1: CONCENTRATION ON SINGULAR VECTORS")
    print("="*80)

    results = {}

    for word, data in h2_data.items():
        h2 = data['h2']  # [n_occur, 3072]

        # 投影到左奇异空间
        left_proj = h2 @ U  # [n_occur, n_sv]

        # 计算方差
        var_per_sv = left_proj.var(axis=0)
        total_var = var_per_sv.sum()

        if total_var < 1e-6:
            continue

        var_ratio = var_per_sv / total_var
        sorted_var = sorted(var_ratio, reverse=True)

        # 集中度
        conc_top1 = sorted_var[0]
        conc_top3 = sum(sorted_var[:3])
        conc_top5 = sum(sorted_var[:5])

        results[word] = {
            'concentration_top1': float(conc_top1),
            'concentration_top3': float(conc_top3),
            'concentration_top5': float(conc_top5),
            'type': data['type'],
        }

    # 打印结果
    print("\nFunction Words (预期: 高集中度, >0.7):")
    for word in sorted([w for w, v in results.items() if v['type'] == 'function'],
                       key=lambda w: results[w]['concentration_top5'], reverse=True):
        r = results[word]
        print(f"  {word:10s}: top1={r['concentration_top1']:.3f}, "
              f"top3={r['concentration_top3']:.3f}, top5={r['concentration_top5']:.3f}")

    print("\nContent Words (预期: 低集中度, <0.3):")
    for word in sorted([w for w, v in results.items() if v['type'] == 'content'],
                       key=lambda w: results[w]['concentration_top5'], reverse=True):
        r = results[word]
        print(f"  {word:10s}: top1={r['concentration_top1']:.3f}, "
              f"top3={r['concentration_top3']:.3f}, top5={r['concentration_top5']:.3f}")

    return results


# ============================================================================
# ANALYSIS 2: LEFT-RIGHT ASYMMETRY
# ============================================================================

def analyze_asymmetry(h2_data, U, S):
    """分析左右奇异空间的不对称性"""
    print("\n" + "="*80)
    print("ANALYSIS 2: LEFT-RIGHT SPACE ASYMMETRY")
    print("="*80)

    results = {}

    for word, data in h2_data.items():
        h2 = data['h2']  # [n_occur, 3072]

        # 左奇异空间投影
        left_proj = h2 @ U  # [n_occur, n_sv]

        # 右奇异空间：通过权重变换
        # W₂ = U @ diag(S) @ Vt
        # 输出 ≈ (h₂ @ U) @ diag(S)
        scaled_proj = left_proj * S.reshape(1, -1)  # [n_occur, n_sv]

        # 计算浓度
        left_var = left_proj.var(axis=0)
        right_var = scaled_proj.var(axis=0)

        left_total = left_var.sum()
        right_total = right_var.sum()

        if left_total < 1e-6 or right_total < 1e-6:
            continue

        # 前3个维度的浓度
        left_top3 = sorted(left_var, reverse=True)[:3].sum() / left_total
        right_top3 = sorted(right_var, reverse=True)[:3].sum() / right_total

        asymmetry_ratio = left_top3 / (right_top3 + 1e-6)

        results[word] = {
            'left_concentration': float(left_top3),
            'right_concentration': float(right_top3),
            'asymmetry_ratio': float(asymmetry_ratio),
            'type': data['type'],
        }

    # 打印结果
    print("\nAsymmetry Ratio (Left/Right):")
    print("(预期: 函数词 ratio > 1.5, 内容词 ratio < 1.0)")
    print()

    for word in sorted(results.keys(), key=lambda w: results[w]['asymmetry_ratio'], reverse=True):
        r = results[word]
        direction = "←LEFT" if r['asymmetry_ratio'] > 1 else "RIGHT→"
        print(f"  {word:10s}: ratio={r['asymmetry_ratio']:6.3f} {direction:10s} "
              f"(L={r['left_concentration']:.3f}, R={r['right_concentration']:.3f}, "
              f"type={r['type']})")

    return results


# ============================================================================
# ANALYSIS 3: STABILITY
# ============================================================================

def analyze_stability(h2_data):
    """分析跨句子稳定性"""
    print("\n" + "="*80)
    print("ANALYSIS 3: CROSS-CONTEXT STABILITY")
    print("="*80)

    results = {}

    for word, data in h2_data.items():
        h2 = data['h2']  # [n_occur, 3072]

        if len(h2) < 2:
            continue

        # 归一化
        norms = np.linalg.norm(h2, axis=1, keepdims=True)
        h2_norm = h2 / (norms + 1e-8)

        # 计算所有对的余弦相似度
        sims = []
        for i in range(len(h2_norm)):
            for j in range(i+1, len(h2_norm)):
                sim = np.dot(h2_norm[i], h2_norm[j])
                sims.append(sim)

        sims = np.array(sims)

        results[word] = {
            'mean_stability': float(sims.mean()),
            'std_stability': float(sims.std()),
            'type': data['type'],
        }

    # 打印结果
    print("\nMean Stability (Cosine Similarity):")
    print("(预期: 函数词 > 0.8, 内容词 < 0.6)")
    print()

    for word in sorted(results.keys(), key=lambda w: results[w]['mean_stability'], reverse=True):
        r = results[word]
        stability_level = "STABLE★" if r['mean_stability'] > 0.8 else ("MEDIUM" if r['mean_stability'] > 0.6 else "VARYING")
        print(f"  {word:10s}: {r['mean_stability']:.3f} ± {r['std_stability']:.3f}  {stability_level:10s}  "
              f"(type={r['type']})")

    return results


# ============================================================================
# ANALYSIS 4: V1 ALIGNMENT
# ============================================================================

def analyze_v1_alignment(h2_data, U):
    """分析与主奇异向量v₁的对齐"""
    print("\n" + "="*80)
    print("ANALYSIS 4: PRINCIPAL DIRECTION (v₁) ALIGNMENT")
    print("="*80)

    v1 = U[:, 0]  # 最强的左奇异向量 [3072]
    v1_norm = v1 / np.linalg.norm(v1)

    results = {}

    for word, data in h2_data.items():
        h2 = data['h2']  # [n_occur, 3072]

        alignments = []
        for h2_vec in h2:
            h2_norm = h2_vec / (np.linalg.norm(h2_vec) + 1e-8)
            alignment = np.dot(h2_norm, v1_norm)
            alignments.append(alignment)

        alignments = np.array(alignments)

        results[word] = {
            'mean_alignment': float(alignments.mean()),
            'std_alignment': float(alignments.std()),
            'max_alignment': float(alignments.max()),
            'type': data['type'],
        }

    # 打印结果
    print("\nAlignment with v₁ (Principal Singular Vector):")
    print("(预期: 函数词强对齐 > 0.5, 内容词弱对齐 < 0.3)")
    print()

    for word in sorted(results.keys(), key=lambda w: results[w]['mean_alignment'], reverse=True):
        r = results[word]
        alignment_level = "STRONG★★" if r['mean_alignment'] > 0.5 else ("MEDIUM" if r['mean_alignment'] > 0.3 else "WEAK")
        print(f"  {word:10s}: {r['mean_alignment']:6.3f} ± {r['std_alignment']:.3f}  "
              f"{alignment_level:10s}  (max={r['max_alignment']:.3f}, type={r['type']})")

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_results(conc_results, asym_results, stab_results, align_results):
    """生成可视化"""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # 分离函数词和内容词
    func_words = [w for w, r in conc_results.items() if r['type'] == 'function']
    cont_words = [w for w, r in conc_results.items() if r['type'] == 'content']

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Concentration
    ax1 = fig.add_subplot(gs[0, 0])
    words_sorted = sorted(conc_results.keys(),
                          key=lambda w: conc_results[w]['concentration_top5'],
                          reverse=True)
    conc_sorted = [conc_results[w]['concentration_top5'] for w in words_sorted]
    colors = ['red' if conc_results[w]['type'] == 'function' else 'blue' for w in words_sorted]

    ax1.barh(range(len(words_sorted)), conc_sorted, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(words_sorted)))
    ax1.set_yticklabels(words_sorted)
    ax1.set_xlabel('Concentration (Top 5 Singular Vectors)')
    ax1.set_title('Analysis 1: Variance Concentration', fontweight='bold')
    ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='threshold')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Plot 2: Asymmetry
    ax2 = fig.add_subplot(gs[0, 1])
    asym_sorted = sorted(asym_results.keys(),
                         key=lambda w: asym_results[w]['asymmetry_ratio'],
                         reverse=True)
    ratio_sorted = [asym_results[w]['asymmetry_ratio'] for w in asym_sorted]
    colors = ['red' if asym_results[w]['type'] == 'function' else 'blue' for w in asym_sorted]

    ax2.barh(range(len(asym_sorted)), ratio_sorted, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(asym_sorted)))
    ax2.set_yticklabels(asym_sorted)
    ax2.set_xlabel('Asymmetry Ratio (Left / Right)')
    ax2.set_title('Analysis 2: Left-Right Asymmetry', fontweight='bold')
    ax2.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3)

    # Plot 3: Stability
    ax3 = fig.add_subplot(gs[1, 0])
    stab_sorted = sorted(stab_results.keys(),
                         key=lambda w: stab_results[w]['mean_stability'],
                         reverse=True)
    stability = [stab_results[w]['mean_stability'] for w in stab_sorted]
    colors = ['red' if stab_results[w]['type'] == 'function' else 'blue' for w in stab_sorted]

    ax3.bar(range(len(stab_sorted)), stability, color=colors, alpha=0.7)
    ax3.set_xticks(range(len(stab_sorted)))
    ax3.set_xticklabels(stab_sorted, rotation=45, ha='right')
    ax3.set_ylabel('Mean Stability (Cosine Similarity)')
    ax3.set_title('Analysis 3: Cross-Context Stability', fontweight='bold')
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='high')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='medium')
    ax3.set_ylim([0, 1.0])
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: V1 Alignment
    ax4 = fig.add_subplot(gs[1, 1])
    align_sorted = sorted(align_results.keys(),
                          key=lambda w: align_results[w]['mean_alignment'],
                          reverse=True)
    alignment = [align_results[w]['mean_alignment'] for w in align_sorted]
    colors = ['red' if align_results[w]['type'] == 'function' else 'blue' for w in align_sorted]

    ax4.bar(range(len(align_sorted)), alignment, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(align_sorted)))
    ax4.set_xticklabels(align_sorted, rotation=45, ha='right')
    ax4.set_ylabel('Mean Alignment with v₁')
    ax4.set_title('Analysis 4: Principal Direction Alignment', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Function Words'),
                       Patch(facecolor='blue', alpha=0.7, label='Content Words')]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))

    plt.savefig('results/exp5_test/exp5_mock_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved: results/exp5_test/exp5_mock_analysis.png")
    plt.close()


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary(conc_results, asym_results, stab_results, align_results):
    """生成汇总报告"""

    report = """
================================================================================
EXPERIMENT 5: FUNCTION WORDS SVD MAPPING - MOCK DATA VALIDATION REPORT
================================================================================

OBJECTIVE:
  Investigate whether function words have massive activations in left vs right
  singular vector spaces of W₂ (MLP down-projection matrix)

================================================================================
KEY FINDINGS
================================================================================

"""

    # Finding 1: Concentration
    report += "\n1️⃣ VARIANCE CONCENTRATION (Low-Dimensionality)\n"
    report += "-" * 80 + "\n"

    func_conc = [conc_results[w]['concentration_top5'] for w in conc_results
                 if conc_results[w]['type'] == 'function']
    cont_conc = [conc_results[w]['concentration_top5'] for w in conc_results
                 if conc_results[w]['type'] == 'content']

    avg_func = np.mean(func_conc)
    avg_cont = np.mean(cont_conc)

    report += f"Function Words Avg Concentration (Top 5): {avg_func:.3f}\n"
    report += f"Content Words Avg Concentration (Top 5):  {avg_cont:.3f}\n"
    report += f"Difference: {avg_func - avg_cont:.3f}\n\n"

    if avg_func > 0.7 and avg_cont < 0.3:
        report += "✅ STRONG EVIDENCE: Function words are highly concentrated (low-dimensional)\n"
        report += "                   Content words are spread across dimensions (high-dimensional)\n"
    else:
        report += "⚠️ WEAK EVIDENCE: Patterns not as clear as expected\n"

    # Finding 2: Asymmetry
    report += "\n2️⃣ LEFT-RIGHT SPACE ASYMMETRY (Information Determination)\n"
    report += "-" * 80 + "\n"

    func_asym = [asym_results[w]['asymmetry_ratio'] for w in asym_results
                 if asym_results[w]['type'] == 'function']
    cont_asym = [asym_results[w]['asymmetry_ratio'] for w in asym_results
                 if asym_results[w]['type'] == 'content']

    avg_func_asym = np.mean(func_asym)
    avg_cont_asym = np.mean(cont_asym)

    report += f"Function Words Avg Asymmetry Ratio:  {avg_func_asym:.3f}\n"
    report += f"Content Words Avg Asymmetry Ratio:   {avg_cont_asym:.3f}\n\n"

    if avg_func_asym > 1.5:
        report += "✅ STRONG EVIDENCE: Function words show left-biased concentration\n"
        report += "                   Information determined in Linear1 (expansion phase)\n"
    elif avg_func_asym > 1.0:
        report += "⚠️ MODERATE EVIDENCE: Left-biased but not dominant\n"
    else:
        report += "❌ WEAK EVIDENCE: No left-side dominance detected\n"

    # Finding 3: Stability
    report += "\n3️⃣ CROSS-CONTEXT STABILITY (Fixed Representation)\n"
    report += "-" * 80 + "\n"

    func_stab = [stab_results[w]['mean_stability'] for w in stab_results
                 if stab_results[w]['type'] == 'function']
    cont_stab = [stab_results[w]['mean_stability'] for w in stab_results
                 if stab_results[w]['type'] == 'content']

    avg_func_stab = np.mean(func_stab)
    avg_cont_stab = np.mean(cont_stab)

    report += f"Function Words Avg Stability:        {avg_func_stab:.3f}\n"
    report += f"Content Words Avg Stability:         {avg_cont_stab:.3f}\n"
    report += f"Difference: {avg_func_stab - avg_cont_stab:.3f}\n\n"

    if avg_func_stab > 0.85 and avg_cont_stab < 0.6:
        report += "✅ STRONG EVIDENCE: Function words have stable, context-independent representations\n"
        report += "                   Content words show high context-dependency\n"
    else:
        report += "⚠️ MODERATE EVIDENCE: Stability difference present but not extreme\n"

    # Finding 4: V1 Alignment
    report += "\n4️⃣ PRINCIPAL DIRECTION (v₁) ALIGNMENT (Massive Activation Driver)\n"
    report += "-" * 80 + "\n"

    func_align = [align_results[w]['mean_alignment'] for w in align_results
                  if align_results[w]['type'] == 'function']
    cont_align = [align_results[w]['mean_alignment'] for w in align_results
                  if align_results[w]['type'] == 'content']

    avg_func_align = np.mean(func_align)
    avg_cont_align = np.mean(cont_align)

    report += f"Function Words Avg v₁ Alignment:     {avg_func_align:.3f}\n"
    report += f"Content Words Avg v₁ Alignment:      {avg_cont_align:.3f}\n"
    report += f"Difference: {avg_func_align - avg_cont_align:.3f}\n\n"

    if avg_func_align > 0.5 and avg_cont_align < 0.3:
        report += "✅ STRONG EVIDENCE: Function words preferentially align with v₁\n"
        report += "                   This explains why they trigger massive activations!\n"
    else:
        report += "⚠️ MODERATE EVIDENCE: Alignment preference present\n"

    # Overall conclusion
    report += "\n\n" + "="*80
    report += "\nOVERALL CONCLUSION\n"
    report += "="*80 + "\n\n"

    evidence_count = 0
    if avg_func > 0.7 and avg_cont < 0.3:
        evidence_count += 1
    if avg_func_asym > 1.5:
        evidence_count += 1
    if avg_func_stab > 0.85 and avg_cont_stab < 0.6:
        evidence_count += 1
    if avg_func_align > 0.5 and avg_cont_align < 0.3:
        evidence_count += 1

    if evidence_count == 4:
        report += "🎯 DEFINITIVE DISCOVERY:\n"
        report += "   Function words have MASSIVE ACTIVATIONS in both left and right SVD spaces!\n\n"
        report += "Theory Confirmed:\n"
        report += "  1. Low-dimensional projections → concentrated on few singular vectors\n"
        report += "  2. Left-biased asymmetry → information determined early (Linear1)\n"
        report += "  3. High stability → fixed semantic role, context-independent\n"
        report += "  4. Strong v₁ alignment → drives massive activations through σ₁ amplification\n\n"
        report += "Mechanism:\n"
        report += "  output ≈ (h₂ · v₁) × σ₁ × u₁\n"
        report += "  where function words have large (h₂ · v₁) → massive output!\n"

    elif evidence_count >= 3:
        report += "✅ STRONG EVIDENCE:\n"
        report += "   Most key patterns confirmed. Function words show specialized behavior.\n"
    elif evidence_count >= 2:
        report += "⚠️ MODERATE EVIDENCE:\n"
        report += "   Some patterns confirmed, but more analysis needed.\n"
    else:
        report += "❌ WEAK EVIDENCE:\n"
        report += "   Patterns not clearly visible. Check data quality or hypotheses.\n"

    return report


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    import os
    os.makedirs('results/exp5_test', exist_ok=True)

    # Generate mock data
    U, S, h2_data = generate_mock_data()

    # Run analyses
    conc_results = analyze_concentration(h2_data, U)
    asym_results = analyze_asymmetry(h2_data, U, S)
    stab_results = analyze_stability(h2_data)
    align_results = analyze_v1_alignment(h2_data, U)

    # Generate visualizations
    plot_results(conc_results, asym_results, stab_results, align_results)

    # Generate summary report
    summary = generate_summary(conc_results, asym_results, stab_results, align_results)
    print(summary)

    # Save summary
    with open('results/exp5_test/EXP5_MOCK_VALIDATION.txt', 'w') as f:
        f.write(summary)

    print("\n" + "="*80)
    print("✅ MOCK VALIDATION COMPLETE")
    print("="*80)
    print("\nFiles saved:")
    print("  - results/exp5_test/exp5_mock_analysis.png")
    print("  - results/exp5_test/EXP5_MOCK_VALIDATION.txt")
