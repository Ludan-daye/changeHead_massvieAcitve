"""
Experiment 6: V矩阵消融实验
测试MLP down_proj右奇异向量(V)对Massive Activation的影响

方式A: 移除前k个V方向
方式B: 只保留前k个V方向
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

# 禁用代理
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

import lib
from lib.model_utils import is_llama_model


def get_mlp_down_proj(layer, model_name):
    """获取MLP的down_proj权重"""
    if "gptj" in model_name:
        return layer.mlp.fc_out
    elif "falcon" in model_name:
        return layer.mlp.dense_4h_to_h
    elif "bloom" in model_name:
        return layer.mlp.dense_4h_to_h
    elif "qwen" in model_name or "mistral" in model_name or is_llama_model(model_name):
        return layer.mlp.down_proj
    elif "opt" in model_name:
        return layer.fc2
    elif "gpt2" in model_name:
        return layer.mlp.c_proj
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_critical_layer(model_name):
    """获取关键层ID"""
    critical_layers = {
        "gptj_6b": 0,
        "falcon_7b": 0,
        "mistral_7b_v03": 0,
        "qwen2.5_7b": 3,
        "bloom_7b1": 28,
        "opt_6.7b": 0,
    }
    return critical_layers.get(model_name, 0)


def ablate_v_remove_top_k(U, S, Vh, k):
    """方式A: 移除前k个V方向 (使用预计算的SVD)"""
    S_ablated = S.copy()
    S_ablated[:k] = 0
    W_ablated = U @ np.diag(S_ablated) @ Vh
    return W_ablated, {"removed_singular_values": S[:k].tolist()}


def ablate_v_keep_top_k(U, S, Vh, k):
    """方式B: 只保留前k个V方向 (使用预计算的SVD)"""
    W_ablated = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]
    return W_ablated, {"kept_singular_values": S[:k].tolist()}


def run_and_collect_ma(model, layers, testseq, layer_id, model_name):
    """运行模型并收集指定层的MA"""
    # 设置hook收集激活
    activations = {}
    
    def make_hook(lid):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activations[lid] = out.detach().cpu().float()
        return hook
    
    layer = layers[layer_id]
    handle = layer.register_forward_hook(make_hook(layer_id))
    
    with torch.no_grad():
        _ = model(testseq)
    
    handle.remove()
    
    # 计算MA指标
    feat = activations[layer_id].numpy()
    if len(feat.shape) == 3:
        feat = feat.reshape(-1, feat.shape[-1])
    
    top1 = np.abs(feat).max()
    max_idx = np.unravel_index(np.abs(feat).argmax(), feat.shape)
    ma_dim = max_idx[1]
    ma_vec = feat[max_idx[0]]
    
    return {
        "top1": float(top1),
        "ma_dim": int(ma_dim),
        "ma_vec": ma_vec,
        "feat_shape": list(feat.shape)
    }


def run_ablation_experiment(args):
    """运行消融实验"""
    print("=" * 60)
    print("EXPERIMENT 6: V矩阵消融实验")
    print("=" * 60)
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    
    # 获取关键层
    critical_layer = get_critical_layer(args.model)
    print(f"关键层: Layer {critical_layer}")
    
    # 获取测试数据
    print("加载测试数据...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, 
                                 seqlen=min(seq_len, 2048), device=device)
    
    # 获取MLP down_proj
    layer = layers[critical_layer]
    down_proj = get_mlp_down_proj(layer, args.model)
    W_original = down_proj.weight.data.cpu().float().numpy()
    
    # GPT-2使用Conv1D，权重需要转置
    if "gpt2" in args.model:
        W_original = W_original.T  # (3072, 768) -> (768, 3072)
    
    print(f"down_proj权重shape: {W_original.shape}")
    
    # SVD分析
    U, S, Vh = np.linalg.svd(W_original, full_matrices=False)
    print(f"奇异值Top5: {S[:5].round(2)}")
    print(f"σ₁/σ₂ = {S[0]/S[1]:.4f}")
    
    # 实验结果
    results = {
        "model": args.model,
        "critical_layer": critical_layer,
        "weight_shape": list(W_original.shape),
        "singular_values_top10": S[:10].tolist(),
        "sigma_ratio": float(S[0]/S[1]),
        "k_values": args.k_values,
        "ablation_results": {
            "remove_top_k": {},
            "keep_top_k": {}
        }
    }
    
    # 1. Baseline (无消融)
    print("\n--- Baseline (无消融) ---")
    baseline_mas = []
    for testseq in tqdm(testseq_list, desc="Baseline"):
        ma_info = run_and_collect_ma(model, layers, testseq, critical_layer, args.model)
        baseline_mas.append(ma_info["top1"])
    
    baseline_avg = np.mean(baseline_mas)
    results["baseline"] = {
        "top1_values": baseline_mas,
        "top1_avg": float(baseline_avg)
    }
    print(f"Baseline Top1平均: {baseline_avg:.2f}")
    
    # 2. 方式A: 移除前k个V
    print("\n--- 方式A: 移除前k个V方向 ---")
    for k in args.k_values:
        print(f"\n  移除前 {k} 个V方向...")
        
        # 消融权重 (使用预计算的SVD)
        W_ablated, info = ablate_v_remove_top_k(U, S, Vh, k)
        
        # 临时替换权重 (GPT-2需要转置回来)
        W_to_set = W_ablated.T if "gpt2" in args.model else W_ablated
        with torch.no_grad():
            down_proj.weight.data = torch.tensor(W_to_set, dtype=down_proj.weight.dtype, 
                                                  device=down_proj.weight.device)
        
        # 测试
        ablated_mas = []
        for testseq in testseq_list:
            ma_info = run_and_collect_ma(model, layers, testseq, critical_layer, args.model)
            ablated_mas.append(ma_info["top1"])
        
        ablated_avg = np.mean(ablated_mas)
        change_pct = (ablated_avg - baseline_avg) / baseline_avg * 100
        
        results["ablation_results"]["remove_top_k"][str(k)] = {
            "top1_values": ablated_mas,
            "top1_avg": float(ablated_avg),
            "change_percent": float(change_pct),
            "removed_sigma_sum": float(np.sum(S[:k])),
            "removed_sigma_ratio": float(np.sum(S[:k]) / np.sum(S))
        }
        
        print(f"    Top1平均: {ablated_avg:.2f} (变化: {change_pct:+.1f}%)")
        
        # 恢复原始权重 (GPT-2需要转置回来)
        W_restore = W_original.T if "gpt2" in args.model else W_original
        with torch.no_grad():
            down_proj.weight.data = torch.tensor(W_restore, dtype=down_proj.weight.dtype,
                                                  device=down_proj.weight.device)
    
    # 3. 方式B: 只保留前k个V
    print("\n--- 方式B: 只保留前k个V方向 ---")
    for k in args.k_values:
        print(f"\n  只保留前 {k} 个V方向...")
        
        # 消融权重 (使用预计算的SVD)
        W_ablated, info = ablate_v_keep_top_k(U, S, Vh, k)
        
        # 临时替换权重 (GPT-2需要转置回来)
        W_to_set = W_ablated.T if "gpt2" in args.model else W_ablated
        with torch.no_grad():
            down_proj.weight.data = torch.tensor(W_to_set, dtype=down_proj.weight.dtype,
                                                  device=down_proj.weight.device)
        
        # 测试
        ablated_mas = []
        for testseq in testseq_list:
            ma_info = run_and_collect_ma(model, layers, testseq, critical_layer, args.model)
            ablated_mas.append(ma_info["top1"])
        
        ablated_avg = np.mean(ablated_mas)
        change_pct = (ablated_avg - baseline_avg) / baseline_avg * 100
        
        results["ablation_results"]["keep_top_k"][str(k)] = {
            "top1_values": ablated_mas,
            "top1_avg": float(ablated_avg),
            "change_percent": float(change_pct),
            "kept_sigma_sum": float(np.sum(S[:k])),
            "kept_sigma_ratio": float(np.sum(S[:k]) / np.sum(S))
        }
        
        print(f"    Top1平均: {ablated_avg:.2f} (变化: {change_pct:+.1f}%)")
        
        # 恢复原始权重 (GPT-2需要转置回来)
        W_restore = W_original.T if "gpt2" in args.model else W_original
        with torch.no_grad():
            down_proj.weight.data = torch.tensor(W_restore, dtype=down_proj.weight.dtype,
                                                  device=down_proj.weight.device)
    
    # 分析结论
    print("\n" + "=" * 60)
    print("实验分析")
    print("=" * 60)
    
    # 分析方式A
    remove_results = results["ablation_results"]["remove_top_k"]
    print("\n方式A (移除前k个V) 分析:")
    for k, data in remove_results.items():
        print(f"  k={k}: 变化 {data['change_percent']:+.1f}% (移除了 {data['removed_sigma_ratio']*100:.1f}% 的奇异值能量)")
    
    # 分析方式B
    keep_results = results["ablation_results"]["keep_top_k"]
    print("\n方式B (只保留前k个V) 分析:")
    for k, data in keep_results.items():
        print(f"  k={k}: 变化 {data['change_percent']:+.1f}% (保留了 {data['kept_sigma_ratio']*100:.1f}% 的奇异值能量)")
    
    # 结论
    results["conclusion"] = analyze_results(results)
    print(f"\n结论: {results['conclusion']['summary']}")
    
    return results


def analyze_results(results):
    """分析实验结果并得出结论"""
    remove_results = results["ablation_results"]["remove_top_k"]
    keep_results = results["ablation_results"]["keep_top_k"]
    
    # 检查移除v1的影响
    remove_1 = remove_results.get("1", {})
    remove_1_change = remove_1.get("change_percent", 0)
    
    # 检查只保留v1能复现多少
    keep_1 = keep_results.get("1", {})
    keep_1_change = keep_1.get("change_percent", 0)
    
    # 判断类型
    if abs(remove_1_change) > 20:
        v1_important = True
        importance = "strong"
    elif abs(remove_1_change) > 5:
        v1_important = True
        importance = "moderate"
    else:
        v1_important = False
        importance = "weak"
    
    # 检查MA是否能被少量V复现
    keep_5 = keep_results.get("5", {})
    keep_5_change = abs(keep_5.get("change_percent", -100))
    
    if keep_5_change < 20:
        ma_concentrated = True
    else:
        ma_concentrated = False
    
    conclusion = {
        "v1_importance": importance,
        "v1_remove_effect": remove_1_change,
        "v1_alone_effect": keep_1_change,
        "ma_concentrated_in_top_v": ma_concentrated,
        "summary": ""
    }
    
    if v1_important and ma_concentrated:
        conclusion["summary"] = f"MA高度依赖V主导方向 (移除v1: {remove_1_change:+.1f}%, 前5个V可复现MA)"
    elif v1_important:
        conclusion["summary"] = f"V1对MA有{importance}影响 (移除v1: {remove_1_change:+.1f}%), 但MA分散在多个V方向"
    else:
        conclusion["summary"] = f"MA与V主导方向关系较弱 (移除v1仅: {remove_1_change:+.1f}%)"
    
    return conclusion


def plot_results(results, savedir):
    """绘制结果图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    baseline = results["baseline"]["top1_avg"]
    k_values = results["k_values"]
    
    # 方式A: 移除前k个V
    ax1 = axes[0]
    remove_changes = [results["ablation_results"]["remove_top_k"][str(k)]["change_percent"] 
                      for k in k_values]
    ax1.bar(range(len(k_values)), remove_changes, color='coral')
    ax1.set_xticks(range(len(k_values)))
    ax1.set_xticklabels([f'k={k}' for k in k_values])
    ax1.set_ylabel('MA Change (%)')
    ax1.set_title(f'方式A: 移除前k个V方向\n(Baseline={baseline:.1f})')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 方式B: 只保留前k个V
    ax2 = axes[1]
    keep_changes = [results["ablation_results"]["keep_top_k"][str(k)]["change_percent"] 
                    for k in k_values]
    ax2.bar(range(len(k_values)), keep_changes, color='steelblue')
    ax2.set_xticks(range(len(k_values)))
    ax2.set_xticklabels([f'k={k}' for k in k_values])
    ax2.set_ylabel('MA Change (%)')
    ax2.set_title(f'方式B: 只保留前k个V方向\n(Baseline={baseline:.1f})')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.suptitle(f'{results["model"]} - Layer {results["critical_layer"]} V矩阵消融实验', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(savedir, 'v_ablation_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"图表已保存: {os.path.join(savedir, 'v_ablation_results.png')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 5, 10, 50, 100])
    parser.add_argument('--savedir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    # 运行实验
    results = run_ablation_experiment(args)
    
    # 保存结果
    results["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(args.savedir, 'v_ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存: {os.path.join(args.savedir, 'v_ablation_results.json')}")
    
    # 绘图
    plot_results(results, args.savedir)
    
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
