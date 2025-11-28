"""
Experiment 6: V矩阵消融实验 (简化版)
直接消融整个右侧奇异矩阵V，对比MA变化

原始: W = UΣVᵀ
消融: W' = UΣ (移除Vᵀ)
"""

import os
import sys
import torch
import numpy as np
import json
import argparse
from datetime import datetime
from tqdm import tqdm

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
        "opt_7b": 0,
        "gpt2": 0,
        "llama2_13b": 3,
    }
    return critical_layers.get(model_name, 0)


def run_and_get_ma(model, layers, testseq, layer_id):
    """运行模型并获取MA"""
    activations = {}
    
    def make_hook(lid):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            activations[lid] = out.detach().cpu().float()
        return hook
    
    handle = layers[layer_id].register_forward_hook(make_hook(layer_id))
    
    with torch.no_grad():
        _ = model(testseq)
    
    handle.remove()
    
    feat = activations[layer_id].numpy()
    if len(feat.shape) == 3:
        feat = feat.reshape(-1, feat.shape[-1])
    
    top1 = np.abs(feat).max()
    return float(top1)


def run_experiment(args):
    """运行消融实验"""
    print("=" * 60)
    print("EXPERIMENT 6: V矩阵消融实验 (简化版)")
    print("=" * 60)
    print("\n方法: 直接移除V矩阵，对比 W=UΣVᵀ vs W'=UΣ")
    
    # 加载模型
    print(f"\n加载模型: {args.model}")
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    
    critical_layer = get_critical_layer(args.model)
    print(f"关键层: Layer {critical_layer}")
    
    # 获取测试数据
    print("加载测试数据...")
    testseq_list = lib.get_data(tokenizer, nsamples=args.nsamples, 
                                 seqlen=min(seq_len, 2048), device=device)
    
    # 获取权重
    layer = layers[critical_layer]
    down_proj = get_mlp_down_proj(layer, args.model)
    W_original = down_proj.weight.data.cpu().float().numpy()
    
    # GPT-2 转置
    is_gpt2 = "gpt2" in args.model
    if is_gpt2:
        W_original = W_original.T
    
    print(f"权重shape: {W_original.shape}")
    
    # SVD分解
    print("\nSVD分解...")
    U, S, Vh = np.linalg.svd(W_original, full_matrices=False)
    print(f"U: {U.shape}, S: {S.shape}, Vh: {Vh.shape}")
    print(f"Top5奇异值: {S[:5].round(2)}")
    
    # 构造消融权重: W' = U @ diag(S) (移除V)
    # 注意: 这会改变输出维度，需要特殊处理
    # 更好的方法: W' = U @ diag(S) @ I (用单位矩阵代替V)
    # 但维度不匹配，所以用: W' = U @ diag(S) @ random_V
    
    # 方法1: 用随机正交矩阵替代V
    print("\n生成随机正交矩阵替代V...")
    random_V = np.random.randn(*Vh.shape)
    random_V, _ = np.linalg.qr(random_V.T)
    random_V = random_V.T
    W_random_V = U @ np.diag(S) @ random_V
    
    # 方法2: 将V置为单位方向 (每个v_i = e_i)
    # W_identity_V = U @ diag(S) @ I[:rank, :]
    
    # 1. Baseline
    print("\n--- Baseline (原始权重) ---")
    baseline_mas = []
    for testseq in tqdm(testseq_list, desc="Baseline"):
        ma = run_and_get_ma(model, layers, testseq, critical_layer)
        baseline_mas.append(ma)
    baseline_avg = np.mean(baseline_mas)
    print(f"Baseline MA平均: {baseline_avg:.2f}")
    
    # 2. 消融V (用随机正交矩阵替代)
    print("\n--- 消融V (随机正交矩阵替代) ---")
    W_to_set = W_random_V.T if is_gpt2 else W_random_V
    with torch.no_grad():
        down_proj.weight.data = torch.tensor(W_to_set, dtype=down_proj.weight.dtype,
                                              device=down_proj.weight.device)
    
    ablated_mas = []
    for testseq in tqdm(testseq_list, desc="V Ablated"):
        ma = run_and_get_ma(model, layers, testseq, critical_layer)
        ablated_mas.append(ma)
    ablated_avg = np.mean(ablated_mas)
    change_pct = (ablated_avg - baseline_avg) / baseline_avg * 100
    print(f"消融后MA平均: {ablated_avg:.2f} (变化: {change_pct:+.1f}%)")
    
    # 恢复原始权重
    W_restore = W_original.T if is_gpt2 else W_original
    with torch.no_grad():
        down_proj.weight.data = torch.tensor(W_restore, dtype=down_proj.weight.dtype,
                                              device=down_proj.weight.device)
    
    # 结果
    results = {
        "model": args.model,
        "critical_layer": critical_layer,
        "weight_shape": list(W_original.shape),
        "baseline": {
            "ma_values": baseline_mas,
            "ma_avg": float(baseline_avg)
        },
        "v_ablated": {
            "method": "random_orthogonal_replacement",
            "ma_values": ablated_mas,
            "ma_avg": float(ablated_avg),
            "change_percent": float(change_pct)
        },
        "conclusion": ""
    }
    
    # 结论
    if abs(change_pct) > 50:
        results["conclusion"] = f"V矩阵对MA影响极大 ({change_pct:+.1f}%)"
    elif abs(change_pct) > 20:
        results["conclusion"] = f"V矩阵对MA有显著影响 ({change_pct:+.1f}%)"
    elif abs(change_pct) > 5:
        results["conclusion"] = f"V矩阵对MA有一定影响 ({change_pct:+.1f}%)"
    else:
        results["conclusion"] = f"V矩阵对MA影响较小 ({change_pct:+.1f}%)"
    
    print(f"\n结论: {results['conclusion']}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--nsamples', type=int, default=5)
    parser.add_argument('--savedir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    results = run_experiment(args)
    
    results["timestamp"] = datetime.now().isoformat()
    with open(os.path.join(args.savedir, 'v_ablation_simple.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存: {os.path.join(args.savedir, 'v_ablation_simple.json')}")
    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
