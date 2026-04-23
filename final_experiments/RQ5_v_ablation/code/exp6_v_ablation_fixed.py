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
    """获取MLP的down_proj权重

    Bug B4 fix (2026-04-21): add glm4 and yi to SwiGLU branch.
    Both use standard `layer.mlp.down_proj` attribute (same as llama/qwen/mistral).
    MoE models (SparseMoeBlock) are explicitly NOT handled here — callers must
    skip MoE layers via the guard in run_ablation_experiment().
    """
    if "gptj" in model_name:
        return layer.mlp.fc_out
    elif "falcon" in model_name:
        return layer.mlp.dense_4h_to_h
    elif "bloom" in model_name:
        return layer.mlp.dense_4h_to_h
    elif ("qwen" in model_name or "mistral" in model_name or is_llama_model(model_name)
            or "glm4" in model_name or "yi" in model_name):
        return layer.mlp.down_proj
    elif "opt" in model_name:
        return layer.fc2
    elif "gpt2" in model_name:
        return layer.mlp.c_proj
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_critical_layer(model_name):
    """获取起源层（origin layer）ID。

    Bug B5 fix (2026-04-21): previously defaulted to L0 for most models,
    causing ablation to happen at wrong layer. Now:
      1. CLI override via OVERRIDE_CRITICAL_LAYER env var (highest priority)
      2. Read paper_experiments/origin_layer/output/L_ORIGIN.json (RQ2c.l_origin_from_step1)
      3. Legacy hardcode fallback (only for 6 models tested historically)
      4. Raise if none found — forces user to run origin_layer/ first or pass --layer_id

    This guarantees all RQ6 ablation happens at the TRUE origin layer.
    """
    import os as _os, json as _json

    # 1. CLI override
    if _os.environ.get('OVERRIDE_CRITICAL_LAYER'):
        return int(_os.environ['OVERRIDE_CRITICAL_LAYER'])

    # 2. Read L_ORIGIN.json — look in several plausible locations
    _here = _os.path.dirname(_os.path.abspath(__file__))
    _candidates = [
        # Relative to this script in changeHead_massvieAcitve/experiments/exp6_v_ablation/
        _os.path.join(_here, '..', '..', '..', 'paper_experiments', 'origin_layer', 'output', 'L_ORIGIN.json'),
        # From repo root
        _os.path.join(_os.getcwd(), 'paper_experiments', 'origin_layer', 'output', 'L_ORIGIN.json'),
        # Common repo checkout paths
        '/root/changeHead_massvieAcitve/paper_experiments/origin_layer/output/L_ORIGIN.json',
        '/home/vicuna/ma/paper_experiments/origin_layer/output/L_ORIGIN.json',
    ]
    for _p in _candidates:
        _p = _os.path.normpath(_p)
        if _os.path.exists(_p):
            with open(_p) as _f:
                _l_origin_map = _json.load(_f)
            if model_name in _l_origin_map:
                return int(_l_origin_map[model_name])

    # 3. Embedded fallback table (values from L_ORIGIN.json as of 2026-04-21 snapshot).
    # Keep this in sync with paper_experiments/origin_layer/output/L_ORIGIN.json.
    # Only used when the JSON file is missing on disk (partial checkout / deployment issue).
    # All values below are the CORRECT L_origin per RQ2c.l_origin_from_step1 — no known-wrong
    # historical defaults retained (that was the whole point of Bug B5).
    embedded_fallback = {
        "bloom_7b1": 3, "falcon_7b": 3, "glm4_32b": 0, "glm4_9b": 1,
        "gpt2": 3, "gptj_6b": 2, "llama2_13b": 0, "llama3.1_8b": 1,
        "mistral_7b_v03": 0, "opt_6.7b": 1, "qwen1.5_14b": 35,
        "qwen2.5_0.5b": 0, "qwen2.5_7b": 3, "qwen2_7b": 3,
        "qwen3.5_27b": 54, "qwen3.5_35b_a3b": 9, "qwen3.5_9b": 22,
        "qwen3_0.6b": 2, "qwen3_1.7b": 2, "qwen3_14b": 6,
        "qwen3_30b_a3b": 1, "qwen3_32b": 6, "qwen3_4b": 6,
        "qwen3_8b": 6, "yi_9b": 8,
    }
    if model_name in embedded_fallback:
        import warnings
        warnings.warn(
            f"L_ORIGIN.json not found; using embedded fallback L={embedded_fallback[model_name]} "
            f"for '{model_name}'. Run `paper_experiments/origin_layer/run.sh` for latest values.",
            UserWarning,
        )
        return embedded_fallback[model_name]

    # 4. Fail-fast — no silent defaulting to L0
    raise ValueError(
        f"No critical_layer for '{model_name}'. "
        f"Run `paper_experiments/origin_layer/run.sh` to generate L_ORIGIN.json, "
        f"or set OVERRIDE_CRITICAL_LAYER env var, "
        f"or pass --layer_id on the command line."
    )


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
    """运行模型并收集**全模型**的真 MA（跨所有层找 top1 峰值）。

    Bug B6 fix (2026-04-21):
      Previously this function measured top1 only at `layer_id` (the ablation layer),
      but MA usually peaks at a LATER layer (after attention broadcast). Consequence:
      RQ6 baseline was orders of magnitude smaller than true MA (e.g. glm4_32b
      RQ2a baseline=298598 vs RQ6 baseline=1.15, 260000× discrepancy),
      making `remove/keep_top_K` percentages meaningless.

      Now scans ALL layers for the global max, aligning RQ6 baseline with RQ2a.
      `layer_id` is retained only for logging (reports which layer was ablated).
    """
    # 挂 hook 到所有层
    activations = {}

    def make_hook(lid):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            activations[lid] = out.detach().cpu().float()
        return hook

    handles = []
    for lid, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_hook(lid)))

    with torch.no_grad():
        _ = model(testseq)

    for h in handles:
        h.remove()

    # 跨所有层找 global MA 峰值
    peak_top1 = 0.0
    peak_layer = -1
    peak_dim = -1
    peak_vec = None
    peak_shape = None
    for lid in range(len(layers)):
        feat = activations.get(lid)
        if feat is None:
            continue
        feat_np = feat.numpy()
        if len(feat_np.shape) == 3:
            feat_np = feat_np.reshape(-1, feat_np.shape[-1])
        layer_top1 = np.abs(feat_np).max()
        if layer_top1 > peak_top1:
            peak_top1 = layer_top1
            peak_layer = lid
            max_idx = np.unravel_index(np.abs(feat_np).argmax(), feat_np.shape)
            peak_dim = int(max_idx[1])
            peak_vec = feat_np[max_idx[0]]
            peak_shape = list(feat_np.shape)

    return {
        "top1": float(peak_top1),
        "ma_dim": int(peak_dim) if peak_dim >= 0 else -1,
        "ma_vec": peak_vec,
        "peak_layer": int(peak_layer),
        "ablation_layer": int(layer_id),  # 记录是哪一层被消融了
        "feat_shape": peak_shape,
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

    # Bug B2 guard (2026-04-21): skip MoE models (Tier C separate analysis).
    # MoE layers have SparseMoeBlock which lacks .down_proj — would crash below.
    _first_mlp = getattr(layers[0], 'mlp', None)
    if _first_mlp is not None and hasattr(_first_mlp, 'experts'):
        print(f"⚠ {args.model} is a MoE model (has layer.mlp.experts).")
        print(f"  Skipping — use Tier C per-expert ablation scripts for MoE analysis.")
        return None

    # 获取起源层 (Bug B5 fix: read from L_ORIGIN.json or --layer_id, not default L0)
    if args.layer_id is not None:
        critical_layer = args.layer_id
        print(f"起源层 (from --layer_id): Layer {critical_layer}")
    else:
        critical_layer = get_critical_layer(args.model)
        print(f"起源层 (from L_ORIGIN.json): Layer {critical_layer}")
    
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
    # Bug B5 fix (2026-04-21): allow explicit origin-layer override.
    # If omitted, falls back to L_ORIGIN.json or legacy hardcode (see get_critical_layer).
    parser.add_argument('--layer_id', type=int, default=None,
                        help='Override origin layer. If None, read from '
                             'paper_experiments/origin_layer/output/L_ORIGIN.json.')
    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)

    # 运行实验
    results = run_ablation_experiment(args)

    # Bug B2 guard returned None for MoE — skip saving
    if results is None:
        print("Skipped (MoE model). No output written.")
        return

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
