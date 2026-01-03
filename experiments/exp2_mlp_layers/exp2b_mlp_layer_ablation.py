#!/usr/bin/env python3
"""
Experiment 2B: MLP Layer-wise Ablation

Objective:
  Identify which MLP layer contributes most to Massive Activation

Method:
  1. Baseline: All MLPs normal
  2. Layer-wise ablation: Suppress one MLP layer at a time, keep others normal, measure activation decrease
  3. Comparison: Find the critical layer with maximum impact after suppression

Data saved to:
  results/models/{model}/exp2b_mlp_layer_ablation/
    ├── baseline.json
    ├── layer_0_disabled.json
    ├── layer_1_disabled.json
    └── ...
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

# Disable proxy
for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
    os.environ.pop(key, None)

import lib
from lib.model_utils import is_llama_model


class MLPLayerDisableHook:
    """MLP layer-wise disable hook"""
    def __init__(self, target_layer_id, mode='disable'):
        self.target_layer_id = target_layer_id
        self.mode = mode  # 'disable' or 'normal'

    def __call__(self, module, input, output):
        if self.mode == 'disable':
            # Zero out this layer's MLP output
            return torch.zeros_like(output)
        return output


def get_mlp_modules(model_name, layer):
    """Get MLP module list for different models"""
    if "opt" in model_name:
        # OPT model's FFN is fc1+fc2
        return [layer.fc1, layer.fc2]
    else:
        # Other models have unified mlp module
        return [layer.mlp]


def run_single_layer_experiment(args, disabled_layer_id=None):
    """
    Run single experiment

    Args:
        disabled_layer_id: MLP layer ID to disable, None means baseline

    Returns:
        dict: Activation statistics for each layer
    """
    # Force cleanup VRAM
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # Load model
    model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
    model.eval()
    n_layers = len(layers)

    # Register hooks
    hooks = []
    if disabled_layer_id is not None:
        # Disable specified layer's MLP
        target_layer = layers[disabled_layer_id]
        mlp_modules = get_mlp_modules(args.model, target_layer)
        hook = MLPLayerDisableHook(disabled_layer_id, mode='disable')
        for mlp_module in mlp_modules:
            handle = mlp_module.register_forward_hook(hook)
            hooks.append(handle)

    # Load data - keep on CPU
    print("Loading data to CPU...")
    testseq_list = lib.get_data(tokenizer=tokenizer, nsamples=args.nsamples, seqlen=seq_len, device='cpu')

    # Force cleanup once
    gc.collect()
    torch.cuda.empty_cache()

    # VRAM usage threshold (70GB)
    VRAM_THRESHOLD = 70 * 1024 * 1024 * 1024  # 70GB in bytes

    # Collect activations
    layer_activations = {i: [] for i in range(n_layers)}
    failed_samples = 0

    with torch.no_grad():
        for idx, testseq in enumerate(tqdm(testseq_list, desc=f"Layer {disabled_layer_id if disabled_layer_id is not None else 'baseline'}")):
            # 检查显存使用
            if torch.cuda.is_available():
                vram_allocated = torch.cuda.memory_allocated(device)
                vram_reserved = torch.cuda.memory_reserved(device)

                # 如果显存使用超过阈值，强制清理
                if vram_reserved > VRAM_THRESHOLD:
                    print(f"\n⚠️ 显存使用过高 ({vram_reserved/1e9:.1f}GB)，强制清理...")
                    gc.collect()
                    torch.cuda.empty_cache()

            try:
                # 将数据移动到GPU
                testseq = testseq.to(device)

                # 前向传播
                outputs = model(testseq, output_hidden_states=True)
                hidden_states = outputs.hidden_states

                # 收集每层的激活值
                for layer_id in range(n_layers):
                    hidden = hidden_states[layer_id + 1]  # +1因为第0个是embedding
                    max_activation = hidden.abs().max().item()
                    layer_activations[layer_id].append(max_activation)

                # 立即清理中间变量，释放显存
                del outputs, hidden_states, testseq
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                failed_samples += 1
                print(f"\n⚠️ Sample {idx} failed: {str(e)[:100]}")
                # 清理显存后继续
                if 'testseq' in locals():
                    del testseq
                gc.collect()
                torch.cuda.empty_cache()
                continue

    # 清理hooks
    for handle in hooks:
        handle.remove()

    # 打印失败样本统计
    if failed_samples > 0:
        print(f"⚠️ {failed_samples}/{len(testseq_list)} 样本失败，已跳过")

    # 统计结果
    results = {}
    for layer_id in range(n_layers):
        activations = layer_activations[layer_id]
        if len(activations) == 0:
            raise RuntimeError(f"Layer {layer_id} has no valid samples!")
        results[layer_id] = {
            'mean': float(np.mean(activations)),
            'max': float(np.max(activations)),
            'min': float(np.min(activations)),
            'std': float(np.std(activations)),
            'n_samples': len(activations)
        }

    # 清理显存
    del model
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    # 强制垃圾回收
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return results


def run_experiment_wrapper(args_tuple):
    """多进程包装函数"""
    args, layer_id = args_tuple
    try:
        results = run_single_layer_experiment(args, layer_id)
        return (layer_id, results, None)
    except Exception as e:
        return (layer_id, None, str(e))


def main():
    parser = argparse.ArgumentParser(
        description='Experiment 2B: MLP Layer-wise Ablation'
    )
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., bloom_7b1, gpt2, llama2_13b)')
    parser.add_argument('--nsamples', type=int, default=10,
                       help='Number of samples to test')
    parser.add_argument('--savedir', type=str, default=None,
                       help='Save directory (default: results/models/{model}/exp2b_mlp_layer_ablation)')
    parser.add_argument('--n_jobs', type=int, default=1,
                       help='Number of parallel jobs (careful with GPU memory!)')
    parser.add_argument('--start_layer', type=int, default=0,
                       help='Start from this layer (for resuming)')

    args = parser.parse_args()

    # 设置保存目录
    if args.savedir is None:
        args.savedir = f'results/models/{args.model}/exp2b_mlp_layer_ablation'
    os.makedirs(args.savedir, exist_ok=True)

    print("="*80)
    print("🔬 Experiment 2B: MLP Layer-wise Ablation")
    print("="*80)
    print(f"模型: {args.model}")
    print(f"样本数: {args.nsamples}")
    print(f"并行任务数: {args.n_jobs}")
    print(f"保存目录: {args.savedir}")
    print("="*80)

    # 获取层数
    print("\n📊 加载模型获取层数...")
    temp_args = argparse.Namespace(**vars(args))
    model, _, _, layers, _, _ = lib.load_llm(temp_args)
    n_layers = len(layers)
    print(f"✅ 模型共 {n_layers} 层")
    del model
    torch.cuda.empty_cache()

    # 1. 运行Baseline（所有MLP正常）
    baseline_file = os.path.join(args.savedir, 'baseline.json')
    if os.path.exists(baseline_file):
        print(f"\n✅ Baseline已存在，跳过")
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
    else:
        print(f"\n🟢 Phase 1: Running Baseline (所有MLP正常)")
        # 清理显存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        baseline_results = run_single_layer_experiment(args, disabled_layer_id=None)

        # 保存baseline
        with open(baseline_file, 'w') as f:
            json.dump({
                'experiment': 'exp2b_mlp_layer_ablation_baseline',
                'model': args.model,
                'date': datetime.now().isoformat(),
                'n_samples': args.nsamples,
                'results': baseline_results
            }, f, indent=2)
        print(f"✅ Baseline已保存: {baseline_file}")

    # 2. 逐层抑制MLP
    print(f"\n🔴 Phase 2: Running Layer-wise MLP Ablation (逐层抑制)")
    print(f"将运行 {n_layers} 个实验（每次抑制一层MLP）")

    # 检查已完成的层
    completed_layers = []
    for layer_id in range(n_layers):
        layer_file = os.path.join(args.savedir, f'layer_{layer_id}_disabled.json')
        if os.path.exists(layer_file):
            completed_layers.append(layer_id)

    if completed_layers:
        print(f"✅ 已完成 {len(completed_layers)} 层: {completed_layers}")

    # 需要运行的层
    pending_layers = [i for i in range(args.start_layer, n_layers) if i not in completed_layers]

    if not pending_layers:
        print(f"✅ 所有层已完成！")
    else:
        print(f"📝 待运行 {len(pending_layers)} 层: {pending_layers[:10]}{'...' if len(pending_layers) > 10 else ''}")

        if args.n_jobs > 1:
            # 多进程并行
            print(f"\n⚡ 使用 {args.n_jobs} 个并行任务")
            print("⚠️  警告: 多进程会占用大量GPU显存，确保显存足够！")

            # 准备参数
            tasks = [(args, layer_id) for layer_id in pending_layers]

            with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
                results_iter = executor.map(run_experiment_wrapper, tasks)

                for layer_id, results, error in results_iter:
                    if error:
                        print(f"❌ Layer {layer_id} 失败: {error}")
                        continue

                    # 保存结果
                    layer_file = os.path.join(args.savedir, f'layer_{layer_id}_disabled.json')
                    with open(layer_file, 'w') as f:
                        json.dump({
                            'experiment': 'exp2b_mlp_layer_ablation',
                            'model': args.model,
                            'disabled_layer': layer_id,
                            'date': datetime.now().isoformat(),
                            'n_samples': args.nsamples,
                            'results': results
                        }, f, indent=2)
                    print(f"✅ Layer {layer_id} 完成并保存")
        else:
            # 单进程顺序执行
            for layer_id in tqdm(pending_layers, desc="逐层抑制MLP"):
                layer_file = os.path.join(args.savedir, f'layer_{layer_id}_disabled.json')

                print(f"\n{'─'*60}")
                print(f"🔴 抑制 Layer {layer_id} 的MLP")

                # 在加载新模型前清理显存
                import gc
                gc.collect()
                torch.cuda.empty_cache()

                results = run_single_layer_experiment(args, disabled_layer_id=layer_id)

                # 保存结果
                with open(layer_file, 'w') as f:
                    json.dump({
                        'experiment': 'exp2b_mlp_layer_ablation',
                        'model': args.model,
                        'disabled_layer': layer_id,
                        'date': datetime.now().isoformat(),
                        'n_samples': args.nsamples,
                        'results': results
                    }, f, indent=2)

                print(f"✅ Layer {layer_id} 完成并保存")

                # 显示当前进度
                completed = len(completed_layers) + (pending_layers.index(layer_id) + 1)
                progress = completed / n_layers * 100
                print(f"📊 总进度: {completed}/{n_layers} ({progress:.1f}%)")

    # 3. 生成汇总报告
    print(f"\n📝 生成汇总报告...")
    generate_summary_report(args.savedir, baseline_results, n_layers, args.model)

    print("\n" + "="*80)
    print("✅ 实验完成！")
    print(f"📁 结果保存在: {args.savedir}")
    print("="*80)


def generate_summary_report(savedir, baseline_results, n_layers, model_name):
    """生成汇总报告"""

    # 收集所有数据
    summary = {
        'model': model_name,
        'baseline': {},
        'ablation': {},
        'contribution': {}
    }

    # Baseline统计
    for layer_id in range(n_layers):
        if str(layer_id) in baseline_results:
            summary['baseline'][layer_id] = baseline_results[str(layer_id)]['max']

    # 逐层抑制结果
    for layer_id in range(n_layers):
        layer_file = os.path.join(savedir, f'layer_{layer_id}_disabled.json')
        if os.path.exists(layer_file):
            with open(layer_file, 'r') as f:
                data = json.load(f)
                # 取最大激活层的max值作为代表
                max_layer_result = max(data['results'].items(),
                                      key=lambda x: x[1]['max'])
                summary['ablation'][layer_id] = max_layer_result[1]['max']

    # 计算贡献度（baseline - ablation）
    for layer_id in range(n_layers):
        if layer_id in summary['baseline'] and layer_id in summary['ablation']:
            contribution = summary['baseline'][layer_id] - summary['ablation'][layer_id]
            summary['contribution'][layer_id] = contribution

    # 保存汇总
    summary_file = os.path.join(savedir, 'summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # 找出关键层
    if summary['contribution']:
        max_contribution_layer = max(summary['contribution'].items(), key=lambda x: x[1])

        print(f"\n{'='*60}")
        print(f"📊 MLP层贡献度汇总")
        print(f"{'='*60}")
        print(f"🔑 关键层: Layer {max_contribution_layer[0]}")
        print(f"   抑制后激活下降: {max_contribution_layer[1]:.2f}")
        print(f"   Baseline激活: {summary['baseline'][max_contribution_layer[0]]:.2f}")
        print(f"   抑制后激活: {summary['ablation'][max_contribution_layer[0]]:.2f}")

        # Top 5贡献层
        top_5 = sorted(summary['contribution'].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n🏆 Top 5 贡献层:")
        for rank, (layer_id, contrib) in enumerate(top_5, 1):
            print(f"   {rank}. Layer {layer_id}: 下降 {contrib:.2f}")

    print(f"\n✅ 汇总已保存: {summary_file}")


if __name__ == '__main__':
    main()
