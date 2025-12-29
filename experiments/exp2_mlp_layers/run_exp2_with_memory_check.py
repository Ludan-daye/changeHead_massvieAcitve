#!/usr/bin/env python3
"""
智能Exp2运行器 - 自动检查显存并调整参数
"""
import subprocess
import re
import sys
import torch

def get_gpu_memory():
    """获取GPU显存信息 (单位: MB)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        total, used, free = map(int, result.stdout.strip().split(','))
        return {'total': total, 'used': used, 'free': free}
    except Exception as e:
        print(f"⚠️ 无法获取GPU信息: {e}")
        return None

def clear_gpu_cache():
    """清理GPU缓存"""
    print("🧹 清理GPU缓存...")
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def calculate_optimal_samples(model_name, free_memory_mb):
    """
    根据空闲显存计算最优样本数

    Args:
        model_name: 模型名称
        free_memory_mb: 空闲显存 (MB)

    Returns:
        nsamples: 推荐样本数
    """
    # 预估每个样本的显存需求 (MB)
    memory_per_sample = {
        'gpt2': 100,
        'gptj_6b': 500,
        'bloom_7b1': 600,
        'falcon_7b': 600,
        'opt_7b': 500,
        'mistral_7b_v03': 600,
        'qwen2.5_7b': 600,
        'llama2_13b': 1500,  # 13B模型需要更多
    }

    # 获取预估值，默认800MB
    mem_per_sample = memory_per_sample.get(model_name, 800)

    # 保留安全边界 (至少10GB)
    safety_margin_mb = 10 * 1024
    usable_memory = max(0, free_memory_mb - safety_margin_mb)

    # 计算可以安全运行的样本数
    max_samples = usable_memory // mem_per_sample

    # 限制在1-10之间
    optimal_samples = max(1, min(10, int(max_samples)))

    print(f"📊 显存分析:")
    print(f"  - 空闲显存: {free_memory_mb/1024:.1f} GB")
    print(f"  - 安全边界: {safety_margin_mb/1024:.1f} GB")
    print(f"  - 可用显存: {usable_memory/1024:.1f} GB")
    print(f"  - 每样本需求: {mem_per_sample} MB")
    print(f"  - 推荐样本数: {optimal_samples}")

    return optimal_samples

def run_experiment(model_name, nsamples=None, force=False):
    """
    运行Exp2实验

    Args:
        model_name: 模型名称
        nsamples: 样本数 (None=自动计算)
        force: 是否强制运行（跳过显存检查）
    """
    print("="*80)
    print(f"🚀 准备运行 {model_name} 的 Exp2 实验")
    print("="*80)

    # 清理GPU缓存
    clear_gpu_cache()

    # 检查显存
    gpu_info = get_gpu_memory()
    if gpu_info is None and not force:
        print("❌ 无法获取GPU信息，请使用 --force 强制运行")
        return False

    if gpu_info:
        print(f"\n💾 当前GPU状态:")
        print(f"  - 总显存: {gpu_info['total']/1024:.1f} GB")
        print(f"  - 已使用: {gpu_info['used']/1024:.1f} GB")
        print(f"  - 空闲: {gpu_info['free']/1024:.1f} GB")

        # 自动计算样本数
        if nsamples is None:
            nsamples = calculate_optimal_samples(model_name, gpu_info['free'])

        # 检查是否有足够显存
        if gpu_info['free'] < 5 * 1024 and not force:
            print(f"\n⚠️ 警告: 空闲显存不足5GB，建议先清理")
            print("使用 --force 强制运行")
            return False
    else:
        # 没有GPU信息时使用保守值
        nsamples = nsamples or 3

    print(f"\n✅ 将使用 {nsamples} 个样本运行实验")

    # 构建命令
    cmd = [
        'python3',
        'experiments/common/exp2b_mlp_layer_ablation.py',
        '--model', model_name,
        '--nsamples', str(nsamples),
        '--n_jobs', '1'
    ]

    print(f"\n📝 执行命令: {' '.join(cmd)}")
    print("="*80)

    # 运行实验
    result = subprocess.run(cmd, cwd='PROJECT_ROOT')

    return result.returncode == 0

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='智能Exp2运行器')
    parser.add_argument('--model', type=str, required=True,
                       help='模型名称 (e.g., llama2_13b)')
    parser.add_argument('--nsamples', type=int, default=None,
                       help='样本数 (None=自动计算)')
    parser.add_argument('--force', action='store_true',
                       help='强制运行，跳过显存检查')

    args = parser.parse_args()

    success = run_experiment(args.model, args.nsamples, args.force)
    sys.exit(0 if success else 1)
