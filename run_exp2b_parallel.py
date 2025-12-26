#!/usr/bin/env python3
"""
并行运行MLP逐层抑制实验 - 充分利用A100 80G显存

策略:
  1. GPT-2 (小模型): 可以同时运行多层
  2. 7B模型: 顺序执行各层
  3. 13B模型: 顺序执行各层

优化:
  - 智能估计显存需求
  - 动态调整并行度
  - 断点续传支持
"""

import subprocess
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time

PROJECT_ROOT = "/home/vicuna/ludan/massActive/changeHead_massvieAcitve"
SCRIPT = f"{PROJECT_ROOT}/experiments/common/exp2b_mlp_layer_ablation.py"

# 模型配置
MODELS_CONFIG = {
    'gpt2': {
        'name': 'gpt2',
        'size': '124M',
        'estimated_vram': 2,  # GB
        'priority': 1  # 优先级
    },
    'bloom_7b1': {
        'name': 'bloom_7b1',
        'size': '7B',
        'estimated_vram': 16,
        'priority': 2
    },
    'falcon_7b': {
        'name': 'falcon_7b',
        'size': '7B',
        'estimated_vram': 16,
        'priority': 2
    },
    'gptj_6b': {
        'name': 'gptj_6b',
        'size': '6B',
        'estimated_vram': 14,
        'priority': 2
    },
    'mistral_7b_v03': {
        'name': 'mistral_7b_v03',
        'size': '7B',
        'estimated_vram': 16,
        'priority': 2
    },
    'opt_7b': {
        'name': 'opt_7b',
        'size': '6.7B',
        'estimated_vram': 15,
        'priority': 2
    },
    'qwen2.5_7b': {
        'name': 'qwen2.5_7b',
        'size': '7B',
        'estimated_vram': 16,
        'priority': 2
    },
    'llama2_13b': {
        'name': 'llama2_13b',
        'size': '13B',
        'estimated_vram': 28,
        'priority': 3
    }
}

# 实验参数
NSAMPLES = 10
TOTAL_VRAM = 80  # A100 80G


def run_model_experiment(model_name, nsamples=10):
    """运行单个模型的实验"""
    print(f"\n{'='*80}")
    print(f"🔬 开始实验: {model_name}")
    print(f"{'='*80}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    cmd = [
        'python3', SCRIPT,
        '--model', model_name,
        '--nsamples', str(nsamples),
        '--n_jobs', '1'
    ]

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"✅ {model_name} 完成")
        print(f"用时: {elapsed/60:.1f} 分钟")
        return (model_name, True, elapsed)

    except subprocess.CalledProcessError as e:
        print(f"❌ {model_name} 失败: {e}")
        return (model_name, False, 0)
    except Exception as e:
        print(f"❌ {model_name} 异常: {e}")
        return (model_name, False, 0)


def check_completed_models():
    """检查已完成的模型"""
    completed = []
    for model_name in MODELS_CONFIG.keys():
        savedir = f"{PROJECT_ROOT}/results/models/{model_name}/exp2b_mlp_layer_ablation"
        summary_file = f"{savedir}/summary.json"
        if os.path.exists(summary_file):
            completed.append(model_name)
    return completed


def main():
    print("="*80)
    print("🚀 MLP逐层抑制实验 - 批量并行运行")
    print("="*80)
    print(f"GPU: A100 80G")
    print(f"总显存: {TOTAL_VRAM}GB")
    print(f"样本数: {NSAMPLES}")
    print(f"模型数: {len(MODELS_CONFIG)}")
    print("="*80)

    # 检查已完成的模型
    completed_models = check_completed_models()
    if completed_models:
        print(f"\n✅ 已完成 {len(completed_models)} 个模型:")
        for m in completed_models:
            print(f"  - {m}")

    # 待运行的模型
    pending_models = [m for m in MODELS_CONFIG.keys() if m not in completed_models]

    if not pending_models:
        print(f"\n✅ 所有模型已完成！")
        return

    print(f"\n📝 待运行 {len(pending_models)} 个模型:")
    for m in pending_models:
        config = MODELS_CONFIG[m]
        print(f"  - {m} ({config['size']}, ~{config['estimated_vram']}GB)")

    # 按优先级排序 (小模型优先)
    pending_models_sorted = sorted(
        pending_models,
        key=lambda m: (MODELS_CONFIG[m]['priority'], MODELS_CONFIG[m]['estimated_vram'])
    )

    print(f"\n🔄 执行顺序:")
    for i, m in enumerate(pending_models_sorted, 1):
        config = MODELS_CONFIG[m]
        print(f"  {i}. {m} ({config['size']})")

    # 开始实验
    start_time = time.time()
    results = []

    print(f"\n{'='*80}")
    print("开始运行实验...")
    print(f"{'='*80}")

    # 策略1: 小模型GPT-2可以尝试并行，但为了稳定性，仍然顺序执行
    # 策略2: 7B和13B模型顺序执行
    for model_name in pending_models_sorted:
        config = MODELS_CONFIG[model_name]

        # 评估是否有足够显存
        if config['estimated_vram'] > TOTAL_VRAM:
            print(f"⚠️  警告: {model_name} 预计需要 {config['estimated_vram']}GB，可能超出显存限制")

        # 运行实验
        result = run_model_experiment(model_name, nsamples=NSAMPLES)
        results.append(result)

        # 短暂休息，清理显存
        print(f"⏸️  休息5秒，清理显存...")
        time.sleep(5)

    # 总结
    elapsed_total = time.time() - start_time
    hours = int(elapsed_total // 3600)
    minutes = int((elapsed_total % 3600) // 60)
    seconds = int(elapsed_total % 60)

    print(f"\n{'='*80}")
    print("📊 实验总结")
    print(f"{'='*80}")
    print(f"总用时: {hours}h {minutes}m {seconds}s")

    successful = [r for r in results if r[1]]
    failed = [r for r in results if not r[1]]

    print(f"\n✅ 成功: {len(successful)}/{len(results)}")
    for model_name, _, elapsed in successful:
        print(f"  - {model_name} ({elapsed/60:.1f}分钟)")

    if failed:
        print(f"\n❌ 失败: {len(failed)}")
        for model_name, _, _ in failed:
            print(f"  - {model_name}")

    print(f"\n📁 结果保存在: results/models/{{model}}/exp2b_mlp_layer_ablation/")
    print("="*80)


if __name__ == '__main__':
    main()
