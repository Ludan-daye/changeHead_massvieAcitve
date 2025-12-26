#!/usr/bin/env python3
"""
实验执行器（带显存管理）
每个实验在独立进程中运行，完成后完全清理显存
"""

import os
import sys
import gc
import torch
import subprocess
import time
from datetime import datetime


def cleanup_gpu_memory():
    """彻底清理GPU显存"""
    print("\n🧹 清理GPU显存...")

    try:
        # 1. PyTorch清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            # 清理所有GPU
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()

            print("  ✓ PyTorch缓存已清理")

        # 2. Python垃圾回收
        gc.collect()
        print("  ✓ Python垃圾回收完成")

        # 3. 等待显存释放
        time.sleep(2)

        # 4. 检查显存
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        total_mem = sum([float(x.strip()) for x in result.stdout.strip().split('\n')])
        print(f"  ✓ 当前显存使用: {total_mem/1024:.2f} GB\n")

    except Exception as e:
        print(f"  ⚠️  清理警告: {e}\n")


def run_exp3_u_ablation(model_name, savedir, nsamples=5, k_values=None):
    """
    运行Exp3 U矩阵消融实验

    Args:
        model_name: 模型名称
        savedir: 保存目录
        nsamples: 样本数
        k_values: k值列表

    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*80}")
    print(f"🔬 开始实验: Exp3 U矩阵消融 - {model_name}")
    print(f"{'='*80}\n")

    try:
        # 导入实验模块（延迟导入避免提前加载模型）
        sys.path.insert(0, '/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve')

        from experiments.common.exp3_u_ablation import run_ablation_experiment
        import argparse

        # 构造参数
        args = argparse.Namespace(
            model=model_name,
            nsamples=nsamples,
            k_values=k_values or [1, 5, 10, 50, 100],
            savedir=savedir,
            access_token=''
        )

        # 运行实验
        results = run_ablation_experiment(args)

        # 保存结果
        import json
        from datetime import datetime

        results["timestamp"] = datetime.now().isoformat()
        json_path = os.path.join(savedir, 'u_ablation_results.json')

        os.makedirs(savedir, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ 实验完成: {model_name}")
        print(f"   结果保存至: {json_path}")

        return True

    except Exception as e:
        print(f"\n❌ 实验失败: {model_name}")
        print(f"   错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 确保清理显存
        cleanup_gpu_memory()


def run_exp6_v_ablation(model_name, savedir, nsamples=5, k_values=None):
    """
    运行Exp6 V矩阵消融实验

    Args:
        model_name: 模型名称
        savedir: 保存目录
        nsamples: 样本数
        k_values: k值列表

    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*80}")
    print(f"🔬 开始实验: Exp6 V矩阵消融 - {model_name}")
    print(f"{'='*80}\n")

    try:
        sys.path.insert(0, '/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/ludan/massActive/changeHead_massvieAcitve')

        from experiments.common.exp6_v_ablation import run_ablation_experiment
        import argparse

        args = argparse.Namespace(
            model=model_name,
            nsamples=nsamples,
            k_values=k_values or [1, 5, 10, 50, 100],
            savedir=savedir,
            access_token=''
        )

        results = run_ablation_experiment(args)

        import json
        from datetime import datetime

        results["timestamp"] = datetime.now().isoformat()
        json_path = os.path.join(savedir, 'v_ablation_results.json')

        os.makedirs(savedir, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✅ 实验完成: {model_name}")
        print(f"   结果保存至: {json_path}")

        return True

    except Exception as e:
        print(f"\n❌ 实验失败: {model_name}")
        print(f"   错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        cleanup_gpu_memory()


def run_single_experiment(experiment_type, model_name, savedir, nsamples=5):
    """
    运行单个实验（入口函数）

    Args:
        experiment_type: 'exp3' 或 'exp6'
        model_name: 模型名称
        savedir: 保存目录
        nsamples: 样本数

    Returns:
        bool: 是否成功
    """
    start_time = datetime.now()

    # 清理启动前的显存
    cleanup_gpu_memory()

    # 根据实验类型调用对应函数
    if experiment_type == 'exp3':
        success = run_exp3_u_ablation(model_name, savedir, nsamples)
    elif experiment_type == 'exp6':
        success = run_exp6_v_ablation(model_name, savedir, nsamples)
    else:
        print(f"❌ 未知实验类型: {experiment_type}")
        return False

    # 计算运行时间
    elapsed = datetime.now() - start_time
    hours = int(elapsed.total_seconds() // 3600)
    minutes = int((elapsed.total_seconds() % 3600) // 60)
    seconds = int(elapsed.total_seconds() % 60)

    if success:
        print(f"\n✅ 实验成功完成")
    else:
        print(f"\n❌ 实验失败")

    print(f"   运行时间: {hours}h {minutes}m {seconds}s")

    return success


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='运行单个实验（带显存管理）')
    parser.add_argument('--experiment', type=str, required=True, choices=['exp3', 'exp6'],
                        help='实验类型')
    parser.add_argument('--model', type=str, required=True, help='模型名称')
    parser.add_argument('--savedir', type=str, required=True, help='保存目录')
    parser.add_argument('--nsamples', type=int, default=5, help='样本数')

    args = parser.parse_args()

    success = run_single_experiment(
        experiment_type=args.experiment,
        model_name=args.model,
        savedir=args.savedir,
        nsamples=args.nsamples
    )

    sys.exit(0 if success else 1)
