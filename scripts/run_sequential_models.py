#!/usr/bin/env python3
"""
串行处理多模型，每个模型内部多线程处理样本
一次只运行一个模型，完成后清理显存再处理下一个
"""

import os
import sys
import gc
import torch
import subprocess
import time
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# 要测试的模型列表 (去掉gptj因为模型文件不完整)
MODELS = [
    ("bloom_7b1", "bloom_7b1", "bloom"),
    ("qwen2.5_7b", "qwen2.5_7b", "qwen"),
    ("mistral_7b_v03", "mistral_7b_v03", "mistral"),
    ("falcon_7b_local", "falcon_7b", "falcon"),
    ("deepseek_v2_lite", "deepseek_v2_lite", "deepseek"),
    # ("gptj_6b", "gptj_6b", "gptj"),  # 模型文件不完整，跳过
]

def clear_gpu():
    """彻底清理GPU显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    time.sleep(3)
    print("✓ GPU显存已清理")

def run_model(model_name, result_dir, log_dir, nsamples=10, num_threads=4):
    """运行单个模型实验"""
    print(f"\n{'='*60}")
    print(f"🚀 开始实验: {model_name}")
    print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")
    print(f"🧵 线程数: {num_threads}")
    print(f"{'='*60}")
    
    result_path = os.path.join(PROJECT_ROOT, f"results/models/{result_dir}/exp1")
    log_path = os.path.join(PROJECT_ROOT, f"logs/{log_dir}")
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, f"exp1_{result_dir}.log")
    
    # 设置多线程环境变量
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(num_threads)
    env['MKL_NUM_THREADS'] = str(num_threads)
    env['NUMEXPR_NUM_THREADS'] = str(num_threads)
    for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        env.pop(key, None)
    
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "experiments/common/exp1_feasibility_test.py"),
        "--model", model_name,
        "--nsamples", str(nsamples),
        "--savedir", result_path,
    ]
    
    start = time.time()
    try:
        # 实时输出到终端和日志
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT, text=True, bufsize=1
            )
            for line in process.stdout:
                print(line, end='')  # 实时打印
                f.write(line)
                f.flush()
            process.wait()
        
        elapsed = time.time() - start
        if process.returncode == 0:
            print(f"\n✅ {model_name} 完成! 耗时: {elapsed/60:.1f}分钟")
            return True
        else:
            print(f"\n❌ {model_name} 失败! 返回码: {process.returncode}")
            return False
    except Exception as e:
        print(f"\n❌ {model_name} 异常: {e}")
        return False
    finally:
        clear_gpu()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nsamples', type=int, default=10)
    parser.add_argument('--threads', type=int, default=8, help='每个实验的线程数')
    args = parser.parse_args()
    
    print("="*60)
    print("多模型串行实验 (每个模型内部多线程)")
    print(f"开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模型数: {len(MODELS)}")
    print(f"样本数: {args.nsamples}")
    print(f"线程数: {args.threads}")
    print("="*60)
    
    results = {}
    total_start = time.time()
    
    for i, (model_name, result_dir, log_dir) in enumerate(MODELS, 1):
        print(f"\n[{i}/{len(MODELS)}] 处理模型: {model_name}")
        success = run_model(model_name, result_dir, log_dir, args.nsamples, args.threads)
        results[model_name] = "✓ 成功" if success else "✗ 失败"
    
    total_time = time.time() - total_start
    
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for model, status in results.items():
        print(f"  {status}: {model}")
    print(f"\n总耗时: {total_time/60:.1f}分钟")
    
    # 保存摘要
    with open(os.path.join(PROJECT_ROOT, "results/models/exp1_summary.txt"), 'w') as f:
        f.write(f"实验完成时间: {datetime.now()}\n")
        f.write(f"总耗时: {total_time/60:.1f}分钟\n\n")
        for model, status in results.items():
            f.write(f"{model}: {status}\n")

if __name__ == "__main__":
    main()
