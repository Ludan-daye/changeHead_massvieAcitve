#!/usr/bin/env python3
"""
Experiment 2: Robust Batch Parallel Runner
健壮的批次并行运行器 - 确保显存彻底释放
"""

import os
import sys
import time
import argparse
import subprocess
import json
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='opt_7b')
    parser.add_argument('--dataset', type=str, default='wikitext')
    parser.add_argument('--nsamples', type=int, default=30)
    parser.add_argument('--savedir', type=str, default='results/exp2_opt_6.7b/')
    parser.add_argument('--start_layer', type=int, default=3, help='Start layer to test')
    parser.add_argument('--end_layer', type=int, default=37, help='End layer to test')
    parser.add_argument('--batch_size', type=int, default=3, help='Number of parallel processes per batch')
    
    args = parser.parse_args()
    
    os.makedirs(args.savedir, exist_ok=True)
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: ROBUST BATCH PARALLEL RUNNER")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Range: Layer {args.start_layer} - {args.end_layer}")
    print(f"Batch Size: {args.batch_size} processes")
    print("="*80)
    
    # 1. 确定需要跑哪些层
    layers_to_process = []
    for layer in range(args.start_layer, args.end_layer + 1):
        result_file = os.path.join(args.savedir, f'layer_{layer}_results.json')
        if os.path.exists(result_file):
            try:
                # 验证文件是否有效
                with open(result_file, 'r') as f:
                    json.load(f)
                print(f"✓ Layer {layer} already completed.")
            except:
                print(f"⚠️ Layer {layer} result corrupted, re-queueing.")
                layers_to_process.append(layer)
        else:
            layers_to_process.append(layer)
            
    print(f"\nTotal layers to process: {len(layers_to_process)}")
    if not layers_to_process:
        print("✅ All layers completed! Exiting.")
        return

    # 2. 分批执行
    total_batches = (len(layers_to_process) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, len(layers_to_process))
        current_batch = layers_to_process[start_idx:end_idx]
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting Batch {batch_idx+1}/{total_batches}")
        print(f"Running Layers: {current_batch}")
        
        processes = []
        logs = []
        
        # 启动当前批次的进程
        for layer in current_batch:
            log_file = f"logs/exp2_layer_{layer}.log"
            os.makedirs("logs", exist_ok=True)
            
            # 调用单层测试脚本
            # 注意：这里我们调用原始的 exp2_single_layer_restoration.py
            # 但需要它支持只跑单层的模式
            # 我们修改调用方式，直接用之前写的 run_single_layer_experiment 函数
            # 或者简单点，让 exp2_parallel.py 的 worker 逻辑变成独立脚本
            
            cmd = [
                "python", "-u", "exp2_single_layer_worker.py",
                "--model", args.model,
                "--dataset", args.dataset,
                "--nsamples", str(args.nsamples),
                "--layer", str(layer),
                "--savedir", args.savedir
            ]
            
            with open(log_file, "w") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                processes.append(p)
                logs.append(log_file)
                
        # 等待当前批次完成
        for p in processes:
            p.wait()
            
        # 检查结果
        failed_layers = []
        for layer in current_batch:
            if not os.path.exists(os.path.join(args.savedir, f'layer_{layer}_results.json')):
                failed_layers.append(layer)
        
        if failed_layers:
            print(f"❌ Batch finished with failures: {failed_layers}")
        else:
            print(f"✅ Batch {batch_idx+1} completed successfully.")
            
        # 显式清理（其实进程结束后OS会自动回收，但为了保险）
        print("🧹 Cleaning up processes and memory...")
        # 这里的清理通过进程结束自然发生
        
        # 简单进度展示
        completed_count = len(layers_to_process[:end_idx]) - len(failed_layers)
        progress = completed_count / len(layers_to_process) * 100
        print(f"Progress: {progress:.1f}% ({completed_count}/{len(layers_to_process)})")
        
        # 稍微停顿确保显存完全释放
        time.sleep(5)

if __name__ == "__main__":
    main()
