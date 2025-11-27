#!/usr/bin/env python3
"""
多线程并行运行多模型实验1
使用ThreadPoolExecutor并行处理，每完成一个模型就清理显存
"""

import os
import sys
import gc
import subprocess
import time
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# 线程锁用于打印
print_lock = threading.Lock()

# 要测试的模型列表
MODELS = [
    # 模型名, 结果目录名, 日志目录
    ("bloom_7b1", "bloom_7b1", "bloom"),
    ("gptj_6b", "gptj_6b", "gptj"),
    ("qwen2.5_7b", "qwen2.5_7b", "qwen"),
    ("mistral_7b_v03", "mistral_7b_v03", "mistral"),
    ("falcon_7b_local", "falcon_7b", "falcon"),
    ("deepseek_v2_lite", "deepseek_v2_lite", "deepseek"),
]

def safe_print(*args, **kwargs):
    """线程安全的打印"""
    with print_lock:
        print(*args, **kwargs)
        sys.stdout.flush()

def run_single_model(model_info, nsamples=10):
    """运行单个模型的实验（在独立线程中）"""
    model_name, result_dir, log_dir = model_info
    thread_id = threading.current_thread().name
    
    safe_print(f"\n[{thread_id}] 🚀 开始: {model_name} @ {datetime.now().strftime('%H:%M:%S')}")
    
    # 创建目录
    result_path = os.path.join(PROJECT_ROOT, f"results/models/{result_dir}/exp1")
    log_path = os.path.join(PROJECT_ROOT, f"logs/{log_dir}")
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    log_file = os.path.join(log_path, f"exp1_{result_dir}.log")
    
    # 构建命令
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, "experiments/common/exp1_feasibility_test.py"),
        "--model", model_name,
        "--nsamples", str(nsamples),
        "--savedir", result_path,
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        env.pop(key, None)
    
    # 运行实验
    start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            safe_print(f"[{thread_id}] ✓ {model_name} 完成! 耗时: {elapsed/60:.1f}分钟")
            return (model_name, True, elapsed)
        else:
            safe_print(f"[{thread_id}] ✗ {model_name} 失败! 返回码: {result.returncode}")
            return (model_name, False, elapsed)
            
    except Exception as e:
        safe_print(f"[{thread_id}] ✗ {model_name} 异常: {e}")
        return (model_name, False, 0)

def main():
    parser = argparse.ArgumentParser(description='多线程并行运行实验1')
    parser.add_argument('--workers', type=int, default=2, 
                        help='并行线程数 (默认2, A100 80GB建议2-3)')
    parser.add_argument('--nsamples', type=int, default=10,
                        help='每个模型的样本数')
    args = parser.parse_args()
    
    print("="*60)
    print("多模型并行实验1 - Massive Activation分析")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"待测模型: {len(MODELS)}个")
    print(f"并行线程: {args.workers}个")
    print(f"样本数: {args.nsamples}")
    print("="*60)
    
    start_time = time.time()
    results = {}
    
    # 使用线程池并行执行
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix='Model') as executor:
        # 提交所有任务
        futures = {
            executor.submit(run_single_model, model_info, args.nsamples): model_info[0] 
            for model_info in MODELS
        }
        
        # 收集结果
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                name, success, elapsed = future.result()
                results[name] = ("成功", elapsed) if success else ("失败", elapsed)
            except Exception as e:
                results[model_name] = ("异常", 0)
                safe_print(f"✗ {model_name} 执行异常: {e}")
    
    total_time = time.time() - start_time
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for model, (status, elapsed) in results.items():
        icon = "✓" if status == "成功" else "✗"
        print(f"  {icon} {model}: {status} ({elapsed/60:.1f}分钟)")
    
    print(f"\n总耗时: {total_time/60:.1f}分钟")
    print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存结果摘要
    summary_file = os.path.join(PROJECT_ROOT, "results/models/exp1_parallel_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"并行实验1结果 (线程数: {args.workers})\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_time/60:.1f}分钟\n\n")
        for model, (status, elapsed) in results.items():
            f.write(f"{model}: {status} ({elapsed/60:.1f}分钟)\n")
    print(f"\n结果摘要已保存: {summary_file}")

if __name__ == "__main__":
    main()
