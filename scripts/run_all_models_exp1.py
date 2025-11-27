#!/usr/bin/env python3
"""
批量运行多模型实验1
每完成一个模型就保存数据并清理显存
"""

import os
import sys
import gc
import subprocess
import time
from datetime import datetime

# 项目根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

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

def clear_gpu_memory():
    """清理GPU显存"""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print("✓ GPU显存已清理")

def run_single_model(model_name, result_dir, log_dir, nsamples=10):
    """运行单个模型的实验"""
    print(f"\n{'='*60}")
    print(f"开始实验: {model_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
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
    
    # 设置环境变量（禁用代理）
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
            print(f"✓ {model_name} 完成! 耗时: {elapsed/60:.1f}分钟")
            return True
        else:
            print(f"✗ {model_name} 失败! 返回码: {result.returncode}")
            print(f"  查看日志: {log_file}")
            return False
            
    except Exception as e:
        print(f"✗ {model_name} 异常: {e}")
        return False
    finally:
        # 清理显存
        clear_gpu_memory()

def main():
    print("="*60)
    print("多模型批量实验1 - Massive Activation分析")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"待测模型: {len(MODELS)}个")
    print("="*60)
    
    results = {}
    
    for model_name, result_dir, log_dir in MODELS:
        success = run_single_model(model_name, result_dir, log_dir)
        results[model_name] = "成功" if success else "失败"
        
        # 每个模型之间等待5秒确保显存完全释放
        time.sleep(5)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for model, status in results.items():
        icon = "✓" if status == "成功" else "✗"
        print(f"  {icon} {model}: {status}")
    
    print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 保存结果摘要
    summary_file = os.path.join(PROJECT_ROOT, "results/models/exp1_batch_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"批量实验1结果\n")
        f.write(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for model, status in results.items():
            f.write(f"{model}: {status}\n")
    print(f"\n结果摘要已保存: {summary_file}")

if __name__ == "__main__":
    main()
