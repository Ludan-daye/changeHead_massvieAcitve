#!/usr/bin/env python3
"""
归因实验批量运行器
自动为每个模型运行Exp5/7/8归因实验
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = 'PROJECT_ROOT'
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)

from gpu_monitor import GPUMonitor
from progress_reporter import ProgressReporter


# 模型的关键层配置（根据之前实验确定的MA关键层）
CRITICAL_LAYERS = {
    'gpt2': 3,
    'gptj_6b': 3,
    'bloom_7b1': 28,
    'falcon_7b': 3,
    'opt_6.7b': 3,
    'mistral_7b_v03': 31,
    'qwen2.5_7b': 3,
    'llama2_13b': 3
}


class AttributionExperimentRunner:
    """归因实验运行器"""

    def __init__(self, max_memory_gb=75, nsamples=5, results_dir=None):
        self.max_memory_gb = max_memory_gb
        self.nsamples = nsamples
        self.results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results/experiments')

        self.gpu_monitor = GPUMonitor(max_memory_gb=max_memory_gb, check_interval=5)
        self.progress_reporter = ProgressReporter(report_interval=30)

        self.task_queue = []
        self.completed_tasks = []
        self.failed_tasks = []

    def add_task(self, experiment_type, model_name):
        """
        添加归因实验任务

        Args:
            experiment_type: 'exp5', 'exp7', 或 'exp8'
            model_name: 模型名称
        """
        if model_name not in CRITICAL_LAYERS:
            print(f"⚠️  警告: {model_name} 没有配置关键层，跳过")
            return

        task_id = f"{experiment_type}_{model_name}"
        layer = CRITICAL_LAYERS[model_name]
        savedir = os.path.join(self.results_dir, experiment_type, model_name)

        task = {
            'id': task_id,
            'experiment': experiment_type,
            'model': model_name,
            'layer': layer,
            'savedir': savedir,
            'nsamples': self.nsamples
        }

        self.task_queue.append(task)

        self.progress_reporter.add_task(task_id, {
            'model': model_name,
            'experiment': experiment_type,
            'layer': layer
        })

        print(f"✓ 任务已添加: {task_id} (Layer {layer})")

    def add_batch_tasks(self, models, experiments=['exp5', 'exp7', 'exp8']):
        """批量添加任务"""
        for model in models:
            for exp in experiments:
                self.add_task(exp, model)

    def run_all(self):
        """运行所有任务"""
        print("\n" + "="*80)
        print("🚀 归因实验批量运行器")
        print("="*80)
        print(f"总任务数: {len(self.task_queue)}")
        print(f"显存限制: {self.max_memory_gb} GB")
        print(f"样本数: {self.nsamples}")
        print(f"结果目录: {self.results_dir}")
        print("="*80 + "\n")

        # 启动监控
        self.gpu_monitor.start_monitoring()
        self.progress_reporter.start_reporting(gpu_monitor=self.gpu_monitor)

        try:
            for i, task in enumerate(self.task_queue, 1):
                print(f"\n{'='*80}")
                print(f"📋 任务 {i}/{len(self.task_queue)}: {task['id']}")
                print(f"{'='*80}\n")

                # 等待显存
                required_memory_gb = 20
                print(f"检查显存... (需要至少 {required_memory_gb}GB 空闲)")

                if not self.gpu_monitor.wait_for_memory(required_memory_gb, timeout=600):
                    print(f"⚠️  显存等待超时，跳过任务: {task['id']}")
                    self.failed_tasks.append(task)
                    self.progress_reporter.complete_task(task['id'], success=False,
                                                         message="显存不足")
                    continue

                print(f"✓ 显存充足，开始任务\n")

                # 开始任务
                self.progress_reporter.start_task(task['id'])

                # 运行任务
                success = self._run_task(task)

                # 完成任务
                if success:
                    self.completed_tasks.append(task)
                    self.progress_reporter.complete_task(task['id'], success=True)
                else:
                    self.failed_tasks.append(task)
                    self.progress_reporter.complete_task(task['id'], success=False,
                                                         message="实验执行失败")

                # 等待显存释放
                print(f"\n等待显存释放...")
                import time
                time.sleep(5)

        finally:
            self.gpu_monitor.stop_monitoring()
            self.progress_reporter.stop_reporting()

        # 生成报告
        self._generate_final_summary()

    def _run_task(self, task):
        """运行单个任务"""
        # 根据实验类型构建命令
        exp_scripts = {
            'exp5': 'exp5_uv_interaction.py',
            'exp7': 'exp7_direction_superposition.py',
            'exp8': 'exp8_decomposed_attribution.py'
        }

        script = exp_scripts.get(task['experiment'])
        if not script:
            print(f"❌ 未知实验类型: {task['experiment']}")
            return False

        cmd = [
            'python',
            os.path.join(PROJECT_ROOT, 'experiments/common', script),
            '--model', task['model'],
            '--layer', str(task['layer']),
            '--savedir', task['savedir'],
            '--nsamples', str(task['nsamples'])
        ]

        # Exp7需要额外的k-values参数
        if task['experiment'] == 'exp7':
            cmd.extend(['--k-values', '1,2,3,5,10,20,50,100'])

        print(f"执行命令: {' '.join(cmd)}\n")

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            for line in process.stdout:
                print(line, end='')

            return_code = process.wait()

            if return_code == 0:
                print(f"\n✅ 任务成功完成")
                return True
            else:
                print(f"\n❌ 任务失败 (返回码: {return_code})")
                return False

        except Exception as e:
            print(f"\n❌ 任务异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_final_summary(self):
        """生成最终汇总"""
        print("\n" + "="*80)
        print("📊 归因实验汇总报告")
        print("="*80 + "\n")

        self.progress_reporter.generate_final_report()

        # 详细汇总
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tasks': len(self.task_queue),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks),
            'success_rate': len(self.completed_tasks) / len(self.task_queue) * 100 if self.task_queue else 0,
            'completed_tasks': [t['id'] for t in self.completed_tasks],
            'failed_tasks': [t['id'] for t in self.failed_tasks]
        }

        summary_path = os.path.join(self.results_dir, 'attribution_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ 详细汇总已保存: {summary_path}\n")

        # Markdown报告
        self._generate_markdown_report(summary)

    def _generate_markdown_report(self, summary):
        """生成Markdown报告"""
        report = f"""# 归因实验批量运行汇总

## 执行信息
- **执行时间**: {summary['timestamp']}
- **总任务数**: {summary['total_tasks']}
- **完成任务**: {summary['completed']} ✅
- **失败任务**: {summary['failed']} ❌
- **成功率**: {summary['success_rate']:.1f}%

## 完成的实验
"""

        if self.completed_tasks:
            for task in self.completed_tasks:
                report += f"- ✅ {task['id']} (Layer {task['layer']})\n"
                report += f"  - 结果目录: `{task['savedir']}`\n"
        else:
            report += "（无）\n"

        if self.failed_tasks:
            report += "\n## 失败的实验\n"
            for task in self.failed_tasks:
                report += f"- ❌ {task['id']}\n"

        report += f"\n---\n*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        report_path = os.path.join(self.results_dir, 'ATTRIBUTION_SUMMARY.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✓ Markdown报告已保存: {report_path}\n")


def main():
    parser = argparse.ArgumentParser(description='归因实验批量运行器')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['gpt2', 'gptj_6b', 'bloom_7b1', 'falcon_7b',
                                 'opt_6.7b', 'mistral_7b_v03', 'qwen2.5_7b'],
                        help='模型列表')
    parser.add_argument('--experiments', type=str, nargs='+',
                        default=['exp5', 'exp7', 'exp8'],
                        help='实验类型列表')
    parser.add_argument('--nsamples', type=int, default=5,
                        help='每个实验的样本数')
    parser.add_argument('--max-memory', type=int, default=75,
                        help='最大显存限制（GB）')

    args = parser.parse_args()

    # 创建运行器
    runner = AttributionExperimentRunner(
        max_memory_gb=args.max_memory,
        nsamples=args.nsamples
    )

    # 添加任务
    runner.add_batch_tasks(args.models, args.experiments)

    # 运行
    runner.run_all()

    print("\n🎉 所有归因实验已完成！\n")


if __name__ == "__main__":
    main()
