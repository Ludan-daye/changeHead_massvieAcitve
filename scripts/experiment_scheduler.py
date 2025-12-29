#!/usr/bin/env python3
"""
实验调度器
管理多个实验的顺序执行，确保显存不超限
"""

import os
import sys
import subprocess
import time
import json
from datetime import datetime
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = 'PROJECT_ROOT'
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, 'scripts')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPTS_DIR)

from gpu_monitor import GPUMonitor
from progress_reporter import ProgressReporter


class ExperimentScheduler:
    """实验调度器"""

    def __init__(self, max_memory_gb=75, nsamples=5, results_dir=None):
        """
        Args:
            max_memory_gb: 最大显存限制（GB）
            nsamples: 每个实验的样本数
            results_dir: 结果保存根目录
        """
        self.max_memory_gb = max_memory_gb
        self.nsamples = nsamples
        self.results_dir = results_dir or os.path.join(PROJECT_ROOT, 'results/experiments')

        # 组件
        self.gpu_monitor = GPUMonitor(max_memory_gb=max_memory_gb, check_interval=5)
        self.progress_reporter = ProgressReporter(report_interval=30)

        # 任务队列
        self.task_queue = []
        self.completed_tasks = []
        self.failed_tasks = []

    def add_task(self, experiment_type, model_name):
        """
        添加实验任务

        Args:
            experiment_type: 'exp3' 或 'exp6'
            model_name: 模型名称
        """
        task_id = f"{experiment_type}_{model_name}"

        savedir = os.path.join(self.results_dir, experiment_type, model_name)

        task = {
            'id': task_id,
            'experiment': experiment_type,
            'model': model_name,
            'savedir': savedir,
            'nsamples': self.nsamples
        }

        self.task_queue.append(task)

        # 添加到进度汇报器
        self.progress_reporter.add_task(task_id, {
            'model': model_name,
            'experiment': experiment_type
        })

        print(f"✓ 任务已添加: {task_id}")

    def add_batch_tasks(self, models, experiments=['exp3', 'exp6']):
        """
        批量添加任务

        Args:
            models: 模型列表
            experiments: 实验类型列表
        """
        for model in models:
            for exp in experiments:
                self.add_task(exp, model)

    def run_all(self):
        """运行所有任务"""
        print("\n" + "="*80)
        print("🚀 实验调度器启动")
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
            # 逐个执行任务
            for i, task in enumerate(self.task_queue, 1):
                print(f"\n{'='*80}")
                print(f"📋 任务 {i}/{len(self.task_queue)}: {task['id']}")
                print(f"{'='*80}\n")

                # 等待显存充足（预留20GB给模型加载）
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

                # 运行任务（独立进程）
                success = self._run_task_subprocess(task)

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
                time.sleep(5)

        finally:
            # 停止监控
            self.gpu_monitor.stop_monitoring()
            self.progress_reporter.stop_reporting()

        # 生成最终报告
        self._generate_final_summary()

    def _run_task_subprocess(self, task):
        """
        在子进程中运行任务

        Args:
            task: 任务字典

        Returns:
            bool: 是否成功
        """
        # 构建命令
        cmd = [
            'python',
            os.path.join(PROJECT_ROOT, 'scripts/experiment_runner.py'),
            '--experiment', task['experiment'],
            '--model', task['model'],
            '--savedir', task['savedir'],
            '--nsamples', str(task['nsamples'])
        ]

        print(f"执行命令: {' '.join(cmd)}\n")

        try:
            # 运行子进程
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # 实时输出
            for line in process.stdout:
                print(line, end='')

            # 等待完成
            return_code = process.wait()

            if return_code == 0:
                print(f"\n✅ 子进程成功完成")
                return True
            else:
                print(f"\n❌ 子进程失败 (返回码: {return_code})")
                return False

        except Exception as e:
            print(f"\n❌ 子进程异常: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _generate_final_summary(self):
        """生成最终汇总报告"""
        print("\n" + "="*80)
        print("📊 最终汇总报告")
        print("="*80 + "\n")

        # 进度汇报器的最终报告
        self.progress_reporter.generate_final_report()

        # 生成详细的JSON报告
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tasks': len(self.task_queue),
            'completed': len(self.completed_tasks),
            'failed': len(self.failed_tasks),
            'success_rate': len(self.completed_tasks) / len(self.task_queue) * 100 if self.task_queue else 0,
            'max_memory_gb': self.max_memory_gb,
            'peak_memory_gb': self.gpu_monitor.peak_usage / 1024,
            'completed_tasks': [t['id'] for t in self.completed_tasks],
            'failed_tasks': [t['id'] for t in self.failed_tasks]
        }

        summary_path = os.path.join(self.results_dir, 'experiment_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"✓ 详细汇总已保存: {summary_path}\n")

        # 生成Markdown报告
        self._generate_markdown_report(summary)

    def _generate_markdown_report(self, summary):
        """生成Markdown格式的汇总报告"""
        report = f"""# 实验批量运行汇总报告

## 执行信息
- **执行时间**: {summary['timestamp']}
- **总任务数**: {summary['total_tasks']}
- **完成任务**: {summary['completed']} ✅
- **失败任务**: {summary['failed']} ❌
- **成功率**: {summary['success_rate']:.1f}%

## 资源使用
- **显存限制**: {summary['max_memory_gb']} GB
- **峰值显存**: {summary['peak_memory_gb']:.1f} GB

## 完成的实验
"""

        if self.completed_tasks:
            for task in self.completed_tasks:
                report += f"- ✅ {task['id']}\n"
                report += f"  - 结果目录: `{task['savedir']}`\n"
        else:
            report += "（无）\n"

        if self.failed_tasks:
            report += "\n## 失败的实验\n"
            for task in self.failed_tasks:
                report += f"- ❌ {task['id']}\n"

        report += f"\n---\n*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"

        report_path = os.path.join(self.results_dir, 'EXPERIMENT_SUMMARY.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"✓ Markdown报告已保存: {report_path}\n")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='实验批量调度器')
    parser.add_argument('--models', type=str, nargs='+',
                        default=['gpt2', 'gptj_6b', 'falcon_7b', 'bloom_7b1', 'opt_6.7b',
                                 'mistral_7b_v03', 'qwen2.5_7b'],
                        help='模型列表')
    parser.add_argument('--experiments', type=str, nargs='+',
                        default=['exp3', 'exp6'],
                        help='实验类型列表')
    parser.add_argument('--nsamples', type=int, default=5,
                        help='每个实验的样本数')
    parser.add_argument('--max-memory', type=int, default=75,
                        help='最大显存限制（GB）')
    parser.add_argument('--results-dir', type=str, default=None,
                        help='结果保存目录')

    args = parser.parse_args()

    # 创建调度器
    scheduler = ExperimentScheduler(
        max_memory_gb=args.max_memory,
        nsamples=args.nsamples,
        results_dir=args.results_dir
    )

    # 添加任务
    scheduler.add_batch_tasks(args.models, args.experiments)

    # 运行所有任务
    scheduler.run_all()

    print("\n🎉 所有实验已完成！\n")


if __name__ == "__main__":
    main()
