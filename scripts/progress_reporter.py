#!/usr/bin/env python3
"""
进度汇报器
每30秒自动汇报实验进度
"""

import time
import threading
from datetime import datetime, timedelta
from collections import defaultdict


class ProgressReporter:
    """实验进度汇报器"""

    def __init__(self, report_interval=30):
        """
        Args:
            report_interval: 汇报间隔（秒）
        """
        self.report_interval = report_interval
        self.reporting = False
        self.report_thread = None

        # 任务状态
        self.tasks = {}  # {task_id: task_info}
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.start_time = None
        self.current_task = None

        # 锁
        self.lock = threading.Lock()

    def add_task(self, task_id, task_info):
        """
        添加任务

        Args:
            task_id: 任务ID
            task_info: 任务信息字典 {'model': ..., 'experiment': ..., 'phase': ...}
        """
        with self.lock:
            self.tasks[task_id] = {
                'info': task_info,
                'status': 'pending',
                'start_time': None,
                'end_time': None,
                'progress': 0,
                'message': ''
            }
            self.total_tasks += 1

    def start_task(self, task_id):
        """开始任务"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = 'running'
                self.tasks[task_id]['start_time'] = datetime.now()
                self.current_task = task_id

    def update_task(self, task_id, progress=None, message=None):
        """
        更新任务进度

        Args:
            task_id: 任务ID
            progress: 进度百分比 (0-100)
            message: 进度消息
        """
        with self.lock:
            if task_id in self.tasks:
                if progress is not None:
                    self.tasks[task_id]['progress'] = progress
                if message is not None:
                    self.tasks[task_id]['message'] = message

    def complete_task(self, task_id, success=True, message=None):
        """完成任务"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]['status'] = 'completed' if success else 'failed'
                self.tasks[task_id]['end_time'] = datetime.now()
                if message:
                    self.tasks[task_id]['message'] = message

                if success:
                    self.completed_tasks += 1
                else:
                    self.failed_tasks += 1

                if self.current_task == task_id:
                    self.current_task = None

    def start_reporting(self, gpu_monitor=None):
        """
        启动进度汇报

        Args:
            gpu_monitor: GPU监控器实例（可选）
        """
        if self.reporting:
            print("⚠️  进度汇报已在运行")
            return

        self.reporting = True
        self.start_time = datetime.now()

        self.report_thread = threading.Thread(
            target=self._report_loop,
            args=(gpu_monitor,),
            daemon=True
        )
        self.report_thread.start()
        print(f"✓ 进度汇报已启动 (间隔: {self.report_interval}秒)")

    def stop_reporting(self):
        """停止进度汇报"""
        self.reporting = False
        if self.report_thread:
            self.report_thread.join(timeout=5)
        print("✓ 进度汇报已停止")

    def _report_loop(self, gpu_monitor):
        """汇报循环（后台线程）"""
        while self.reporting:
            time.sleep(self.report_interval)
            if self.reporting:  # 再次检查，避免刚好停止时汇报
                self._generate_report(gpu_monitor)

    def _generate_report(self, gpu_monitor=None):
        """生成并打印进度报告"""
        with self.lock:
            print("\n" + "="*80)
            print(f"📊 进度汇报 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80)

            # 总体进度
            elapsed = datetime.now() - self.start_time if self.start_time else timedelta(0)
            completed_pct = (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0

            print(f"总体进度: {self.completed_tasks}/{self.total_tasks} 任务完成 ({completed_pct:.1f}%)")
            print(f"运行时间: {self._format_timedelta(elapsed)}")

            if self.failed_tasks > 0:
                print(f"失败任务: {self.failed_tasks} ❌")

            # ETA估算
            if self.completed_tasks > 0 and self.completed_tasks < self.total_tasks:
                avg_time_per_task = elapsed.total_seconds() / self.completed_tasks
                remaining_tasks = self.total_tasks - self.completed_tasks
                eta_seconds = avg_time_per_task * remaining_tasks
                eta = timedelta(seconds=int(eta_seconds))
                print(f"预计剩余时间: {self._format_timedelta(eta)}")

            # 当前任务
            if self.current_task:
                task = self.tasks[self.current_task]
                info = task['info']
                task_elapsed = datetime.now() - task['start_time']

                print(f"\n当前任务:")
                print(f"  模型: {info.get('model', 'N/A')}")
                print(f"  实验: {info.get('experiment', 'N/A')}")
                if 'phase' in info:
                    print(f"  阶段: {info['phase']}")
                print(f"  进度: {task['progress']}%")
                if task['message']:
                    print(f"  状态: {task['message']}")
                print(f"  运行时长: {self._format_timedelta(task_elapsed)}")

            # GPU状态
            if gpu_monitor:
                print(f"\n{gpu_monitor.get_status_string()}")

            # 待完成任务列表
            pending = [tid for tid, t in self.tasks.items() if t['status'] == 'pending']
            if pending and len(pending) <= 5:
                print(f"\n待运行任务:")
                for tid in pending[:5]:
                    info = self.tasks[tid]['info']
                    print(f"  - {info.get('model', 'N/A')} / {info.get('experiment', 'N/A')}")

            print("="*80 + "\n")

    def _format_timedelta(self, td):
        """格式化时间差"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def generate_final_report(self):
        """生成最终汇总报告"""
        with self.lock:
            print("\n" + "="*80)
            print("📈 最终实验汇总报告")
            print("="*80)

            total_time = datetime.now() - self.start_time if self.start_time else timedelta(0)

            print(f"\n总任务数: {self.total_tasks}")
            print(f"完成: {self.completed_tasks} ✓")
            print(f"失败: {self.failed_tasks} ✗")
            print(f"总运行时间: {self._format_timedelta(total_time)}")

            # 成功率
            success_rate = (self.completed_tasks / self.total_tasks * 100) if self.total_tasks > 0 else 0
            print(f"成功率: {success_rate:.1f}%")

            # 详细任务列表
            print(f"\n详细任务列表:")
            print("-"*80)

            completed = [(tid, t) for tid, t in self.tasks.items() if t['status'] == 'completed']
            failed = [(tid, t) for tid, t in self.tasks.items() if t['status'] == 'failed']

            if completed:
                print("\n✓ 完成的任务:")
                for tid, task in completed:
                    info = task['info']
                    duration = task['end_time'] - task['start_time']
                    print(f"  [{self._format_timedelta(duration)}] {info.get('model', 'N/A')} / {info.get('experiment', 'N/A')}")

            if failed:
                print("\n✗ 失败的任务:")
                for tid, task in failed:
                    info = task['info']
                    print(f"  {info.get('model', 'N/A')} / {info.get('experiment', 'N/A')}")
                    if task['message']:
                        print(f"    错误: {task['message']}")

            print("="*80 + "\n")


def test_reporter():
    """测试进度汇报器"""
    reporter = ProgressReporter(report_interval=5)

    # 添加测试任务
    reporter.add_task('task1', {'model': 'gpt2', 'experiment': 'exp3'})
    reporter.add_task('task2', {'model': 'bloom', 'experiment': 'exp3'})
    reporter.add_task('task3', {'model': 'llama', 'experiment': 'exp6'})

    # 启动汇报
    reporter.start_reporting()

    # 模拟任务执行
    print("开始模拟任务执行...")

    reporter.start_task('task1')
    for i in range(5):
        time.sleep(2)
        reporter.update_task('task1', progress=i*20, message=f"Phase {i+1}/5")

    reporter.complete_task('task1', success=True)

    reporter.start_task('task2')
    time.sleep(3)
    reporter.complete_task('task2', success=False, message="模拟失败")

    reporter.start_task('task3')
    time.sleep(5)
    reporter.complete_task('task3', success=True)

    # 停止汇报
    reporter.stop_reporting()

    # 最终报告
    reporter.generate_final_report()


if __name__ == "__main__":
    test_reporter()
