#!/usr/bin/env python3
"""
GPU显存监控工具
实时监控GPU显存使用，确保不超过75GB限制
"""

import subprocess
import time
import threading
from datetime import datetime


class GPUMonitor:
    """GPU显存监控器"""

    def __init__(self, max_memory_gb=75, check_interval=5):
        """
        Args:
            max_memory_gb: 最大显存限制（GB）
            check_interval: 检查间隔（秒）
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_mb = max_memory_gb * 1024
        self.check_interval = check_interval
        self.monitoring = False
        self.monitor_thread = None
        self.current_usage = {}
        self.peak_usage = 0

    def get_gpu_memory(self):
        """
        获取GPU显存使用情况

        Returns:
            dict: {gpu_id: memory_used_mb}
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )

            memory_usage = {}
            for line in result.stdout.strip().split('\n'):
                if line:
                    gpu_id, memory_mb = line.split(',')
                    memory_usage[int(gpu_id.strip())] = float(memory_mb.strip())

            return memory_usage
        except Exception as e:
            print(f"⚠️  获取GPU显存失败: {e}")
            return {}

    def get_total_memory(self):
        """获取所有GPU的总显存使用"""
        memory_usage = self.get_gpu_memory()
        return sum(memory_usage.values())

    def is_memory_available(self, required_gb=None):
        """
        检查显存是否充足

        Args:
            required_gb: 需要的显存（GB），如果None则检查是否超过上限

        Returns:
            bool: True如果显存充足
        """
        total_used_mb = self.get_total_memory()

        if required_gb is not None:
            required_mb = required_gb * 1024
            available_mb = self.max_memory_mb - total_used_mb
            return available_mb >= required_mb
        else:
            return total_used_mb < self.max_memory_mb

    def wait_for_memory(self, required_gb, timeout=300):
        """
        等待显存释放到满足要求

        Args:
            required_gb: 需要的显存（GB）
            timeout: 超时时间（秒）

        Returns:
            bool: True如果显存满足，False如果超时
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_memory_available(required_gb):
                return True

            total_used_gb = self.get_total_memory() / 1024
            print(f"⏳ 等待显存释放... 当前使用: {total_used_gb:.1f}GB / {self.max_memory_gb}GB")
            time.sleep(self.check_interval)

        return False

    def start_monitoring(self, callback=None):
        """
        启动后台监控线程

        Args:
            callback: 回调函数，当显存超限时调用
        """
        if self.monitoring:
            print("⚠️  监控已在运行")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(callback,),
            daemon=True
        )
        self.monitor_thread.start()
        print(f"✓ GPU监控已启动 (限制: {self.max_memory_gb}GB)")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        print("✓ GPU监控已停止")

    def _monitor_loop(self, callback):
        """监控循环（后台线程）"""
        while self.monitoring:
            try:
                self.current_usage = self.get_gpu_memory()
                total_mb = sum(self.current_usage.values())
                total_gb = total_mb / 1024

                # 更新峰值
                if total_mb > self.peak_usage:
                    self.peak_usage = total_mb

                # 检查是否超限
                if total_mb > self.max_memory_mb:
                    print(f"🔴 显存超限! {total_gb:.1f}GB / {self.max_memory_gb}GB")
                    if callback:
                        callback(total_gb)

                time.sleep(self.check_interval)
            except Exception as e:
                print(f"⚠️  监控异常: {e}")
                time.sleep(self.check_interval)

    def get_status_string(self):
        """获取显存状态字符串"""
        total_mb = sum(self.current_usage.values())
        total_gb = total_mb / 1024
        peak_gb = self.peak_usage / 1024
        usage_pct = (total_gb / self.max_memory_gb) * 100

        # 根据使用率选择图标
        if usage_pct < 50:
            icon = "🟢"
        elif usage_pct < 80:
            icon = "🟡"
        else:
            icon = "🔴"

        status = f"{icon} GPU显存: {total_gb:.1f}GB / {self.max_memory_gb}GB ({usage_pct:.0f}%)"

        if len(self.current_usage) > 1:
            # 多GPU详细信息
            details = " | ".join([f"GPU{i}: {mb/1024:.1f}GB" for i, mb in sorted(self.current_usage.items())])
            status += f"\n   {details}"

        status += f" | 峰值: {peak_gb:.1f}GB"

        return status

    def clear_cache(self):
        """尝试清理GPU缓存"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("✓ GPU缓存已清理")
        except ImportError:
            pass


def test_monitor():
    """测试GPU监控"""
    monitor = GPUMonitor(max_memory_gb=75, check_interval=2)

    print("="*60)
    print("GPU显存监控测试")
    print("="*60)

    # 获取当前显存
    memory_usage = monitor.get_gpu_memory()
    print(f"\n当前显存使用:")
    for gpu_id, mem_mb in memory_usage.items():
        print(f"  GPU {gpu_id}: {mem_mb/1024:.2f} GB")

    total_gb = monitor.get_total_memory() / 1024
    print(f"\n总显存使用: {total_gb:.2f} GB")
    print(f"显存限制: {monitor.max_memory_gb} GB")

    # 检查是否可用
    available = monitor.is_memory_available()
    print(f"显存状态: {'✓ 充足' if available else '✗ 不足'}")

    # 启动监控10秒
    print(f"\n启动监控10秒...")
    monitor.start_monitoring()

    for i in range(5):
        time.sleep(2)
        print(monitor.get_status_string())

    monitor.stop_monitoring()
    print("\n测试完成!")


if __name__ == "__main__":
    test_monitor()
