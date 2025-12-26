# 批量实验执行指南

## 🎯 功能特性

本系统提供**智能批量实验调度**，自动管理显存和进度汇报：

✅ **显存智能管理** - 自动监控，确保不超过75GB限制
✅ **自动进度汇报** - 每30秒汇报当前进度和ETA
✅ **完全显存清理** - 每个实验独立进程，完成后彻底释放显存
✅ **任务队列管理** - 顺序执行所有任务，失败自动记录
✅ **详细报告生成** - 自动生成JSON和Markdown格式的汇总报告

---

## 🚀 快速开始

### 1. 运行所有模型的Exp3和Exp6

```bash
./run_all_experiments.sh
```

这会运行：
- 7个模型: gpt2, gptj_6b, falcon_7b, bloom_7b1, opt_6.7b, mistral_7b_v03, qwen2.5_7b
- 2个实验: Exp3 (U矩阵消融), Exp6 (V矩阵消融)
- **共14个任务**，顺序执行

---

### 2. 自定义运行

#### 只运行部分模型

```bash
./run_all_experiments.sh --models gpt2,bloom_7b1,qwen2.5_7b
```

#### 只运行Exp3

```bash
./run_all_experiments.sh --experiments exp3
```

#### 调整样本数和显存限制

```bash
./run_all_experiments.sh --nsamples 10 --max-memory 60
```

#### 组合使用

```bash
./run_all_experiments.sh \
    --models gpt2,falcon_7b \
    --experiments exp3,exp6 \
    --nsamples 10 \
    --max-memory 70
```

---

## 📊 进度监控

### 实时输出示例

每30秒会自动输出进度报告：

```
================================================================================
📊 进度汇报 - 2025-01-15 14:23:45
================================================================================
总体进度: 3/14 任务完成 (21.4%)
运行时间: 1h 23m 15s
预计剩余时间: 5h 02m 30s

当前任务:
  模型: bloom_7b1
  实验: exp3
  进度: 60%
  状态: Phase 3/5 - 方式B消融测试
  运行时长: 12m 34s

🟢 GPU显存: 45.3GB / 75GB (60%)
   GPU0: 22.1GB | GPU1: 23.2GB | 峰值: 48.7GB

待运行任务:
  - falcon_7b / exp3
  - gptj_6b / exp3
  - mistral_7b_v03 / exp3
  - qwen2.5_7b / exp3
  - opt_6.7b / exp3
================================================================================
```

### 显存状态图标

- 🟢 绿色：显存使用 < 50%
- 🟡 黄色：显存使用 50-80%
- 🔴 红色：显存使用 > 80%

---

## 📁 结果文件

### 目录结构

```
results/experiments/
├── experiment_summary.json       # JSON格式汇总
├── EXPERIMENT_SUMMARY.md         # Markdown格式汇总
│
├── exp3/                         # Exp3结果
│   ├── gpt2/
│   │   ├── u_ablation_results.json
│   │   ├── u_ablation_results.png
│   │   └── U_ABLATION_SUMMARY.md
│   ├── bloom_7b1/
│   │   └── ...
│   └── ...
│
└── exp6/                         # Exp6结果
    ├── gpt2/
    │   ├── v_ablation_results.json
    │   ├── v_ablation_results.png
    │   └── V_ABLATION_SUMMARY.md
    └── ...
```

### 汇总报告内容

**experiment_summary.json**:
```json
{
  "timestamp": "2025-01-15T18:45:32",
  "total_tasks": 14,
  "completed": 13,
  "failed": 1,
  "success_rate": 92.86,
  "max_memory_gb": 75,
  "peak_memory_gb": 68.3,
  "completed_tasks": ["exp3_gpt2", "exp6_gpt2", ...],
  "failed_tasks": ["exp3_llama2_13b"]
}
```

**EXPERIMENT_SUMMARY.md**:
- 执行信息（时间、任务数、成功率）
- 资源使用（显存限制、峰值）
- 完成的实验列表
- 失败的实验列表

---

## ⚙️ 高级配置

### Python脚本直接调用

```python
from scripts.experiment_scheduler import ExperimentScheduler

# 创建调度器
scheduler = ExperimentScheduler(
    max_memory_gb=75,
    nsamples=5,
    results_dir='results/experiments'
)

# 添加任务
scheduler.add_task('exp3', 'gpt2')
scheduler.add_task('exp6', 'gpt2')
scheduler.add_task('exp3', 'bloom_7b1')

# 或批量添加
scheduler.add_batch_tasks(
    models=['gpt2', 'bloom_7b1', 'falcon_7b'],
    experiments=['exp3', 'exp6']
)

# 运行
scheduler.run_all()
```

### 调整监控间隔

编辑 `scripts/experiment_scheduler.py`:

```python
# GPU监控间隔（默认5秒）
self.gpu_monitor = GPUMonitor(max_memory_gb=max_memory_gb, check_interval=5)

# 进度汇报间隔（默认30秒）
self.progress_reporter = ProgressReporter(report_interval=30)
```

---

## 🔧 故障排除

### 问题1: 显存不足

**现象**: 任务因显存不足被跳过

**解决**:
```bash
# 降低显存限制或减少样本数
./run_all_experiments.sh --max-memory 60 --nsamples 3
```

### 问题2: 某个模型总是失败

**现象**: 特定模型的任务总是失败

**排查**:
```bash
# 单独运行该任务查看详细错误
python scripts/experiment_runner.py \
    --experiment exp3 \
    --model problematic_model \
    --savedir results/test \
    --nsamples 1
```

### 问题3: 进度汇报不显示

**检查**:
- 确认 `scripts/gpu_monitor.py` 和 `scripts/progress_reporter.py` 存在
- 确认 `nvidia-smi` 可用
- 查看是否有Python异常输出

### 问题4: 显存未释放

**手动清理**:
```bash
# 杀死所有Python进程
pkill -9 python

# 等待显存释放
sleep 5

# 检查显存
nvidia-smi
```

---

## 📈 性能优化建议

### 1. 根据GPU数量调整样本数

- **单GPU (24GB)**: `--nsamples 3`
- **双GPU (48GB)**: `--nsamples 5`  ✅ 推荐
- **四GPU (96GB)**: `--nsamples 10`

### 2. 优先运行小模型

```bash
# 先运行小模型，节省时间
./run_all_experiments.sh --models gpt2,gptj_6b,falcon_7b
# 再运行大模型
./run_all_experiments.sh --models bloom_7b1,opt_6.7b,mistral_7b_v03,qwen2.5_7b
```

### 3. 夜间批量运行

```bash
# 后台运行，输出重定向
nohup ./run_all_experiments.sh > experiments.log 2>&1 &

# 查看实时日志
tail -f experiments.log
```

---

## 📞 技术支持

### 系统架构

```
run_all_experiments.sh
    ↓
experiment_scheduler.py (主调度器)
    ├── gpu_monitor.py (显存监控)
    ├── progress_reporter.py (进度汇报)
    └── experiment_runner.py (单个实验执行)
            ├── exp3_u_ablation.py
            └── exp6_v_ablation.py
```

### 关键组件

| 组件 | 功能 | 文件 |
|-----|------|------|
| **调度器** | 任务队列管理 | `experiment_scheduler.py` |
| **GPU监控** | 显存实时监控 | `gpu_monitor.py` |
| **进度汇报** | 30秒汇报进度 | `progress_reporter.py` |
| **实验执行** | 独立进程运行 | `experiment_runner.py` |

### 日志位置

- **实时输出**: 终端标准输出
- **汇总报告**: `results/experiments/EXPERIMENT_SUMMARY.md`
- **详细JSON**: `results/experiments/experiment_summary.json`

---

## 🎉 快速测试

运行一个快速测试（只用2个样本）：

```bash
./run_all_experiments.sh \
    --models gpt2 \
    --experiments exp3 \
    --nsamples 2
```

预计运行时间：~10分钟

---

**Happy Experimenting! 🚀**
