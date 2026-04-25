# RQ2 MLP Source — 辅助：per-layer MA scan

> 本目录为 RQ2 的辅助实验，用于**逐层扫描**找"真起源层"。主 RQ2a（MLP 全消融）在 `../RQ2a_mlp/`。

## 目的

补 RQ2b（逐层 MLP 消融）和 per-layer MA scan，用于：
1. **发现真起源层**：当 RQ2c 的 L_origin 与 RQ2b 的 critical_layer 冲突时，跑 per-layer MA 找最早的 MA 增量层
2. **确定真起源层**：bloom_7b1 就是通过这个方法发现真起源 L=7（非 RQ2c 的 L=3），之后 RQ4/RQ5 用 L=7 定位
3. **opt_6.7b 诊断**：补数据确定真起源（task #15 pending）

## 内容

```
RQ2_mlp_source/
├── README.md                   ← 本文件
├── per_layer_scan/
│   ├── bloom_7b1/
│   │   └── bloom_per_layer_ma.json   ← L0-L29 每层 MA 值（用于找真起源 L=7）
│   ├── qwen3.5_9b_rq2b/        ← 完整 32 层 RQ2b scan
│   └── per_layer_mistral.log
└── results/
    └── opt_6.7b/secondary/     ← opt_6.7b 补跑中
```

## 定位案例

### bloom_7b1
- 原 RQ2c L_origin=3，但 RQ4 @ L=3 R²=0.0001（失败）
- per-layer MA scan 发现 L=7 才是真起源
- RQ4 @ L=7 R²=0.9999（定位）+ RQ5 K=10 ΔMA=-67% ✅

### qwen1.5_14b
- RQ2c L=35 vs RQ2b L=2（冲突 33 层）
- 选 L=2 重跑：RQ4 R²=0.9999，RQ5 mean ΔMA=-76%（接近 -80% 阈值）

### qwen3.5_27b
- 直接跑 L=54：RQ4 R²=0.9923，RQ5 单层 -78% 接近阈值

## 未完成

- **opt_6.7b**：RQ2a hook 修复后仍异常（+250%），需补 per-layer scan 确定真起源
- **qwen3.5_9b**：32 层 scan 数据已采，需分析找真起源层

## 参考

- 主 RQ2a：[`../RQ2a_mlp/README.md`](../RQ2a_mlp/README.md)
- RQ2b 脚本（老仓库）：`changeHead_massvieAcitve/experiments/exp2_mlp_layers/exp2b_mlp_layer_ablation.py`
- RQ2c 脚本：`paper_experiments/RQ6_single_layer_activation/exp6_progressive_ablation.py`
- 起源层判定：`paper_experiments/origin_layer/`
