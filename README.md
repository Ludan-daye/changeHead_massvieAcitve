# Massive Activations: Mechanism and Empirical Evidence

> **Function Words as Geometric Anchors** — 25 个 LLM + ViT 的 MA 机制研究
> 核心论点：MA 是 MLP 在功能词位置写入的"语法重音 mark"，经 attention 广播并由模型调节为稳态。

---

## 一、核心机制：功能词 mark → attention 广播 → 分岔（可解释性链）

这是贯穿所有实验（RQ1–RQ6）的**单一因果链**。每条实验都是对这条链上某一步的检验。

### 步骤 1 — Mark 形成（生成端）

训练动力学决定了"在哪写 MA"：

```
功能词三重性质（Zipf 三合一）:
  - 高频     → 梯度信号充足
  - 低熵     → 任务简单，loss 快速下降
  - 低维语义 → hidden 里大量维度空闲

↓

MLP 发现 "功能词位置 × 空闲维度" 是写标记的最佳载体
→ 在 v₁ 方向把激活推到 300–3000×
→ MA = mark
```

#### 两种生成模式（Mark 写入方式）

**同一个"写 mark"的任务，不同模型用不同方式完成**。按单层消融 ΔMA 判定：

| 模式 | 判定 | 机制 | 例子 | RQ4/5 怎么做 |
|:-:|:-:|---|---|---|
| **模式 A — 单层主导** | 某层 ΔMA ≥ 85% | 一个早期 MLP 层一次性把完整 MA 写入 | GPT-J (L2 -90%)、BLOOM (L3 -95%)、Falcon (L3 -87%)、Llama-3.1 (L1 -87%)、Yi-9B (L1 -88%) | 对该起源层做 SVD、消融 v₁ |
| **模式 B — 多层协作** | 所有单层 ΔMA < 30% | 多个 MLP 层同向接力，每层写一小部分，合力形成 MA | GPT-2（L0-L4 每层最多 -6.7%，u₁ 对齐 0.856）、OPT-6.7B、Qwen3-32B、Qwen3.5-27B | macro-SVD：对 Δh₂ 多层累加做 SVD，得到 macro v₁ |
| 中间形态 | 30%–85% | 疑似"单层 + 调节"或"双层主导"，需 RQ2c 累积消融进一步判定 | Mistral-7B、Qwen1.5-14B、GLM4-9B 等 9 个 | 待判定后决定 |

**关键区别**：
```
模式 A: ablation L_origin → MA 大降
         ablation 其他层 → MA 几乎不变
         → 有明确的"起源层"

模式 B: ablation 任何单层 → MA 小降
         ablation 多层组合 → MA 大降（贪心）
         多层 W_down 的 u₁ 方向高度一致（> 0.8）
         → 整个前段是"分布式起源"
```

**不能把模式 B 简化为"模式 A + 调节"** —— 它们是真实不同的写入方式。GPT-2 的 macro-SVD 验证：η=3.48, u₁ 对齐 0.856, R²=0.870，这些数字只在把多层 Δ 累加后才出现，单层视角看不到。

**对应实验**：
- RQ2（禁 MLP → 证 MLP 是写入者，对 A 和 B 都成立）
- RQ2b（逐层消融 → 判模式 A / B）
- RQ2c（累积消融 → 确认模式判定、解决中间形态）
- RQ3（功能词 vs 内容词 v₁ 投影 → 证位置在功能词）
- RQ4 / RQ5（单层 SVD + v₁ 消融，仅对模式 A 有效）
- RQ6（macro-SVD，对模式 B 和中间形态必需）

### 步骤 2 — Attention sink（softmax 指数放大）

```
attention_weight(i→j) = softmax(Q_i · K_j / √d)

当 K_j 在 v₁ 维度携带 MA (K_j[v₁] ≈ 2000):
  Q_i · K_j ≈ 2000    ← 被这一维主导
  exp(2000/√d) ≫ exp(其他 token 的内积)
→ attention 权重几乎全部聚焦到 MA token
```

这不是模型"设计"了 attention sink，是 **softmax + MA 的必然耦合**。功能词成为 attention sink。

### 步骤 3 — 广播（V 向量横向传播）

```
output_i = Σ_j attention_weight(i→j) · V_j
        ≈ V_{MA_token}   （权重几乎全在 MA token 上）

→ 所有 token 的 attention 输出都被拉向 V_{功能词}
→ 结构信号横向传到整个序列
→ 几何上：所有 token 在 v₁ 轴上"对齐"
```

### 步骤 4 — 分岔（Generative vs Suppressive）

attention head 读到 mark 后，有两种响应方式：

| 模式 | 模型数 | V 投射方向 | 禁用 attention 后 ΔMA | 例子 |
|:-:|:-:|:-:|:-:|---|
| **Generative** | 17 | ∥ +v₁（同向）| **下降**（-20% ~ -98%） | GPT-J, BLOOM, Falcon, Llama-3.1 |
| **Suppressive** | 7 | ∥ −v₁（反向）| **暴涨**（+27% ~ +266%） | OPT-6.7B, Qwen2.5-7B, Yi-9B |

**关键**：两种方向都锁在 v₁ 轴上——没有哪种模型的 attention 会忽略 MA 去别的方向投射。这就是 MA 的 **几何锁定性**：一旦 v₁ 确定，所有下游模块只能在这个轴上选"支持"或"反对"。

#### 生成模式 × 调节模式：两个正交维度

**步骤 1 的生成模式（A 单层 / B 多层）** 和 **步骤 4 的调节模式（Generative / Suppressive）** 是**彼此独立、可自由组合**的两个维度——不能混为一谈：

|  | Generative（attention 放大）| Suppressive（attention 抑制）|
|---|---|---|
| **模式 A 单层主导** | GPT-J, BLOOM, Falcon, Llama-3.1, Qwen2.5-0.5B, Qwen3-0.6B/1.7B | Yi-9B, Qwen2-7B, Qwen2.5-7B |
| **模式 B 多层协作** | GPT-2 | OPT-6.7B, Qwen3-32B, Qwen3.5-35B-a3b |

→ 不同模型组合不同。**MLP 既参与生成（起源层），又可能参与调节（其他层负值输出抑制 MA）**。生成发生在训练初期，调节是训练后期为了稳态新学到的；两件事在架构里同时存在但可独立分析。

### 步骤 5 — 稳态维持

```
残差流      : 跨层垂直传递 MA
LayerNorm  : 限制绝对幅度上限（被动硬约束）
GELU       : 防止单维无限放大（软门控）

→ MA 收敛到某个稳定量级
```

**训练动力学实证**（Pythia-160M）：
```
step 1      : MA ≈ 2.1    （初始噪声）
step 32k    : MA ≈ 622    （达峰）
step 143k   : MA ≈ 293    （回落稳定）
```

---

## 二、机制链对实验的映射

| 步骤 | 对应 RQ | 检测什么 |
|:-:|---|---|
| 1. Mark 形成 | **RQ2** 禁 MLP → MA 消失 | MLP 是写入者 |
| 1. Mark 位置 | **RQ3** 功能词 vs 内容词 v₁ 投影 | 标记集中在功能词 |
| 1. Mark 几何 | **RQ4 / RQ5** σ₁/σ₂ + 消融 v₁ | v₁ 方向因果必要 |
| 1. Mark 多层 | **RQ6** macro-SVD | Δ 多层累加还原 macro v₁ |
| 2–3. Sink + 广播 | **RQ1** 禁 attn 看 ΔMA | attention 读取 mark |
| 4. 分岔方向 | **RQ1 mode** | Generative / Suppressive 二分 |
| 5. 稳态 | **RQ7**（本轮排除） | 训练步维度 MA 演化 |

---

## 三、三个关键推论

### 推论 A：Peak 层测 RQ3 出现 Cohen's d 反向 — 是广播步骤的实证

```
L_origin      : 功能词 h₂[v₁] ≫ 内容词 h₂[v₁]      → Cohen's d ≫ 0
L_origin + k  : attention 把 V_{功能词} 广播到所有位置
                内容词位置被"染色"
L_peak        : 所有 token 都有 +v₁ 成分，差距摊薄   → Cohen's d ≤ 0
```

当前 ALL_EXPERIMENTS_SUMMARY 中 9/16 模型在 peak 层出现 Cohen's d 负值，正是这一步的直接实证。这也是 **RQ3 必须在起源层重跑** 的物理理由。

### 推论 B：信息分轨 — 禁 v₁ 不破坏语言建模

```
hidden dim (4096)
 ├── 主导几维 (v₁ / v₂，~0.1%)
 │    = 结构信号通道 (MA)
 │    = 经 attention sink + 广播 传递
 │    
 └── 其余维度 (~99.9%)
      = 语义信号通道
      = 正常 attention，与 MA sink 不交互
```

两条通路正交不干扰。禁 v₁（RQ5）把 MA 砍 -90% 以上，但 **PPL 几乎不变**。这是把"MA 是附加结构标签、不占语义"的主张做实的关键证据。

### 推论 C：MA 的几何锁定性 — attention 没有"忽略"的选择

步骤 4 里不存在"attention 忽略 MA"的第三选项。一旦 v₁ 在训练中确定，任何 attention head 若想让输出有意义，就必须响应 v₁ 维度——要么放大（generative），要么抑制（suppressive）。这解释了为什么 25/25 模型都有明确的 Mode 归属，没有一个模型出现 ΔMA ≈ 0% 的"中立"情况。

---

## 四、仓库结构

```
ma/
├── README.md                        ← 本文件（机制可解释性）
├── paper_experiments/               ← 实验代码（按 RQ 组织）
│   ├── docs/                        ← 理论文档与进度追踪
│   │   ├── MA_FRAMEWORK.md          生成/传递/调节三部分框架
│   │   ├── MA_CONCLUSIONS_AND_ARGUMENTS.md  完整结论论证
│   │   ├── MA_WHY.md                理论解释（15 章）
│   │   ├── EXPERIMENT_PLAN.md       逐 RQ 讨论 + 机制链附录 A
│   │   ├── PROGRESS_MATRIX.md       25 模型 × 6 RQ 进度矩阵
│   │   ├── TODO_EXPERIMENTS.md      待补实验清单
│   │   ├── FINDINGS_TRACKING.md     核心发现追踪
│   │   ├── CONFLICTS.md             数据冲突记录
│   │   └── CONCLUSIONS.md           阶段性结论
│   ├── RQ1_attention_contribution/  禁 attention 实验
│   ├── RQ2_mlp_source/              禁/逐层 MLP 实验
│   ├── RQ3_function_words/          功能词 SVD 映射
│   ├── RQ4_svd_alignment/           SVD 对齐分析
│   ├── RQ5_v_matrix_ablation/       v₁ 消融因果
│   ├── RQ6_single_layer_activation/ 单层/macro-SVD
│   ├── RQ7_training_dynamics/       训练动力学（本轮排除）
│   ├── lib/                         共享库：模型加载 / hooks / 评估 / 绘图
│   ├── monkey_patch/                激活捕获 hook
│   ├── main_llm.py, main_vit.py     统一入口
│   └── results/                     实验输出
├── paper/
│   ├── Function Words as Geometric Anchors.pdf   主论文
│   ├── acl_source/                  ACL 投稿 LaTeX 源
│   └── notes_zh/                    中文笔记、rebuttal、审稿意见
├── changeHead_massvieAcitve/        老代码库（submodule）
├── figures/                         实验输出图（exp1–7 + combined）
├── archives/                        zip 归档
└── scripts/                         顶层可视化脚本
```

---

## 五、支持的模型（25 个 LLM + 若干 ViT）

### LLM（已在本框架下测试）

| 模式 | 数量 | 模型 |
|:-:|:-:|---|
| A 单层主导 | 10 | GPT-J-6B, BLOOM-7B, Falcon-7B, Llama-3.1-8B, Yi-9B, Qwen2-7B, Qwen2.5-7B, Qwen2.5-0.5B, Qwen3-0.6B, Qwen3-1.7B |
| 中间形态（待 RQ2c 精细判定） | 9 | Mistral-7B-v03, Qwen1.5-14B, GLM4-9B, Qwen3-4B/8B/14B, Qwen3.5-9B, Qwen3-30B-a3b(MoE) |
| B 多层协作 | 4 | GPT-2, OPT-6.7B, Qwen3-32B, Qwen3.5-27B, Qwen3.5-35B-a3b(MoE) |
| 数据待修复 | 2 | Llama-2-13B（缺 RQ2）、GLM4-32B（fp16 溢出） |

**Attention 分岔**：17 generative / 7 suppressive。

### ViT（changeHead_massvieAcitve 里）
- MAE (base/large/huge), CLIP, DINOv2, DINOv2-reg

---

## 六、Quick Start（外部部署）

```bash
# 1. Clone 含子模块（RQ2b 脚本在子模块里）
git clone --recurse-submodules https://github.com/Ludan-daye/changeHead_massvieAcitve.git ma
cd ma

# 或者先 clone 再补 submodule
# git clone https://github.com/Ludan-daye/changeHead_massvieAcitve.git ma
# cd ma && git submodule update --init --recursive

# 2. 安装依赖
cd paper_experiments
bash setup.sh          # 创建 conda 环境 + 装 requirements + spaCy

# 3. Pilot 验证：GPT-J-6B 在起源层 L2 做 RQ5（~30 min，HF 自动下载 24GB）
python RQ5_v_matrix_ablation/exp5_v_ablation.py \
    --model gptj_6b --layer_id 2 --nsamples 30 \
    --savedir results/wikitext_run/RQ5_origin/gptj_6b
# 预期: delta_ma.top1_mean_pct ≤ -85%

# 4. 批量跑所有 23 个模型的 RQ3/4/5 在起源层（~12h 双卡）
bash run_rq345_origin_layer.sh "" all
```

### 环境变量（可选）

| 变量 | 默认 | 作用 |
|---|---|---|
| `HF_CACHE_DIR` | `./model_weights` | HuggingFace 权重缓存目录 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 国内镜像源（用户已默认）|

### 模型路径策略

`lib/model_dict.py` 每个模型都有:
- `model_id`：本地路径（作者服务器约定）
- `hf_fallback`：HuggingFace 公开 ID（外部用户自动走这个）

本地路径不存在时，加载器会**自动 fallback 到 HF 下载**——详见 `lib/model_dict.py` 的 `resolve_model_id()`。

### 不能直接跑的模型（5 个）

下列模型没有公开 HF 发布，外部用户无法部署，本仓库暂用占位路径：
- `glm4_32b`, `qwen3.5_9b`, `qwen3.5_27b`, `qwen3.5_35b_a3b` (MoE)
- 内部用户的 `qwen3_30b_a3b` 用 FP8 权重（需设本地路径）

### 执行计划

- 主手册：`paper_experiments/docs/EXECUTION_PLAN.md`（批次顺序、验收阈值）
- 按 RQ 独立：`paper_experiments/RQ{1..6}_*/PLAN.md`（每个 RQ 自包含）

---

## 七、引用

---

## 八、引用

```
@article{...,
  title={Function Words as Geometric Anchors in Massive Activations},
  author={...},
  year={2026}
}
```

详见 `paper/Function Words as Geometric Anchors.pdf`。
