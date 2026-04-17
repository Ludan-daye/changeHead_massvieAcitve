# 模型 × Benchmark 完整对照表

## 一、Benchmark 分类

### A. 困惑度评估（PPL）

| ID | Benchmark | 语言 | 说明 | 领域使用率 |
|----|-----------|------|------|-----------|
| P1 | WikiText-2 | 英 | 语言建模标准 | 20/20 论文 |
| P2 | C4 | 英 | 大规模语料 | 18/20 |
| P3 | PG-19 | 英 | 长文本 | 4/20 |
| P4 | PTB (Penn Treebank) | 英 | 经典 PPL | 6/20 |

### B. 零样本推理任务

| ID | Benchmark | 语言 | 说明 | 领域使用率 |
|----|-----------|------|------|-----------|
| Z1 | HellaSwag | 英 | 常识推理 | 15/20 |
| Z2 | PIQA | 英 | 物理常识 | 14/20 |
| Z3 | ARC-Easy | 英 | 科学推理（简单）| 14/20 |
| Z4 | ARC-Challenge | 英 | 科学推理（难）| 14/20 |
| Z5 | WinoGrande | 英 | 指代消解 | 13/20 |
| Z6 | LAMBADA | 英 | 语言建模/预测 | 8/20 |
| Z7 | BoolQ | 英 | 是否问答 | 7/20 |

### C. 少样本评估

| ID | Benchmark | 语言 | 说明 | 领域使用率 |
|----|-----------|------|------|-----------|
| F1 | MMLU (5-shot) | 英 | 多任务知识(57科) | 10/20 |

### D. 多语言评估

| ID | Benchmark | 语言 | 说明 | 领域使用率 |
|----|-----------|------|------|-----------|
| M1 | mMMLU | 14语种 | 多语言 MMLU | 1/20（首创机会）|
| M2 | cc100-zh | 中 | 中文语料 PPL | — |
| M3 | cc100-ar | 阿 | 阿拉伯语 PPL | — |
| M4 | XCOPA | 11语种 | 跨语言因果推理 | — |

### E. 视觉评估

| ID | Benchmark | 说明 | 领域使用率 |
|----|-----------|------|-----------|
| V1 | ImageNet (Top-1) | 图像分类 | 所有 ViT 论文 |
| V2 | ADE20k (mIoU) | 语义分割 | Registers 论文 |
| V3 | CIFAR-10/100 | 小规模分类 | 2/20 |

### F. 视觉语言评估

| ID | Benchmark | 说明 | 领域使用率 |
|----|-----------|------|-----------|
| VL1 | VQAv2 | 视觉问答 | AWQ 论文 |
| VL2 | TextVQA | 文字识别问答 | AWQ 论文 |
| VL3 | MMBench | 多模态综合 | — |
| VL4 | POPE | 幻觉检测 | — |

### G. MA 专用分析

| ID | Benchmark | 说明 |
|----|-----------|------|
| MA1 | RedPajama (100 seq) | 激活幅度分析 |
| MA2 | 自定义探针句 | 功能词触发率 |

---

## 二、模型 × Benchmark 对照表

### 类别 1：论文原有模型（8 个，复现验证）

| 模型 | 路径 | 卡数 | P1 | P2 | P3 | P4 | Z1 | Z2 | Z3 | Z4 | Z5 | Z6 | Z7 | F1 | MA1 | MA2 |
|------|------|------|----|----|----|----|----|----|----|----|----|----|----|----|-----|-----|
| GPT-2 (124M) | 需下载 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | — | ✅ | ✅ |
| LLaMA-2-7B | `/model/ModelScope/shakechen/Llama-2-7b-hf` | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| BLOOM-7B1 | 需下载 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GPT-J-6B | 需下载 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Qwen2.5-7B | `/model/ModelScope/Qwen/Qwen2.5-7B-Instruct` | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| OPT-6.7B | 需下载 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Falcon-7B | 需下载 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Mistral-7B | `/model/ModelScope/AI-ModelScope/Ministral-8B-Instruct-2410` | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### 类别 2：新增 Dense LLM（跨架构扩展）

| 模型 | 路径 | 卡数 | P1 | P2 | Z1-Z7 | F1 | M1 | M2 | MA1 | MA2 |
|------|------|------|----|----|--------|----|----|----|----|------|
| Qwen3-8B | `/model/ModelScope/Qwen/Qwen3-8B` | 1 | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| GLM-4-9B-Chat | `/model/ModelScope/ZhipuAI/glm-4-9b-chat` | 1 | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gemma-3-12B-IT | `/model/HuggingFace/google/gemma-3-12b-it` | 1 | ✅ | ✅ | 全✅ | ✅ | — | — | ✅ | ✅ |
| Hunyuan-7B | `/model/HuggingFace/tencent/Hunyuan-7B-Instruct` | 1 | ✅ | ✅ | 全✅ | ✅ | — | ✅ | ✅ | ✅ |
| DeepSeek-R1-Distill-7B | `/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 1 | ✅ | ✅ | 全✅ | ✅ | — | — | ✅ | ✅ |
| LLaMA-3.1-8B | `/model/llm/Meta-Llama-3.1-8B-Instruct` | 1 | ✅ | ✅ | 全✅ | ✅ | — | — | ✅ | ✅ |
| Phi-3.5-mini | 已有 | 1 | ✅ | ✅ | 全✅ | ✅ | — | — | ✅ | ✅ |

### 类别 3：MoE 架构（回应 Reviewer 问题 9）

| 模型 | 路径 | 卡数 | P1 | P2 | Z1-Z7 | F1 | MA1 | MA2 | 特殊分析 |
|------|------|------|----|----|--------|----|----|------|---------|
| Qwen3-30B-A3B-FP8 | `/model/HuggingFace/.../Qwen3-30B-A3B-Thinking-2507-FP8` | 1 | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | Expert 间 η 对比 |
| GLM-4.7-Flash | `/model/ModelScope/ZhipuAI/GLM-4.7-Flash` | 2 | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | MoE Lite vs Dense |
| DeepSeek-V3.1 (小) | `/model/HuggingFace/deepseek-ai/DeepSeek-V3.1` | 1 | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | DeepSeek MoE |

### 类别 4：规模缩放系列（同架构不同规模）

| 模型系列 | 可用规模 | P1 | P2 | Z1-Z7 | F1 | MA1 | MA2 | 分析点 |
|---------|---------|----|----|--------|----|----|------|--------|
| Qwen3 Dense | 0.6B, 1.7B, 4B, 8B, 14B, 32B | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | η 与规模的关系 |
| Hunyuan Dense | 0.5B, 1.8B, 4B, 7B | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | 新架构 MA 模式 |
| DeepSeek-R1-Distill | 1.5B, 7B, 14B, 32B | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | 蒸馏对 MA 的影响 |
| Gemma-3 | 1B, 4B, 12B, 27B | ✅ | ✅ | 全✅ | ✅ | ✅ | ✅ | Google 架构 |

### 类别 5：多语言验证（回应 Reviewer 问题 9）

| 模型 | 路径 | 卡数 | P1 | M1 | M2 | M3 | M4 | MA1 | MA2 | 分析语言 |
|------|------|------|----|----|----|----|----|----|------|---------|
| Qwen2.5-7B | 已有 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 中/英/阿/日 |
| Qwen3-8B | 已有 | 1 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 中/英/阿/日 |
| GLM-4-9B | 已有 | 1 | ✅ | ✅ | ✅ | — | ✅ | ✅ | ✅ | 中/英 |
| BLOOM-7B1 | 需下载 | 1 | ✅ | ✅ | — | ✅ | ✅ | ✅ | ✅ | 法/阿/西 |

### 类别 6：视觉语言模型（VLM）

| 模型 | 路径 | 卡数 | P1 | Z1-Z7 | VL1 | VL2 | VL3 | VL4 | MA1 | 特殊分析 |
|------|------|------|----|----|------|------|------|------|-----|---------|
| Qwen2.5-VL-7B | `/model/ModelScope/Qwen/Qwen2.5-VL-7B-Instruct` | 1 | ✅ | 全✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 图文 token MA 对比 |
| Qwen3-VL-8B | `/model/ModelScope/Qwen/Qwen3-VL-8B-Instruct` | 1 | ✅ | 全✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 最新 VL |
| MiniCPM-V-2_6 | `/model/ModelScope/OpenBMB/MiniCPM-V-2_6` | 1 | ✅ | — | ✅ | ✅ | ✅ | ✅ | ✅ | 轻量 VLM |
| LLaVA-Llama-3-8B | `/model/HuggingFace/xtuner/llava-llama-3-8b-v1_1-transformers` | 1 | ✅ | — | ✅ | ✅ | ✅ | — | ✅ | 经典 VLM |

### 类别 7：视觉模型（ViT，已有代码基础）

| 模型 | 卡数 | V1 | V2 | V3 | MA1 | 特殊分析 |
|------|------|----|----|----|----|---------|
| DINOv2 ViT-B/L | 1 | ✅ | ✅ | ✅ | ✅ | Register token MA |
| MAE ViT-B/L/H | 1 | ✅ | — | ✅ | ✅ | 自监督 MA |
| CLIP ViT-L | 1 | ✅ | — | — | ✅ | 对比学习 MA |

---

## 三、实验优先级建议

### 第一轮（必做，直接回应 Reviewer）

| 实验 | 模型 | Benchmark | 目的 |
|------|------|-----------|------|
| 跨语言 | Qwen2.5-7B + Qwen3-8B | M1(mMMLU) + M2(cc100-zh) + MA1+MA2 | 功能词触发率在中文中是否一致 |
| MoE | Qwen3-30B-A3B-FP8 | P1+P2+Z1-Z7+MA1+MA2 | 不同 expert 的 η 和 MA 模式 |
| 消融 PPL | 原有 8 模型 | P1+P2+Z1-Z7 | V-matrix 消融后性能变化 |

### 第二轮（推荐，增强论文深度）

| 实验 | 模型 | Benchmark | 目的 |
|------|------|-----------|------|
| 规模缩放 | Qwen3 0.6B→32B | P1+P2+Z1-Z7+F1+MA1+MA2 | η 与模型规模的关系 |
| 新架构 | GLM-4-9B, Gemma-3-12B, Hunyuan-7B | P1+P2+Z1-Z7+MA1+MA2 | MA 是否是普遍现象 |
| VLM | Qwen2.5-VL-7B | VL1-VL4+MA1+MA2 | 图像 token 是否触发 MA |

### 第三轮（加分项）

| 实验 | 模型 | Benchmark | 目的 |
|------|------|-----------|------|
| 蒸馏影响 | DeepSeek-R1-Distill 系列 | P1+P2+MA1+MA2 | 蒸馏是否改变 MA 几何 |
| 更多语言 | BLOOM + Qwen3 | M3(阿拉伯)+M4(XCOPA) | 富形态语言验证 |
| ViT 扩展 | DINOv2 + MAE | V1+V3+MA1 | 跨模态验证（已有代码）|

---

## 四、Benchmark 获取方式

| Benchmark | 获取命令 | 大小 |
|-----------|---------|------|
| WikiText-2 | `datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')` | ~12MB |
| C4 | `datasets.load_dataset('allenai/c4', 'en', split='validation')` | ~360MB |
| PTB | `datasets.load_dataset('ptb_text_only')` | ~5MB |
| PG-19 | `datasets.load_dataset('emozilla/pg19', split='validation')` | ~500MB |
| HellaSwag/PIQA/... | `lm-eval --tasks hellaswag,piqa,...` | 自动下载 |
| MMLU | `lm-eval --tasks mmlu` | ~400MB |
| mMMLU | `datasets.load_dataset('cais/mmlu', 'all')` + 翻译版 | ~500MB |
| cc100-zh | `datasets.load_dataset('cc100', lang='zh')` | ~大 |
| ImageNet | 需本地准备 | ~150GB |
| VQAv2 | `datasets.load_dataset('HuggingFaceM4/VQAv2')` | ~25GB |
| MMBench | `datasets.load_dataset('opencompass/MMBench')` | ~1GB |

---

## 五、统计总览

| 类别 | 模型数 | Benchmark 覆盖 |
|------|--------|---------------|
| 论文原有 | 8 | PPL(4) + 零样本(7) + 少样本(1) + MA(2) = 14 |
| 新增 Dense | 7 | 同上 |
| MoE | 3 | 同上 + Expert 分析 |
| 规模缩放 | 4 系列(~18 模型) | 同上 |
| 多语言 | 4 | PPL + mMMLU + cc100 + XCOPA + MA = 8 |
| VLM | 4 | PPL + VQA/MMBench/POPE + MA = 8 |
| ViT | 3 系列 | ImageNet + CIFAR + MA = 4 |
| **总计** | **~45 模型** | **~20 种 Benchmark** |
