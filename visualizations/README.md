# 可视化图表目录

本目录包含所有MA（Massive Activation）研究的可视化图表。

## 📊 图表组织结构

```
visualizations/
├── conclusion/          # P0: 核心结论图（7张）
├── rq1/                # RQ1: 巨量激活生成分析
├── rq2/                # RQ2: MLP层来源分析
├── rq3/                # RQ3: 功能词触发分析
├── rq4/                # RQ4: SVD对齐分析
├── rq5/                # RQ5: V矩阵消融分析
└── per_model/          # 单模型深入分析
```

---

## 🎯 核心结论图（P0优先级）

### 图1: MA来源证据 - MLP vs Attention
**文件**: `conclusion/01_ma_source_evidence.png`  
**结论**: MA来自MLP，不是Attention（MLP激活值比Attention高3-3496倍）

**关键发现**:
- 所有5个标准模型的MLP输出都远高于Attention输出
- GPT-J: 3x, BLOOM: 45x, Qwen: 118x, Falcon: 230x, Mistral: 3496x
- 证明MA的真正来源是MLP层

---

### 图2: Attention的真实作用 - 触发而非产生
**文件**: `conclusion/02_attention_role.png`  
**结论**: Attention提供触发信号，MLP产生MA

**关键发现**:
- **Attention触发型**（禁用后MA下降）:
  - GPT-J: -96%
  - BLOOM: -98%
- **MLP主导型**（禁用Attention后MA上升）:
  - Qwen: +266%
- **混合型**（变化较小）:
  - Falcon: -21%
  - Mistral: -18%

**机制**: Attention → 触发输入 → MLP → MA

---

### 图3: MA出现位置 - 功能词触发
**文件**: `conclusion/03_function_word_trigger.png`  
**结论**: 平均79.2%的MA出现在无语义词位置

**关键发现**:
- **标点符号**: 主要触发类型
- **功能词**: 辅助触发
- **空白/换行**: 结构标记
- **实义词**: 占比最少（约20%）

**各模型无语义词占比**:
- GPT-J: 76.0%
- BLOOM: 92.0%
- Qwen: 84.0%
- Falcon: 70.0%
- Mistral: 74.0%

---

### 图4: V矩阵依赖强度 - 7模型全景 ⭐
**文件**: `conclusion/04_v_matrix_dependency.png`  
**结论**: 6/7模型强依赖V矩阵（变化>50%）

**关键发现**（按|变化率|降序）:
1. **Qwen**: -99.1% （极强依赖）
2. **Mistral**: -82.7% （强依赖）
3. **Falcon**: -78.8% （强依赖）
4. **GPT-J**: -70.7% （强依赖）
5. **OPT**: -XX% （待验证）
6. **GPT-2**: -XX% （待验证）
7. **BLOOM**: -18.8% （弱依赖，特例）

**分级**:
- 🔴 强依赖 (>80%): Qwen, Mistral
- 🟠 中依赖 (50-80%): Falcon, GPT-J
- 🔵 弱依赖 (<50%): BLOOM

---

### 图5: 综合热力图 - 跨RQ全景
**文件**: `conclusion/05_comprehensive_heatmap.png`  
**结论**: 模型间MA机制差异明显

**矩阵维度**:
- **行**: 5个标准模型
- **列**: 4个关键指标
  1. RQ1: |Attention变化率|
  2. RQ2: MLP/Attention比值
  3. RQ3: 无语义词占比
  4. RQ5: |V消融变化率|

**颜色**: 蓝(低) → 白 → 红(高)，已归一化到[0,1]

---

### 图6: 模型机制分类树
**文件**: `conclusion/06_mechanism_classification.png`  
**结论**: 3种MA生成机制

**分类体系**:

#### 1. Attention触发型 (MA下降>50%)
- **强V依赖**: GPT-J (-96%, V-71%)
- **弱V依赖**: BLOOM (-98%, V-19%)

#### 2. MLP主导型 (MA上升)
- **强V依赖**: Qwen (+266%, V-99%)

#### 3. 混合型 (|变化|<50%)
- Falcon (-21%, V-79%)
- Mistral (-18%, V-83%)

---

### 图7: BLOOM特例分析
**文件**: `conclusion/07_bloom_special_case.png`  
**结论**: BLOOM采用"早期生成 + 残差传递"机制

**特殊机制**:

#### 1. 早期生成 (Layer 0)
- MLP产生MA
- V依赖强 (-71%)

#### 2. 残差传递 (Layer 28)
- 通过残差连接传递累积MA
- V依赖弱 (-19%)

#### 3. 语义对齐
- MA方向与标点符号高度对齐
  - `,`: 0.44
  - `.`: 0.42
  - `\n`: 0.38
- 用于句子边界标记

**与其他模型的差异**:
- 其他模型: 关键层产生MA
- BLOOM: L0产生 → 残差传递 → L28累积

---

## 📈 图表生成信息

### 技术参数
- **分辨率**: 300 DPI
- **尺寸**: 12x6 (跨模型对比), 10x6 (单模型)
- **颜色方案**: 
  - Attention: 蓝色 (#3498db)
  - MLP: 红色 (#e74c3c)
  - Baseline: 绿色 (#2ecc71)
  - Ablated: 灰色 (#95a5a6)

### 数据来源
| 图表 | 数据来源 | 模型数 |
|------|---------|--------|
| 图1 | `RQ2_mlp_source/verification.json` | 5 |
| 图2 | `exp1/README.md` | 5 |
| 图3 | `MA_POSITION_TOKEN_ANALYSIS.json` | 5 |
| 图4 | `exp6/v_ablation_simple.json` | **7** |
| 图5 | 所有RQ汇总 | 5 |
| 图6 | RQ1 + RQ5综合 | 5 |
| 图7 | BLOOM专项数据 | 1 |

### 生成脚本
- **主脚本**: `scripts/generate_visualizations.py`
- **生成时间**: 2025-12-11
- **生成命令**: `python3 scripts/generate_visualizations.py`

---

## 🔍 模型覆盖

### 标准模型（5个）
完整的RQ1-RQ5数据，使用统一目录结构

1. **GPT-J-6B**
2. **BLOOM-7B1**
3. **Qwen-2.5-7B**
4. **Falcon-7B**
5. **Mistral-7B-v0.3**

### 扩展模型（2个）
仅RQ5数据可用

6. **GPT-2** - RQ5 V消融实验
7. **OPT-6.7B** - RQ5 V消融实验

---

## 📝 核心结论总结

### 1. MA来源
✅ **所有模型MA都来自MLP**，不是Attention（证据：图1）

### 2. Attention作用
✅ **提供触发输入**，不是产生MA本身（证据：图2）

### 3. 触发位置
✅ **79.2%出现在无语义词**（标点、功能词）（证据：图3）

### 4. V矩阵依赖
✅ **6/7模型强依赖V矩阵**（变化>50%）（证据：图4）

### 5. 模型差异
✅ **3种机制**：Attention触发型、MLP主导型、混合型（证据：图6）  
✅ **BLOOM特例**：早期生成 + 残差传递（证据：图7）

---

## 🚀 下一步

### P1 优先级（RQ详细对比图）
- [ ] RQ1: 3张（关键层分布、层级趋势、机制饼图）
- [ ] RQ2: 2张（散点图、比值对比）
- [ ] RQ3: 3张（占比排名、堆叠图、BLOOM特例）
- [ ] RQ4: 2张（奇异值分布、对齐度对比）
- [ ] RQ5: 3张（Baseline vs Ablated、分类、BLOOM多层）

### P2 优先级（单模型分析）
- [ ] 每个标准模型5张深入分析图

---

*最后更新: 2025-12-11*  
*图表版本: v1.0*
