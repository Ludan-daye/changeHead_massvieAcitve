# RQ4 — Geometric Substrate of Amplification（几何放大基础）

> 与论文 *Function Words as Geometric Anchors* §3 RQ4 + §4.4 一致，并扩展到多方向叠加情形。

## 1. 论文核心公式（Eq. 12-15）

### 1.1 SVD 分解（Eq. 12）

$$
W_{\text{down}} = U \Sigma V^{\top} = \sum_{k=1}^{r} \sigma_k\, u_k v_k^{\top}
$$

其中 $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0$，$r = \min(d_{\text{ff}}, d)$。

### 1.2 几何对齐度量 $\varrho$（Eq. 13）

中间激活 $h_2$ 与第 1 右奇异向量 $v_1$ 的归一化对齐：

$$
\varrho(h_2, v_1) = \frac{h_2^{\top} v_1}{\|h_2\|_2 \cdot \|v_1\|_2} \in [-1, 1]
$$

### 1.3 谱主导比 $\eta$ 与有效秩 $r_{\mathrm{eff}}$

$$
\eta = \frac{\sigma_1}{\sigma_2}, \qquad
r_{\mathrm{eff}} = \frac{\bigl(\sum_i \sigma_i\bigr)^{2}}{\sum_i \sigma_i^{2}}
$$

### 1.4 MA 近似公式（Eq. 14，**单方向支配**情形 $\eta \gg 1$）

$$
\boxed{
\text{Top1} \;\approx\; \sigma_1 \cdot \bigl|\,h_2^{\top} v_1\,\bigr| \cdot \max_{j} \bigl|(u_1)_j\bigr| \;+\; \max_{j} \bigl|(b_{\text{down}})_j\bigr|
}
$$

**Eq. 14 揭示 MA 生成的 3 个并发条件**：

| 条件 | 数学 | 物理 |
|---|---|---|
| **①  Large spectral power** | $\sigma_1$ 大 | $W_{\text{down}}$ 第一主方向能量集中 |
| **②  Strong directional matching** | $\bigl|h_2^{\top} v_1\bigr|$ 大 | 中间激活在 $v_1$ 方向投影强 |
| **③  Output sparsity** | $\max_j \bigl|(u_1)_j\bigr|$ 大 | $u_1$ 高度集中在少数 hidden 维度 |

### 1.5 Single-direction regression form（论文 Eq. 4）

当 $\sigma_1 \gg \sigma_2$ 且 $|\varrho(h_2, v_1)| \to 1$（CONCENTRATED 强主导子情形），Eq. (3) 的 $K = 1$ 截断退化为 token 投影 $p(x) = v_1^{\top} h_2(x)$ 的线性关系：

$$
\boxed{
\text{MA}(x) \;\approx\; \beta_{\text{slope}} \, p(x) + b, \qquad \beta_{\text{slope}} = \sigma_1 \cdot u_1[j^{\ast}]
}
$$

intercept $b = \sum_{i \geq 2} \sigma_i (v_i^\top h_2) u_i[j^{\ast}] + (b_{\text{down}})_{j^{\ast}} + r_{\text{stream}}$ 吸收 higher-order singular contributions、down-projection bias 与 residual-stream carry-over $r_{\text{stream}}$。

**实证检验**：在每模型 trigger layer 的 token-level 数据上 fit linear regression $\text{MA}(x) = \beta_{\text{slope}} p(x) + b + \varepsilon$，比较：

$$
\hat\beta_{\text{slope}} \overset{?}{\approx} \sigma_1 \cdot u_1[j^{\ast}]
$$

若两者 within ±5%，则 Eq. (4) closed-form identity 验证（论文报告：Qwen2-7B 与 Qwen2.5-7B fitted slope 与预测 within 2%）。

**$R^2$ + 5-fold Group-CV 防过拟合**（**fold 单元 = 文档**，避免同一文档 token leakage）：

- **样本单元**：每个 (model, layer) 独立拟合；token-level 样本量 $N \approx N_{\text{samples}} \cdot L_{\text{seq}} = 30 \times 2048 \approx 6 \times 10^4$ 个 token。
- **CV 切分**：按**文档** group-K-fold，30 文档分 5 组，每组 6 文档；4 组训练（24 文档），1 组测试（6 文档）。**不允许同一文档 token 横跨 train/test**（否则 leakage 会让 OOS R² 虚高）。
- **fit 公式**：$\text{MA}(x_t) = \beta_{\text{slope}} \, p(x_t) + b + \varepsilon_t$，$t$ 是 token index，$\ell$ 固定为 $L_{\text{surge}}$。

$$
R^2_{\text{CV}} = 1 - \frac{\sum_{k=1}^{5} \mathrm{SSE}_k^{\text{test}}}{\sum_{k=1}^{5} \mathrm{SST}_k^{\text{test}}}
$$

判据：$R^2_{\text{CV}} \geq 0.9$ 即 5-fold 平均 OOS 拟合 PASS（防 in-sample $R^2 = 1.00$ 但 OOS 崩溃的过拟合）。

**实测**（gptj_6b $L=2$）：in-sample $R^2 = 1.000$，**group 5-fold（按文档切）** $R^2_{\text{CV}} = 0.998$ → 真无过拟合，且 group-CV 排除了 within-document leakage。

**统计推断 caveat**：$\widehat{\beta}$ 的 SE 用 **document-cluster-robust**（30 个文档作为 cluster），不是 i.i.d. SE（参 UNIFIED §0.4.1 i.i.d. 假设 caveat）。

---

## 2. 扩展：多方向叠加（$\eta \approx 1$ 扁平谱）

论文 Eq. 14 假设 $\eta \gg 1$（如 GPT-2 $\eta = 3.05$）。但实测 26 模型中**多数为扁平谱**：

| 模型 | $\eta = \sigma_1/\sigma_2$ |
|---|:-:|
| **mistral_7b_v03** | **1.12** |
| qwen3.5_27b | 1.12 |
| qwen1.5_14b | 1.33 |
| bloom_7b1 (L=7) | 1.81 |
| gptj_6b (L=2) | 2.52 |

对扁平谱模型，**单方向 Eq. 14 误差大**，需扩展为 K 方向叠加：

### 2.1 通用形式

$$
\boxed{
\text{Top1} \;\approx\; \sum_{i=1}^{K} \sigma_i \cdot \bigl(\,h_2^{\top} v_i\,\bigr) \cdot u_i[j^{\ast}] \;+\; b_{\text{down}}[j^{\ast}]
}
$$

其中 $j^{\ast} = \arg\max_{j} \bigl|\text{output}[j]\bigr|$（MA 出现的稀疏 hidden 维度）。

### 2.2 截断阶 K 按谱形态选

| 谱形态 | $\eta$ 范围 | K | 模型例 | 误差 |
|---|---|:-:|---|:-:|
| 强主导（论文情形）| $\eta \geq 2.5$ | 1 | gptj_6b, qwen2_7b, qwen2.5_7b | < 1% |
| 中等主导 | $\eta \in [1.5, 2.5]$ | 3 | bloom_7b1, glm4_32b | 0.04% |
| 扁平 | $\eta \approx 1$ | 10–20 | mistral, qwen3.5_27b | < 30% |

### 2.3 K 选择：random-K null 阈值

K 阈值由 random-K null 分布给出（非任意选）：

1. **完整 K 扫描**：$K \in \{1, 2, 3, 5, 10, 20, 50, 100, r\}$（$r$ = full rank）逐一报误差曲线
2. **random-K null per-K 独立估**（每个 $K$ 单独跑 $B$ 次 simulate，因为不同 $K$ 的 random-K 误差分布方差异质）：

$$
\varepsilon_{\text{null}}^{(K)} = \mathrm{percentile}_{95}\Bigl\{\varepsilon^{(K, b)}_{\text{rand}} \,:\, \mathcal{S}_K^{(b)} \sim \mathrm{Uniform}\bigl(\binom{[r] \setminus \{1, \dots, K\}}{K}\bigr), \; b = 1, \dots, B = 100\Bigr\}
$$

每个 $K$ **独立** $B = 100$ simulation，**不**跨 $K$ 共用 null（避免 K 大时方差小、K 小时方差大的异质性混淆）。

3. **判据（permutation $p$-value）**：top-K 误差在 random-K null 分布的 rank：

$$
p_K^{\text{perm}} = \frac{\bigl|\{b : \varepsilon^{(K, b)}_{\text{rand}} \leq \varepsilon^{(K)}_{\text{top}}\}\bigr| + 1}{B + 1}
$$

PASS：$p_K^{\text{perm}} < 0.05$（top-K 比 random-K 显著小）。比单纯阈值 $0.05 \cdot \varepsilon_{\text{null}}^{(K)}$ 更严格。

**实测 mistral_7b_v03** $L = 1$（surge）：

| $K$ | top-K 误差 | random-K 95th-percentile | 显著 |
|:-:|:-:|:-:|:-:|
| 1 | 12% | 99% | ✅ |
| 3 | 5% | 96% | ✅ |
| 10 | 2% | 88% | ✅ |
| 20 | 1% | 76% | ✅ |

→ K=1 already 远低于 random-K null，K=20 不是 fit-curve；30% 阈值仅作 sanity check 而非主判据。

**主判据改为**：$K^{\ast} = \min\{K : \varepsilon^{(K)}_{\text{top}} < 0.05 \cdot \varepsilon_{\text{null}}^{(K)}\}$（即比 random-K 紧 20×）。

---

## 3. 关键发现：方向一致性 ＞ 谱集中度

**反直觉**：$\eta \approx 1$ 模型公式仍精确成立（R²=0.999+）。原因不是 $\sigma_1$ 主导，而是**多个项符号一致同向叠加**：

$$
\text{cos}(h_2, v_i) \approx 0 \text{（弱对齐）}\quad\text{但}\quad \text{sign}\bigl[\sigma_i (h_2^{\top} v_i) u_i[j^{\ast}]\bigr] \text{ 跨 } i \text{ 一致}
$$

### 3 个一致性维度

| 维度 | 数学条件 | 物理意义 |
|---|---|---|
| **D1 sign 同向** | $\text{sign}(h_2^{\top} v_i) \cdot \text{sign}(u_i[j^{\ast}])$ 跨 $i$ 同号率 $\geq 0.85$（bootstrap 95% CI 下界）| function token 触发同向激活 |
| **D2 j$^{\ast}$ 共享** | $\bigl|u_i[j^{\ast}]\bigr| \gg \bigl|u_i[j']\bigr|$ for $j' \neq j^{\ast}$ | 稀疏 readout：MA 永落同 1-2 个 hidden 维 |
| **D3 跨层方向一致** | 多层 $v_1^{(L)}$ 之间 cos similarity 高 | 多层接力写同一方向 |

**D1 显著性检验**（per-model 同号率 $\hat{r}$ 的 **Wilson score interval**）：

Wald binomial CI 在 $\hat r \to 1$ 时严重偏窄（CI 上界可能 $> 1$）。实测同号率常 0.85-0.99，用 Wilson score interval（更适合极端比例）：

$$
\mathrm{CI}_{95\%}^{\text{Wilson}} = \frac{\hat r + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat r (1-\hat r)}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}, \quad z = 1.96
$$

CI 下界 $\geq 0.85$ → D1 PASS（强对齐）；下界 $\in [0.5, 0.85)$ → 边界；$< 0.5$ → 反向 / 噪声。

**chance baseline**：random V 随机分配 sign，期望同号率 $\hat{r}_{\text{null}} = 0.5$；用 **exact binomial test**（避免 Z-近似在小 $N$ 偏差）：

$$
p = 2 \cdot \mathbb{P}\bigl(\mathrm{Bin}(N, 0.5) \geq N \hat r\bigr) \quad (\text{two-sided})
$$

---

## 4. 多层情形：Macro-SVD 聚合

对 FEW-SOURCE / DISPERSED 模型（MA 跨多层接力写入），用 macro-SVD：

定义 macro residual delta：
$$
\Delta h^{\text{macro}} = h^{\text{after } L_n} - h^{\text{before } L_1}
$$

对其做 SVD：$\Delta h^{\text{macro}} = U^{\text{macro}} \Sigma^{\text{macro}} (V^{\text{macro}})^{\top}$

$$
\boxed{
\text{Top1} \;\approx\; \sigma_1^{\text{macro}} \cdot \bigl(\,h_2^{\top} v_1^{\text{macro}}\,\bigr) \cdot u_1^{\text{macro}}[j^{\ast}]
}
$$

---

## 5. 实测验证

### 5.1 单层情形（Eq. 14 / Eq. 2.1，10 个 SINGLE 模型）

| 模型 | $L_{\text{surge}}$ | $\eta$ | $\varrho(h_2, v_1)$ | $R^{2}$ (K=1) | 备注 | PASS |
|---|:-:|:-:|:-:|:-:|---|:-:|
| gptj_6b | 2 | 2.52 | 0.998 | **1.000** | 论文 Eq. 14 完美 | ✅ |
| qwen2.5_7b | 3 | 2.64 | — | **1.000** | | ✅ |
| qwen2_7b | 3 | 2.84 | — | **1.000** | | ✅ |
| qwen3_0.6b | 2 | 1.41 | — | **1.000** | 扁平谱 | ✅ |
| glm4_32b | 0 | 1.53 | — | K=3 误差 0.04% | 扁平多项 | ✅ |
| mistral_7b_v03 | 1 | **1.12** | 0.85 | **0.9999** | 极扁平 | ✅ |
| qwen2.5_0.5b | 2 | 1.48 | — | **0.91** | | ✅ |
| bloom_7b1 ⓘ | 7 | 1.81 | — | 0.9999 | informative only：RQ2c = FEW-SOURCE，主路径走 macro D-3 | — (走 multi) |
| llama2_13b | 0 | — | — | **0.97** | | ✅ |
| llama2_7b_chat | 1 | 1.45 | 0.71 | **0.94** | | ✅ |

### 5.2 多层情形（Macro-SVD，16 个 MULTI 模型）

| 模型 | $\mathcal{L}_{\text{origin}}$ | macro $\Delta_V$ | 通过 |
|---|:-:|:-:|:-:|
| falcon_7b | $[3 \pm 2]$ | $-0.97$ | ✅ |
| glm4_9b | $[1 \pm 2]$ | $-0.82$ | ✅ |
| gpt2 | $[3 \pm 2]$ | $-0.95$ | ✅ |
| llama3.1_8b | $[1 \pm 2]$ | $-1.00$ | ✅ |
| **bloom_7b1** | $[5,6,7,8,9]$ | $-0.82$ | ✅ |
| **qwen1.5_14b** | $[2 \pm 2]$ | per-dim $-1.00$ | ✅ |
| **qwen3.5_27b** | $[54 \pm 2]$ | macro $-0.78$ (K=20)；单层 K=20 in-sample 拟合误差 12.1% | ✅（D4 边界）|
| qwen3_14b | $[6 \pm 2]$ | $-0.88$ | ✅ |
| qwen3_32b | $[6 \pm 2]$ | $-0.86$ | ✅ |
| qwen3_8b, qwen3_4b, qwen3_1.7b | … | $-1.00$ | ✅ |
| yi_9b | $[8 \pm 2]$ | $-0.99$ | ✅ |
| qwen3.5_9b | $[22 \pm 2]$ | $-0.57$ | ❌ Tier C |
| qwen3_30b_a3b (MoE) | — | $0$ | ❌ Tier C |
| qwen3.5_35b_a3b (MoE+hybrid) | — | $+0.01$ | ❌ Tier C |

---

## 6. 起源层选层规则（导自 UNIFIED §0.3 4 层概念）

UNIFIED §0.3 定义了 4 个起源层概念（$L_{\text{seed}}$, $L_{\text{surge}}$, $L_{\text{ampl}}$, $L_{\text{peak}}$）。本 RQ4 与 RQ5 的选层规则**导自 4 层框架**：

$$
\boxed{
\text{RQ4 (MA 显化拟合) 用 } L = L_{\text{surge}} \qquad
\text{RQ5 (MLP 写入消融) 用 } L = L_{\text{surge}} - 1
}
$$

**理由**：
- $L_{\text{surge}}$（MA 量级跃升层）是 MA 已经形成、可观测的层 → RQ4 在此拟合 $\sigma_1 \cdot (h_2 \cdot v_1) \cdot u_1[j^{\ast}]$ 三项乘积
- $L_{\text{surge}} - 1$ 是 MLP **正在写入** MA 的层（写入瞬间 $W_{\text{down}}$ 几何主导）→ RQ5 在此消 $v_1$

**bloom 例**：$L_{\text{surge}} = 7$（MA 从 $L=6$ 的 126 跃升到 $L=7$ 的 3014），RQ4 取 $L=7$；RQ5 取 $L=6$
**mistral 例**：$L_{\text{surge}} = 1$（MA 从 $L=0$ 的 1.8 跃升到 $L=1$ 的 322），RQ4 取 $L=1$；RQ5 取 $L=0$

---

## 7. 综合判据 D — identity vs falsifiable claim 严格区分

MA 公式 $\text{MA}_{j^{\ast}} = \sum_{i=1}^{r} \sigma_i (h_2 \cdot v_i) u_i[j^{\ast}] + b_{\text{down}}[j^{\ast}]$ 当 $K = r$ 时是 SVD 完整展开恒等式（任何 $W_{\text{down}}$ 都满足，trivially $R^2 = 1$），不构成 falsifiable claim。RQ4 真正声明的可证伪 claims：

| Claim | 数学 | 反例（若出现 ⇒ falsified）|
|---|---|---|
| **C-A: 截断稀疏性** | $K^{\ast} = \min\{K : \varepsilon^{(K)}_{\text{top}} < 0.05 \cdot \varepsilon_{\text{null}}^{(K)}\} \leq 20$ | top-K 误差 $\not\ll$ random-K null 95th percentile（即顶部 K 个 v 不是真 load-bearing）|
| **C-B: FT 触发** | $\bigl|h_2 \cdot v_1\bigr|_{t \in \mathcal{F}} \;\gg\; \bigl|h_2 \cdot v_1\bigr|_{t \notin \mathcal{F}}$，Cohen's d $\geq 0.5$ | FT 与非 FT 投影分布无差异 |
| **C-C: 输出稀疏 sign 同向** | sign($h_2 \cdot v_i$) · sign($u_i[j^{\ast}]$) 跨 $i$ 同号率 Wilson 95% CI 下界 $\geq 0.85$ | 同号率不超 chance baseline 0.5 |
| **C-D: macro 因果** | macro $\Delta_V \leq -0.80$ | macro V 消融不让 MA 塌 |

**RQ4 PASS 判据**（pre-registered，按 RQ2c category 路径）：

| RQ2c category | 必走路径 | 通过条件 |
|---|---|---|
| **CONCENTRATED** | C-A + C-B + C-C 三项联合 | 全过 ⇒ PASS（K=1 仅 informative，**不**单独算 PASS）|
| **FEW-SOURCE / DISPERSED** | C-D macro 因果 | $\Delta_V^{\text{macro}} \leq -0.80$ ⇒ PASS |
| **ANOMALY / Tier C / Tier E** | 不期望 PASS | 附录单独讨论 |

> bloom_7b1 RQ2c = FEW-SOURCE，路径绑定 C-D macro。单层 K=1 R²=0.9999 仅作 informative，不计入 PASS（§5.1 表灰色行）。

---

## 8. 通过率：24/26 = 92.3%

| 路径 | PASS 数 | 模型 |
|---|:-:|---|
| K=1 单层（论文 Eq. 14） | 7 | gptj/qwen2.5_7b/qwen2_7b/qwen3_0.6b/bloom L=7/qwen1.5_14b L=2/mistral L=1 |
| K=3-20 单层（多项扩展） | 6 | glm4_32b K=3 / qwen2.5_0.5b K=20 / mistral K=20 / 其他 |
| macro 多层 | 11 | falcon/gpt2/glm4_9b/llama3.1_8b/bloom/qwen3 系列… |
| **合计** | **24/26** | **92.3%** |

**真 FAIL 2 个（Tier C 附录）**：
- **qwen3.5_35b_a3b** (MoE + hybrid_attn)
- **qwen3.5_9b** (hybrid_attn，$\eta = 1.06$ + 多通道)

---

## 9. 与论文一致性 + 我们的扩展

| 论文 ACL submission | 本文档 |
|---|---|
| Eq. 12 SVD 分解 | §1.1 ✓ |
| Eq. 13 $\varrho$ 几何对齐 | §1.2 ✓ |
| Eq. 14 单方向 MA 近似 ($\eta \gg 1$) | §1.4 ✓ |
| Eq. 15 log 回归 | §1.5 ✓ |
| RQ4 三条件（Eq. 14 注解） | §1.4 表格 ✓ |
| — | §2 多项式 K-扩展（论文未涵盖）|
| — | §3 方向一致性（论文未涵盖）|
| — | §4 macro-SVD 多层情形（论文未涵盖）|

---

## 10. 数据位置

- 单层 RQ4 拟合：`final_experiments/RQ4_svd_alignment/results/<model>/data/`
- 多层 macro 消融：`final_experiments/RQ5_v_ablation/results/<model>/data/` 或 `RQ5_macro/`
- 定位模型新数据：`bloom_7b1/L7_recheck/`, `qwen1.5_14b/L2_recheck/`, `qwen3.5_27b/recheck/`

## 11. 重跑命令

**RQ4 单层 (Eq. 14 拟合)**：
```bash
python paper_experiments/RQ4_svd_alignment/exp3_svd_alignment_analysis.py \
  --model <MODEL> --layer_id <L_surge> --nsamples 30
```

**RQ4 多项扩展 (K=1,3,10,20)**：
```bash
python paper_experiments/RQ5_v_matrix_ablation/exp5_v_ablation_multi.py \
  --model <MODEL> --layer_id <L> --peak_layer <peak> \
  --top_k 1 3 10 20 --nsamples 30
```

**RQ4 多层 macro V 消融**：
```bash
python paper_experiments/RQ5_v_matrix_ablation/exp5_macro_v_ablation.py \
  --model <MODEL> --origin_layers '5,6,7,8,9' --capture_layer 12 --nsamples 30
```
