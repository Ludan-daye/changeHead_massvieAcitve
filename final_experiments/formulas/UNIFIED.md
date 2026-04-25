# MA 机制统一公式纲要（按流程 / 整体性）

> 本文档把 RQ1-RQ6 的所有公式**按 MA 生成流程串联**，用统一符号体系，从 token 输入到 MA 因果验证一气呵成。
>
> 每个公式既有数学定义，也有该步在 MA 链中的物理意义。

## 论文 6 RQ 命名（§3.2）

| RQ | 名称 | 角色 |
|:-:|---|---|
| **RQ1 Source** | Attention 是否生成 MA | attention 是 regulator 而非 physical writer |
| **RQ2 Localization** | MLP down-projection 是否 substrate | $W_{\text{down}}$ 是物理基础 |
| **RQ3 Trigger** | Function token 是否优先激活 | FT 是 geometric anchor |
| **RQ4 Mechanism** | SVD 公式是否预测 MA magnitude | $\sigma_i (h_2 \cdot v_i) u_i[j^{\ast}]$ |
| **RQ5 Causality** | V-matrix geometry 是否 causally necessary（载体性）| 破坏 v₁ ⇒ MA 塌 |
| **RQ6 Sufficiency** | Single-layer recovery 测 residual-stream dependence | 反向 sufficiency 验证 |

---

## 0. 全局符号体系（跨 RQ 共享）

### 0.1 张量与索引

| 符号 | 含义 | 维度 |
|---|---|---|
| $\mathbf{H}_{\ell, t, j}$ | 第 $\ell$ 层、token 位置 $t$、hidden 维度 $j$ 的激活 | 标量 |
| $\mathbf{H}_{\ell} \in \mathbb{R}^{B \times L \times d}$ | 第 $\ell$ 层 hidden state 张量 | (batch, seq, hidden) |
| $\mathbf{H}_{\ell}^{\text{attn}}$ | 第 $\ell$ 层 attention sub-layer 输出 | 同上 |
| $\mathbf{H}_{\ell}^{\text{mlp}}$ | 第 $\ell$ 层 MLP sub-layer 输出 | 同上 |
| $\mathcal{I} = \{1,\dots,B\} \times \{1,\dots,L\} \times \{1,\dots,d\}$ | 全索引集 | — |

### 0.2 MA 关键位置（RQ3 + RQ4）

| 符号 | 定义 | 出现在 |
|---|---|---|
| $i^{\ast}$ | MA 出现的 **token 位置** | RQ3 |
| $j^{\ast}$ | MA 出现的 **hidden 维度** | RQ3-RQ6 |
| $\mathcal{F}$ | 广义 function token 集合 | RQ3 |

**Definition 1 (Massive activation)**（论文）：

$$
T = P_{0.999}(|\mathbf{A}|), \qquad \mathcal{M} = \{a \in \mathbf{A} : |a| > T\}
$$

即 99.9 percentile 阈值上的激活值集合。这是 operational candidate definition；orders-of-magnitude scale 的 claim 仅引用 $\text{Top1}$ 而不是仅靠 percentile rule。

### 0.3 起源层 4 层概念（RQ2 扩展，控制 RQ4-RQ6 选层）

| 概念 | 含义 | 检测 |
|---|---|---|
| $L_{\text{seed}}$ | MA 第一次出现的层（小信号）| per-layer scan first $> 100$ |
| $L_{\text{surge}}$ | **MA 量级跃升的层（真起源）** | first 5× jump |
| $L_{\text{ampl}}$ | RQ2b critical_layer（关掉降最多）| 逐层 MLP 消融 |
| $L_{\text{peak}}$ | MA 绝对值最大的层 | $\arg\max_{\ell} \max\bigl|\mathbf{H}_{\ell}\bigr|$ |

**关键**：

$$
\boxed{
\text{RQ4 用 } L = L_{\text{surge}} \;\;(\text{MA 显化层});\;\;
\text{RQ5 用 } L = L_{\text{surge}} - 1 \;\;(\text{MLP 写入层})
}
$$

### 0.4 MLP 内部（RQ2 + RQ4 + RQ5）

| 符号 | 含义 | 维度 |
|---|---|---|
| $W_{\text{up}}$ | up-projection（hidden → intermediate）| $d_{\text{ff}} \times d$ |
| $W_{\text{down}}$ | down-projection（intermediate → hidden）| $d \times d_{\text{ff}}$ |
| $b_{\text{up}}, b_{\text{down}}$ | 偏置（老架构如 BLOOM/OPT/GPT-2 非零；新架构 LLaMA/Qwen 系 = 0）| 向量 |
| $\phi$ | 激活函数（GELU / SiLU）| — |
| $h_2 = \phi(W_{\text{up}} x + b_{\text{up}})$ | MLP 中间激活 | $\mathbb{R}^{d_{\text{ff}}}$ |
| $W_{\text{down}} = U \Sigma V^{\top}$ | down-projection 的 SVD | — |
| $\sigma_i, u_i, v_i$ | 第 $i$ 组奇异值 / 左 / 右奇异向量 | $\sigma_i \in \mathbb{R}, u_i \in \mathbb{R}^{d}, v_i \in \mathbb{R}^{d_{\text{ff}}}$ |
| $\eta = \sigma_1 / \sigma_2$ | 谱主导比 | — |
| $r_{\mathrm{eff}} = \frac{(\sum_i \sigma_i)^{2}}{\sum_i \sigma_i^{2}}$ | 有效秩 | — |
| $r$（rank）| $= \min(d, d_{\text{ff}})$ | — |

### 0.4.0 全局统计协议（**必读**：所有 PASS rate / CI 都遵守这套规则）

**A. 主报告基本单元 = 模型** （而非 model × RQ × K × 判据 600+ 假设）

每个模型在每个 RQ 只算 **1 个 PASS**（取该 RQ 多路径 D1/D2/D3/D4 的 best path），全局共 26 × 6 = 156 检验，跨 RQ 用 Benjamini-Hochberg FDR $q < 0.05$ 校正。

**B. 单层 vs 多层判据按 RQ2c.category 绑定**

模型按 RQ2c.category 分类，对应判据：

| RQ2c category | 主路径 | 备用路径 |
|---|---|---|
| CONCENTRATED | D1 (单层 V 消融 $\Delta_V \leq -0.80$) | D2 (per_dim) |
| FEW-SOURCE / DISPERSED | D3 (macro V 消融) | D4 (边界 -0.78) |
| ANOMALY (opt_6.7b) | Tier E 附录 | — |
| Tier C (MoE/hybrid) | Tier C 附录 | — |

例：bloom_7b1 RQ2c 分类 = FEW-SOURCE → RQ5 走 D3 macro，$\Delta_V = -0.82$ PASS。

**C. Top1 极值统计量必须报 95% CI**

所有 $\text{Top1}^{*}$、$r_{\text{res}}$、$\Delta_V$、$\tau$、$R^2$ 配 bootstrap 95% CI（按文档 cluster resample $B = 1000$）：

$$
r_{\text{res}}(\text{gptj\_6b}) = 0.017 \, [0.012, 0.024], \quad \Delta_V = -0.99 \, [-0.993, -0.985]
$$

**Max-bias caveat**：$\text{Top1} = \max_{(b,t,j)} |\mathbf{H}|$ 是极值统计量，在有限样本下偏低估计 $\mathbb{E}[\text{Top1}]$，量级 $O(1/\log N_{\text{tokens}}) \approx 5\%$（$N \sim 6 \times 10^4$）。bootstrap CI 估 sampling variance 但不修正 finite-sample max-bias；GEV tail-fitting 可作 unbiased estimator（future work）。

**Cluster bootstrap caveat**：$N_{\text{samples}} = 30$ 文档 cluster 偏紧（经验法则 $G \geq 50$，Cameron & Miller 2015）。推荐 **wild-cluster bootstrap-t**（Cameron, Gelbach & Miller 2008）作为主推断：

$$
t^{\ast(b)} = \frac{\hat\theta^{\ast(b)} - \hat\theta}{\widehat{\mathrm{SE}}_{\text{CR}}^{\ast(b)}}, \quad \mathrm{CI}_{95\%} = \hat\theta \pm t_{0.975}^{(B)} \cdot \widehat{\mathrm{SE}}_{\text{CR}}
$$

$\hat\theta^{\ast(b)}$ 用 Rademacher weights ($+1/-1$) 在文档级 score residual 上重抽，$B = 999$ replications。

### **D. 全局多重比较校正**

26 模型 × 6 RQ × 多 K × 多判据 ≈ 600+ 检验。每模型每 RQ 取**最佳路径** PASS（按 §0.4.0.B pre-register），全局 156 检验做 BH-FDR 校正：

$$
q_i^{\text{BH}} = \min_{j \geq i} \frac{p_{(j)} \cdot 156}{j}, \qquad \text{PASS} \iff q_i^{\text{BH}} < 0.05
$$

### 0.4.1 采样协议（**必读**：所有 PASS rate / CI 都基于此）

**数据**：wikitext-103 验证集，文档独立采样。

| 参数 | 默认值 | 说明 |
|---|:-:|---|
| $N_{\text{samples}}$ | **30** | 抽 30 个独立文档（i.i.d.）|
| $L_{\text{seq}}$ | 2048 | 每个文档截 2048 token |
| $N_{\text{samples}}^{\text{boundary}}$ | **60** | 边界除零模型（如 qwen2_7b baseline ≈ 0）用 60 复测 |
| Bootstrap $B$ | 1000 | resample documents 计算 95% CI |

**i.i.d. 假设caveat**：跨文档独立，但**同一文档跨 token / 跨层不独立**。所有跨 layer / 跨 token 的统计推断（如 R²、Wald SE）需用 **document-cluster-robust SE**（即按文档聚类）。

### 0.5 各 RQ 度量（统一 Top1 命名）

> **统一约定**（**全局唯一命名**）：所有公式中"baseline / disabled / V-ablated 后的最大激活值"统一记为 `\text{Top1}^{...}`（首字母大写，无 lowercase 变体）。这与论文 Eq. 1（MA 现象定义）、Eq. 2（候选标量量级）、Eq. 3（per-layer Top1 = $\max |\mathbf{H}_{\ell}|$）一致。

| 符号 | RQ | 定义 | 论文 Eq. |
|---|---|---|---|
| $\text{Top1}_{\ell}$ | 全部 | 第 $\ell$ 层最大激活绝对值 | Eq. 3 |
| $r_{\text{res}}$ | RQ1 | residual ratio：关 attention 后 MA 残留 | Eq. 6 派生 |
| $\Delta_{\text{attn}}$ | RQ1 | $(\text{Top1}^{\text{dis,attn}} - \text{Top1}^{\text{base}}) / \text{Top1}^{\text{base}}$ | Eq. 6 |
| $\Phi_{\text{Attn}}$ | RQ1 | attention 消融算子 $\mathbf{H}^{\text{attn}} \to \mathbf{0}$ | Eq. 5 |
| $\rho_{\ell}$ | RQ2 | MLP / attention 主导比 | Eq. 8 |
| $\tau$ | RQ2a | 关全 MLP 后保留率 | §4.3 |
| $\pi_{\text{func}}$ | RQ3 | function token trigger rate | Eq. 11 |
| $\varrho(h_2, v_1)$ | RQ4 | 几何对齐（cosine） | Eq. 13 |
| $\Delta_V$ | RQ5 | V 消融后 MA 变化率 | Eq. 18 |
| $r_{\text{recovery}}$ | RQ6 | top-K 保留后 MA 恢复率 | —（项目扩展） |

---

## 1. MA 生成机制全链（流程图）

```
       Input tokens
              ↓
       Embedding (L=0)
              ↓ ──────────────────────────────────────────────────────────────┐
                                                                                │
   For each layer ℓ ∈ [0, n-1]:                                                 │
                                                                                │  RQ1
       H_{ℓ-1} → Attention → H^{attn}_ℓ                                         │  (attn 是放大器/抑制器，
              ↓                                                                  │   非起源)
       (H_{ℓ-1} + H^{attn}_ℓ) → LayerNorm                                       │
              ↓                                                                  │
       MLP 内部 ↓ ── (W_up, b_up, φ, W_down, b_down) ─────── RQ2  ──────────────┤
                                                                                │  RQ2
       h_2 = φ(W_up · x + b_up)                                                 │  (MLP 是物理基础
              ↓                                                                  │   ρ_ℓ > 1, τ ≤ 0.10)
       W_down 的 SVD: U Σ V^T = Σ σ_i u_i v_i^T  ── RQ4 几何对齐 ───────────────┤
              ↓                                                                  │  RQ3 + RQ4
       Token i* = FT (∈ F): h_2 在 v_1 投影强 ── RQ3 触发 ────────────────────┤  (FT 触发 + 几何对齐)
              ↓                                                                  │
       Output[j*] = Σ σ_i (h_2 · v_i) u_i[j*] + b_down[j*] ──── RQ4 主公式 ──┤
              ↓                                                                  │
       MA 形成：稀疏 hidden 维度 j* 上的极值                                      │
              ↓                                                                  │
   ============== 因果验证 ==============                                       │  RQ5
       破坏 v_1 / macro v_1 → MA 塌 (Δ_V ≤ -0.80) ── RQ5 ────────────────────┤  (load-bearing；sufficient for elimination)
              ↓                                                                  │
       保留单层 top-K → MA 复 (r_recovery ≥ 0.30) ── RQ6 ──────────────────────┘  RQ6 (反向恢复)
```

---

## 2. RQ1 → RQ2: 起源定位（attention 不是 / MLP 是）

### 2.1 RQ1：Attention 消融证伪 H₀

**实验**：禁用全部 attention 头，测 MA 是否归零。

**Top1 统一定义**（论文 Eq. 1 / Eq. 2 / Eq. 3 → 本文档统一记号）：

$$
\text{Top1}_{\ell} = \max_{(b,t,j) \in \mathcal{I}}\bigl|\mathbf{H}_{\ell, t, j}\bigr|, \qquad
\text{Top1} = \max_{\ell} \text{Top1}_{\ell}
$$

定义 baseline / disabled：

$$
\text{Top1}^{\text{base}} = \max_{(b,t,j) \in \mathcal{I}}\bigl|\mathbf{H}_{\ell, t, j}\bigr|, \qquad
\text{Top1}^{\text{dis,attn}} = \max_{(b,t,j) \in \mathcal{I}}\bigl|\mathbf{H}_{\ell, t, j}\bigr|_{\,\Phi_{\text{Attn}}}
$$

**消融算子**（论文 Eq. 5）：$\Phi_{\text{Attn}}: \mathbf{H}_{\ell}^{\text{attn}} \to \mathbf{0}$（所有层 attention sub-layer 输出清零）。

**主判据**（残留率，对应论文 Eq. 6 $\Delta_{\text{Top1}}$ 的归一化形式）：

$$
\boxed{r_{\text{res}} = \frac{\text{Top1}^{\text{dis,attn}}}{\text{Top1}^{\text{base}}}, \qquad r_{\text{res}} > 0 \;\Longrightarrow\; \text{H}_0 \text{ 证伪}}
$$

**模式分类**（Generative vs Suppressive）：

$$
\Delta_{\text{attn}} = \frac{\text{Top1}^{\text{dis,attn}} - \text{Top1}^{\text{base}}}{\text{Top1}^{\text{base}}}
$$

- $\Delta_{\text{attn}} < 0$：Generative（17 模型，attention 是放大器）
- $\Delta_{\text{attn}} > 0$：Suppressive（8 模型，attention 是抑制器；如 OPT $\Delta_{\text{attn}} = +7.44$）

**通过率**：26/26 = 100%（所有模型 attention 关掉 MA 都未归零）。

### 2.2 RQ2：MLP 是物理基础

**结构**（论文 Eq. 7）：

$$
\boxed{
\text{MLP}(\mathbf{x}) = W_{\text{down}} \cdot \phi\bigl(W_{\text{up}} \mathbf{x} + b_{\text{up}}\bigr) + b_{\text{down}}
}
$$

**MLP / Attention 主导比**（论文 Eq. 8）：

$$
\rho_{\ell} = \frac{\max_{(b,t,j) \in \mathcal{I}} \bigl|\mathbf{H}_{\ell, t, j}^{\text{mlp}}\bigr|}{\max_{(b,t,j) \in \mathcal{I}} \bigl|\mathbf{H}_{\ell, t, j}^{\text{attn}}\bigr|}
$$

**假设检验**：$H_0: \rho_{\ell} = 1$ vs $H_1: \rho_{\ell} > 1$（论文 26/26 验证 $H_1$）。

**消融判据**（关全部 MLP）：

$$
\boxed{
\tau = \frac{\text{Top1}^{\text{dis,mlp}}}{\text{Top1}^{\text{base}}}
}
$$

**主报告改连续指标 + bootstrap CI**（详见 RQ2 §2.2）：dense 22 模型 $\tau$ median = $0.020$（IQR $[0.005, 0.038]$）。离散判据（$\tau \leq 0.10$ 严格 / $\tau \leq 0.15$ 边界）作为 sanity 附录，不作主报告。

**通过率**：dense 22 模型 $\tau \leq 0.15$ 22 个；$\tau \leq 0.10$ 21 个（glm4_32b $\tau = 0.126$ 边界）。

---

## 3. RQ3 → RQ4: 触发机制（FT 在 v₁ 投影强）

### 3.1 RQ3：广义 function token 触发

**MA 位置定位**（论文 Eq. 9）：

$$
\boxed{
i^{\ast} = \arg\max_{i \in [1, L]} \max_{d \in [1, D]} \bigl|\mathbf{H}_{\ell, i, d}\bigr|
}
$$

**广义 function token 集合**（扩展论文 spaCy POS）：

$$
\mathcal{F} = \mathcal{F}_{\text{paper}} \,\cup\, \mathcal{F}_{\text{struct}} \,\cup\, \mathcal{F}_{\text{digit}} \,\cup\, \mathcal{F}_{\text{bpe-frag}}
$$

其中：
- $\mathcal{F}_{\text{paper}} = \{\text{ADP, DET, AUX, CONJ, PRON}\}$（spaCy POS）
- $\mathcal{F}_{\text{struct}}$：换行 `\n\n`、标点 `.,!?`、特殊符号 `@#`
- $\mathcal{F}_{\text{digit}}$：数字
- $\mathcal{F}_{\text{bpe-frag}}$：短 BPE 碎片

**主判据**：

$$
\text{PASS} \iff \text{token}(i^{\ast}) \in \mathcal{F}
$$

**辅助 1**（论文 Eq. 10 Fisher's exact test）：

$$
p = \frac{\binom{n_{1+}}{n_{11}} \binom{n_{2+}}{n_{21}}}{\binom{N}{n_{+1}}}, \qquad p < 0.001 \Rightarrow \text{显著}
$$

**辅助 2**（论文 Eq. 11 trigger rate）：

$$
\pi_{\text{func}} = \frac{n_{11}}{n_{+1}}
$$

**辅助 3**（u₁ decode）：

$$
\text{logits}_{u_1} = W_U \cdot u_1, \qquad \text{Top-K} = \mathrm{topk}_{t \in \text{vocab}} \bigl(\text{logits}_{u_1}[t]\bigr)
$$

实测：6/7 CONC 模型 v₁/v₂ Top-1 反解 token 是同一 FT。

**通过率**：24/26 = 92.3%。

### 3.2 RQ4：几何对齐 + MA 生成主公式

**SVD 分解**（论文 Eq. 12，对应 §0.4）：

$$
W_{\text{down}} = U \Sigma V^{\top} = \sum_{i=1}^{r} \sigma_i \, u_i v_i^{\top}
$$

**几何对齐**（论文 Eq. 13）：

$$
\varrho(h_2, v_1) = \frac{h_2^{\top} v_1}{\|h_2\|_2 \cdot \|v_1\|_2} \in [-1, 1]
$$

**MA 单方向近似**（论文 Eq. 14，$\eta \gg 1$）：

$$
\text{Top1} \approx \sigma_1 \cdot \bigl|h_2^{\top} v_1\bigr| \cdot \max_j \bigl|(u_1)_j\bigr| + \max_j \bigl|(b_{\text{down}})_j\bigr|
$$

**3 个并发条件**（来自 Eq. 14）：
1. **大谱能量** $\sigma_1$ 大
2. **强方向匹配** $|h_2^{\top} v_1|$ 大（**RQ3 的 FT 触发就是这个！**）
3. **输出稀疏** $\max_j |(u_1)_j|$ 大（**RQ3 的 u₁ decode 集中在 FT 维度就是这个！**）

**多项式扩展**（应对 $\eta \approx 1$ 扁平谱）：

$$
\boxed{
\text{MA}_{j^{\ast}} = \sum_{i=1}^{K} \sigma_i \cdot \bigl(h_2 \cdot v_i\bigr) \cdot u_i[j^{\ast}] + b_{\text{down}}[j^{\ast}]
}
$$

K 按谱形态选：$\eta \geq 2.5$ 用 K=1；$\eta \in [1.5, 2.5]$ 用 K=3；$\eta \approx 1$ 用 K=10~20。

**多层情形**（macro-SVD）：

$$
\Delta h^{\text{macro}} = h^{\text{after } L_n} - h^{\text{before } L_1}, \qquad \Delta h^{\text{macro}} = U^{\text{macro}} \Sigma^{\text{macro}} (V^{\text{macro}})^{\top}
$$

$$
\text{MA}_{j^{\ast}} \approx \sigma_1^{\text{macro}} \cdot \bigl(h_2 \cdot v_1^{\text{macro}}\bigr) \cdot u_1^{\text{macro}}[j^{\ast}]
$$

**方向一致性**（关键扩展）：扁平谱模型公式仍精确成立，因为：

$$
\text{cos}(h_2, v_i) \approx 0 \;\text{但}\; \text{sign}\bigl[\sigma_i (h_2 \cdot v_i) u_i[j^{\ast}]\bigr] \;\text{跨 } i \;\text{一致}
$$

**通过率**：24/26 = 92.3%。

---

## 4. RQ4 → RQ5: 因果验证（破坏 v₁ → MA 塌）

### 4.1 RQ5：V 消融

**论文方法**（Eq. 16，随机正交替换）：

$$
\tilde{W}_{\text{down}} = U \Sigma \tilde{V}^{\top}, \qquad \tilde{V} \sim \text{QR}\bigl(\mathcal{N}(0, 1)^{d_{\text{ff}} \times d_{\text{ff}}}\bigr)
$$

**理论预期**（Eq. 17）：

$$
\mathbb{E}[\Delta_V] \approx 1 - \frac{1}{\sqrt{d_{\text{ff}}}}
$$

**实测变化率**（Eq. 18）：

$$
\boxed{
\Delta_V = \frac{\text{Top1}^{V\text{-ablated}} - \text{Top1}^{\text{base}}}{\text{Top1}^{\text{base}}}, \qquad \Delta_V \leq -0.80 \;\Rightarrow\; \text{PASS}
}
$$

**扩展 1：multi-K 投影消除**（替代论文随机替换）：

$$
\tilde{W}_{\text{down}} = W_{\text{down}} \cdot \biggl(\mathbf{I} - \sum_{i=1}^{K} v_i v_i^{\top}\biggr) = \sum_{i=K+1}^{r} \sigma_i u_i v_i^{\top}
$$

**扩展 2：macro V 消融**（多层情形）：

$$
\tilde{W}_{\text{down}}^{(\ell)} = W_{\text{down}}^{(\ell)} \cdot \bigl(\mathbf{I} - v_1^{\text{macro}} (v_1^{\text{macro}})^{\top}\bigr) \quad \forall \ell \in \mathcal{L}_{\text{origin}}
$$

**扩展 3：bias 消融对照**（区分 σ·v·u 因果 vs bias 贡献）：

$$
\Delta^{\text{V-only}}: \tilde{W}, \, b_{\text{down}}\text{ 不动} \quad\text{vs}\quad \Delta^{\text{V+bias}}: \tilde{W}, \, b_{\text{down}} \to \mathbf{0}
$$

实测 gptj：$\Delta^{\text{V-only}} = -0.99$，$\Delta^{\text{V+bias}} = -0.898$ → bias 占 ~10%。

**综合判据 D1-D4**（任一过 PASS）：

| 路径 | 公式 | 阈值 |
|:-:|---|:-:|
| D1 单层 multi-K | $\Delta_V$ K=1~K=20 | $\leq -0.80$ |
| D2 per_dim 强证据 | 主 MA dim ΔMA | $\leq -1.00$ |
| D3 macro 多层 | $\Delta_V$ 跨层 | $\leq -0.80$ |
| D4 边界放宽 | $\Delta_V$ | $\leq -0.78$ |

**通过率**：dense 主体 (pre-registered 22) 20/22 = **90.9%**；全 26 模型 21/26 = 80.8%。

---

## 5. RQ5 → RQ6: 反向验证（保 top-K → MA 复）

### 5.1 RQ6：Top-K Recovery

对起源层保留 top-K 激活：

$$
\tilde{h}^{(L)}_{j} = \begin{cases} h^{(L)}_{j}, & j \in \mathrm{topk}_{j'} \bigl|h^{(L)}_{j'}\bigr| \\ 0, & \text{otherwise} \end{cases}
$$

**Recovery rate**：

$$
\boxed{
r_{\text{recovery}} = \frac{\text{Top1}^{\text{topk-keep}}}{\text{Top1}^{\text{base}}}
}
$$

**分层判据**（双向证伪）：

$$
\text{PASS} \iff
\begin{cases}
r_{\text{recovery}} \geq 0.30, & \text{若 CONCENTRATED（单层主导）} \\
r_{\text{recovery}} < 0.30, & \text{若多层（一致性，单层不足以恢复）}
\end{cases}
$$

**RQ6 ↔ RQ5 互证关系**：

$$
\begin{aligned}
&\text{单层 RQ5 PASS}\;(\Delta_V \leq -0.80) \\
\iff \;& \text{单层 RQ6 PASS}\;(r_{\text{recovery}} \geq 0.30)
\end{aligned}
$$

**通过率**：6/6 ⭐⭐⭐ 双过 = 2 个（gptj_6b 76% + llama3.1_8b 49%）；dense 一致性 16/23 = 70%。

---

## 6. 整体 MA 链：从 token 到 MA 的统一公式

把所有公式串联，**MA 在第 $\ell = L_{\text{surge}}$ 层第 $j^{\ast}$ 维生成**：

$$
\boxed{
\text{MA}_{\ell = L_{\text{surge}}, \, t = i^{\ast}, \, j = j^{\ast}}
\;=\;
\sum_{i=1}^{K}
\underbrace{\sigma_i}_{\text{RQ4 ① 谱能量}}
\cdot
\underbrace{\bigl(h_2(t = i^{\ast}) \cdot v_i\bigr)}_{\text{RQ3 ② FT 触发：FT 投影强}}
\cdot
\underbrace{u_i[j^{\ast}]}_{\text{RQ4 ③ 输出稀疏}}
+
\underbrace{b_{\text{down}}[j^{\ast}]}_{\text{Bias 项 (老架构)}}
}
$$

**逐项解释**：

| 项 | 物理意义 | 验证 RQ |
|---|---|:-:|
| $\sigma_i$ | $W_{\text{down}}$ 第 $i$ 主方向能量（多大）| RQ4 § |
| $h_2 \cdot v_i$ | 中间激活 $h_2$ 在右奇异向量 $v_i$ 上投影（**FT 触发关键**）| RQ3 + RQ4 ② |
| $u_i[j^{\ast}]$ | 左奇异向量 $u_i$ 在稀疏维度 $j^{\ast}$ 的分量（**MA 落点**） | RQ3 u₁ decode + RQ4 ③ |
| $b_{\text{down}}[j^{\ast}]$ | down-projection bias（老架构非零，gptj ~10% 贡献）| RQ5 §4 |

**MA 写在 $i^{\ast}$ 位置 = FT 位置**（RQ3）；**MA 显化在 $L_{\text{surge}}$**（RQ4）；**MLP 是写入器**（RQ2）；**attention 是放大器/抑制器**（RQ1）；**消 v₁ 让 MA 塌**（RQ5）；**保 top-K 让 MA 复**（RQ6）。

---

## 7. 通过率全表（边界放宽口径）

| RQ | 主公式 | 判据 | PASS / 总 | 率 |
|:-:|---|---|:-:|:-:|
| **RQ1** | $r_{\text{res}} = \text{Top1}^{\text{dis,attn}} / \text{Top1}^{\text{base}}$ | $r_{\text{res}} > 0$ | **26 / 26** | **100%** |
| **RQ2** (dense 22) | $\rho_{\ell} > 1$ + $\tau$ 连续指标（详见 RQ2 §2.2）| dense 主体 PASS | **21 / 22** | **95.5%** |
| **RQ3** | token($i^{\ast}$) ∈ $\mathcal{F}$ | Top-1 ∈ 广义 FT 集合 | **24 / 26** | **92.3%** |
| **RQ4** | $\text{MA} = \sum \sigma_i (h_2 v_i) u_i[j^{\ast}] + b$ | K=1 R²≥0.9 / K=20 误差≤0.30 / macro≤-0.80 任一 | **24 / 26** | **92.3%** |
| **RQ5（单层组 10）** | 单层 $\Delta_V \leq -0.80$ OR per_dim ≤ -1.00 | CONC 类 V 消融 | **9 / 10** | **90%** |
| **RQ5（多层组 16）** | macro $\Delta_V \leq -0.80$ | FS+DISP 类 macro V 消融 | **12 / 16** | **75%** |
| **RQ5 合计** (dense 22) | 单层 / 多层 pre-register 路径任一过 | 20/22 = 90.9% | **20 / 22** | **90.9%** |
| **RQ6（单层组 10）** | $r_{\text{recovery}} \geq 0.30$（期望高） | CONC 单层主导反向恢复 | **1 / 10** | **10%** |
| **RQ6（多层组 16）** | $r_{\text{recovery}} < 0.30$（期望低，一致性） | 多层接力，单层不足 | **15 / 16** | **94%** |
| **RQ6 dense 23** | 分层判据综合（去 3 架构特异） | RQ5 ↔ RQ6 互证 | **16 / 23** | **70%** |

### dense 主体定义

**全局 pre-registered exclusion list**（4 个架构特异模型）：

$$
\mathcal{M}_{\text{anomaly}} = \{\text{opt\_6.7b (Tier E), qwen3.5\_9b, qwen3.5\_35b\_a3b, qwen3\_30b\_a3b}\} \;\;(\text{4 个：1 OPT + 2 hybrid + 1 MoE})
$$

$$
\boxed{
\mathcal{M}_{\text{dense}} = \mathcal{M}_{\text{all}} \setminus \mathcal{M}_{\text{anomaly}}, \qquad |\mathcal{M}_{\text{dense}}| = 26 - 4 = 22
}
$$

**所有 RQ 的 PASS rate 都用 22 作分母**（不再随 RQ 浮动），在主报告中：

| RQ | $|\mathcal{M}_{\text{dense}}|$ | PASS | 率 |
|:-:|:-:|:-:|:-:|
| RQ1 | 22 | 22 | 100% |
| RQ2a | 22 | 21 | 95.5%（glm4_32b $\tau = 0.126$ 边界）|
| RQ3 | 22 | 21 | 95.5%（qwen2.5_0.5b 严格 POS 边界）|
| RQ4 | 22 | 21 | 95.5%（qwen2.5_0.5b 边界；按 §0.4.0.B pre-register 路径）|
| RQ5 | 22 | 20 | 90.9%（qwen2.5_0.5b $\Delta_V^{\text{mean}} = -0.55$ 边界 + qwen1.5_14b D2-PASS）|
| RQ6 | 22 | 仅 2 直测 | — (data-incomplete) |

**4 个 anomaly 模型在论文 Appendix Tier C / E 单独讨论**，不计入 main result PASS rate。

**关键禁止条款**（bloom 例）：
- bloom_7b1 RQ2c category = **FEW-SOURCE**（pre-registered）
- 必走 macro V 消融路径 D3：$\Delta_V^{\text{macro}} = -0.82$ ✅ PASS
- **单层 RQ4 R² = 0.9999 仅作 informative table（标灰）**，不算 PASS 路径
- 同理 qwen1.5_14b、qwen3.5_27b 也不允许"两边都试"

---

## 8. 与论文 Eq. 一一对应

| 论文 Eq. | 论文章节 | 本文档 § | 我们扩展 |
|:-:|---|:-:|---|
| Eq. 3 | RQ pre | §0.5（Top1 定义）| — |
| Eq. 4 | RQ1 | §2.1 残差流分解 | — |
| Eq. 5 | RQ1 | §2.1 $\Phi_{\text{Attn}}: \mathbf{H}^{\text{attn}} \to \mathbf{0}$ | — |
| Eq. 6 | RQ1 | §2.1 $\Delta_{\text{Top1}}$ | — |
| Eq. 7 | RQ2 | §2.2 MLP 内部结构 | — |
| Eq. 8 | RQ2 | §2.2 $\rho_{\ell}$ 主导比 | — |
| Eq. 9 | RQ3 | §3.1 $i^{\ast}$ 定位 | — |
| Eq. 10 | RQ3 | §3.1 Fisher's exact test | — |
| Eq. 11 | RQ3 | §3.1 $\pi_{\text{func}}$ | — |
| Eq. 12 | RQ4 | §3.2 SVD 分解 | — |
| Eq. 13 | RQ4 | §3.2 $\varrho$ 几何对齐 | — |
| Eq. 14 | RQ4 | §3.2 单方向 MA 近似 | — |
| Eq. 15 | RQ4 | §3.2 log 回归 | — |
| Eq. 16 | RQ5 | §4.1 随机 V 替换 | multi-K + macro + bias 扩展 |
| Eq. 17 | RQ5 | §4.1 理论预期 | — |
| Eq. 18 | RQ5 | §4.1 $\Delta_V$ 实测 | D1-D4 多路径判据 |
| —（论文无）| RQ6 | §5 | top-K recovery（辅助）|
| —（论文无）| 起源层 4 层 | §0.3 | 实证扩展 |
| —（论文无）| 多项式 K-扩展 | §3.2 | 应对扁平谱 |
| —（论文无）| macro-SVD 多层 | §3.2 + §4.1 | FS/DISP 模型 |
| —（论文无）| bias 消融对照 | §4.1 | 区分 σ·v·u vs bias |

---

## 8.1 Reviewer 全 14 问题逐项回应

### 公式 / 数学相关（**已加进 formulas/**）

| # | Reviewer 提问 | 类型 | 回应位置 |
|:-:|---|:-:|---|
| **#1** qarC | Eq. 17 数学不一致 | solved | RQ5 §1.2 完整 4 步推导（球面方向矩 + Jensen 上界 + MA 比值）|
| **#3** bDKj | LayerNorm 混淆变量 | solved | RQ2 §1.2 Pre-LN/Post-LN 残差流 + LN 不变性论证 |
| **#4** daTc | 功能词 ≠ 高频词解耦 | ask | RQ3 §2.2 4 象限 + Logistic 回归 + PMI（双向不对称证伪频率假说）|
| **#5** daTc | RQ5 消融下游影响 | ask | RQ5 §4.4 $\Delta_{\text{PPL}}$ 公式 + 正向支持论证（PPL 升高 ⇒ MA 是关键特征）|
| **#6** daTc | 归一化 / 激活函数差异 | solved | RQ2 §1.2.1 8 模型多变体对照表 |
| **#9** | 跨语言 / 架构泛化 | solved | 本节（限制：英语 + dense ≤ 14B；机制依赖 $W_{\text{down}}$ 频谱几何，跨语言不否定）|
| **#14** daTc | 量化/压缩应用 | — | RQ5 §4.4.1 明确"验证性实验，非部署方案" |

### 非公式相关（**叙事 / 数据 / 编辑层**，不影响 formulas/）

| # | Reviewer 提问 | 处理位置 |
|:-:|---|---|
| **#2** bDKj | 样本量 / 文档独立性 | 论文 Methodology 章节补充 nsamples=30, seqlen=2048, wikitext 独立采样说明 |
| **#7** daTc | Figure 4 光谱可视化 | 论文 Figure 4 重画；formulas/ 改用 $\eta = \sigma_1/\sigma_2$ 数值表代替主观视觉 |
| **#8** daTc/qarC | Qwen2.5 语义 token | RQ3 §4.1 Tab 摘录已标 ⚠️；论文叙事承认架构演进 |
| **#10** bDKj | 弱化"profound discovery" | 论文文风调整（如 "we find" 替代 "we fundamentally reveal"）|
| **#11** qarC | 拼写 "Valur/Bath/Dracton" | 论文 Figure 2/6 修正 |
| **#12** qarC | "76% to 100%" vs Tab 1 | 论文统一用 Tab 1 精确数字 |
| **#13** bDKj | 论文结构（方法+结果交织）| 论文章节重组（不影响公式集）|

### 总览

公式相关 7 项**已全部回应到** formulas/；非公式 7 项需在论文 LaTeX 主稿 (`acl_source/`) 处理，不影响本文档集合。

## 9. 主结论一句话

> **MA 是 MLP 在 surge_layer 的 down-projection 上、由广义 function token 在 $v_1$ 方向触发、经 $\sigma_1$ 谱能量放大、落到 $u_1[j^{\ast}]$ 稀疏维度形成的极端激活；attention 仅作为下游调节器（放大器或抑制器）。**
>
> **统一公式**：$\text{MA}_{j^{\ast}} = \sum_{i=1}^{K} \sigma_i (h_2 \cdot v_i) u_i[j^{\ast}] + b_{\text{down}}[j^{\ast}]$
>
> **6 RQ 联合验证**：
> - RQ1 attention 不是起源（100%）
> - RQ2 MLP 是物理基础（88.5%）
> - RQ3 FT 触发（92.3%）
> - RQ4 公式拟合（92.3%）
> - RQ5 V 消融因果（80.8%）
> - RQ6 top-K 反向恢复（dense 70%）
>
> **dense 主体（pre-registered 22 = 26 − 4 anomaly）综合 PASS：~91-95% per-RQ**（每 RQ 1-2 个边界）；4 anomaly 模型在 Tier C / E 附录单独讨论。

---

## 10. 各 RQ 详细文档索引

| RQ | 详细公式 | 通过率 |
|:-:|---|:-:|
| RQ1 | [RQ1_formula.md](RQ1_formula.md) | 26/26 = 100% |
| RQ2 | [RQ2_formula.md](RQ2_formula.md) | 23/26 = 88.5% |
| RQ3 | [RQ3_formula.md](RQ3_formula.md) | 24/26 = 92.3% |
| RQ4 | [RQ4_formula.md](RQ4_formula.md) | 24/26 = 92.3% |
| RQ5 | [RQ5_formula.md](RQ5_formula.md) | 21/26 = 80.8% |
| RQ6 | [RQ6_formula.md](RQ6_formula.md) | dense 16/23 = 70% |
