# MA 公式集（formulas/）

每个 RQ 对应的核心数学公式 + 验证证据 + 通过率，论文写作的主参考。

## 🎯 推荐入口：[UNIFIED.md](UNIFIED.md)

**整体公式纲要**：把 RQ1-RQ6 按 MA 生成流程串联，统一符号体系，从 token 输入到 MA 因果验证一气呵成。每个公式既有数学定义，也有该步在 MA 链中的物理意义。

## 各 RQ 详细索引（论文 §3.2 命名）

| RQ | 名称 | 公式主题 | 文件 | 通过率（dense 22）|
|:-:|---|---|---|:-:|
| **RQ1** | Source | Attention 消融判据（attention 是 regulator）| [RQ1_formula.md](RQ1_formula.md) | **22/22 = 100%** |
| **RQ2** | Localization | MLP 是 substrate + ρ 主导比 + 起源层 | [RQ2_formula.md](RQ2_formula.md) | 21/22 = **95.5%** |
| **RQ3** | Trigger | Function token 触发 + Fisher + u₁ decode | [RQ3_formula.md](RQ3_formula.md) | 21/22 = **95.5%** |
| **RQ4** | Mechanism | SVD 公式 + multi-K 截断 + macro-SVD | [RQ4_formula.md](RQ4_formula.md) | 21/22 = **95.5%** |
| **RQ5** | Causality | V 消融（multi-K + macro + bias + per_dim）| [RQ5_formula.md](RQ5_formula.md) | 20/22 = **90.9%** |
| **RQ6** | Sufficiency | Top-K recovery（case-study）| [RQ6_formula.md](RQ6_formula.md) | 2 直测 + 14-16 间接一致 |

## 主公式（统一形式）

$$
\boxed{\text{MA}_{j^{\ast}} = \sum_{i=1}^{K} \sigma_i \cdot (h_2 \cdot v_i) \cdot u_i[j^{\ast}] + b[j^{\ast}]}
$$

- $\sigma_i, v_i, u_i$：$W_{\text{down}}$ 的 SVD 第 $i$ 组奇异值/右/左奇异向量
- $h_2$：MLP 中间激活
- $j^{\ast}$：MA 出现的稀疏 hidden 维度
- $b[j^{\ast}]$：down-projection bias（老架构如 OPT/BLOOM/GPT-2 非零；新架构 LLaMA/Qwen 系 = 0）
- $K$：截断阶数（按 $\sigma_1/\sigma_2$ 谱形态选）
