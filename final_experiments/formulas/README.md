# MA 公式集（formulas/）

每个 RQ 对应的核心数学公式 + 验证证据 + 通过率，论文写作的主参考。

## 索引

| RQ | 公式主题 | 文件 | 通过率 |
|:-:|---|---|:-:|
| **RQ1** | Attention 消融判据（H₀ 证伪）| [RQ1_formula.md](RQ1_formula.md) | **26/26 = 100%** |
| **RQ2** | MLP 是物理基础 + ρ 主导比 + 起源层定位 | [RQ2_formula.md](RQ2_formula.md) | 23/26 = **88.5%** |
| **RQ3** | 广义 function token 触发 + Fisher + u₁ decode | [RQ3_formula.md](RQ3_formula.md) | 24/26 = **92.3%** |
| **RQ4** | MA 生成多项式公式（单层 + 多层 + 方向一致性）| [RQ4_formula.md](RQ4_formula.md) | 24/26 = **92.3%** |
| **RQ5** | V 消融因果（multi-K + macro + bias 对照 + per_dim 强证据）| [RQ5_formula.md](RQ5_formula.md) | 21/26 = **80.8%** |
| RQ6 | 多层 macro V 消融因果判据 | 待写 | 13/16 = 81.2% |

## 主公式（统一形式）

$$
\boxed{\text{MA}_{j^{\ast}} = \sum_{i=1}^{K} \sigma_i \cdot (h_2 \cdot v_i) \cdot u_i[j^{\ast}] + b[j^{\ast}]}
$$

- $\sigma_i, v_i, u_i$：$W_{\text{down}}$ 的 SVD 第 $i$ 组奇异值/右/左奇异向量
- $h_2$：MLP 中间激活
- $j^{\ast}$：MA 出现的稀疏 hidden 维度
- $b[j^{\ast}]$：down-projection bias（老架构如 OPT/BLOOM/GPT-2 非零；新架构 LLaMA/Qwen 系 = 0）
- $K$：截断阶数（按 $\sigma_1/\sigma_2$ 谱形态选）
