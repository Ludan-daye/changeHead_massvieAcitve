# SVD矩阵运算详细推导 - Exp 5 Appendix

## 内容导航

1. [矩阵乘法的详细展开](#矩阵乘法的详细展开)
2. [投影向量的计算](#投影向量的计算)
3. [左右奇异空间的变换](#左右奇异空间的变换)
4. [特征值和特征向量](#特征值和特征向量)
5. [协方差矩阵和SVD的关系](#协方差矩阵和svd的关系)
6. [主成分分析(PCA)视角](#主成分分析pca视角)

---

## 矩阵乘法的详细展开

### 1.1 基本矩阵乘法

**问题**: 给定 $h_2 \in \mathbb{R}^{3072}$ 和 $W_2 \in \mathbb{R}^{3072 \times 768}$，计算 $output = h_2 @ W_2$

**展开形式**:

```
output[j] = Σ(i=0 to 3071) h₂[i] × W₂[i,j]

j ∈ {0, 1, ..., 767}

因此:
  output[0] = h₂[0]×W₂[0,0] + h₂[1]×W₂[1,0] + ... + h₂[3071]×W₂[3071,0]
  output[1] = h₂[0]×W₂[0,1] + h₂[1]×W₂[1,1] + ... + h₂[3071]×W₂[3071,1]
  ...
  output[767] = h₂[0]×W₂[0,767] + h₂[1]×W₂[1,767] + ... + h₂[3071]×W₂[3071,767]
```

### 1.2 SVD分解后的乘法

**定义**: $W_2 = U @ Σ @ V^T$

其中:
- $U \in \mathbb{R}^{3072 \times 768}$: 左奇异向量
- $Σ \in \mathbb{R}^{768 \times 768}$: 对角矩阵
- $V^T \in \mathbb{R}^{768 \times 768}$: 右奇异向量的转置

**逐步计算**:

$$output = h_2 @ W_2 = h_2 @ (U @ Σ @ V^T)$$

**Step 1**: 计算 $temp_1 = h_2 @ U$
```
temp₁ ∈ ℝ^{768}
temp₁[k] = Σ(i=0 to 3071) h₂[i] × U[i,k]
         = h₂ · u_k  (u_k是U的第k列)
```

**Step 2**: 计算 $temp_2 = temp_1 @ Σ$
```
temp₂ ∈ ℝ^{768}
temp₂[k] = temp₁[k] × σ_k
         = (h₂ · u_k) × σ_k
```

**Step 3**: 计算 $output = temp_2 @ V^T$
```
output ∈ ℝ^{768}
output[j] = Σ(k=0 to 767) temp₂[k] × V^T[k,j]
          = Σ(k=0 to 767) (h₂ · u_k) × σ_k × V[j,k]
```

### 1.3 通用形式

综合Step 1-3:

$$output[j] = \sum_{k=0}^{767} (h_2 \cdot u_k) \times \sigma_k \times V[j,k]$$

或者用向量形式:

$$output = \sum_{k=0}^{767} (h_2 \cdot u_k) \times \sigma_k \times v_k$$

其中 $v_k$ 是V的第k列

---

## 投影向量的计算

### 2.1 到奇异向量的投影

**问题**: 给定向量 $h_2$ 和奇异向量 $v_k$，计算投影

**定义**:

$$\text{projection}_k = h_2 \cdot v_k$$

**计算**:

由于 $v_k \in \mathbb{R}^{3072}$ 且被标准化（$|v_k| = 1$），投影为:

$$h_2 \cdot v_k = \sum_{i=0}^{3071} h_2[i] \times v_k[i]$$

**几何意义**:

投影 $h_2 \cdot v_k$ 表示向量 $h_2$ 在 $v_k$ 方向上的"长度"

$$\text{projection}_k = |h_2| \times \cos(\angle(h_2, v_k))$$

### 2.2 投影的主成分

在SVD框架中，我们通常关注前几个最强的奇异向量:

$$h_2 = \sum_{k=0}^{767} (h_2 \cdot v_k) \times v_k$$

(完全重构)

或者用前K个:

$$h_2 \approx \sum_{k=0}^{K-1} (h_2 \cdot v_k) \times v_k$$

(近似重构)

**例子** (K=5):

$$h_2 \approx (h_2 \cdot v_0) \times v_0 + (h_2 \cdot v_1) \times v_1 + ... + (h_2 \cdot v_4) \times v_4$$

### 2.3 方差和投影

若有N个向量 $\{h_2^{(1)}, h_2^{(2)}, ..., h_2^{(N)}\}$，则:

$$\text{variance}_k = \text{Var}(h_2^{(1)} \cdot v_k, h_2^{(2)} \cdot v_k, ..., h_2^{(N)} \cdot v_k)$$

$$= \frac{1}{N} \sum_{n=1}^{N} [(h_2^{(n)} \cdot v_k) - \bar{p}_k]^2$$

其中 $\bar{p}_k = \frac{1}{N} \sum_{n=1}^{N} (h_2^{(n)} \cdot v_k)$ 是投影的平均值

**方差集中性指标**:

$$C_K = \frac{\sum_{k=0}^{K-1} \text{variance}_k}{\sum_{k=0}^{767} \text{variance}_k}$$

这就是"集中度"分析的数学基础！

---

## 左右奇异空间的变换

### 3.1 左奇异空间 (Left Singular Space)

当我们计算 $h_2 @ U$ 时，得到的是左奇异空间的坐标:

$$h_2^{\text{left}} = h_2 @ U \in \mathbb{R}^{768}$$

$$h_2^{\text{left}}[k] = h_2 \cdot u_k$$

其中 $u_k$ 是U的第k列

**物理含义**:
- 这是h₂在左奇异向量基下的坐标
- 第k个坐标表示h₂沿$u_k$方向的强度
- $|h_2^{\text{left}}|$ 通常很大（3072维的影响）

### 3.2 经过Σ加权的空间

$$h_2^{\text{weighted}} = (h_2 @ U) @ Σ \in \mathbb{R}^{768}$$

$$h_2^{\text{weighted}}[k] = (h_2 \cdot u_k) \times \sigma_k$$

**物理含义**:
- 每个坐标按对应的奇异值放大
- 强奇异向量(大σ)对应的坐标被放大更多
- 弱奇异向量(小σ)对应的坐标被放大较少

### 3.3 右奇异空间 (Right Singular Space)

最终输出在右奇异空间:

$$output = h_2^{\text{weighted}} @ V^T \in \mathbb{R}^{768}$$

$$output[j] = \sum_{k=0}^{767} (h_2 \cdot u_k) \times \sigma_k \times V[j,k]$$

其中$V[j,k]$是V矩阵的第j行第k列（或者说$v_k$的第j个分量）

**物理含义**:
- 这是最终的768维输出
- 是左空间中加权坐标的线性组合
- 每个$v_k$与σ加权后的投影贡献到输出

### 3.4 左右空间的浓度比较

**左空间浓度**:

对于第k维（最强的维度，k=0）:

$$\text{concentration}_{\text{left},0} = |h_2 \cdot u_0| / \sum_{k=0}^{767} |h_2 \cdot u_k|$$

**右空间浓度**:

对于第j维的输出（通常对应最强的输出方向）:

$$\text{concentration}_{\text{right},j} = |output[j]| / \sum_{j'=0}^{767} |output[j']|$$

**不对称比**:

$$\text{asymmetry\_ratio} = \frac{\text{concentration}_{\text{left},0}}{\text{concentration}_{\text{right},j^*}}$$

其中$j^*$是最强输出维度

**为什么会不对称？**

对于函数词，左空间的信息集中在前几个维度（对应大的σ值），经过Σ加权后，这些信息被放大。但当投影回768维输出空间时（通过$V^T$），这种集中性被分散，因为$V^T$是一般性的变换矩阵。

---

## 特征值和特征向量

### 4.1 W₂Wᵀ的特征值分解

**定义**:

$$A = W_2 @ W_2^T \in \mathbb{R}^{3072 \times 3072}$$

**谱定理**:

$$A = \sum_{k=0}^{767} \sigma_k^2 \times u_k u_k^T$$

其中$u_k u_k^T$是秩-1矩阵

**特征值和特征向量**:
- 特征值: $\lambda_k = \sigma_k^2$
- 特征向量: $u_k$ (即左奇异向量)

**验证**:

$$A @ u_k = (W_2 @ W_2^T) @ u_k = W_2 @ (W_2^T @ u_k) = W_2 @ v_k @ \sigma_k = \sigma_k^2 @ u_k$$

因此 $u_k$ 是 $W_2 W_2^T$ 的特征向量，特征值为 $\sigma_k^2$

### 4.2 WᵀW的特征值分解

**定义**:

$$B = W_2^T @ W_2 \in \mathbb{R}^{768 \times 768}$$

**谱定理**:

$$B = \sum_{k=0}^{767} \sigma_k^2 \times v_k v_k^T$$

**特征值和特征向量**:
- 特征值: $\lambda_k = \sigma_k^2$
- 特征向量: $v_k$ (即右奇异向量)

**验证**:

$$B @ v_k = (W_2^T @ W_2) @ v_k = W_2^T @ (W_2 @ v_k) = W_2^T @ u_k @ \sigma_k = \sigma_k^2 @ v_k$$

### 4.3 Rayleigh商

对于任意向量 $x$，Rayleigh商定义为:

$$R(x) = \frac{x^T @ A @ x}{x^T @ x}$$

**性质**:

$$\lambda_{\min} \leq R(x) \leq \lambda_{\max}$$

等号成立当且仅当 $x$ 是对应的特征向量

**在我们的情况下**:

对于 $h_2$ 和 $A = W_2 @ W_2^T$:

$$R(h_2) = \frac{|W_2^T @ h_2|^2}{|h_2|^2}$$

当$h_2 \parallel u_k$时（平行于某个特征向量），$R(h_2) = \sigma_k^2$

这解释了为什么沿着主奇异方向的激活会最大！

---

## 协方差矩阵和SVD的关系

### 5.1 数据矩阵的SVD

假设我们有N个数据点（词的多个出现），构成矩阵 $X \in \mathbb{R}^{N \times 3072}$:

```
X = [h_2^(1)]
    [h_2^(2)]
    [   ...  ]
    [h_2^(N)]
```

### 5.2 中心化

首先中心化数据:

$$\tilde{X} = X - \bar{X}$$

其中 $\bar{X}$ 是均值（每列的平均）

### 5.3 SVD分解

对中心化的数据进行SVD:

$$\tilde{X} = U' @ Σ' @ V'^T$$

其中:
- $U' \in \mathbb{R}^{N \times M}$: 左奇异向量
- $Σ' \in \mathbb{R}^{M \times 768}$: 奇异值矩阵
- $V'^T \in \mathbb{R}^{768 \times 768}$: 右奇异向量的转置
- $M = \min(N, 768)$

### 5.4 样本协方差矩阵

样本协方差矩阵定义为:

$$Cov = \frac{1}{N-1} \tilde{X}^T @ \tilde{X}$$

### 5.5 关键关系

$$Cov = \frac{1}{N-1} \tilde{X}^T @ \tilde{X} = \frac{1}{N-1} V' @ Σ'^T @ Σ' @ V'^T$$

由于 $Σ'^T @ Σ'$ 是对角矩阵（奇异值的平方），我们有:

$$Cov = V' @ \Lambda @ V'^T$$

其中:
- $\Lambda[k,k] = \frac{\sigma_k'^2}{N-1}$: 方差
- $V'$ 的列: 协方差矩阵的特征向量

**关键结论**: **SVD提供的$V'$的列就是PCA的主成分方向！**

### 5.6 主成分分析(PCA)视角

$$\text{主成分}_k = \sqrt{\frac{\sigma_k'^2}{N-1}}$$

这正好对应方差！

因此，"集中度"分析实际上是在看数据在各个主成分方向上的方差分布。

---

## 主成分分析(PCA)视角

### 6.1 PCA的数学形式

**目标**: 找到方向 $w$ 使得投影方差最大

$$\max_w \text{Var}(\tilde{X} @ w) \quad \text{s.t.} \quad |w| = 1$$

### 6.2 解

通过Lagrange乘数法:

$$\nabla \text{Var}(\tilde{X} @ w) = \lambda \nabla (|w|^2)$$

$$\frac{1}{N-1} \tilde{X}^T @ \tilde{X} @ w = \lambda w$$

$$Cov @ w = \lambda w$$

**因此**: $w$ 是协方差矩阵的特征向量，$\lambda$ 是对应的特征值（方差）

### 6.3 排序和集中度

若按方差排序，$\lambda_1 \geq \lambda_2 \geq ... \geq \lambda_{768}$，则:

$$C_K = \frac{\sum_{k=1}^{K} \lambda_k}{\sum_{k=1}^{768} \lambda_k}$$

**特点**:
- 函数词: $C_5 \approx 78.6\%$ （前5个PC捕捉大部分方差）
- 内容词: $C_5 \approx 27.3\%$ （方差分散在多个PC）

### 6.4 Exp 5中的PCA视角

当我们计算投影 $(h_2 \cdot v_k)$ 到奇异向量 $v_k$ 时，实际上就是在计算主成分！

因此整个Exp 5的分析框架可以理解为:

**主要问题**: 函数词和内容词的主成分分布有何不同？

**答案**:
- 函数词: 集中在少数几个强主成分
- 内容词: 均匀分布在多个弱主成分

这正是它们产生不同激活水平的根本原因！

---

## 数学推导总结表

| 概念 | 数学表达 | 物理含义 |
|------|---------|---------|
| **SVD分解** | $W_2 = U @ Σ @ V^T$ | 矩阵的奇异值分解 |
| **激活输出** | $output = (h_2 @ U) @ Σ @ V^T$ | MLP的线性变换 |
| **主方向近似** | $output ≈ (h_2 \cdot v_1) × σ_1 × u_1$ | 由最强奇异向量主导 |
| **放大因子** | $σ_1 = 38.26$ | 沿主方向的乘法因子 |
| **对齐强度** | $(h_2 \cdot v_1)$ | 投影系数(0-1) |
| **激活倍数** | $(h_2 \cdot v_1) × σ_1$ | 相对于输入的放大 |
| **方差集中** | $C_K = \sum_{i=1}^{K} \lambda_i / \sum_i \lambda_i$ | 前K个PC的方差占比 |
| **不对称比** | $\text{left conc} / \text{right conc}$ | 信息何时确定 |
| **稳定性** | $\text{mean}(\cos \text{ sim})$ | 跨句表示一致性 |
| **主成分** | 协方差矩阵的特征向量 | SVD的$V$列向量 |

---

**推导完成**：2024年10月29日

**用途**：学术论文数学附录、理论课程讲义、研究方法论文档

