We thank all reviewers for the constructive feedback.

## Sample type and breadth (bDKj1, qarC4)

1) This study focuses on mechanistic interpretability, not distribution statistics or generalization evaluation. Previous work already observed Massive Activations on large-scale corpora; our goal is to analyze their generation path and internal geometry. We intentionally chose small-scale controlled inputs to minimize distribution-level interference, thereby revealing the causal relationship between weight structure and activation amplification.

2) The Massive Activations mechanism relies on the spectral geometry of the MLP down-projection weight matrix rather than language-specific lexical features. Different languages may alter trigger token distributions, but this does not negate the geometric amplification mechanism.

## LayerNorm as Confound (bDKj2)

The $\gamma$ parameter of LayerNorm may scale downstream module amplitudes unevenly. However, our results show Massive Activations are not dominated by LayerNorm gain. In RQ4, Massive Activations align with the principal singular direction ($v_1$) of the MLP down-projection matrix, not with the $\gamma$ vector. If LayerNorm were the main source, activation direction should be dominated by $\gamma$, not the principal SVD direction. In RQ5, random orthogonal replacements on the $V$ matrix of $W_{down}$—keeping singular value spectrum and LayerNorm parameters unchanged—caused activation intensity to collapse by 78.8%–99.1% across models. This demonstrates dependence on weight geometric orientation rather than scale amplification. LayerNorm gain may modulate overall scale but is not the fundamental cause.

## Co-occurrence bias/aggregation effects of high-frequency tokens (daTc1)

The geometric structure analyzed here lies in the singular orientation space of the MLP down-projection matrix, not the input embedding space. If driven solely by word frequency, all high-frequency words should show similar trigger probabilities. Yet Massive Activations concentrate in function words, indicating category selectivity. With embedding space and input distribution unchanged, randomizing the $V$ matrix orientation of $W_{down}$ caused 78%–99% activation collapse across models. Massive Activations are better explained as a structural anchoring effect of syntactic features in MLP geometric space rather than high-frequency aggregation bias.

## Quantization or model compression (daTc2)

This paper reveals the geometric-causal generation mechanism of Massive Activations as mechanistic interpretability research, not proposing new quantization or compression algorithms. The applicability statement intends to highlight explanatory basis for the structural origin of anomalous activations, potentially inspiring future engineering methods—not claiming verified performance improvements.

## How MA influence the models (daTc3)

This paper does not advocate suppressing Massive Activations as an optimization objective. The RQ5 perturbation experiment is a causal verification tool—a structural necessity test, not a deployable modification. Its impact on Perplexity or downstream performance does not refute the mechanism conclusion. Performance changes from extreme perturbations only indicate coupling between Massive Activations and internal representation structure.

## Classification on the models (daTc4)

This paper covers Post-LN (GPT-2), Pre-LN + RMSNorm (LLaMA-2, Qwen2.5), and activation functions (GELU, SiLU, SwiGLU). As shown in Table 3, models are classified into Generative-Promoting, Inhibitory-Regulatory, and Mixed/Hybrid based on triggering mechanism, reflecting regulation mode differences across architectures.

## Analysis on Qwen2.5 (daTc5, qarC3)

The term "extreme" may appear visually overstated for some models. However, spectral anisotropy is evaluated via $\eta = \sigma_1/\sigma_2$ (Table 1), not subjective curve inspection. GPT-2 has $\eta=3.05$, Qwen-2.5 has $2.64$, indicating dominant first singular directions. Different architectures show varying spectral concentration: Qwen-2.5 exhibits concentrated structure while GPT-J and Mistral show distributed geometries, supporting the RQ5 distinction between concentrated and distributed modes.

The increased semantic token triggers in Qwen-2.5 may reflect architectural shifts in MA representation. Cross-model analysis shows a shift from function-word-dominated toward semantic triggering; detailed linguistic analysis remains future work.

## Theoretical inconsistency in formalism (qarC1)

We thank the reviewers for noting the scale representation issue in Eq. 17. In high-dimensional space, the expected squared cosine similarity between a random unit vector and a fixed vector is $1/d_{ff}$, with cosine magnitude $1/\sqrt{d_{ff}}$.

## Presentation quality (qarC2)

We thank the reviewers for pointing out spelling and formatting errors in Figures 2 and 6. We will correct these in the revised version.
