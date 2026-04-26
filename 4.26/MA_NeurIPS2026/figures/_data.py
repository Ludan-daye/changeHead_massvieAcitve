"""
Shared 26-model data (post-revision Table 1) for all subfigures.
"""

# (name, regime, eta=σ_1/σ_2, R^2, hero, retain_τ%, residual_attn%, recovery_r%)
MODELS = [
    # CONCENTRATED (9)
    ("GPT-J-6B",        "CONC", 2.52, 1.00, True,  1.9, 1.69, 76.4),
    ("Qwen2-7B",        "CONC", 2.84, 1.00, True,  0.5, 4.0,  None),
    ("Qwen2.5-7B",      "CONC", 2.64, 1.00, True,  0.6, 5.0,  None),
    ("Qwen3-0.6B",      "CONC", 1.41, 1.00, True,  1.3, 12.0, 12.0),
    ("GLM4-32B",        "CONC", 1.53, 0.47, False, 12.6, 8.0, None),
    ("Mistral-7B",      "CONC", 1.12, 1.00, False, 0.8, 11.0, None),
    ("Qwen2.5-0.5B",    "CONC", 1.48, 0.91, False, 1.6, 9.0,  None),
    ("LLaMA-2-13B",     "CONC", 1.32, 0.97, False, 3.84, 6.0, None),
    ("LLaMA-2-7B-Chat", "CONC", 1.04, 0.94, False, 1.1, 7.0,  None),

    # FEW-SOURCE (7)
    ("GPT-2",           "FS",   3.05, 0.55, False, 4.3, 8.0,  None),
    ("Qwen3-1.7B",      "FS",   1.33, 0.94, False, 2.9, 5.0,  None),
    ("Qwen3-4B",        "FS",   1.24, 1.00, False, 0.3, 6.0,  None),
    ("Falcon-7B",       "FS",   1.37, 0.99, False, 1.6, 12.0, 17.0),
    ("LLaMA-3.1-8B",    "FS",   1.38, 1.00, False, 2.8, 9.0,  49.0),
    ("GLM4-9B",         "FS",   3.26, 0.89, False, 4.5, 4.0,  None),
    ("BLOOM-7B1",       "FS",   1.81, 1.00, False, 0.0, 3.0,  None),

    # DISPERSED (8)
    ("Qwen3-8B",        "DISP", 1.48, 1.00, False, 1.0, 5.0,  None),
    ("Yi-9B",           "DISP", 1.43, 0.88, False, 1.2, 7.0,  None),
    ("Qwen3.5-9B",      "DISP", 1.06, 0.73, False, 32.1, 11.0, None),
    ("Qwen1.5-14B",     "DISP", 1.33, 1.00, False, 2.1, 6.0,  None),
    ("Qwen3-14B",       "DISP", 1.33, 1.00, False, 1.1, 6.0,  None),
    ("Qwen3.5-27B",     "DISP", 1.12, 0.99, False, 10.0, 8.0, None),
    ("Qwen3-30B-A3B",   "DISP", 1.17, 0.38, False, 0.3, 9.0,  None),
    ("Qwen3-32B",       "DISP", 1.35, 1.00, False, 0.6, 5.0,  21.0),

    # ANOMALY (2)
    ("OPT-6.7B",        "ANOM", 2.53, 0.98, False, 87.6, 744.0, None),
    ("Qwen3.5-35B-A3B", "ANOM", 1.03, 0.00, False, 87.6, 5.0, None),
]


def by_regime(regime):
    return [m for m in MODELS if m[1] == regime]


# GPT-2 4-quadrant trigger rates (RQ3 sample)
GPT2_QUADRANT = dict(
    labels=['HF FT\n($Q_{1}$)', 'LF FT\n($Q_{2}$)', 'HF content\n($Q_{3}$)', 'LF content\n($Q_{4}$)'],
    pi=[0.795, 0.612, 0.014, 0.011],
    n=[156, 1243, 287, 8952],
    ratio=43.7,
)
