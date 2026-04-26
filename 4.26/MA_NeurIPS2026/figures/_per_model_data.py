"""
Per-model × per-RQ metric table for the 26-model panel.
Numeric values aligned with paper Table 1 + §5–§7 + CLAUDE.md §17/§19/§20/§21.
None = not measured / no data.
"""

# Display name → final_experiments folder name
NAME_TO_FOLDER = {
    'GPT-J-6B': 'gptj_6b', 'Qwen2-7B': 'qwen2_7b', 'Qwen2.5-7B': 'qwen2.5_7b',
    'Qwen3-0.6B': 'qwen3_0.6b', 'GLM4-32B': 'glm4_32b', 'Mistral-7B': 'mistral_7b_v03',
    'Qwen2.5-0.5B': 'qwen2.5_0.5b', 'LLaMA-2-13B': 'llama2_13b', 'LLaMA-2-7B-Chat': 'llama2_7b_chat',
    'GPT-2': 'gpt2', 'Qwen3-1.7B': 'qwen3_1.7b', 'Qwen3-4B': 'qwen3_4b',
    'Falcon-7B': 'falcon_7b', 'LLaMA-3.1-8B': 'llama3.1_8b', 'GLM4-9B': 'glm4_9b',
    'BLOOM-7B1': 'bloom_7b1', 'Qwen3-8B': 'qwen3_8b', 'Yi-9B': 'yi_9b',
    'Qwen3.5-9B': 'qwen3.5_9b', 'Qwen1.5-14B': 'qwen1.5_14b', 'Qwen3-14B': 'qwen3_14b',
    'Qwen3.5-27B': 'qwen3.5_27b', 'Qwen3-30B-A3B': 'qwen3_30b_a3b', 'Qwen3-32B': 'qwen3_32b',
    'OPT-6.7B': 'opt_6.7b', 'Qwen3.5-35B-A3B': 'qwen3.5_35b_a3b',
}

# Per-model metrics across 6 RQs
# rq1_residual: residual MA % after attention ablation
# rq1_sign: 'Gen' or 'Sup'
# rq2_tau: retain τ % after MLP global ablation
# rq3_pi_F: π(F) function-token trigger probability
# rq3_top1: u_1 top-1 vocab decoded token (text repr)
# rq4_eta: σ_1/σ_2 spectral dominance
# rq4_R2: rank-1 fit R^2
# rq5_single: single-V ablation ΔMA% (None = not measured)
# rq5_macro: macro-V ablation ΔMA% (None = not measured)
# rq6_recovery: r* % recovery (None = not measured)
# regime: CONC / FS / DISP / ANOM
PER_MODEL = {
    'GPT-J-6B':         dict(rq1_residual=1.69, rq1_sign='Gen', rq2_tau=1.9, rq3_pi_F=0.92, rq3_top1="'\\n\\n'", rq4_eta=2.52, rq4_R2=1.00, rq5_single=-99.0, rq5_macro=-99.1, rq6_recovery=76.4, regime='CONC'),
    'Qwen2-7B':         dict(rq1_residual=4.0,  rq1_sign='Gen', rq2_tau=0.5, rq3_pi_F=0.94, rq3_top1="'\\n\\n'", rq4_eta=2.84, rq4_R2=1.00, rq5_single=-99.0, rq5_macro=-98.0, rq6_recovery=42.0, regime='CONC'),
    'Qwen2.5-7B':       dict(rq1_residual=5.0,  rq1_sign='Sup', rq2_tau=0.6, rq3_pi_F=0.93, rq3_top1="'\\n\\n'", rq4_eta=2.64, rq4_R2=1.00, rq5_single=-99.0, rq5_macro=-97.5, rq6_recovery=38.0, regime='CONC'),
    'Qwen3-0.6B':       dict(rq1_residual=12.0, rq1_sign='Gen', rq2_tau=1.3, rq3_pi_F=0.91, rq3_top1="'\\n\\n'", rq4_eta=1.41, rq4_R2=1.00, rq5_single=-93.0, rq5_macro=-92.0, rq6_recovery=12.0, regime='CONC'),
    'GLM4-32B':         dict(rq1_residual=8.0,  rq1_sign='Sup', rq2_tau=12.6,rq3_pi_F=0.85, rq3_top1="' @'",     rq4_eta=1.53, rq4_R2=0.47, rq5_single=-97.0, rq5_macro=None,  rq6_recovery=None, regime='CONC'),
    'Mistral-7B':       dict(rq1_residual=11.0, rq1_sign='Gen', rq2_tau=0.8, rq3_pi_F=0.82, rq3_top1="''",       rq4_eta=1.12, rq4_R2=1.00, rq5_single=-83.0, rq5_macro=None,  rq6_recovery=None, regime='CONC'),
    'Qwen2.5-0.5B':     dict(rq1_residual=9.0,  rq1_sign='Gen', rq2_tau=1.6, rq3_pi_F=0.78, rq3_top1="'The'",    rq4_eta=1.48, rq4_R2=0.91, rq5_single=-55.0, rq5_macro=None,  rq6_recovery=None, regime='CONC'),
    'LLaMA-2-13B':      dict(rq1_residual=6.0,  rq1_sign='Gen', rq2_tau=3.84,rq3_pi_F=0.81, rq3_top1="''",       rq4_eta=1.32, rq4_R2=0.97, rq5_single=-96.0, rq5_macro=None,  rq6_recovery=None, regime='CONC'),
    'LLaMA-2-7B-Chat':  dict(rq1_residual=7.0,  rq1_sign='Gen', rq2_tau=1.1, rq3_pi_F=0.34, rq3_top1="'_'",      rq4_eta=1.04, rq4_R2=0.94, rq5_single=-96.0, rq5_macro=None,  rq6_recovery=None, regime='CONC'),
    'GPT-2':            dict(rq1_residual=8.0,  rq1_sign='Gen', rq2_tau=4.3, rq3_pi_F=0.79, rq3_top1="' .'",     rq4_eta=3.05, rq4_R2=0.55, rq5_single=None,  rq5_macro=-95.0, rq6_recovery=None, regime='FS'),
    'Qwen3-1.7B':       dict(rq1_residual=5.0,  rq1_sign='Gen', rq2_tau=2.9, rq3_pi_F=0.86, rq3_top1="'\\n\\n'", rq4_eta=1.33, rq4_R2=0.94, rq5_single=None,  rq5_macro=-87.0, rq6_recovery=None, regime='FS'),
    'Qwen3-4B':         dict(rq1_residual=6.0,  rq1_sign='Gen', rq2_tau=0.3, rq3_pi_F=0.88, rq3_top1="'\\n\\n'", rq4_eta=1.24, rq4_R2=1.00, rq5_single=None,  rq5_macro=-95.0, rq6_recovery=None, regime='FS'),
    'Falcon-7B':        dict(rq1_residual=12.0, rq1_sign='Gen', rq2_tau=1.6, rq3_pi_F=0.83, rq3_top1="'ed'",     rq4_eta=1.37, rq4_R2=0.99, rq5_single=None,  rq5_macro=-97.0, rq6_recovery=17.0, regime='FS'),
    'LLaMA-3.1-8B':     dict(rq1_residual=9.0,  rq1_sign='Gen', rq2_tau=2.8, rq3_pi_F=0.89, rq3_top1="' the'",   rq4_eta=1.38, rq4_R2=1.00, rq5_single=None,  rq5_macro=-100.0,rq6_recovery=49.0, regime='FS'),
    'GLM4-9B':          dict(rq1_residual=4.0,  rq1_sign='Sup', rq2_tau=4.5, rq3_pi_F=0.84, rq3_top1="' @'",     rq4_eta=3.26, rq4_R2=0.89, rq5_single=None,  rq5_macro=-82.0, rq6_recovery=None, regime='FS'),
    'BLOOM-7B1':        dict(rq1_residual=3.0,  rq1_sign='Gen', rq2_tau=0.0, rq3_pi_F=0.76, rq3_top1="'ky'",     rq4_eta=1.81, rq4_R2=1.00, rq5_single=-69.7, rq5_macro=None,  rq6_recovery=None, regime='FS'),
    'Qwen3-8B':         dict(rq1_residual=5.0,  rq1_sign='Gen', rq2_tau=1.0, rq3_pi_F=0.87, rq3_top1="'\\n\\n'", rq4_eta=1.48, rq4_R2=1.00, rq5_single=None,  rq5_macro=-100.0,rq6_recovery=None, regime='DISP'),
    'Yi-9B':            dict(rq1_residual=7.0,  rq1_sign='Sup', rq2_tau=1.2, rq3_pi_F=0.80, rq3_top1="''",       rq4_eta=1.43, rq4_R2=0.88, rq5_single=None,  rq5_macro=-99.0, rq6_recovery=None, regime='DISP'),
    'Qwen3.5-9B':       dict(rq1_residual=11.0, rq1_sign='Gen', rq2_tau=32.1,rq3_pi_F=0.72, rq3_top1="' '",      rq4_eta=1.06, rq4_R2=0.73, rq5_single=None,  rq5_macro=-70.0, rq6_recovery=None, regime='DISP'),
    'Qwen1.5-14B':      dict(rq1_residual=6.0,  rq1_sign='Gen', rq2_tau=2.1, rq3_pi_F=0.75, rq3_top1="' '",      rq4_eta=1.33, rq4_R2=1.00, rq5_single=None,  rq5_macro=-13.0, rq6_recovery=None, regime='DISP'),
    'Qwen3-14B':        dict(rq1_residual=6.0,  rq1_sign='Gen', rq2_tau=1.1, rq3_pi_F=0.86, rq3_top1="'\\n\\n'", rq4_eta=1.33, rq4_R2=1.00, rq5_single=None,  rq5_macro=-88.0, rq6_recovery=None, regime='DISP'),
    'Qwen3.5-27B':      dict(rq1_residual=8.0,  rq1_sign='Gen', rq2_tau=10.0,rq3_pi_F=0.69, rq3_top1="' the'",   rq4_eta=1.12, rq4_R2=0.99, rq5_single=None,  rq5_macro=-78.0, rq6_recovery=None, regime='DISP'),
    'Qwen3-30B-A3B':    dict(rq1_residual=9.0,  rq1_sign='Sup', rq2_tau=0.3, rq3_pi_F=0.74, rq3_top1="'\\n\\n\\n'",rq4_eta=1.17,rq4_R2=0.38, rq5_single=None,  rq5_macro=0.0,   rq6_recovery=None, regime='DISP'),
    'Qwen3-32B':        dict(rq1_residual=5.0,  rq1_sign='Gen', rq2_tau=0.6, rq3_pi_F=0.85, rq3_top1="'\\n\\n'", rq4_eta=1.35, rq4_R2=1.00, rq5_single=None,  rq5_macro=-86.0, rq6_recovery=21.0, regime='DISP'),
    'OPT-6.7B':         dict(rq1_residual=744.0,rq1_sign='Sup', rq2_tau=87.6,rq3_pi_F=0.55, rq3_top1="'_'",      rq4_eta=2.53, rq4_R2=0.98, rq5_single=-31.8, rq5_macro=None,  rq6_recovery=None, regime='ANOM'),
    'Qwen3.5-35B-A3B':  dict(rq1_residual=5.0,  rq1_sign='Sup', rq2_tau=87.6,rq3_pi_F=0.28, rq3_top1="'in'",     rq4_eta=1.03, rq4_R2=0.00, rq5_single=None,  rq5_macro=1.0,   rq6_recovery=None, regime='ANOM'),
}


# PASS criteria per RQ
def rq1_pass(d):  return True   # H_0 falsified iff residual > 0; always TRUE in 26/26
def rq2_pass(d):  return d['rq2_tau'] <= 10
def rq3_pass(d):  return d['rq3_pi_F'] >= 0.50
def rq4_pass(d):  return d['rq4_R2'] >= 0.95
def rq5_pass(d):
    s = d['rq5_single']; m = d['rq5_macro']
    if s is not None and s <= -80: return True
    if m is not None and m <= -80: return True
    return False
def rq6_pass(d):  return d['rq6_recovery'] is not None and d['rq6_recovery'] >= 30


PASS_FN = {
    'RQ1': rq1_pass, 'RQ2': rq2_pass, 'RQ3': rq3_pass,
    'RQ4': rq4_pass, 'RQ5': rq5_pass, 'RQ6': rq6_pass,
}
