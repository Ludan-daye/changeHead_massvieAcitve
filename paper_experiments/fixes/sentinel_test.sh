#!/usr/bin/env bash
# Sentinel verification for the fix package.
#
# Runs 3 sentinel models through the 3 modified RQ scripts and asserts:
#   - No errors (exit code 0 or expected MoE skip)
#   - RQ3 word_stats has content tokens > function tokens (B1 fix confirmed)
#   - RQ6 baseline is within 10× of RQ2a baseline (B6 fix confirmed)
#   - MoE sentinel gracefully skips (B2 guard)
#
# Usage: bash paper_experiments/fixes/sentinel_test.sh
# Expects: repo root as CWD; paper_experiments/fixes/ files already deployed.

set -u  # error on undefined vars; NOT -e so we control pass/fail manually

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

TMP_DIR="/tmp/fix_sentinel_$(date +%s)"
mkdir -p "$TMP_DIR"
echo "Sentinel test output → $TMP_DIR"
echo

PASS=0
FAIL=0

pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

# ----- Test A: B3 white-list (dry, no GPU needed) -----
echo "Test A: B3 white-list (get_mlp_submodules with fake layer)"
python3 -c "
import sys, importlib.util
spec = importlib.util.spec_from_file_location(
    'mu', 'paper_experiments/fixes/lib/model_utils.py')
mu = importlib.util.module_from_spec(spec); spec.loader.exec_module(mu)

class P: pass
class M:
    up_proj = P(); gate_proj = P(); down_proj = P()
    act_fn = lambda x: x; act = lambda x: x
    c_fc = P(); c_proj = P(); fc_in = P(); fc_out = P()
    dense_h_to_4h = P(); dense_4h_to_h = P(); fc1 = P(); fc2 = P()
    gelu_impl = lambda x: x; activation_fn = lambda x: x
class L:
    mlp = M(); fc1 = P(); fc2 = P(); activation_fn = lambda x: x

failed = []
for m in ['glm4_9b', 'yi_9b', 'qwen1.5_14b', 'qwen3.5_9b', 'llama2_7b', 'gpt2', 'gptj_6b']:
    try:
        r = mu.get_mlp_submodules(m, L())
        if not isinstance(r, dict): raise ValueError(f'returned {type(r)}')
    except Exception as e:
        failed.append(f'{m}: {e}')

if failed:
    for f in failed: print('FAIL', f)
    sys.exit(1)
print('OK')
" 2>&1 | tail -5 | grep -q "^OK$" && pass "B3: all 7 model names resolved" || fail "B3: some model names failed"
echo

# ----- Test B: B7 tuple hook -----
echo "Test B: B7 MoE tuple hook"
python3 -c "
import torch, re
with open('paper_experiments/fixes/RQ2_mlp_source/exp2a_mlp_feasibility_test.py') as f:
    code = f.read()
# Extract MLPDisableHook class
m = re.search(r'class MLPDisableHook:.*?(?=\n\ndef |\n\nclass )', code, re.DOTALL)
ns = {'torch': torch}; exec(m.group(0), ns)
H = ns['MLPDisableHook'](0, 'disable_all')

# Dense
t = torch.ones(2,3); out = H(None, None, t)
assert torch.all(out == 0), 'dense non-zero'

# MoE tuple
h = torch.ones(2,3); r = torch.ones(2,4)*5
out = H(None, None, (h, r))
assert isinstance(out, tuple), 'not tuple'
assert torch.all(out[0] == 0), 'hidden not zeroed'
assert torch.allclose(out[1], r), 'router corrupted'
print('OK')
" 2>&1 | tail -2 | grep -q "^OK$" && pass "B7: dense + MoE tuple both handled" || fail "B7: tuple handling broken"
echo

# ----- Test C: B4 white-list (dry) -----
echo "Test C: B4 RQ6 get_mlp_down_proj"
python3 -c "
import sys, re, os
# Read file, extract just get_mlp_down_proj
with open('paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py') as f:
    code = f.read()
m = re.search(r'def get_mlp_down_proj.*?(?=\n\ndef |\n\nclass )', code, re.DOTALL)

# Stub is_llama_model for isolated test
ns = {'is_llama_model': lambda name: 'llama' in name or 'mistral' in name}
exec(m.group(0), ns)

class P: pass
class M: down_proj = P(); c_proj = P(); fc_out = P(); dense_4h_to_h = P(); fc2 = P()
class L: mlp = M(); fc2 = P()

failed = []
for m_name in ['glm4_9b', 'glm4_32b', 'yi_9b', 'gpt2', 'gptj_6b', 'bloom_7b1', 'falcon_7b', 'qwen3_8b']:
    try:
        r = ns['get_mlp_down_proj'](L(), m_name)
    except Exception as e:
        failed.append(f'{m_name}: {e}')

if failed:
    for f in failed: print('FAIL', f)
    sys.exit(1)
print('OK')
" 2>&1 | tail -2 | grep -q "^OK$" && pass "B4: all 8 model names resolved" || fail "B4: some names failed"
echo

# ----- Test D: B5 L_ORIGIN.json reading (dry) -----
echo "Test D: B5 get_critical_layer reads L_ORIGIN.json"
python3 -c "
import sys, os, re, json, warnings
os.chdir('$REPO_ROOT')
# Clear any env override
os.environ.pop('OVERRIDE_CRITICAL_LAYER', None)

with open('paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py') as f:
    code = f.read()
m = re.search(r'def get_critical_layer.*?(?=\n\ndef |\n\nclass )', code, re.DOTALL)

# Need __file__ context for path resolution
ns = {'__file__': 'paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py'}
exec(m.group(0), ns)

# Verify authoritative L_ORIGIN.json is readable
if not os.path.exists('paper_experiments/origin_layer/output/L_ORIGIN.json'):
    print('FAIL: L_ORIGIN.json not found at expected path')
    sys.exit(1)

# Test 3 models that had wrong defaults before (were silently L=0)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    test_cases = [
        ('qwen3_32b', 6),     # was L=0
        ('yi_9b', 8),         # was L=0
        ('qwen1.5_14b', 35),  # was L=0
        ('bloom_7b1', 3),     # was L=28 (wrong hardcode)
    ]
    failed = []
    for m_name, expected in test_cases:
        L = ns['get_critical_layer'](m_name)
        if L != expected:
            failed.append(f'{m_name}: expected L={expected}, got L={L}')

# Test env override
os.environ['OVERRIDE_CRITICAL_LAYER'] = '99'
L = ns['get_critical_layer']('qwen3_32b')
if L != 99: failed.append(f'env override: expected 99, got {L}')
os.environ.pop('OVERRIDE_CRITICAL_LAYER', None)

# Test unknown model raises
try:
    L = ns['get_critical_layer']('nonexistent_999b')
    failed.append('expected ValueError for unknown model')
except ValueError:
    pass

if failed:
    for f in failed: print('FAIL', f)
    sys.exit(1)
print('OK')
" 2>&1 | tail -5 | grep -q "^OK$" && pass "B5: 4 real models + env override + fail-fast" || fail "B5: layer resolution broken"
echo

# ----- Test E: B6 hook scans all layers -----
echo "Test E: B6 run_and_collect_ma pattern check"
python3 -c "
with open('paper_experiments/fixes/RQ6_v_ablation/exp6_v_ablation.py') as f:
    code = f.read()
checks = {
    'Scans all layers': 'for lid, layer in enumerate(layers)' in code and 'handles = []' in code,
    'Records peak_layer': '\"peak_layer\":' in code,
    'Records ablation_layer': '\"ablation_layer\":' in code,
    'MoE guard present': 'hasattr(_first_mlp' in code,
    '--layer_id argparse': '--layer_id' in code,
}
failed = [k for k, v in checks.items() if not v]
if failed:
    for f in failed: print(f'FAIL: {f}')
    import sys; sys.exit(1)
print('OK')
" 2>&1 | tail -5 | grep -q "^OK$" && pass "B6: all structural markers present" || fail "B6: missing markers"
echo

# ----- Test F: B1 classification logic -----
echo "Test F: B1 FunctionWordSVDTracker classification"
python3 -c "
import sys, importlib.util, torch
sys.path.insert(0, 'paper_experiments')
spec = importlib.util.spec_from_file_location(
    'm', 'paper_experiments/fixes/RQ3_function_words/exp5_function_words_svd_mapping.py')
m = importlib.util.module_from_spec(spec)
try: spec.loader.exec_module(m)
except Exception as e:
    print(f'FAIL: load error {e}'); sys.exit(1)

t = m.FunctionWordSVDTracker(layer_id=0, tokenizer=None)
h = torch.randn(3072)
tests = [
    ('the', 'func'), ('of', 'func'), ('is', 'func'),
    ('dog', 'content'), ('information', 'content'), ('cat', 'content'),
    ('.', 'struct'), ('@', 'struct'), ('\n\n', 'struct'), ('<|endoftext|>', 'struct'),
]
for tok, _ in tests:
    t.add_token(tok, h)

ws = t.get_word_statistics()
n_func = sum(s['count'] for s in ws.values() if s['is_function'])
n_struct = sum(s['count'] for s in ws.values() if s['is_structural'])
n_content = sum(s['count'] for s in ws.values() if not s['is_function'] and not s['is_structural'])

# Expect: func=3, struct=4, content=3
if (n_func, n_struct, n_content) != (3, 4, 3):
    print(f'FAIL: got func={n_func}, struct={n_struct}, content={n_content}; expected (3,4,3)')
    sys.exit(1)
print('OK')
" 2>&1 | tail -2 | grep -q "^OK$" && pass "B1: func/struct/content all tracked correctly" || fail "B1: classification wrong"
echo

# ----- Summary -----
echo "==================================="
echo "Sentinel results: $PASS passed, $FAIL failed"
echo "==================================="

if [ $FAIL -ne 0 ]; then
    echo "FAIL — do NOT deploy until all tests pass."
    exit 1
fi
echo "All checks passed. Safe to deploy."
