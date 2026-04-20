# 脚本修复工作日志（Bug Fixes）

> **目的**：记录本轮发现的 7 个脚本 bug（B1-B7）的**修复代码**。按执行顺序组织，每个 bug 附：位置、症状、修法（前/后代码）、验证。
>
> **状态标记**：
> - `[ ]` = 未修
> - `[~]` = 正在修
> - `[x]` = 已修 + 已验证

---

## 总览

| Bug | 位置 | 影响 | 优先级 | 状态 |
|:-:|---|---|:-:|:-:|
| B1 | `RQ3_function_words/exp5_function_words_svd_mapping.py::add_token` | RQ3/4 丢内容词 | 🔥 最高 | `[ ]` |
| B2 | `RQ3_function_words/exp5_*.py` 直接访问 `.up_proj/.down_proj` | RQ3 MoE 破 | 🔥 最高 | `[ ]` |
| B3 | `lib/model_utils.py::get_mlp_submodules()` | RQ3 缺 glm4/qwen1.5/qwen3.5/yi 白名单 | 🔥 最高 | `[ ]` |
| B4 | `changeHead/experiments/exp6_v_ablation/exp6_v_ablation.py::get_mlp_down_proj` | RQ6 缺 glm4/MoE 分支 | ⭐ 高 | `[ ]` |
| B5 | 同上 `::get_critical_layer` | RQ6 默认 L0 不读 L_origin | ⭐ 高 | `[ ]` |
| B6 | 同上 baseline 测量逻辑 | RQ6 baseline 错层非真 MA | ⭐ 高 | `[ ]` |
| B7 | `RQ2_mlp_source/exp2a_mlp_feasibility_test.py::MLPDisableHook` | RQ2a MoE tuple 未处理 | ○ 低 | `[ ]` |

**合计修复时间估算**：~2h（串行）

---

## Bug B1：`add_token` 只存功能词，丢内容词

**位置**：`paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py` 第 99-104 行

**症状**：内容词 h₂ 从未被记录 → RQ3 的 "func vs content Cohen's d" 实际比较的是"常见 func vs 边缘 func"，不是 func vs content。

**当前代码**：

```python
def add_token(self, token_text, h2_vector):
    """Add token representation"""
    self.total_token_count += 1
    if self.is_function_word(token_text):
        # h2_vector shape: [3072]
        self.word_data[token_text].append((self.context_counter, h2_vector.cpu().detach().numpy()))
```

**修复代码**：

```python
def add_token(self, token_text, h2_vector):
    """Add token representation (both function AND content words, tagged with is_function)"""
    self.total_token_count += 1
    is_func = self.is_function_word(token_text)
    # Structural token check (扩展:新增，包含标点/换行/特殊符号)
    is_struct = self.is_structural_token(token_text)
    
    # 存所有 token,但附加类别标签
    self.word_data[token_text].append({
        'context_id': self.context_counter,
        'h2': h2_vector.cpu().detach().numpy(),
        'is_function': is_func,
        'is_structural': is_struct,
    })
```

**附加方法**（放在 `is_function_word` 下方）：

```python
STRUCTURAL_TOKENS = {
    # 标点
    '.', ',', '!', '?', ';', ':', '"', "'", '(', ')', '[', ']', '{', '}',
    '-', '—', '–', '/', '\\', '|', '*', '&', '@', '#', '$', '%', '^', '~',
    # 换行/空白
    '\n', '\n\n', '\t', '\r',
    # 特殊 token
    '<bos>', '<eos>', '<pad>', '<unk>', '<|endoftext|>', '<|im_start|>', '<|im_end|>',
}

def is_structural_token(self, token_text):
    """Check if token is a structural/non-semantic token (punctuation, newline, special)."""
    clean = token_text.strip().lstrip('Ġ ')
    if clean in STRUCTURAL_TOKENS:
        return True
    # 所有字符都是标点/空白
    if clean and all(not c.isalnum() for c in clean):
        return True
    return False
```

**下游改动**（`get_word_statistics` 里把 list-of-dicts 兼容原先的 list-of-tuple 格式）：

```python
def get_word_statistics(self):
    """Get stats for each word (返回兼容旧格式的字典)"""
    stats_dict = {}
    for word, records in self.word_data.items():
        # records 是新格式的 list of dicts
        occurrences = [(r['context_id'], r['h2']) for r in records]
        stats_dict[word] = {
            'count': len(records),
            'contexts': len(set(r['context_id'] for r in records)),
            'occurrences': occurrences,
            'is_function': records[0]['is_function'] if records else False,
            'is_structural': records[0]['is_structural'] if records else False,
        }
    return stats_dict
```

**验证**：
```bash
python RQ3_function_words/exp5_function_words_svd_mapping.py \
    --model gpt2 --nsamples 5 --savedir /tmp/rq3_test
# 检查 word_stats 里：
python3 -c "import json; d=json.load(open('/tmp/rq3_test/exp5_detailed_results.json')); 
funcs = [w for w,s in d['word_stats'].items() if s.get('is_function')]; 
contents = [w for w,s in d['word_stats'].items() if not s.get('is_function')];
print(f'func: {len(funcs)}, content: {len(contents)}')"
# 预期: content 数量 >> func 数量(词汇的大头)
```

---

## Bug B2：MoE `SparseMoeBlock` 无 `.up_proj/.down_proj`

**位置**：`paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py` + `lib/` 里访问 MLP 层的地方

**症状**（server 日志）：
```
AttributeError: 'Qwen3MoeSparseMoeBlock' object has no attribute 'up_proj'
AttributeError: 'Qwen3_5MoeSparseMoeBlock' object has no attribute 'up_proj'
```

**修法**：新增 MoE 适配函数，自动检测 MoE 并聚合 experts。

**新增到 `lib/model_utils.py`**：

```python
def is_moe_block(module):
    """Detect if a module is a MoE-style block with experts list."""
    return hasattr(module, 'experts') and hasattr(module, 'gate')

def get_moe_down_projs(mlp_module):
    """Extract all expert down_proj weights from a MoE block.
    Returns: list of (expert_idx, down_proj_weight_tensor)
    """
    if not is_moe_block(mlp_module):
        raise TypeError("Not a MoE block")
    return [(i, e.down_proj) for i, e in enumerate(mlp_module.experts)]

def get_mlp_down_proj_universal(layer, model_name):
    """Universal getter: handles dense + MoE.
    For MoE: returns list of per-expert down_projs
    For dense: returns single down_proj (wrapped in list of len 1)
    """
    mlp = getattr(layer, 'mlp', None)
    if mlp is None:
        raise ValueError(f"Layer has no 'mlp' attribute: {type(layer)}")
    if is_moe_block(mlp):
        return get_moe_down_projs(mlp)  # list of (idx, down_proj)
    # Dense fallback
    for attr in ['down_proj', 'fc_out', 'dense_4h_to_h', 'fc2', 'c_proj']:
        if hasattr(mlp, attr):
            return [(0, getattr(mlp, attr))]
    raise ValueError(f"Cannot find down_proj in {type(mlp)}")
```

**在 RQ3 脚本里使用**：

```python
# 原代码（第 590 行附近）
W2 = lib.get_mlp_down_proj(args.model, layer).cpu().float().t()

# 修后：
down_projs = get_mlp_down_proj_universal(layers[layer], args.model)
if len(down_projs) == 1:
    # Dense: normal single SVD
    W2 = down_projs[0][1].weight.cpu().float().t()
else:
    # MoE: aggregate per-expert (average or concatenate)
    W2 = sum(dp.weight for _, dp in down_projs) / len(down_projs)
    W2 = W2.cpu().float().t()
    # 或: 对每个 expert 单独做 SVD（Tier C per-expert 版本）
```

**MoE per-expert 版本**（Tier C 专用，新脚本 `exp5_moe_per_expert.py`）：

```python
# 对每个 expert 单独跑 SVD + 功能词对齐
for expert_idx, down_proj in down_projs:
    W2_exp = down_proj.weight.cpu().float().t()
    U_e, S_e, Vh_e = torch.linalg.svd(W2_exp, full_matrices=False)
    # 只对路由到该 expert 的 token 做统计
    ...
```

---

## Bug B3：`get_mlp_submodules()` 缺白名单

**位置**：`paper_experiments/lib/model_utils.py::get_mlp_submodules`（约第 342 行）

**症状**：
```
ValueError: Cannot identify MLP submodules for model 'glm4_9b'. Add support in model_utils.py.
```

**修法**：扩展白名单，加 glm4/qwen1.5/qwen3.5/yi。

**当前代码（推测）**：

```python
def get_mlp_submodules(model_name, layer):
    if 'gpt2' in model_name:
        return {'up_proj': layer.mlp.c_fc, 'activation': layer.mlp.act, 'down_proj': layer.mlp.c_proj}
    elif 'gptj' in model_name:
        return {'up_proj': layer.mlp.fc_in, 'activation': layer.mlp.act, 'down_proj': layer.mlp.fc_out}
    # ... 其他已支持的
    else:
        raise ValueError(f"Cannot identify MLP submodules for model '{model_name}'. Add support in model_utils.py.")
```

**修复：添加以下分支**：

```python
    # 新增：glm4 系列（使用 SwiGLU，结构与 llama 接近）
    elif 'glm4' in model_name:
        mlp = layer.mlp
        return {
            'up_proj': getattr(mlp, 'gate_up_proj', None) or mlp.dense_h_to_4h,
            'activation': mlp.activation_func if hasattr(mlp, 'activation_func') else None,
            'down_proj': mlp.down_proj if hasattr(mlp, 'down_proj') else mlp.dense_4h_to_h,
        }
    
    # 新增：qwen1.5 和 qwen3.5 系列（标准 SwiGLU）
    elif 'qwen1.5' in model_name or 'qwen3.5' in model_name:
        return {
            'up_proj': layer.mlp.up_proj,
            'gate_proj': layer.mlp.gate_proj,
            'activation': layer.mlp.act_fn,
            'down_proj': layer.mlp.down_proj,
        }
    
    # 新增：yi 系列（LLaMA-style）
    elif 'yi' in model_name:
        return {
            'up_proj': layer.mlp.up_proj,
            'gate_proj': layer.mlp.gate_proj,
            'activation': layer.mlp.act_fn,
            'down_proj': layer.mlp.down_proj,
        }
    
    # 新增：MoE 路由（Qwen3 MoE / Qwen3.5 MoE）
    elif 'a3b' in model_name or 'moe' in model_name.lower():
        from lib.model_utils import get_mlp_down_proj_universal
        # 返回特殊标记,调用方需自己迭代 experts
        return {
            '_moe_experts': get_mlp_down_proj_universal(layer, model_name),
            '_is_moe': True,
        }
```

**验证**：

```python
# Test:
from lib.model_utils import get_mlp_submodules
# 对每个模型试一次
for m in ['gpt2', 'gptj_6b', 'glm4_9b', 'qwen1.5_14b', 'qwen3.5_9b', 'yi_9b', 'qwen3_30b_a3b']:
    # ... load model, layer
    try:
        parts = get_mlp_submodules(m, layer)
        print(f'{m}: OK ({list(parts.keys())})')
    except Exception as e:
        print(f'{m}: FAIL - {e}')
```

---

## Bug B4：`get_mlp_down_proj` 缺 glm4/MoE 分支

**位置**：`changeHead_massvieAcitve/experiments/exp6_v_ablation/exp6_v_ablation.py::get_mlp_down_proj`（第 30 行）

**当前代码**：

```python
def get_mlp_down_proj(layer, model_name):
    if "gptj" in model_name: return layer.mlp.fc_out
    elif "falcon" in model_name: return layer.mlp.dense_4h_to_h
    elif "bloom" in model_name: return layer.mlp.dense_4h_to_h
    elif "qwen" in model_name or "mistral" in model_name or is_llama_model(model_name):
        return layer.mlp.down_proj
    elif "opt" in model_name: return layer.fc2
    elif "gpt2" in model_name: return layer.mlp.c_proj
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

**修复代码**：

```python
def get_mlp_down_proj(layer, model_name):
    """获取MLP的down_proj权重(扩展支持 glm4/yi/MoE)."""
    model_lower = model_name.lower()
    
    # MoE: 返回 list of per-expert (调用方需单独处理)
    if hasattr(layer.mlp, 'experts'):
        return [(i, e.down_proj) for i, e in enumerate(layer.mlp.experts)]
    
    # Dense models
    if "gptj" in model_lower:
        return layer.mlp.fc_out
    elif "falcon" in model_lower:
        return layer.mlp.dense_4h_to_h
    elif "bloom" in model_lower:
        return layer.mlp.dense_4h_to_h
    elif "glm4" in model_lower:
        # GLM4 uses SwiGLU with down_proj
        return layer.mlp.down_proj
    elif "yi" in model_lower:
        return layer.mlp.down_proj
    elif "qwen" in model_lower or "mistral" in model_lower or is_llama_model(model_name):
        return layer.mlp.down_proj
    elif "opt" in model_lower:
        return layer.fc2
    elif "gpt2" in model_lower:
        return layer.mlp.c_proj
    else:
        raise ValueError(f"Unknown model: {model_name}")
```

---

## Bug B5：`get_critical_layer` 默认 L0 不读 L_origin

**位置**：同上文件 `::get_critical_layer`（第 48 行）

**当前代码**：

```python
def get_critical_layer(model_name):
    critical_layers = {
        "bloom_7b1": 28,
        "opt_6.7b": 0,
    }
    return critical_layers.get(model_name, 0)
```

**修复代码**：

```python
def get_critical_layer(model_name, l_origin_json='paper_experiments/origin_layer/output/L_ORIGIN.json'):
    """读取 RQ2c 生成的 L_ORIGIN 作为 critical_layer."""
    import os, json
    # 支持通过命令行 --layer_id 参数覆盖
    if os.environ.get('OVERRIDE_CRITICAL_LAYER'):
        return int(os.environ['OVERRIDE_CRITICAL_LAYER'])
    
    # 读 L_ORIGIN 文件
    if os.path.exists(l_origin_json):
        with open(l_origin_json) as f:
            l_origin_map = json.load(f)
        if model_name in l_origin_map:
            return int(l_origin_map[model_name])
    
    # Fallback 到硬编码(legacy)
    legacy = {"bloom_7b1": 28, "opt_6.7b": 0}
    if model_name in legacy:
        return legacy[model_name]
    
    raise ValueError(f"No critical_layer for '{model_name}'; run origin_layer/ first or set OVERRIDE_CRITICAL_LAYER env var")
```

**或者直接从命令行传参**（更简单）：

```python
# 在 main() 里
parser.add_argument('--layer_id', type=int, default=None,
                    help='Critical layer (L_origin). If None, read from origin_layer/output/L_ORIGIN.json')
args = parser.parse_args()

critical_layer = args.layer_id if args.layer_id is not None else get_critical_layer(args.model)
```

---

## Bug B6：RQ6 baseline 只在 critical_layer 测（非真 MA）

**位置**：同上文件 `run_and_collect_ma` 函数（第 75-112 行）

**症状**：
- glm4_32b RQ6 baseline = 1.15 vs RQ2a baseline = 298598（差 260000×）
- 在 `critical_layer` 单层测 top1，不是 MA 真正的峰值层

**当前代码**（第 90 行附近）：

```python
layer = layers[layer_id]  # layer_id = critical_layer
handle = layer.register_forward_hook(make_hook(layer_id))
# ... forward ...
feat = activations[layer_id].numpy()
top1 = np.abs(feat).max()
```

**修复代码**：

```python
def run_and_collect_ma(model, layers, testseq, model_name, all_layers=True):
    """跑模型,在所有层找真正的 MA 极值(和 RQ2a 方式一致)."""
    activations = {}
    
    def make_hook(lid):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            activations[lid] = out.detach().cpu().float()
        return hook
    
    # 挂 hook 到所有层
    handles = []
    for lid, layer in enumerate(layers):
        h = layer.register_forward_hook(make_hook(lid))
        handles.append(h)
    
    with torch.no_grad():
        _ = model(testseq)
    
    for h in handles: h.remove()
    
    # 跨所有层找 top1
    peak_top1 = 0
    peak_layer = -1
    peak_dim = -1
    for lid in range(len(layers)):
        feat = activations[lid].numpy()
        if len(feat.shape) == 3:
            feat = feat.reshape(-1, feat.shape[-1])
        top1 = np.abs(feat).max()
        if top1 > peak_top1:
            peak_top1 = top1
            peak_layer = lid
            max_idx = np.unravel_index(np.abs(feat).argmax(), feat.shape)
            peak_dim = int(max_idx[1])
    
    return {
        "top1": float(peak_top1),
        "peak_layer": int(peak_layer),
        "ma_dim": peak_dim,
    }
```

**验证**：

```bash
# 修完后对 glm4_9b 重跑 baseline
python exp6_v_ablation.py --model glm4_9b --nsamples 5 --save-baseline-only
# 预期输出: baseline ≈ 2250（和 RQ2a 一致），不是 4.58
```

---

## Bug B7：RQ2a `MLPDisableHook` 未处理 tuple 输出

**位置**：`paper_experiments/RQ2_mlp_source/exp2a_mlp_feasibility_test.py` 第 45-53 行

**症状**：qwen3.5_35b_a3b RQ2a retain=81%（异常高），可能是 tuple 输出未正确 zero。

**当前代码**：

```python
class MLPDisableHook:
    def __init__(self, layer_id, mode='disable_all'):
        self.layer_id = layer_id
        self.mode = mode

    def __call__(self, module, input, output):
        if self.mode == 'disable_all':
            return torch.zeros_like(output)
        else:
            return output
```

**修复代码**（参照 `exp6_progressive_ablation.py` line 33-35）：

```python
class MLPDisableHook:
    def __init__(self, layer_id, mode='disable_all'):
        self.layer_id = layer_id
        self.mode = mode

    def __call__(self, module, input, output):
        if self.mode == 'disable_all':
            # 处理 MoE SparseMoeBlock 返回 tuple (hidden, router_logits) 的情况
            if isinstance(output, tuple):
                return (torch.zeros_like(output[0]),) + output[1:]
            return torch.zeros_like(output)
        return output
```

**验证**：

```bash
python RQ2_mlp_source/exp2a_mlp_feasibility_test.py \
    --model qwen3.5_35b_a3b --nsamples 5 --savedir /tmp/rq2a_moe_test
# 预期: retention_pct 应该显著 < 81%（接近 qwen3_30b_a3b 的 2.85%）
```

---

## 修复后 Sentinel 测试

修完所有 bug 后跑一次，确认无回归：

```bash
# 3 个 sentinel 模型：dense + glm4 + MoE
for model in gpt2 glm4_9b qwen3_30b_a3b; do
    echo "=== Testing $model ==="
    # RQ2a
    python RQ2_mlp_source/exp2a_mlp_feasibility_test.py --model $model --nsamples 5 --savedir /tmp/fix_test/$model/RQ2a
    # RQ3
    python RQ3_function_words/exp5_function_words_svd_mapping.py --model $model --nsamples 5 --layer_id 2 --savedir /tmp/fix_test/$model/RQ3
    # RQ6 (V ablation)
    python changeHead_massvieAcitve/experiments/exp6_v_ablation/exp6_v_ablation.py --model $model --nsamples 5 --layer_id 2 --savedir /tmp/fix_test/$model/RQ6
done

# 检查：
# 1. 都不报错
# 2. RQ3 的 word_stats 里 content words 数量 > func words
# 3. RQ6 的 baseline 和 RQ2a baseline 量级接近（不是差几千倍）
```

---

## 修复顺序建议

按**依赖链 + 受益最大**排序：

1. **B3**（白名单扩展，~10 min）—— 所有 glm4/qwen1.5/qwen3.5/yi 后续才能跑
2. **B1**（add_token 存所有 token，~15 min）—— RQ3 主 bug
3. **B4 + B5 + B6**（RQ6 一组改动，~40 min）—— RQ6 整体重做前提
4. **B2**（MoE 适配，~30 min）—— Tier C 前提
5. **B7**（RQ2a tuple，~5 min）—— 顺手修，不阻塞

**合计时间 ~2h**，修完后跑 sentinel 测试（~15 min）验证无回归。

---

## 日志

记录每个 bug 修复的实际时间 + 遇到的问题：

| Bug | 开始时间 | 结束时间 | 实际用时 | 备注 |
|:-:|---|---|:-:|---|
| B1 | — | — | — | — |
| B2 | — | — | — | — |
| B3 | — | — | — | — |
| B4 | — | — | — | — |
| B5 | — | — | — | — |
| B6 | — | — | — | — |
| B7 | — | — | — | — |

（修完一个填一个）
