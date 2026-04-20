# `fixes/RQ3_function_words/` — exp5_function_words_svd_mapping.py 修复

## 修了什么

### Bug B1（主要）：只存功能词，丢内容词

原代码第 99-104 行：

```python
def add_token(self, token_text, h2_vector):
    self.total_token_count += 1
    if self.is_function_word(token_text):   # ← 过滤器
        self.word_data[token_text].append(...)
    # 非功能词直接丢弃！
```

**后果**：`word_data` 里**只有功能词**。所有 RQ3 下游分析（Cohen's d、SVD projection、alignment）**实际上在比较核心功能词 vs 边缘功能词**，**从未测过功能词 vs 内容词的对比**。16 模型的 Cohen's d 结论**方法学无效**。

### 副带修复 I2/I3

- **I2**：`r_func` 原用 `w.strip().lower() in FUNCTION_WORDS` 字符串匹配，漏掉 BPE `Ġ`-前缀 token（如 `'Ġthe'`），导致统计偏低。修复改用 tracker 的 `is_function` 标签。
- **I3**：原日志 "Collected X function word occurrences" 在 B1 修复后不准确，改为 "Collected X total tokens (F func, S struct, C content)"。

## 修复方法

### 1. 新增 `STRUCTURAL_TOKENS` 集合（文件第 78-93 行）

```python
STRUCTURAL_TOKENS = {
    # 标点
    '.', ',', '!', '?', ';', ':', '"', "'", '`',
    '(', ')', '[', ']', '{', '}',
    '-', '—', '–', '/', '\\', '|', '*', '&', '@', '#', '$', '%', '^', '~',
    # 换行 / 空白
    '\n', '\n\n', '\t', '\r', ' ',
    # 特殊 token
    '<bos>', '<eos>', '<pad>', '<unk>',
    '<|endoftext|>', '<|im_start|>', '<|im_end|>',
    '<s>', '</s>', '<|system|>', '<|user|>', '<|assistant|>',
}
```

### 2. 新增 `FunctionWordSVDTracker.is_structural_token()` 方法

```python
def is_structural_token(self, token_text):
    clean = token_text.lstrip('Ġ ').strip()
    if not clean: return True                      # 纯空白
    if clean in STRUCTURAL_TOKENS: return True     # 已知特殊 token
    if all(not c.isalnum() for c in clean): return True  # 全非字母数字
    return False
```

### 3. 改写 `add_token` 存所有 token

```python
def add_token(self, token_text, h2_vector):
    self.total_token_count += 1
    is_func = self.is_function_word(token_text)
    is_struct = self.is_structural_token(token_text)
    self.word_data[token_text].append({
        'context_id': self.context_counter,
        'h2': h2_vector.cpu().detach().numpy(),
        'is_function': is_func,
        'is_structural': is_struct,
    })
```

### 4. 改写 `get_word_statistics` 保持下游兼容

`occurrences` 字段仍返回 `[(context_id, h2_array), ...]` 格式；新增 `is_function` / `is_structural` 字段。

## 部署

```bash
cd <repo-root>
cp paper_experiments/fixes/RQ3_function_words/exp5_function_words_svd_mapping.py \
   paper_experiments/RQ3_function_words/exp5_function_words_svd_mapping.py
```

注：`STRUCTURAL_TOKENS` 直接内联在脚本里，**不需要单独的 `structural_tokens.py` 文件**。

## 怎么跑（以 glm4_9b 为例）

```bash
cd paper_experiments
python RQ3_function_words/exp5_function_words_svd_mapping.py \
    --model glm4_9b \
    --nsamples 30 \
    --layer_id 1 \
    --savedir results/wikitext_run/RQ3_origin/glm4_9b
```

**注意**：`--layer_id` 必须用 **RQ2c 给的起源层**（从 `paper_experiments/origin_layer/output/L_ORIGIN.json` 读取）。

### 批量跑 24 dense 模型

```bash
# 先 source 起源层映射
source paper_experiments/origin_layer/output/L_ORIGIN.sh

# 按 L_ORIGIN 每个模型跑一次
for model in bloom_7b1 falcon_7b glm4_9b glm4_32b gpt2 gptj_6b ... yi_9b; do
    L="${L_ORIGIN[$model]}"
    python RQ3_function_words/exp5_function_words_svd_mapping.py \
        --model $model --nsamples 30 --layer_id "$L" \
        --savedir results/wikitext_run/RQ3_origin/$model
done
```

### 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model` | 必填 | 模型名 |
| `--layer_id` | 必填 | **起源层**（见 L_ORIGIN.json）|
| `--nsamples` | 30 | 样本数 |
| `--savedir` | 必填 | 输出目录 |

## 预期输出

```
results/wikitext_run/RQ3_origin/<model>/
├── exp5_detailed_results.json   # 含 word_stats (带 is_function/is_structural 标签)
├── table1_rq3.json              # 含 func_pct / struct_pct / content_pct
├── exp5_alignment_v1.png
├── exp5_asymmetry_analysis.png
├── exp5_concentration_top5.png
├── exp5_stability_analysis.png
└── EXP5_SUMMARY.txt
```

### `word_stats` 新字段

每个 word 的 entry 现在有：

```json
{
  "the": {
    "count": 2945,
    "contexts": 30,
    "occurrences": [(ctx_id, h2_array), ...],
    "is_function": true,        // 新增
    "is_structural": false      // 新增
  }
}
```

### `table1_rq3.json` 新字段

```json
{
  "model": "glm4_9b",
  "func_pct": 28.5,              // 已有
  "struct_pct": 12.3,            // 新增
  "content_pct": 59.2,           // 新增
  "func_tokens": 17427,
  "struct_tokens": 7531,         // 新增
  "content_tokens": 36482,       // 新增
  "total_tokens": 61440
}
```

## 验证

```bash
bash paper_experiments/fixes/sentinel_test.sh     # Test F 检查分类
```

**手动检查**：

```bash
python3 -c "
import json
d = json.load(open('results/wikitext_run/RQ3_origin/glm4_9b/exp5_detailed_results.json'))
ws = d['word_stats']
n_func = sum(s['count'] for s in ws.values() if s['is_function'])
n_struct = sum(s['count'] for s in ws.values() if s['is_structural'])
n_content = sum(s['count'] for s in ws.values() if not s['is_function'] and not s['is_structural'])
print(f'func={n_func}  struct={n_struct}  content={n_content}')
# 预期: content > func > struct
"
```

## MoE 模型（qwen3_30b_a3b / qwen3.5_35b_a3b）

**本脚本不适配 MoE** —— 访问 `.up_proj` 会 `AttributeError`。Tier C 专项（MoE per-expert 分析）另写脚本。**批量跑时跳过 MoE**。

## 下游影响

如果有其他脚本读 `exp5_detailed_results.json` 的 `word_stats`：
- 老字段（count / contexts / occurrences）**完全兼容**
- 新字段（is_function / is_structural）**可选读**（ignore 不会坏）
- `table1_rq3.json` 增加 3 个字段，读的时候注意 KeyError 兼容
