#!/usr/bin/env python3
"""
determine_origin_layer.py

从 `results/ALL_EXPERIMENTS_SUMMARY_v2.json` 的 exp2c 字段，为每个模型**自动推导**
正确的起源层 / 起源层集合，输出可直接粘贴到 `run_rq345_origin_layer.sh` 的
`L_ORIGIN` 和 `ORIGIN_LAYERS_MACRO` 关联数组。

为什么需要这个脚本
====================
旧的 `run_rq345_origin_layer.sh` 里 `L_ORIGIN` 的值来自 v1 JSON 的
`exp2.critical_layer`（单层最强消融层）。对 CONCENTRATED/FEW-SOURCE 模型 OK，
但对 DISPERSED 模型严重失真——例如 glm4_9b 在 v1 里 critical_layer=17，
而 exp2c 的贪心消融在 L1 就砍掉 78% MA。用 L17 做 RQ3/4/5 根本碰不到真实起源。

本脚本自动产出两份起源层表:
- 单层实验 (RQ3/4/5-single):   用 exp2c.l_origin_from_step1
- macro 实验 (RQ5b/RQ6-macro): 用 exp2c.final_disabled_set（DISPERSED 模式取前半）

用法
====
    python determine_origin_layer.py                # 打印到 stdout
    python determine_origin_layer.py --json         # JSON 格式输出
    python determine_origin_layer.py --bash         # 只输出 bash 关联数组
    python determine_origin_layer.py --compare      # 对比旧 L_ORIGIN 与新的差异

    python determine_origin_layer.py --bash > /tmp/L_ORIGIN_v2.sh
    # 然后把输出粘贴进 run_rq345_origin_layer.sh 替换原 L_ORIGIN
"""

import os
import re
import json
import argparse
from typing import Optional

# 脚本默认读的 JSON 路径（相对于脚本所在目录）
# 此脚本在 paper_experiments/origin_layer/ 下，数据在 paper_experiments/results/
DEFAULT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', 'results', 'ALL_EXPERIMENTS_SUMMARY_v2.json'
)

# 默认输出目录（paper_experiments/origin_layer/output/）
DEFAULT_OUTDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'output'
)

# 旧 L_ORIGIN 表（从 v1 critical_layer 来的，已在 run_rq345_origin_layer.sh 内）
OLD_L_ORIGIN = {
    'bloom_7b1': 3, 'falcon_7b': 3, 'gptj_6b': 2,
    'llama3.1_8b': 1, 'qwen2.5_0.5b': 0, 'qwen2.5_7b': 3,
    'qwen2_7b': 3, 'qwen3_0.6b': 2, 'qwen3_1.7b': 2,
    'yi_9b': 1, 'mistral_7b_v03': 1, 'qwen1.5_14b': 2,
    'qwen3_4b': 5, 'qwen3_8b': 6, 'qwen3_14b': 6,
    'qwen3.5_9b': 26, 'glm4_9b': 17, 'qwen3_30b_a3b': 2,
    'gpt2': 3, 'opt_6.7b': 1, 'qwen3_32b': 43,
    'qwen3.5_27b': 54, 'qwen3.5_35b_a3b': 39,
}


def load_v2(path: str) -> dict:
    """读 v2 JSON，处理 NaN/Infinity。"""
    with open(path) as f:
        text = f.read()
    # json 不支持 NaN/Infinity，替换成 null 再 parse
    text = re.sub(r'\bNaN\b', 'null', text)
    text = re.sub(r'\bInfinity\b', 'null', text)
    text = re.sub(r'\b-Infinity\b', 'null', text)
    return json.loads(text)


def derive_single_layer(model: str, entry: dict) -> Optional[int]:
    """
    推导**单层实验**的起源层。

    优先级:
      1. exp2c.l_origin_from_step1  (最准，贪心消融第一步)
      2. exp2.critical_layer        (v1 方法，回退)
      3. None                       (无法确定)
    """
    e2c = entry.get('exp2c')
    if e2c and e2c.get('l_origin_from_step1') is not None:
        return int(e2c['l_origin_from_step1'])
    e2 = entry.get('exp2')
    if e2 and e2.get('critical_layer') is not None:
        return int(e2['critical_layer'])
    return None


def derive_macro_layers(model: str, entry: dict) -> Optional[list]:
    """
    推导 **macro 实验** 的起源层集合。

    优先级:
      1. exp2c.final_disabled_set  (贪心完整集合)
         - DISPERSED 模型: 取前 50%（避免包含弱贡献层）
         - 否则: 全部
      2. **fallback**: 从 exp2.critical_layer L 推导:
         - L ≤ 5: 取 [0..5]  (前段模型，覆盖早期起源段)
         - L > 5: 取 [L-2..L+2] (深层起源，以 L 为中心±2)
         这是经验值，供外部用户在没跑 exp2c 时也能启动 macro 实验
      3. None（连 critical_layer 都没有）

    注意 DISPERSED 模型用完整 final_disabled_set 可能效果差（见 qwen1.5_14b、
    qwen3.5_9b、qwen3.5_27b 的 RQ5b 弱结果）。这里保守取前半。
    """
    # 主路径: 用 exp2c 的 final_disabled_set
    e2c = entry.get('exp2c')
    if e2c:
        fset = e2c.get('final_disabled_set') or []
        if fset:
            category = e2c.get('category', '')
            if category == 'DISPERSED':
                n = max(1, len(fset) // 2)
                return sorted(set(fset[:n]))
            return sorted(set(fset))

    # Fallback: 用 v1 critical_layer 推导合理的 macro 窗口
    e2 = entry.get('exp2')
    if e2 and e2.get('critical_layer') is not None:
        L = int(e2['critical_layer'])
        n_layers = e2.get('num_layers_tested')
        if L <= 5:
            # 早期起源：覆盖 L0 到 L5
            return list(range(0, 6))
        # 深层起源：L-2 到 L+2 的 5 层窗口
        lo = max(0, L - 2)
        hi = L + 2
        if n_layers is not None:
            hi = min(hi, int(n_layers) - 1)
        return list(range(lo, hi + 1))

    return None


def macro_source(entry: dict) -> str:
    """返回 macro_layers 的来源标签（供 SUMMARY 展示）。"""
    e2c = entry.get('exp2c')
    if e2c and e2c.get('final_disabled_set'):
        return 'exp2c'
    e2 = entry.get('exp2')
    if e2 and e2.get('critical_layer') is not None:
        return 'fallback_window'
    return 'none'


def build_table(data: dict) -> dict:
    """给每个模型计算 single + macro 起源层。"""
    out = {}
    for model, entry in data.items():
        sl = derive_single_layer(model, entry)
        ml = derive_macro_layers(model, entry)
        msrc = macro_source(entry)
        e2c = entry.get('exp2c') or {}
        out[model] = {
            'macro_source': msrc,
            'single_layer': sl,
            'macro_layers': ml,
            'category': e2c.get('category'),
            'steps_to_kill': e2c.get('steps_to_kill'),
            'total_drop_pct': e2c.get('total_drop_pct'),
        }
    return out


def print_bash(table: dict):
    """输出可直接粘贴到 shell 的关联数组。"""
    print("# === 来自 determine_origin_layer.py 的自动产出 ===")
    print("# 来源: results/ALL_EXPERIMENTS_SUMMARY_v2.json 的 exp2c")
    print("# 更新时: 替换 run_rq345_origin_layer.sh 中原 L_ORIGIN 数组")
    print()
    print("# 单层实验用的起源层 (RQ3 / RQ4 / RQ5-single)")
    print("declare -A L_ORIGIN=(")
    for m in sorted(table.keys()):
        sl = table[m]['single_layer']
        cat = table[m].get('category', '?') or '?'
        if sl is not None:
            print(f"    [{m}]={sl:<3}  # {cat}")
        else:
            print(f"    # [{m}]=??  # {cat}, 无 exp2c 数据")
    print(")")
    print()
    print("# macro 实验用的起源层集合 (RQ5b / RQ6 macro-SVD)")
    print("declare -A ORIGIN_LAYERS_MACRO=(")
    for m in sorted(table.keys()):
        ml = table[m]['macro_layers']
        cat = table[m].get('category', '?') or '?'
        if ml:
            layers_str = ','.join(str(x) for x in ml)
            print(f'    [{m}]="{layers_str}"  # {cat}')
        else:
            print(f'    # [{m}]=""  # {cat}, 无 macro 层数据')
    print(")")


def print_compare(table: dict):
    """
    对比新旧 L_ORIGIN，高亮差异。

    注：对 DISPERSED 模型，单层实验本质上就是弱结果（无论新旧层）——
    Δ 很大不是"修好了"而是"都不对"，真实故事在 macro 实验里。
    """
    print(f"\n{'模型':<22} {'旧':>5} {'新':>5} {'Δ':>5} {'类别':<15} {'steps':>6} {'drop%':>8}  {'注意':<25}")
    print("-" * 105)
    for m in sorted(table.keys()):
        if m not in OLD_L_ORIGIN:
            continue
        old = OLD_L_ORIGIN[m]
        new = table[m]['single_layer']
        cat = table[m].get('category', '?') or '?'
        steps = table[m].get('steps_to_kill')
        drop = table[m].get('total_drop_pct')
        if new is None:
            diff = '-'
        else:
            diff = abs(new - old)
            diff = f"{diff}" + ("!" if diff > 3 else "")
        drop_str = f"{drop:.1f}" if drop is not None else '-'
        note = ''
        if cat == 'DISPERSED':
            note = '多层分散, 单层必弱'
        elif new is None:
            note = '无 exp2c'
        elif isinstance(diff, str) and '!' in diff:
            note = '新旧差大, 用新的'
        print(f"{m:<22} {old:>5} {('-' if new is None else str(new)):>5} {diff:>5} {cat:<15} {str(steps or '-'):>6} {drop_str:>8}  {note:<25}")


def print_json(table: dict):
    print(json.dumps(table, indent=2, ensure_ascii=False))


def print_pretty(table: dict):
    """人类可读的概要表。"""
    print(f"\n{'模型':<22} {'单层起源':>8} {'macro 起源集合':<34} {'类别':<15}")
    print("-" * 85)
    for m in sorted(table.keys()):
        sl = table[m]['single_layer']
        ml = table[m]['macro_layers']
        cat = table[m].get('category', '?') or '?'
        sl_str = str(sl) if sl is not None else '-'
        ml_str = '[' + ','.join(str(x) for x in ml) + ']' if ml else '-'
        if len(ml_str) > 33:
            ml_str = ml_str[:30] + '...'
        print(f"{m:<22} {sl_str:>8} {ml_str:<34} {cat:<15}")


def dump_all(table: dict, outdir: str):
    """
    一键产出所有格式到 outdir/:
      - L_ORIGIN.json             单层起源层（model -> int）
      - L_ORIGIN.sh               单层起源层 bash 数组
      - ORIGIN_LAYERS_MACRO.json  macro 起源层集合（model -> list[int]）
      - ORIGIN_LAYERS_MACRO.sh    macro 起源层 bash 数组
      - compare_v1_vs_v2.txt      和旧 L_ORIGIN 的对比
      - SUMMARY.md                人类可读的汇总报告
    """
    os.makedirs(outdir, exist_ok=True)

    # 1. L_ORIGIN.json
    single = {m: t['single_layer'] for m, t in table.items() if t['single_layer'] is not None}
    with open(os.path.join(outdir, 'L_ORIGIN.json'), 'w') as f:
        json.dump(single, f, indent=2, ensure_ascii=False, sort_keys=True)

    # 2. ORIGIN_LAYERS_MACRO.json
    macro = {m: t['macro_layers'] for m, t in table.items() if t['macro_layers']}
    with open(os.path.join(outdir, 'ORIGIN_LAYERS_MACRO.json'), 'w') as f:
        json.dump(macro, f, indent=2, ensure_ascii=False, sort_keys=True)

    # 3. L_ORIGIN.sh（bash 数组，可 source 或粘贴）
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_bash(table)
    full = buf.getvalue()
    # 拆分为两个文件
    part1 = full.split('# macro 实验用的起源层集合')[0]
    part2_header = '# macro 实验用的起源层集合 (RQ5b / RQ6 macro-SVD)\n'
    part2 = part2_header + full.split(part2_header)[1] if part2_header in full else ''
    with open(os.path.join(outdir, 'L_ORIGIN.sh'), 'w') as f:
        f.write(part1)
    with open(os.path.join(outdir, 'ORIGIN_LAYERS_MACRO.sh'), 'w') as f:
        f.write(part2)

    # 4. compare_v1_vs_v2.txt
    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        print_compare(table)
    with open(os.path.join(outdir, 'compare_v1_vs_v2.txt'), 'w') as f:
        f.write(buf2.getvalue())

    # 5. SUMMARY.md（人类可读的起源层汇总）
    # 先统计覆盖情况
    total = len(table)
    with_exp2c = sum(1 for t in table.values() if t.get('category'))
    v1_fallback = sum(1 for t in table.values()
                      if t['single_layer'] is not None and not t.get('category'))
    no_data = sum(1 for t in table.values() if t['single_layer'] is None)
    macro_exp2c = sum(1 for t in table.values() if t.get('macro_source') == 'exp2c')
    macro_fallback = sum(1 for t in table.values() if t.get('macro_source') == 'fallback_window')
    missing_models = [m for m, t in sorted(table.items()) if t['single_layer'] is None]

    lines = ['# 起源层自动判定结果（SUMMARY）', '',
             '> 由 `determine_origin_layer.py` 从 `ALL_EXPERIMENTS_SUMMARY_v2.json` 的 `exp2c` 自动产出',
             '',
             '## 覆盖情况',
             '',
             f'- JSON 中总模型数：**{total}**',
             f'',
             f'### 单层起源层（L_ORIGIN.json）',
             f'- 用 exp2c 数据推导：**{with_exp2c}** 个（最准）',
             f'- 无 exp2c、回退到 v1 `exp2.critical_layer`：**{v1_fallback}** 个',
             f'- 无任何可用数据 → 未产出：**{no_data}** 个']
    if missing_models:
        lines.append(f'  - 缺失模型列表：{", ".join(f"`{m}`" for m in missing_models)}')
    lines += [f'- **总计 `L_ORIGIN.json` 含 {with_exp2c + v1_fallback} 个模型**',
              '',
              f'### Macro 起源集合（ORIGIN_LAYERS_MACRO.json）',
              f'- 用 exp2c `final_disabled_set` 推导：**{macro_exp2c}** 个（最准）',
              f'- 无 exp2c、用 `critical_layer ± 2` 或 `[0..5]` 启发式窗口 fallback：**{macro_fallback}** 个',
              f'- 无任何可用数据：**{no_data}** 个',
              f'- **总计 `ORIGIN_LAYERS_MACRO.json` 含 {macro_exp2c + macro_fallback} 个模型**',
              '',
              '> 单层和 macro 都已为可推导的模型全部确定。',
              '',
              '## 所有模型的起源层',
             '',
             '| 模型 | 单层 L_ORIGIN | Macro 起源层集合 | macro 来源 | 类别 | 消融步数 | MA 总降 % |',
             '|---|:-:|---|:-:|:-:|:-:|:-:|']
    for m in sorted(table.keys()):
        t = table[m]
        sl = t['single_layer']
        ml = t['macro_layers']
        cat = t.get('category', '?') or '?'
        steps = t.get('steps_to_kill', '?')
        drop = t.get('total_drop_pct')
        msrc = t.get('macro_source', 'none')
        sl_str = str(sl) if sl is not None else '—'
        ml_str = '[' + ','.join(str(x) for x in ml) + ']' if ml else '—'
        drop_str = f"{drop:.1f}%" if drop is not None else '—'
        msrc_label = {
            'exp2c': '✓ exp2c',
            'fallback_window': '⚠ 启发式',
            'none': '—',
        }.get(msrc, msrc)
        lines.append(f'| `{m}` | **{sl_str}** | {ml_str} | {msrc_label} | {cat} | {steps} | {drop_str} |')

    lines += ['', '## 分类说明',
              '',
              '`category` 由 exp2c 实验直接写入，**仅按 `steps_to_kill` 划分**：',
              '',
              '| 类别 | steps_to_kill | 典型含义 |',
              '|:-:|:-:|---|',
              '| CONCENTRATED | = 1 | 单层主导，该层就是起源（对应模式 A）|',
              '| FEW-SOURCE | 2 – 5 | 少数层共同主导 |',
              '| DISPERSED | > 5 | 多层分散贡献（对应模式 B，单层实验必弱）|',
              '',
              '> 注：`total_drop_pct` 不参与分类——例如 qwen3.5_35b_a3b 总降仅 15.9% 仍标 FEW-SOURCE',
              '> （因 3 步就耗尽消融预算，但单步贡献不够大）。',
              '',
              '## 和 v1 L_ORIGIN 的对比',
              '',
              '详见 `compare_v1_vs_v2.txt`。关键差异：',
              '',
              '| 模型 | v1 | v2 | 类别 |',
              '|---|:-:|:-:|:-:|',
              ]
    big_diffs = []
    for m in sorted(table.keys()):
        if m not in OLD_L_ORIGIN:
            continue
        old = OLD_L_ORIGIN[m]
        new = table[m]['single_layer']
        if new is None:
            continue
        if abs(new - old) > 3:
            big_diffs.append((m, old, new, table[m].get('category', '?') or '?'))
    for m, o, n, cat in big_diffs:
        lines.append(f'| `{m}` | {o} | **{n}** | {cat} |')

    lines += ['', '## 使用这些数据的方式',
              '',
              '```bash',
              '# 方式一：直接 source 到 shell',
              'source output/L_ORIGIN.sh',
              'echo "${L_ORIGIN[gptj_6b]}"  # → 2',
              '',
              '# 方式二：JSON 程序化读取',
              'python3 -c "import json; d=json.load(open(\'output/L_ORIGIN.json\')); print(d[\'gptj_6b\'])"',
              '',
              '# 方式三：粘贴 bash 数组到 run_rq345_origin_layer.sh',
              'cat output/L_ORIGIN.sh  # 复制 declare -A 部分',
              '```',
              '']

    with open(os.path.join(outdir, 'SUMMARY.md'), 'w') as f:
        f.write('\n'.join(lines))

    print(f"✓ 输出已写入 {outdir}/")
    for fn in ['SUMMARY.md', 'L_ORIGIN.json', 'L_ORIGIN.sh',
               'ORIGIN_LAYERS_MACRO.json', 'ORIGIN_LAYERS_MACRO.sh',
               'compare_v1_vs_v2.txt']:
        path = os.path.join(outdir, fn)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"    {fn:<32} ({size} bytes)")


def main():
    p = argparse.ArgumentParser(
        description="从 exp2c 自动推导每个模型的起源层，服务 RQ3/4/5/6 实验"
    )
    p.add_argument('--json-path', default=DEFAULT_JSON,
                   help=f'v2 JSON path (default: {DEFAULT_JSON})')
    p.add_argument('--bash', action='store_true',
                   help='只输出 bash 关联数组')
    p.add_argument('--json', action='store_true',
                   help='以 JSON 输出（程序化使用）')
    p.add_argument('--compare', action='store_true',
                   help='对比旧 L_ORIGIN 与新的差异')
    p.add_argument('--dump-all', nargs='?', const=DEFAULT_OUTDIR,
                   help=f'一次性产出所有格式到目录（默认 {DEFAULT_OUTDIR}）')
    args = p.parse_args()

    if not os.path.exists(args.json_path):
        raise SystemExit(f"找不到 JSON: {args.json_path}")

    data = load_v2(args.json_path)
    table = build_table(data)

    if args.dump_all is not None:
        dump_all(table, args.dump_all)
    elif args.bash:
        print_bash(table)
    elif args.json:
        print_json(table)
    elif args.compare:
        print_compare(table)
    else:
        print_pretty(table)
        print_compare(table)


if __name__ == '__main__':
    main()
