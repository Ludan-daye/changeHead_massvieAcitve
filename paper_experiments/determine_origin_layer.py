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
DEFAULT_JSON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'results/ALL_EXPERIMENTS_SUMMARY_v2.json'
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
      2. 对 DISPERSED 模型：取 final_disabled_set 的**前 50%**，
         否则全部
      3. None

    注意 DISPERSED 模型用完整 final_disabled_set 可能效果差（见 qwen1.5_14b、
    qwen3.5_9b、qwen3.5_27b 的 RQ5b 弱结果）。这里保守取前半。
    """
    e2c = entry.get('exp2c')
    if not e2c:
        return None
    fset = e2c.get('final_disabled_set') or []
    if not fset:
        return None
    category = e2c.get('category', '')

    if category == 'DISPERSED':
        # 前 50% 层
        n = max(1, len(fset) // 2)
        layers = sorted(set(fset[:n]))
    else:
        # CONCENTRATED / FEW-SOURCE: 全部
        layers = sorted(set(fset))
    return layers


def build_table(data: dict) -> dict:
    """给每个模型计算 single + macro 起源层。"""
    out = {}
    for model, entry in data.items():
        sl = derive_single_layer(model, entry)
        ml = derive_macro_layers(model, entry)
        e2c = entry.get('exp2c') or {}
        out[model] = {
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json-path', default=DEFAULT_JSON,
                   help=f'v2 JSON path (default: {DEFAULT_JSON})')
    p.add_argument('--bash', action='store_true',
                   help='只输出 bash 关联数组')
    p.add_argument('--json', action='store_true',
                   help='以 JSON 输出（程序化使用）')
    p.add_argument('--compare', action='store_true',
                   help='对比旧 L_ORIGIN 与新的差异')
    args = p.parse_args()

    if not os.path.exists(args.json_path):
        raise SystemExit(f"找不到 JSON: {args.json_path}")

    data = load_v2(args.json_path)
    table = build_table(data)

    if args.bash:
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
