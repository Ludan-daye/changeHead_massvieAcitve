#!/usr/bin/env python3
"""
分析功能词在MA中出现的占比
为所有8个模型生成详细统计表格
"""

import json
import os
import sys
from pathlib import Path
from collections import Counter

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 功能词列表（英文常见功能词）
FUNCTION_WORDS = {
    # 冠词
    'the', 'a', 'an',
    # 代词
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what',
    'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'themselves',
    # 介词
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
    'over', 'out', 'up', 'down', 'off', 'across', 'along', 'around', 'behind',
    # 连词
    'and', 'or', 'but', 'nor', 'so', 'yet', 'for', 'because', 'although', 'though',
    'while', 'if', 'unless', 'until', 'when', 'where', 'whether', 'as', 'than',
    # 助动词
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could', 'must',
    # 其他功能词
    'not', 'no', 'yes', 'there', 'here', 'then', 'now', 'just', 'only', 'also',
    'very', 'too', 'more', 'most', 'less', 'least', 'much', 'many', 'some', 'any',
    'all', 'both', 'each', 'every', 'either', 'neither', 'other', 'another',
}

# 标点符号
PUNCTUATION = set('.,;:!?\'\"()[]{}/-–—…·•@#$%^&*+=<>|\\`~')

def classify_token(token):
    """分类token类型"""
    token_clean = token.strip().lower()
    
    # 空白/换行
    if not token_clean or token_clean in ['\n', '\n\n', '\t', ' ', '  ']:
        return '空白/换行'
    
    # 标点符号
    if all(c in PUNCTUATION or c.isspace() for c in token_clean):
        return '标点符号'
    
    # 功能词
    if token_clean in FUNCTION_WORDS:
        return '功能词'
    
    # 实义词
    return '实义词'


def load_ma_tokens_from_exp1(model_key):
    """从exp1实验结果加载MA token数据"""
    exp1_dir = PROJECT_ROOT / 'results' / 'experiments' / 'exp1' / model_key
    
    # 尝试多个可能的文件
    possible_files = [
        exp1_dir / 'ma_tokens.json',
        exp1_dir / 'top_ma_tokens.json',
        exp1_dir / 'summary.json',
    ]
    
    for f in possible_files:
        if f.exists():
            with open(f, 'r') as fp:
                data = json.load(fp)
                if 'tokens' in data:
                    return data['tokens']
                if 'top_tokens' in data:
                    return data['top_tokens']
    
    return None


def load_ma_tokens_from_archive(model_key):
    """从archive加载MA token数据"""
    archive_dir = PROJECT_ROOT / 'results' / 'archive' / 'by_model' / model_key
    
    possible_files = [
        archive_dir / 'exp1_ma_position' / 'ma_tokens.json',
        archive_dir / 'exp1_ma_position' / 'top_ma_tokens.json',
    ]
    
    for f in possible_files:
        if f.exists():
            with open(f, 'r') as fp:
                data = json.load(fp)
                if 'tokens' in data:
                    return data['tokens']
                if 'top_tokens' in data:
                    return data['top_tokens']
    
    return None


def analyze_model(model_key, model_display):
    """分析单个模型的功能词占比"""
    # 尝试从不同位置加载数据
    tokens = load_ma_tokens_from_exp1(model_key)
    if tokens is None:
        tokens = load_ma_tokens_from_archive(model_key)
    
    if tokens is None:
        return None
    
    # 统计token类型
    type_counter = Counter()
    total = 0
    
    for item in tokens:
        if isinstance(item, dict):
            token = item.get('token', '')
            count = item.get('count', 1)
        else:
            token = str(item)
            count = 1
        
        token_type = classify_token(token)
        type_counter[token_type] += count
        total += count
    
    if total == 0:
        return None
    
    # 计算百分比
    result = {
        'model': model_display,
        'total_samples': total,
        'type_statistics': {},
    }
    
    for token_type, count in type_counter.items():
        result['type_statistics'][token_type] = {
            'count': count,
            'percentage': round(count / total * 100, 1)
        }
    
    # 计算语义无关占比（功能词 + 标点符号 + 空白/换行）
    semantic_free = 0
    for t in ['功能词', '标点符号', '空白/换行']:
        semantic_free += type_counter.get(t, 0)
    
    result['semantic_free_percentage'] = round(semantic_free / total * 100, 1)
    
    return result


def main():
    """主函数"""
    print("="*70)
    print("功能词在MA中出现占比分析")
    print("="*70)
    
    # 8个模型配置
    models = [
        ('gpt2', 'GPT-2'),
        ('gptj_6b', 'GPT-J-6B'),
        ('bloom_7b1', 'BLOOM-7B1'),
        ('falcon_7b', 'Falcon-7B'),
        ('opt_6.7b', 'OPT-6.7B'),
        ('mistral_7b_v03', 'Mistral-7B'),
        ('qwen2.5_7b', 'Qwen2.5-7B'),
        ('llama2_13b', 'LLaMA2-13B'),
    ]
    
    results = {}
    
    for model_key, model_display in models:
        print(f"\n分析 {model_display}...")
        result = analyze_model(model_key, model_display)
        if result:
            results[model_key] = result
            print(f"  ✓ 总样本: {result['total_samples']}")
            print(f"  ✓ 语义无关占比: {result['semantic_free_percentage']}%")
        else:
            print(f"  ⚠ 数据未找到")
    
    # 保存结果
    output_file = PROJECT_ROOT / 'results' / 'analysis' / 'function_word_ratio_all_models.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    # 打印表格
    print("\n" + "="*70)
    print("详细统计表格")
    print("="*70)
    
    print(f"\n{'模型':<15} {'功能词%':<10} {'标点%':<10} {'空白%':<10} {'实义词%':<10} {'语义无关%':<12}")
    print("-"*70)
    
    for model_key, model_display in models:
        if model_key in results:
            r = results[model_key]
            stats = r['type_statistics']
            func_pct = stats.get('功能词', {}).get('percentage', 0)
            punct_pct = stats.get('标点符号', {}).get('percentage', 0)
            space_pct = stats.get('空白/换行', {}).get('percentage', 0)
            content_pct = stats.get('实义词', {}).get('percentage', 0)
            semantic_free = r['semantic_free_percentage']
            
            print(f"{model_display:<15} {func_pct:<10.1f} {punct_pct:<10.1f} {space_pct:<10.1f} {content_pct:<10.1f} {semantic_free:<12.1f}")
        else:
            print(f"{model_display:<15} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12}")
    
    print("-"*70)
    
    return results


if __name__ == '__main__':
    main()
