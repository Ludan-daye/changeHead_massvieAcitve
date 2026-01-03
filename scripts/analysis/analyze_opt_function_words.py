#!/usr/bin/env python3
"""
为OPT-6.7B分析功能词在MA中出现的占比
独立脚本，不依赖lib模块
"""

import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
from collections import Counter
import json
from pathlib import Path

# 禁用代理，使用本地缓存
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"

# 功能词列表
FUNCTION_WORDS = {
    # 冠词
    'the', 'a', 'an',
    # 代词
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs',
    'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what',
    # 介词
    'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under',
    # 连词
    'and', 'or', 'but', 'nor', 'so', 'yet', 'because', 'although', 'though',
    'while', 'if', 'unless', 'until', 'when', 'where', 'whether', 'as', 'than',
    # 助动词
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'will', 'would', 'shall', 'should', 'may', 'might', 'can', 'could', 'must',
    # 其他
    'not', 'no', 'yes', 'there', 'here', 'then', 'now', 'just', 'only', 'also',
}

PUNCTUATION = set('.,;:!?\'\"()[]{}/-–—…·•@#$%^&*+=<>|\\`~')


def classify_token(token):
    """分类token类型"""
    token_clean = token.strip().lower()
    
    # 特殊token（如</s>, <s>, <pad>等）归类为标点符号
    if token_clean.startswith('<') and token_clean.endswith('>'):
        return '标点符号'
    
    if not token_clean or token_clean in ['\n', '\n\n', '\t', ' ', '  ']:
        return '空白/换行'
    
    if all(c in PUNCTUATION or c.isspace() for c in token_clean):
        return '标点符号'
    
    if token_clean in FUNCTION_WORDS:
        return '功能词'
    
    return '实义词'


def analyze_opt_model():
    """分析OPT-6.7B的功能词占比"""
    print("="*60)
    print("OPT-6.7B 功能词在MA中出现占比分析")
    print("="*60)
    
    # 加载模型和tokenizer
    model_name = "facebook/opt-6.7b"
    print(f"\n加载模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    
    # 加载数据
    print("\n加载WikiText-2数据集...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # 收集MA位置的token（只在最后一层找全局最大MA位置）
    ma_tokens = []
    n_samples = 50
    
    print(f"\n分析 {n_samples} 个样本...")
    
    sample_idx = 0
    for item in tqdm(dataset, total=n_samples):
        if sample_idx >= n_samples:
            break
            
        text = item['text']
        if len(text.strip()) < 50:
            continue
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(model.device)
        
        if input_ids.shape[1] < 10:
            continue
        
        # Forward pass with hooks to capture activations
        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # 只在最后一层找全局最大MA位置
            last_hs = hidden_states[-1][0]  # [seq_len, hidden_dim]
            
            # 找到绝对值最大的激活
            abs_hs = torch.abs(last_hs)
            max_val = torch.max(abs_hs).item()
            
            # 找到最大激活的位置
            max_pos = torch.argmax(abs_hs).item()
            token_pos = max_pos // last_hs.shape[1]
            
            # 记录这个位置的token
            token_id = input_ids[0, token_pos].item()
            token = tokenizer.decode([token_id])
            ma_tokens.append({
                'token': token,
                'ma_value': max_val
            })
        
        sample_idx += 1
    
    # 统计token类型
    print(f"\n收集到 {len(ma_tokens)} 个MA token")
    
    # 打印前20个token用于调试
    print("\n前20个MA token:")
    for i, item in enumerate(ma_tokens[:20]):
        token = item['token']
        token_type = classify_token(token)
        print(f"  {i+1}. '{repr(token)}' -> {token_type}")
    
    type_counter = Counter()
    for item in ma_tokens:
        token_type = classify_token(item['token'])
        type_counter[token_type] += 1
    
    total = sum(type_counter.values())
    if total == 0:
        print("警告：没有收集到足够的MA token")
        return None
    
    # 计算百分比
    result = {
        'model': 'OPT-6.7B',
        'total_samples': n_samples,
        'ma_token_count': total,
        'type_statistics': {},
    }
    
    for token_type, count in type_counter.items():
        result['type_statistics'][token_type] = {
            'count': count,
            'percentage': round(count / total * 100, 1)
        }
    
    # 计算语义无关占比
    semantic_free = sum(type_counter.get(t, 0) for t in ['功能词', '标点符号', '空白/换行'])
    result['semantic_free_percentage'] = round(semantic_free / total * 100, 1)
    
    # 打印结果
    print("\n" + "="*60)
    print("统计结果:")
    print("="*60)
    for token_type, stats in result['type_statistics'].items():
        print(f"  {token_type}: {stats['count']} ({stats['percentage']}%)")
    print(f"\n  语义无关总占比: {result['semantic_free_percentage']}%")
    
    # Save results
    output_dir = Path(__file__).resolve().parents[2] / 'results' / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'opt_6.7b_function_word_analysis.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 结果已保存: {output_file}")
    
    return result


if __name__ == '__main__':
    analyze_opt_model()
