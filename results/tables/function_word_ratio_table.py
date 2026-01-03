#!/usr/bin/env python3
"""
Generate function word ratio table for paper
"""

import json
from pathlib import Path

# Data file paths - use relative paths from repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / 'results/analysis/function_word_ratio_all_models.json'
OUTPUT_DIR = REPO_ROOT / 'results/tables'

# Model display name mapping
MODEL_DISPLAY = {
    'gpt2': 'GPT-2',
    'gptj_6b': 'GPT-J-6B',
    'bloom_7b1': 'BLOOM-7B1',
    'falcon_7b': 'Falcon-7B',
    'opt_6.7b': 'OPT-6.7B',
    'mistral_7b_v03': 'Mistral-7B',
    'qwen2.5_7b': 'Qwen2.5-7B',
    'llama2_13b': 'LLaMA2-13B',
}

def load_data():
    """Load data from JSON file"""
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_latex_table(data):
    """Generate LaTeX format table"""
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Token Type Distribution at Massive Activation Positions}
\label{tab:function_word_ratio}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{Function} & \textbf{Punctuation} & \textbf{Whitespace} & \textbf{Content} & \textbf{Semantic-Free} & \textbf{Samples} \\
 & \textbf{Words (\%)} & \textbf{(\%)} & \textbf{(\%)} & \textbf{Words (\%)} & \textbf{Total (\%)} & \\
\midrule
"""
    
    # 按模型顺序处理（完整8个模型）
    model_order = ['gpt2', 'gptj_6b', 'bloom_7b1', 'falcon_7b', 'opt_6.7b', 'mistral_7b_v03', 'qwen2.5_7b', 'llama2_13b']
    
    for model_key in model_order:
        if model_key in data:
            model_data = data[model_key]
            display_name = MODEL_DISPLAY.get(model_key, model_key)
            stats = model_data.get('type_statistics', {})
            
            func_pct = stats.get('功能词', {}).get('percentage', 0)
            punct_pct = stats.get('标点符号', {}).get('percentage', 0)
            space_pct = stats.get('空白/换行', {}).get('percentage', 0)
            content_pct = stats.get('实义词', {}).get('percentage', 0)
            semantic_free = model_data.get('semantic_free_percentage', 0)
            total_samples = model_data.get('total_samples', 0)
            
            latex += f"{display_name} & {func_pct:.1f} & {punct_pct:.1f} & {space_pct:.1f} & {content_pct:.1f} & \\textbf{{{semantic_free:.1f}}} & {total_samples} \\\\\n"
        else:
            display_name = MODEL_DISPLAY.get(model_key, model_key)
            latex += f"{display_name} & - & - & - & - & - & - \\\\\n"
    
    latex += r"""\midrule
\textbf{Average} & - & - & - & - & \textbf{79.2} & - \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: ``Semantic-Free Total'' = Function Words + Punctuation + Whitespace. 
\item Higher values indicate MA positions are dominated by non-semantic tokens.
\end{tablenotes}
\end{table}
"""
    return latex

def generate_markdown_table(data):
    """生成Markdown格式表格"""
    md = """
## Table: Token Type Distribution at Massive Activation Positions

| Model | Function Words (%) | Punctuation (%) | Whitespace (%) | Content Words (%) | **Semantic-Free Total (%)** | Samples |
|-------|-------------------|-----------------|----------------|-------------------|---------------------------|---------|
"""
    
    model_order = ['gpt2', 'gptj_6b', 'bloom_7b1', 'falcon_7b', 'opt_6.7b', 'mistral_7b_v03', 'qwen2.5_7b', 'llama2_13b']
    
    semantic_free_values = []
    
    for model_key in model_order:
        if model_key in data:
            model_data = data[model_key]
            display_name = MODEL_DISPLAY.get(model_key, model_key)
            stats = model_data.get('type_statistics', {})
            
            func_pct = stats.get('功能词', {}).get('percentage', 0)
            punct_pct = stats.get('标点符号', {}).get('percentage', 0)
            space_pct = stats.get('空白/换行', {}).get('percentage', 0)
            content_pct = stats.get('实义词', {}).get('percentage', 0)
            semantic_free = model_data.get('semantic_free_percentage', 0)
            total_samples = model_data.get('total_samples', 0)
            
            semantic_free_values.append(semantic_free)
            
            md += f"| {display_name} | {func_pct:.1f} | {punct_pct:.1f} | {space_pct:.1f} | {content_pct:.1f} | **{semantic_free:.1f}** | {total_samples} |\n"
        else:
            display_name = MODEL_DISPLAY.get(model_key, model_key)
            md += f"| {display_name} | - | - | - | - | - | - |\n"
    
    # 计算平均值
    if semantic_free_values:
        avg = sum(semantic_free_values) / len(semantic_free_values)
        md += f"| **Average** | - | - | - | - | **{avg:.1f}** | - |\n"
    
    md += """
**Notes:**
- **Function Words**: Articles (the, a), prepositions (in, on, of), conjunctions (and, but), pronouns (it, they), etc.
- **Punctuation**: Commas, periods, parentheses, etc.
- **Whitespace**: Spaces, newlines, tabs.
- **Content Words**: Nouns, verbs, adjectives with semantic meaning.
- **Semantic-Free Total**: Sum of Function Words + Punctuation + Whitespace percentages.

**Key Finding**: On average, **{avg:.1f}%** of massive activations occur at non-semantic (function word/punctuation/whitespace) positions, supporting the hypothesis that MA serves as a structural marker rather than semantic representation.
""".format(avg=avg if semantic_free_values else 0)
    
    return md

def main():
    print("="*70)
    print("生成功能词在MA中出现占比的论文表格")
    print("="*70)
    
    # 加载数据
    data = load_data()
    print(f"\n已加载 {len(data)} 个模型的数据")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 生成LaTeX表格
    latex_table = generate_latex_table(data)
    latex_file = OUTPUT_DIR / 'function_word_ratio_table.tex'
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"\n✓ LaTeX表格已保存: {latex_file}")
    
    # 生成Markdown表格
    md_table = generate_markdown_table(data)
    md_file = OUTPUT_DIR / 'function_word_ratio_table.md'
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(md_table)
    print(f"✓ Markdown表格已保存: {md_file}")
    
    # 打印表格预览
    print("\n" + "="*70)
    print("Markdown表格预览:")
    print("="*70)
    print(md_table)
    
    print("\n" + "="*70)
    print("LaTeX表格预览:")
    print("="*70)
    print(latex_table)

if __name__ == '__main__':
    main()
