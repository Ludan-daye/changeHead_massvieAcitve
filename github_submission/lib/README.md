# lib/ — 共享库

所有实验共用的模型加载、数据处理、评估和可视化工具。

## 模块说明

| 文件 | 功能 | 主要 API |
|------|------|---------|
| `load_model.py` | 统一模型加载 | `load_llm(args)` → (model, tokenizer, device, layers, hidden_size, seq_len) |
| | | `load_vit(args)` → (model, layers, val_transform) |
| `model_dict.py` | 模型注册表 | `MODEL_DICT_LLMs` — HuggingFace model ID 映射 |
| `hook.py` | 干预钩子 | `setup_intervene_hook(layer, model_name, reset_type)` — V-matrix 消融实现 |
| `load_data.py` | 数据加载 | `get_data(tokenizer, nsamples, seqlen)` — WikiText/RedPajama |
| `eval_utils.py` | 评估工具 | `eval_ppl(dataset, model, tokenizer, seed)` — 计算困惑度 |
| | | `test_imagenet(model, dataloader)` — ImageNet Top-1 准确率 |
| `plot_utils_llm.py` | LLM 可视化 | `plot_3d_feat()`, `plot_layer_ax()`, `plot_attn()` |
| `plot_utils_vit.py` | ViT 可视化 | `plot_3d_feat_vit()`, `plot_layer_ax_vit()` |

## 模型加载流程

```python
import lib

# 加载 LLM
model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)

# 加载数据
data = lib.get_data(tokenizer, nsamples=50, seqlen=2048)

# 评估 PPL
ppl = lib.eval_ppl("wikitext", model, tokenizer, seed=0)
```

## 添加新模型

编辑 `model_dict.py`，在 `MODEL_DICT_LLMs` 中添加：
```python
"your_model": {
    "model_id": "org/model-name",   # HuggingFace ID
    "cache_dir": CACHE_DIR_BASE
}
```

同时需要在 `load_model.py` 的 `load_llm()` 中添加层提取逻辑（如果模型结构不同于已有模型）。
