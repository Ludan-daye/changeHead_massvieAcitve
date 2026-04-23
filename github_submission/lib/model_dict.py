"""
MODEL_DICT_LLMs — 模型注册表。

每个条目:
    "model_key": {
        "model_id": "<local_path_or_hf_id>",   # 默认尝试的路径或 HF ID
        "hf_fallback": "<hf_id>",              # 可选：本地路径不存在时自动下载
        "cache_dir": CACHE_DIR_BASE,
    }

外部用户:
  - 本地路径不存在 → load_model.py 自动走 `hf_fallback`（如果有）
  - 没有 hf_fallback 的条目 → 必须手动设置本地权重

本地用户（服务器上有权重）:
  - 保持原 /model/... 路径不变，无感
  - 可通过环境变量 `LOCAL_MODELS_DIR` 统一覆盖前缀（见 README）
"""

import os

CACHE_DIR_BASE = os.environ.get("HF_CACHE_DIR", "./model_weights")


def _entry(local_path: str, hf_fallback: str = "") -> dict:
    """构造一个模型条目。本地路径不存在时 load_model.py 会走 hf_fallback。"""
    return {
        "model_id": local_path,
        "hf_fallback": hf_fallback,
        "cache_dir": CACHE_DIR_BASE,
    }


MODEL_DICT_LLMs = {
    ### ============================================================
    ### ALL_EXPERIMENTS_SUMMARY.json 里的 25 模型（本轮重点）
    ### ============================================================

    # —— 公开 HF 模型（无需本地路径也能跑）——
    "gpt2":             _entry("openai-community/gpt2",     "openai-community/gpt2"),
    "gptj_6b":          _entry("EleutherAI/gpt-j-6b",       "EleutherAI/gpt-j-6b"),
    "opt_6.7b":         _entry("facebook/opt-6.7b",         "facebook/opt-6.7b"),
    "bloom_7b1":        _entry("bigscience/bloom-7b1",      "bigscience/bloom-7b1"),
    "falcon_7b":        _entry("tiiuae/falcon-7b",          "tiiuae/falcon-7b"),
    "mistral_7b_v03":   _entry("mistralai/Mistral-7B-v0.3", "mistralai/Mistral-7B-v0.3"),
    "llama2_13b":       _entry("meta-llama/Llama-2-13b-hf", "meta-llama/Llama-2-13b-hf"),

    # —— 服务器本地路径 + HF fallback ——
    "llama3.1_8b":      _entry("/model/llm/Meta-Llama-3.1-8B-Instruct",               "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen2.5_0.5b":     _entry("/model/ModelScope/Qwen/Qwen2.5-0.5B",                 "Qwen/Qwen2.5-0.5B"),
    "qwen2.5_7b":       _entry("/model/ModelScope/Qwen/Qwen2.5-7B-Instruct",          "Qwen/Qwen2.5-7B-Instruct"),
    "qwen2_7b":         _entry("/model/ModelScope/Qwen/Qwen2-7B-Instruct",            "Qwen/Qwen2-7B-Instruct"),
    "qwen3_0.6b":       _entry("/model/ModelScope/Qwen/Qwen3-0.6B",                   "Qwen/Qwen3-0.6B"),
    "qwen3_1.7b":       _entry("/model/ModelScope/Qwen/Qwen3-1.7B",                   "Qwen/Qwen3-1.7B"),
    "qwen3_4b":         _entry("/model/ModelScope/Qwen/Qwen3-4B",                     "Qwen/Qwen3-4B"),
    "qwen3_8b":         _entry("/model/ModelScope/Qwen/Qwen3-8B",                     "Qwen/Qwen3-8B"),
    "qwen3_14b":        _entry("/model/ModelScope/Qwen/Qwen3-14B",                    "Qwen/Qwen3-14B"),
    "qwen3_32b":        _entry("/model/ModelScope/Qwen/Qwen3-32B",                    "Qwen/Qwen3-32B"),
    "qwen1.5_14b":      _entry("/model/ModelScope/Qwen/Qwen1.5-14B",                  "Qwen/Qwen1.5-14B"),
    "yi_9b":            _entry("/model/ModelScope/01ai/Yi-1.5-9B-Chat",               "01-ai/Yi-1.5-9B-Chat"),
    "glm4_9b":          _entry("/model/ModelScope/ZhipuAI/glm-4-9b-chat",             "THUDM/glm-4-9b-chat"),

    # —— 暂无公开 HF 的（占位路径；外部用户目前无法运行这些）——
    "glm4_32b":         _entry("/model/ModelScope/ZhipuAI/glm-4-32b-chat",            ""),
    "qwen3.5_9b":       _entry("/model/ModelScope/Qwen/Qwen3.5-9B",                   ""),
    "qwen3.5_27b":      _entry("/model/ModelScope/Qwen/Qwen3.5-27B",                  ""),
    "qwen3_30b_a3b":    _entry("/model/HuggingFace/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8", "Qwen/Qwen3-30B-A3B"),
    "qwen3.5_35b_a3b":  _entry("/model/HuggingFace/Qwen/Qwen3.5-35B-A3B",             ""),

    ### ============================================================
    ### 历史遗留（非本轮 25 模型范围；保留兼容）
    ### ============================================================

    "llama2_7b":        _entry("/model/ModelScope/shakechen/Llama-2-7b-hf",  "meta-llama/Llama-2-7b-hf"),
    "llama2_7b_chat":   _entry("/model/ModelScope/shakechen/Llama-2-7b-chat-hf", "meta-llama/Llama-2-7b-chat-hf"),
    "mistral_7b":       _entry("mistralai/Mistral-7B-v0.1",  "mistralai/Mistral-7B-v0.1"),
    "mistral_8b":       _entry("/model/ModelScope/AI-ModelScope/Ministral-8B-Instruct-2410", "mistralai/Ministral-8B-Instruct-2410"),
    "gemma3_12b":       _entry("/model/HuggingFace/google/gemma-3-12b-it",            "google/gemma-3-12b-it"),
    "hunyuan_7b":       _entry("/model/HuggingFace/tencent/Hunyuan-7B-Instruct",      "tencent/Hunyuan-7B-Instruct"),
    "deepseek_r1_7b":   _entry("/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
    "deepseek_coder_7b":_entry("/model/ModelScope/deepseek-ai/deepseek-coder-6.7b-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct"),
    "glm4.7_flash":     _entry("/model/ModelScope/ZhipuAI/GLM-4.7-Flash", ""),
    "deepseek_v3.1":    _entry("/model/HuggingFace/deepseek-ai/DeepSeek-V3.1", "deepseek-ai/DeepSeek-V3.1"),
    "gpt2_medium":      _entry("gpt2-medium",                "gpt2-medium"),
    "gpt2_large":       _entry("gpt2-large",                 "gpt2-large"),
    "gpt2_xl":          _entry("gpt2-xl",                    "gpt2-xl"),
    "bloom_7b":         _entry("bigscience/bloom-7b1",       "bigscience/bloom-7b1"),  # alias
    "opt_7b":           _entry("facebook/opt-6.7b",          "facebook/opt-6.7b"),      # alias
    "opt_125m":         _entry("facebook/opt-125m",          "facebook/opt-125m"),
    "pythia_7b":        _entry("EleutherAI/pythia-6.9b",     "EleutherAI/pythia-6.9b"),
}


def resolve_model_id(key: str) -> str:
    """
    返回实际应使用的 model_id：
      - 若 "model_id" 是本地路径且该路径存在，用它
      - 否则若有 "hf_fallback"，用 HF ID
      - 否则返回 "model_id" 本身（可能会报错——提示用户配好权重）
    """
    entry = MODEL_DICT_LLMs[key]
    mid = entry["model_id"]
    fallback = entry.get("hf_fallback", "")

    # 是本地路径吗？（以 / 开头的视为绝对路径）
    if mid.startswith("/"):
        if os.path.isdir(mid):
            return mid
        # 本地路径不存在，尝试 fallback
        if fallback:
            print(f"[model_dict] local path for '{key}' not found: {mid}")
            print(f"[model_dict] falling back to HuggingFace: {fallback}")
            return fallback
        print(f"[model_dict] WARNING: local path for '{key}' not found and no hf_fallback set.")
        print(f"[model_dict] Returning '{mid}' anyway; may fail at load time.")
        return mid

    # 不是本地路径（本来就是 HF ID 之类），直接用
    return mid
