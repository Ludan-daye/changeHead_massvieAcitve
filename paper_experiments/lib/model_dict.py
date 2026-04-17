CACHE_DIR_BASE = "./model_weights"

MODEL_DICT_LLMs = {
    ### ============================================================
    ### 论文原有 8 模型
    ### ============================================================

    "llama2_7b": {
        "model_id": "/model/ModelScope/shakechen/Llama-2-7b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "llama2_7b_chat": {
        "model_id": "/model/ModelScope/shakechen/Llama-2-7b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "qwen2.5_7b": {
        "model_id": "/model/ModelScope/Qwen/Qwen2.5-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_8b": {
        "model_id": "/model/ModelScope/AI-ModelScope/Ministral-8B-Instruct-2410",
        "cache_dir": CACHE_DIR_BASE,
    },
    # 以下 4 个服务器上没有，保留 HuggingFace ID（需下载）
    "gpt2": {
        "model_id": "openai-community/gpt2",
        "cache_dir": CACHE_DIR_BASE
    },
    "bloom_7b": {
        "model_id": "bigscience/bloom-7b1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "gptj_6b": {
        "model_id": "EleutherAI/gpt-j-6b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_7b": {
        "model_id": "facebook/opt-6.7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon_7b": {
        "model_id": "tiiuae/falcon-7b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### ============================================================
    ### 新增 Dense LLM（服务器本地路径）
    ### ============================================================

    "llama3.1_8b": {
        "model_id": "/model/llm/Meta-Llama-3.1-8B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen3_8b": {
        "model_id": "/model/ModelScope/Qwen/Qwen3-8B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen2_7b": {
        "model_id": "/model/ModelScope/Qwen/Qwen2-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "glm4_9b": {
        "model_id": "/model/ModelScope/ZhipuAI/glm-4-9b-chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "gemma3_12b": {
        "model_id": "/model/HuggingFace/google/gemma-3-12b-it",
        "cache_dir": CACHE_DIR_BASE,
    },
    "hunyuan_7b": {
        "model_id": "/model/HuggingFace/tencent/Hunyuan-7B-Instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "deepseek_r1_7b": {
        "model_id": "/model/HuggingFace/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "deepseek_coder_7b": {
        "model_id": "/model/ModelScope/deepseek-ai/deepseek-coder-6.7b-instruct",
        "cache_dir": CACHE_DIR_BASE,
    },
    "yi_9b": {
        "model_id": "/model/ModelScope/01ai/Yi-1.5-9B-Chat",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen1.5_14b": {
        "model_id": "/model/ModelScope/Qwen/Qwen1.5-14B",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### ============================================================
    ### Qwen3 规模缩放系列
    ### ============================================================

    "qwen3_0.6b": {
        "model_id": "/model/ModelScope/Qwen/Qwen3-0.6B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen3_1.7b": {
        "model_id": "/model/ModelScope/Qwen/Qwen3-1.7B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen3_4b": {
        "model_id": "/model/ModelScope/Qwen/Qwen3-4B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen3_14b": {
        "model_id": "/model/ModelScope/Qwen/Qwen3-14B",
        "cache_dir": CACHE_DIR_BASE,
    },
    "qwen3_32b": {
        "model_id": "/model/ModelScope/Qwen/Qwen3-32B",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### ============================================================
    ### MoE 架构
    ### ============================================================

    "qwen3_30b_a3b": {
        "model_id": "/model/HuggingFace/Qwen/Qwen3-30B-A3B-Thinking-2507-FP8",
        "cache_dir": CACHE_DIR_BASE,
    },
    "glm4.7_flash": {
        "model_id": "/model/ModelScope/ZhipuAI/GLM-4.7-Flash",
        "cache_dir": CACHE_DIR_BASE,
    },
    "deepseek_v3.1": {
        "model_id": "/model/HuggingFace/deepseek-ai/DeepSeek-V3.1",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### ============================================================
    ### 其他保留（HuggingFace ID，需下载）
    ### ============================================================

    "gpt2_medium": {
        "model_id": "gpt2-medium",
        "cache_dir": CACHE_DIR_BASE
    },
    "gpt2_large": {
        "model_id": "gpt2-large",
        "cache_dir": CACHE_DIR_BASE
    },
    "gpt2_xl": {
        "model_id": "gpt2-xl",
        "cache_dir": CACHE_DIR_BASE
    },
    "llama2_13b": {
        "model_id": "meta-llama/Llama-2-13b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_125m": {
        "model_id": "facebook/opt-125m",
        "cache_dir": CACHE_DIR_BASE,
    },
    "pythia_7b": {
        "model_id": "EleutherAI/pythia-6.9b",
        "cache_dir": CACHE_DIR_BASE,
    },
}
