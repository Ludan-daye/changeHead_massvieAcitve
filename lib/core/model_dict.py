CACHE_DIR_BASE = "./model_weights"
LOCAL_MODELS_DIR = "LOCAL_MODELS_DIR"

MODEL_DICT_LLMs = {
    ### ============ 新下载的模型 (本地路径) ============ ###
    
    # Qwen2.5-7B (GQA架构)
    "qwen2.5_7b": {
        "model_id": f"{LOCAL_MODELS_DIR}/Qwen2.5-7B",
        "cache_dir": None,  # 本地模型不需要cache
    },
    
    # DeepSeek-V2-Lite (MoE架构)
    "deepseek_v2_lite": {
        "model_id": f"{LOCAL_MODELS_DIR}/DeepSeek-V2-Lite",
        "cache_dir": None,
    },
    
    # Mistral-7B-v0.3 (Sliding Window + GQA)
    "mistral_7b_v03": {
        "model_id": f"{LOCAL_MODELS_DIR}/Mistral-7B-v0.3",
        "cache_dir": None,
    },
    
    # Falcon-7B (Multi-Query + ALiBi + Parallel)
    "falcon_7b_local": {
        "model_id": f"{LOCAL_MODELS_DIR}/Falcon-7B",
        "cache_dir": None,
    },
    
    # BLOOM-7B1 (ALiBi位置编码)
    "bloom_7b1": {
        "model_id": f"{LOCAL_MODELS_DIR}/BLOOM-7B1",
        "cache_dir": None,
    },
    
    # GPT-J-6B (Parallel Attn+FFN + RoPE)
    "gptj_6b": {
        "model_id": f"{LOCAL_MODELS_DIR}/GPT-J-6B",
        "cache_dir": None,
    },
    
    ### ============ 原有模型 (HuggingFace缓存) ============ ###
    ### llama2 model
    "llama2_7b": {
        "model_id": "meta-llama/Llama-2-7b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "llma2b": {  # Alias spelling requested by users
        "model_id": "meta-llama/Llama-2-7b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "llama2_13b": {
        "model_id": "meta-llama/Llama-2-13b-hf",
        "cache_dir": CACHE_DIR_BASE
    },
    "llama2_70b": {
        "model_id": "meta-llama/Llama-2-70b-hf", 
        "cache_dir": CACHE_DIR_BASE
    },

    ### llama2 chat model
    "llama2_7b_chat": {
        "model_id": "meta-llama/Llama-2-7b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 
    "llama2_13b_chat": {
        "model_id": "meta-llama/Llama-2-13b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 
    "llama2_70b_chat": {
        "model_id": "meta-llama/Llama-2-70b-chat-hf",
        "cache_dir": CACHE_DIR_BASE
    }, 

    ### mistral model 
    "mistral_7b": {
        "model_id": "mistralai/Mistral-7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_moe": {
        "model_id": "mistralai/Mixtral-8x7B-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_7b_instruct":{
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mistral_moe_instruct": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "cache_dir": CACHE_DIR_BASE,
    }, 

    ### phi-2
    "phi-2": {
        "model_id": "microsoft/phi-2",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### falcon model
    "falcon_7b": {
        "model_id": "tiiuae/falcon-7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "falcon_40b": {
        "model_id": "tiiuae/falcon-40b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### mpt model 
    "mpt_7b": {
        "model_id": "mosaicml/mpt-7b",
        "cache_dir": CACHE_DIR_BASE,
    },
    "mpt_30b": {
        "model_id": "mosaicml/mpt-30b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### opt model 
    "opt_125m": {
        "model_id": "facebook/opt-125m", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_350m": {
        "model_id": "facebook/opt-350m", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_1.3b": {
        "model_id": "facebook/opt-1.3b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_2.7b": {
        "model_id": "facebook/opt-2.7b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_7b": {
        "model_id": "facebook/opt-6.7b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_13b": {
        "model_id": "facebook/opt-13b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_30b": {
        "model_id": "facebook/opt-30b", 
        "cache_dir": CACHE_DIR_BASE,
    },
    "opt_66b": {
        "model_id": "facebook/opt-66b",
        "cache_dir": CACHE_DIR_BASE,
    },

    ### gpt2 model
    "gpt2": {
        "model_id": "openai-community/gpt2",
        "cache_dir": CACHE_DIR_BASE
    },
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
}
