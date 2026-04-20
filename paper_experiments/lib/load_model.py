import os

import torch
try:
    import timm
except ImportError:
    timm = None
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel

from .model_dict import MODEL_DICT_LLMs, resolve_model_id


# ---------------------------------------------------------------------------
# transformers 5.x compatibility shim for older remote-code models (ChatGLM).
# Multiple internal paths (`infer_auto_device_map`, `caching_allocator_warmup`,
# `get_total_byte_count`, …) read `model.all_tied_weights_keys`, which only
# exists on PreTrainedModel subclasses *shipped with* transformers 5.x. Remote
# trust-remote-code models (e.g. ChatGLM) predate this API and crash with
# AttributeError. We inject an empty dict as the class-level default so any
# subclass that doesn't override inherits "no tied weights" — correct for
# byte-count/device-map purposes; a subclass that *does* define its own will
# shadow this default naturally.
# ---------------------------------------------------------------------------
if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
    PreTrainedModel.all_tied_weights_keys = {}


def load_llm(args):
    print(f"loading model {args.model}")
    # resolve_model_id: 若 model_id 是本地路径且不存在，自动 fallback 到 HF ID（若已登记）
    model_name = resolve_model_id(args.model)
    cache_dir = MODEL_DICT_LLMs[args.model]["cache_dir"]

    # Only use token if it's not the default placeholder
    token_kwargs = {} if args.access_token == "type in your access token here" else {"token": args.access_token}

    # For attention analysis, we need attn_implementation='eager'
    attn_impl = getattr(args, 'attn_implementation', 'eager')

    # dtype 选择（对齐旧仓库 lib/core/load_model.py 的稳定策略）:
    # - bf16 集合: 数值范围与 fp32 相当，避免 fp16 溢出 Inf，速度远快于 fp32
    #   · glm4_32b / glm4_9b : ChatGLM 的 MA 数值极大，fp16 溢出
    #   · qwen3.5_9b         : 之前 fp16 路径下 routing/数值不稳
    #   · qwen3_30b_a3b      : MoE 从 FP8 反量化就是 bf16，统一保留
    # - 其他模型: fp16（默认，速度最快）
    # 用户可通过 --force_fp32 强制走 fp32
    force_fp32 = getattr(args, 'force_fp32', False)
    bf16_models = {"glm4_32b", "glm4_9b", "qwen3.5_9b", "qwen3_30b_a3b", "qwen3.5_35b_a3b"}
    if force_fp32:
        dtype = torch.float32
        print(f"  → using dtype=float32 (--force_fp32)")
    elif args.model in bf16_models:
        dtype = torch.bfloat16
        print(f"  → using dtype=bfloat16 (stable, avoids fp16 Inf overflow on {args.model})")
    else:
        dtype = torch.float16
        print(f"  → using dtype=float16 (default)")

    # glm4 需要 trust_remote_code=True，走独立分支
    if "glm4" in args.model.lower():
        # ChatGLM modeling code requires `config.max_length` which was removed
        # in transformers 5.x. Pre-load the config and inject the attribute so
        # the remote-trust-code model instantiates cleanly.
        glm_cfg = AutoConfig.from_pretrained(
            model_name, cache_dir=cache_dir, trust_remote_code=True, **token_kwargs
        )
        if not hasattr(glm_cfg, "max_length") or glm_cfg.max_length is None:
            fallback = (getattr(glm_cfg, "seq_length", None)
                        or getattr(glm_cfg, "max_position_embeddings", None)
                        or 2048)
            glm_cfg.max_length = int(fallback)
            print(f"  → injected config.max_length={glm_cfg.max_length} for glm (ChatGLM shim)")
        # Other attributes the ChatGLM remote code reads via `self.config.X` but
        # which transformers 5.x's PretrainedConfig no longer auto-populates:
        _glm_defaults = {
            "use_cache": False,            # we never generate; False = no KV cache overhead
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        for _k, _v in _glm_defaults.items():
            if not hasattr(glm_cfg, _k):
                setattr(glm_cfg, _k, _v)
        # NOTE: device_map="auto" triggers transformers 5.x `infer_auto_device_map`,
        # which reads `model.all_tied_weights_keys` — an attribute only present on
        # the new PreTrainedModel base but NOT on ChatGLM's shipped remote code.
        # Use explicit `{"": 0}` to skip auto-inference; glm4_9b (20GB bf16) and
        # glm4_32b (64GB bf16) both fit on a single H100-80GB.
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=glm_cfg, torch_dtype=dtype, cache_dir=cache_dir,
            low_cpu_mem_usage=True, device_map={"": 0}, trust_remote_code=True,
            attn_implementation=attn_impl, **token_kwargs,
        )
    elif "falcon" in args.model or "mpt" in args.model or "phi" in args.model:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True, attn_implementation=attn_impl, **token_kwargs)
    elif "mistral" in args.model or "pythia" in args.model:
        model = AutoModelForCausalLM.from_pretrained(model_name, revision=args.revision, torch_dtype=dtype, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", attn_implementation=attn_impl, **token_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, cache_dir=cache_dir, low_cpu_mem_usage=True, device_map="auto", attn_implementation=attn_impl, **token_kwargs)
    model.eval()

    # glm4 / falcon / mpt / phi ship a custom tokenizer → needs trust_remote_code.
    tok_kwargs = dict(token_kwargs)
    if any(t in args.model.lower() for t in ("glm4", "falcon", "mpt", "phi")):
        tok_kwargs["trust_remote_code"] = True
    if "mpt" in args.model or "pythia" in args.model:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, **tok_kwargs)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, **tok_kwargs)

    device = torch.device("cuda:0")
    if "mpt_30b" in args.model:
        device = model.hf_device_map["transformer.wte"]
    elif "30b" in args.model or "65b" in args.model or "70b" in args.model or "40b" in args.model:
        # Historically used multi-GPU device_map; on a single big GPU (e.g. H100-80GB)
        # the whole model fits and `hf_device_map` is not populated — fall back to cuda:0.
        if hasattr(model, "hf_device_map") and "lm_head" in getattr(model, "hf_device_map", {}):
            device = torch.device("cuda:" + str(model.hf_device_map["lm_head"]))
        else:
            device = torch.device("cuda:0")

    if "llama2_13b" == args.model:
        # device = torch.device("cuda:"+str(model.hf_device_map["lm_head"]))
        device = torch.device("cuda:0")  # 8.138 single GPU

    seq_len=4096
    if "llama" in args.model or "mistral" in args.model:
        layers = model.model.layers
        hidden_size = model.config.hidden_size
    elif "qwen" in args.model:
        layers = model.model.layers
        hidden_size = model.config.hidden_size
    elif "glm" in args.model.lower():
        # GLM4 有两种结构：
        # - 老版本/ChatGLM3: model.transformer.encoder.layers
        # - transformers 新集成: model.model.layers（和 Qwen 一样）
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'encoder'):
            layers = model.transformer.encoder.layers
        elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        else:
            raise ValueError(f"Cannot locate layers for glm model {args.model}")
        hidden_size = model.config.hidden_size
    elif "falcon" in args.model:
        layers = model.transformer.h
        hidden_size = model.config.hidden_size
    elif "bloom" in args.model:
        layers = model.transformer.h
        hidden_size = model.config.hidden_size
        seq_len = 2048
    elif "gptj" in args.model or "gpt-j" in args.model:
        layers = model.transformer.h
        hidden_size = model.config.hidden_size
        seq_len = 2048
    elif "mpt" in args.model:
        layers = model.transformer.blocks
        hidden_size = model.config.d_model
        seq_len=2048
    elif "opt" in args.model:
        layers = model.model.decoder.layers
        hidden_size = model.config.hidden_size
        seq_len = 2048
    elif "gpt2" in args.model:
        layers = model.transformer.h
        hidden_size = model.transformer.embed_dim
        seq_len = 1024
    elif "pythia" in args.model:
        layers = model.gpt_neox.layers
        hidden_size = model.gpt_neox.config.hidden_size
        seq_len = 2048
    elif "phi-2" in args.model:
        layers = model.model.layers
        hidden_size = model.config.hidden_size
    else:
        # Fallback: try common attribute patterns
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            layers = model.model.layers
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            layers = model.transformer.h
        else:
            raise ValueError(f"Unsupported model architecture: {args.model}. Add layer extraction logic in load_model.py.")
        hidden_size = model.config.hidden_size 

    return model, tokenizer, device, layers, hidden_size, seq_len

def load_vit(args):
    if args.model_family == "mae":
        patch_size=14 if args.model_size == "huge" else 16
        model = timm.create_model(f'vit_{args.model_size}_patch{patch_size}_224.mae', pretrained=True)
    elif args.model_family == "openai_clip":
        patch_size=14 if args.model_size == "large" else 16
        model = timm.create_model(f"vit_{args.model_size}_patch{patch_size}_clip_224.openai", pretrained=True)
    elif args.model_family == "dinov2":
        model = timm.create_model(f"vit_{args.model_size}_patch14_dinov2.lvd142m", pretrained=True, num_classes=1000)
    elif args.model_family == "dinov2_reg":
        model = timm.create_model(f"vit_{args.model_size}_patch14_reg4_dinov2.lvd142m", pretrained=True, num_classes=1000)

    model = model.cuda()
    model = model.eval()

    layers = model.blocks

    data_config = timm.data.resolve_model_data_config(model)
    val_transform = timm.data.create_transform(**data_config, is_training=False)

    return model, layers, val_transform

def load_dinov2_linear_head(args):
    assert "dinov2" in args.model_family, "this function is only for dinov2 models"
    if args.model_family == "dinov2_reg":
        linear_head_path = os.path.join(args.linear_head_path, f"dinov2_vit{args.model_size[0]}14_reg4_linear_head.pth")
    elif args.model_family == "dinov2":
        linear_head_path = os.path.join(args.linear_head_path, f"dinov2_vit{args.model_size[0]}14_linear_head.pth")

    linear_head_weights = torch.load(linear_head_path)
    return linear_head_weights