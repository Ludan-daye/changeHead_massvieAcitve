def is_llama_model(model_name: str) -> bool:
    """
    Determine whether the provided model name refers to a LLaMA-family model.

    We treat both the canonical spellings (e.g., ``llama2_7b``) and the short-hand
    alias requested by users (e.g., ``llma2b``) as LLaMA models.

    Also includes LLaMA-architecture variants like Mistral and Qwen.
    """
    if not model_name:
        return False

    lowered = model_name.lower()
    # Check for LLaMA and its architecture variants
    llama_variants = ["llama", "llma", "mistral", "qwen"]
    return any(variant in lowered for variant in llama_variants)


__all__ = ["is_llama_model"]
