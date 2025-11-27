def is_llama_model(model_name: str) -> bool:
    """
    Determine whether the provided model name refers to a LLaMA-family model.

    We treat both the canonical spellings (e.g., ``llama2_7b``) and the short-hand
    alias requested by users (e.g., ``llma2b``) as LLaMA models.
    """
    if not model_name:
        return False

    lowered = model_name.lower()
    return "llama" in lowered or lowered.startswith("llma")


__all__ = ["is_llama_model"]
