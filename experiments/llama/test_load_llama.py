#!/usr/bin/env python3
"""
Minimal test: Load LLaMA-2-13B and perform a simple forward pass
Used to verify that the current proxy + token configuration can successfully pull models from HuggingFace
"""

import argparse
import torch
import lib

def main():
    print("="*80)
    print("TEST: Load LLaMA-2-13B and run simple forward pass")
    print("="*80)
    
    # Construct arguments (same way as experiment scripts)
    args = argparse.Namespace(
        model='llama2_13b',
        access_token='type in your access token here',  # Will automatically use HF_TOKEN environment variable
        attn_implementation='eager',
        revision='main',
    )
    
    print("\n[1/3] Loading model via lib.load_llm...")
    try:
        model, tokenizer, device, layers, hidden_size, seq_len = lib.load_llm(args)
        print(f"✓ Model loaded successfully!")
        print(f"  - Device: {device}")
        print(f"  - Layers: {len(layers)}")
        print(f"  - Hidden size: {hidden_size}")
        print(f"  - Seq len: {seq_len}")
    except Exception as e:
        print(f"✗ Failed to load model:")
        print(f"  {type(e).__name__}: {e}")
        return
    
    print("\n[2/3] Tokenizing a test sentence...")
    test_text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(test_text, return_tensors='pt').to(device)
    print(f"✓ Tokenized: {inputs.input_ids.shape}")
    
    print("\n[3/3] Running forward pass...")
    with torch.no_grad():
        outputs = model(**inputs)
    print(f"✓ Forward pass completed!")
    print(f"  - Logits shape: {outputs.logits.shape}")
    
    print("\n" + "="*80)
    print("✅ TEST PASSED: LLaMA-2-13B loaded and working correctly!")
    print("="*80)

if __name__ == '__main__':
    main()
