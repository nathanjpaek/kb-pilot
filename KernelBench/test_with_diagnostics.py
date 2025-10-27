#!/usr/bin/env python3
"""
Test with detailed diagnostics to see WHERE the outputs differ
"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

import torch
import torch.nn as nn

def detailed_comparison_test():
    """Run test with detailed output comparison"""
    
    # Import reference and custom models
    ref_context = {}
    custom_context = {}
    
    # Load reference model (problem 76)
    with open('KernelBench/level2/76_Gemm_Add_ReLU.py', 'r') as f:
        ref_src = f.read()
    exec(ref_src, ref_context)
    
    # Load custom model  
    with open('src/prompts/correct_cute/level2/2_76.py', 'r') as f:
        custom_src = f.read()
    exec(custom_src, custom_context)
    
    Model = ref_context['Model']
    ModelNew = custom_context['ModelNew']
    get_init_inputs = ref_context['get_init_inputs']
    get_inputs = ref_context['get_inputs']
    
    # Create both models with same seed
    torch.manual_seed(42)
    init_inputs = get_init_inputs()
    model_ref = Model(*init_inputs)
    
    torch.manual_seed(42)  # Reset seed
    model_new = ModelNew(*init_inputs)
    
    # Move to CUDA if available (won't work on Mac but shows the pattern)
    if torch.cuda.is_available():
        model_ref = model_ref.cuda().half()
        model_new = model_new.cuda().half()
        
        # Get input
        torch.manual_seed(100)  # Different seed for input
        inputs = get_inputs()
        x = inputs[0].cuda().half()
        
        # Run both models
        with torch.no_grad():
            out_ref = model_ref(x)
            out_new = model_new(x)
        
        # Detailed comparison
        print("="*80)
        print("DETAILED OUTPUT COMPARISON")
        print("="*80)
        print(f"Reference output shape: {out_ref.shape}, dtype: {out_ref.dtype}")
        print(f"Custom output shape: {out_new.shape}, dtype: {out_new.dtype}")
        print(f"\nReference output stats:")
        print(f"  Min: {out_ref.min().item():.6f}")
        print(f"  Max: {out_ref.max().item():.6f}")
        print(f"  Mean: {out_ref.mean().item():.6f}")
        print(f"  Std: {out_ref.std().item():.6f}")
        print(f"\nCustom output stats:")
        print(f"  Min: {out_new.min().item():.6f}")
        print(f"  Max: {out_new.max().item():.6f}")
        print(f"  Mean: {out_new.mean().item():.6f}")
        print(f"  Std: {out_new.std().item():.6f}")
        
        # Difference analysis
        diff = (out_ref - out_new).abs()
        print(f"\nAbsolute difference:")
        print(f"  Max: {diff.max().item():.6f}")
        print(f"  Mean: {diff.mean().item():.6f}")
        print(f"  Median: {diff.median().item():.6f}")
        
        # Relative difference
        rel_diff = diff / (out_ref.abs() + 1e-8)
        print(f"\nRelative difference:")
        print(f"  Max: {rel_diff.max().item():.6f}")
        print(f"  Mean: {rel_diff.mean().item():.6f}")
        
        # Show first few values
        print(f"\nFirst 10 reference values: {out_ref.flatten()[:10]}")
        print(f"First 10 custom values:    {out_new.flatten()[:10]}")
        print(f"First 10 differences:      {diff.flatten()[:10]}")
        
        # Check weight matching
        print("\n" + "="*80)
        print("WEIGHT COMPARISON")
        print("="*80)
        
        # For nn.Linear structure
        if hasattr(model_ref, 'gemm') and hasattr(model_new, 'gemm'):
            w_diff = (model_ref.gemm.weight - model_new.gemm.weight).abs()
            print(f"Weight difference (max): {w_diff.max().item():.10f}")
            print(f"Weight difference (mean): {w_diff.mean().item():.10f}")
        elif hasattr(model_ref, 'linear') and hasattr(model_new, 'linear'):
            w_diff = (model_ref.linear.weight - model_new.linear.weight).abs()
            print(f"Weight difference (max): {w_diff.max().item():.10f}")
            print(f"Weight difference (mean): {w_diff.mean().item():.10f}")
        
        # Check if outputs match within tolerance
        if torch.allclose(out_ref, out_new, rtol=1e-2, atol=1e-2):
            print("\n✅ PASS: Outputs match within tolerance!")
        else:
            print("\n❌ FAIL: Outputs don't match!")
            
            # Show WHERE they differ most
            diff_flat = diff.flatten()
            max_diff_idx = diff_flat.argmax()
            print(f"\nLargest difference at index {max_diff_idx}:")
            print(f"  Reference: {out_ref.flatten()[max_diff_idx].item():.6f}")
            print(f"  Custom: {out_new.flatten()[max_diff_idx].item():.6f}")
            print(f"  Difference: {diff_flat[max_diff_idx].item():.6f}")
    else:
        print("❌ CUDA not available - can't run test")
        print("This diagnostic script needs to run on Modal or a CUDA machine")

if __name__ == "__main__":
    detailed_comparison_test()

