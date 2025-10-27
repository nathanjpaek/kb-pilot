#!/usr/bin/env python3
"""
Local test for problem 2_9 to debug correctness issues
"""
import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn

# Import reference model
from KernelBench.level2 import _9_Matmul_Subtract_Multiply_ReLU as ref_module

# Import our implementation
from prompts.correct_cute.level2 import _2_9 as our_module

def test_correctness():
    print("Testing 2_9 correctness...")
    
    batch_size = 128
    in_features = 1024
    out_features = 512
    subtract_value = 2.0
    multiply_value = 1.5
    
    # Create both models
    print("Creating models...")
    model_ref = ref_module.Model(in_features, out_features, subtract_value, multiply_value).cuda()
    model_new = our_module.ModelNew(in_features, out_features, subtract_value, multiply_value).cuda()
    
    # Use float16 for both
    model_ref = model_ref.half()
    model_new = model_new.half()
    
    # Copy weights from reference to our model
    print("Copying weights...")
    model_new.linear.weight.data.copy_(model_ref.linear.weight.data)
    model_new.linear.bias.data.copy_(model_ref.linear.bias.data)
    
    # Test input
    print("Creating test input...")
    x = torch.randn(batch_size, in_features, dtype=torch.float16, device='cuda')
    
    # Run both
    print("Running reference model...")
    with torch.no_grad():
        y_ref = model_ref(x)
    
    print("Running CuTe model...")
    with torch.no_grad():
        y_new = model_new(x)
    
    # Compare
    print(f"\nReference output shape: {y_ref.shape}, dtype: {y_ref.dtype}")
    print(f"CuTe output shape: {y_new.shape}, dtype: {y_new.dtype}")
    print(f"Reference output sample: {y_ref[0, :5]}")
    print(f"CuTe output sample: {y_new[0, :5]}")
    
    diff = (y_ref - y_new).abs()
    print(f"\nMax absolute difference: {diff.max().item()}")
    print(f"Mean absolute difference: {diff.mean().item()}")
    print(f"Relative error: {(diff / (y_ref.abs() + 1e-5)).mean().item()}")
    
    # Check if close enough
    if torch.allclose(y_ref, y_new, rtol=1e-2, atol=1e-2):
        print("\n✅ PASS: Outputs match!")
        return True
    else:
        print("\n❌ FAIL: Outputs don't match!")
        return False

if __name__ == "__main__":
    test_correctness()

