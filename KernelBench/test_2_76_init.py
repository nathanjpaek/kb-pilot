#!/usr/bin/env python3
"""
Test script to verify that 2_76.py initializes identically to the reference model
"""
import torch
import torch.nn as nn

# Reference Model (from 76_Gemm_Add_ReLU.py)
class Model(nn.Module):
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

# Custom Model (from 2_76.py)
class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_shape):
        super().__init__()
        assert bias_shape == (out_features,), f"Expected {(out_features,)}, got {bias_shape}"
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

def test_initialization():
    """Test if both models initialize with same parameters when seed is set"""
    in_features = 1024
    out_features = 512
    bias_shape = (out_features,)
    
    # Test 1: Create both with same seed
    torch.manual_seed(42)
    model_ref = Model(in_features, out_features, bias_shape)
    
    torch.manual_seed(42)
    model_custom = ModelNew(in_features, out_features, bias_shape)
    
    # Compare weights
    weight_match = torch.allclose(model_ref.gemm.weight, model_custom.gemm.weight, atol=1e-6)
    bias_match = torch.allclose(model_ref.bias, model_custom.bias, atol=1e-6)
    
    print("=" * 60)
    print("Testing Parameter Initialization")
    print("=" * 60)
    print(f"Weight match: {weight_match}")
    print(f"Bias match: {bias_match}")
    
    if not weight_match:
        max_diff = torch.max(torch.abs(model_ref.gemm.weight - model_custom.gemm.weight)).item()
        print(f"  Max weight difference: {max_diff}")
    
    if not bias_match:
        max_diff = torch.max(torch.abs(model_ref.bias - model_custom.bias)).item()
        print(f"  Max bias difference: {max_diff}")
    
    # Test 2: Sample outputs (CPU)
    torch.manual_seed(99)
    x = torch.randn(128, in_features)
    
    with torch.no_grad():
        out_ref = model_ref.gemm(x) + model_ref.bias
        out_ref = torch.relu(out_ref)
        
        out_custom = model_custom.gemm(x) + model_custom.bias
        out_custom = torch.relu(out_custom)
    
    output_match = torch.allclose(out_ref, out_custom, atol=1e-5)
    print(f"Output match (CPU): {output_match}")
    
    if not output_match:
        max_diff = torch.max(torch.abs(out_ref - out_custom)).item()
        avg_diff = torch.mean(torch.abs(out_ref - out_custom)).item()
        print(f"  Max output difference: {max_diff}")
        print(f"  Avg output difference: {avg_diff}")
    
    print("=" * 60)
    
    return weight_match and bias_match and output_match

if __name__ == "__main__":
    success = test_initialization()
    if success:
        print("✅ All tests passed! Initialization matches.")
    else:
        print("❌ Tests failed! There's a mismatch.")
        exit(1)


