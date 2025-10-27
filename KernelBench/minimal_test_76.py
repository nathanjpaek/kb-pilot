"""
Minimal test to compare reference vs CuTe kernel
"""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')

import torch
import torch.nn as nn

# Load both implementations
ref_code = open('KernelBench/level2/76_Gemm_Add_ReLU.py').read()
ref_ctx = {}
exec(ref_code, ref_ctx)
Model = ref_ctx['Model']

cute_code = open('src/prompts/correct_cute/level2/2_76.py').read()
cute_ctx = {}
exec(cute_code, cute_ctx)
ModelNew = cute_ctx['ModelNew']

# Create both models with same seed
torch.manual_seed(42)
model_ref = Model(1024, 512, (512,))

torch.manual_seed(42)
model_new = ModelNew(1024, 512, (512,))

print("Reference model structure:")
print(f"  gemm.weight: {model_ref.gemm.weight.shape}, {model_ref.gemm.weight.dtype}")
print(f"  bias: {model_ref.bias.shape}, {model_ref.bias.dtype}")

print("\nCustom model structure:")
print(f"  gemm.weight: {model_new.gemm.weight.shape}, {model_new.gemm.weight.dtype}")  
print(f"  bias: {model_new.bias.shape}, {model_new.bias.dtype}")

# Check if weights match
w_match = torch.allclose(model_ref.gemm.weight, model_new.gemm.weight, rtol=1e-5, atol=1e-5)
b_match = torch.allclose(model_ref.bias, model_new.bias, rtol=1e-5, atol=1e-5)
print(f"\nWeights match: {w_match}")
print(f"Bias match: {b_match}")

if torch.cuda.is_available():
    print("\n" + "="*60)
    print("Testing on CUDA...")
    
    model_ref = model_ref.cuda().half()
    model_new = model_new.cuda().half()
    
    # Test input
    torch.manual_seed(100)
    x = torch.randn(128, 1024).cuda().half()
    
    with torch.no_grad():
        out_ref = model_ref(x)
        out_new = model_new(x)
    
    print(f"Reference output: shape={out_ref.shape}, dtype={out_ref.dtype}")
    print(f"Custom output: shape={out_new.shape}, dtype={out_new.dtype}")
    print(f"Outputs match: {torch.allclose(out_ref, out_new, rtol=1e-2, atol=1e-2)}")
    
    if not torch.allclose(out_ref, out_new, rtol=1e-2, atol=1e-2):
        diff = (out_ref - out_new).abs()
        print(f"Max diff: {diff.max().item()}")
        print(f"Mean diff: {diff.mean().item()}")
else:
    print("\nCUDA not available, skipping GPU test")

