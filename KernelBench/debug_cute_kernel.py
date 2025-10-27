#!/usr/bin/env python3
"""
Debug CuTe kernel by comparing intermediate values with PyTorch reference
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# Copy the kernel from 2_76.py
@cute.kernel
def _linear_bias_relu_kernel(
    gA     : cute.Tensor,           # (B, K)
    gBTv   : cute.Tensor,           # ((1,V), (K, O/V))
    gBiasv : cute.Tensor,           # ((1,V), (O/V))
    gCv    : cute.Tensor,           # ((1,V), (B, O/V))
    K      : cutlass.Int32,
):
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi = bidy
    og = bidx * bdimx + tidx

    B  = gCv.shape[1][0]
    Og = gCv.shape[1][1]

    if bi < B and og < Og:
        c_out = gCv[(None, (bi, og))]

        b_frag       = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag    = cute.make_fragment_like(c_out, gA.element_type)
        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()

        # GEMM reduction
        for k in range(K):
            a_val_f32 = cutlass.Float32(gA[bi, k])
            b_vec_gmem = gBTv[(None, (k, og))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec_f32 = b_frag.load().to(cutlass.Float32)
            acc = acc + a_val_f32 * b_vec_f32

        # Add bias
        bias_vec_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_vec_gmem, bias_frag)
        bias_f32 = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # ReLU
        zero = cutlass.Float32(0.0)
        acc = cute.where(acc > zero, acc, zero)

        # Store
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


@cute.jit
def _linear_bias_relu_host(
    mA    : cute.Tensor,
    mBT   : cute.Tensor,
    mBias : cute.Tensor,
    mC    : cute.Tensor,
    V     : cutlass.Constexpr,
    K     : cutlass.Constexpr,
):
    B, O = mA.shape[0], mBT.shape[1]

    gBTv   = cute.zipped_divide(mBT,   (1, V))
    gBiasv = cute.zipped_divide(mBias, (V,))
    gCv    = cute.zipped_divide(mC,    (1, V))

    threads_per_block = 256
    O_groups = O // V
    grid_x = cute.ceil_div(O_groups, threads_per_block)
    grid_y = B

    _linear_bias_relu_kernel(
        mA, gBTv, gBiasv, gCv, cutlass.Int32(K)
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


def test_cute_kernel():
    """Test CuTe kernel with small, controlled inputs"""
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    print("="*80)
    print("Testing CuTe kernel with small inputs")
    print("="*80)
    
    # Small test case
    B, K, O = 4, 8, 16
    V = 8  # Vector width for fp16
    
    torch.manual_seed(42)
    
    # Create simple test data
    x = torch.randn(B, K, dtype=torch.float16, device='cuda')
    W = torch.randn(O, K, dtype=torch.float16, device='cuda')
    bias = torch.randn(O, dtype=torch.float16, device='cuda')
    
    # Reference computation
    with torch.no_grad():
        out_ref = torch.nn.functional.linear(x, W, bias)
        out_ref = torch.relu(out_ref)
    
    print(f"Input shape: {x.shape}, dtype: {x.dtype}")
    print(f"Weight shape: {W.shape}, dtype: {W.dtype}")
    print(f"Bias shape: {bias.shape}, dtype: {bias.dtype}")
    print(f"Reference output shape: {out_ref.shape}")
    print(f"Reference output sample: {out_ref[0, :4]}")
    
    # CuTe kernel computation
    y = torch.empty((B, O), dtype=torch.float16, device='cuda')
    WT = W.t().contiguous()
    
    mA    = from_dlpack(x,    assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    mBT   = from_dlpack(WT,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    mBias = from_dlpack(bias, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
    mC    = from_dlpack(y,    assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
    
    print(f"\nCuTe tensor shapes:")
    print(f"  mA.shape: {mA.shape}")
    print(f"  mBT.shape: {mBT.shape}")
    print(f"  mBias.shape: {mBias.shape}")
    print(f"  mC.shape: {mC.shape}")
    
    # Compile and run
    kernel = cute.compile(_linear_bias_relu_host, mA, mBT, mBias, mC, V, K)
    kernel(mA, mBT, mBias, mC)
    
    print(f"\nCuTe output shape: {y.shape}")
    print(f"CuTe output sample: {y[0, :4]}")
    
    # Compare
    diff = (out_ref - y).abs()
    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"Max difference: {diff.max().item():.6f}")
    print(f"Mean difference: {diff.mean().item():.6f}")
    print(f"Outputs match: {torch.allclose(out_ref, y, rtol=1e-2, atol=1e-2)}")
    
    if not torch.allclose(out_ref, y, rtol=1e-2, atol=1e-2):
        print(f"\n❌ KERNEL BUG DETECTED")
        print(f"\nFirst element comparison:")
        print(f"  Reference: {out_ref[0, 0].item():.6f}")
        print(f"  CuTe: {y[0, 0].item():.6f}")
        print(f"  Difference: {diff[0, 0].item():.6f}")
        
        # Check intermediate steps
        print(f"\nChecking intermediate computation manually:")
        gemm_result = torch.nn.functional.linear(x, W, None)  # No bias
        print(f"  After GEMM (no bias): {gemm_result[0, 0].item():.6f}")
        after_bias = gemm_result + bias
        print(f"  After bias: {after_bias[0, 0].item():.6f}")
        after_relu = torch.relu(after_bias)
        print(f"  After ReLU: {after_relu[0, 0].item():.6f}")
    else:
        print(f"\n✅ KERNEL WORKS!")

if __name__ == "__main__":
    test_cute_kernel()

