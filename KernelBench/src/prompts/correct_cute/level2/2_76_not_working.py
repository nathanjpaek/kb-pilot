"""
CuTe Python DSL implementation for Level 2 Problem 76: Gemm_Add_ReLU
Operation: Y = ReLU(X @ W^T + bias)
"""

import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _linear_bias_relu_kernel(
    gA:     cute.Tensor,   # (B, K)
    gBTv:   cute.Tensor,   # ((1,V), (K, O/V))
    gBiasv: cute.Tensor,   # ((V,), (O/V))
    gCv:    cute.Tensor,   # ((1,V), (B, O/V))
    K:      cutlass.Int32  # Pass K as Int32 parameter
):
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi = bidy
    og = bidx * bdimx + tidx

    B  = gCv.shape[1][0]
    Og = gCv.shape[1][1]

    if (bi < B) and (og < Og):
        c_out = gCv[(None, (bi, og))]

        b_frag    = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag = cute.make_fragment_like(c_out, gA.element_type)
        acc_frag  = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_frag.fill(0.0)

        # GEMM reduction - K is an Int32 parameter
        for k in range(K):
            a_val = cutlass.Float32(gA[bi, k])
            
            b_vec_gmem = gBTv[(None, (k, og))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec = b_frag.load()
            
            # Accumulate in FP32
            b_vec_f32 = b_vec.to(cutlass.Float32)
            acc_loaded = acc_frag.load()
            acc_loaded = acc_loaded + a_val * b_vec_f32
            acc_frag.store(acc_loaded)

        # Add bias
        bias_vec_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_vec_gmem, bias_frag)
        bias_vec = bias_frag.load()
        
        acc_loaded = acc_frag.load()
        acc_loaded = acc_loaded + bias_vec.to(cutlass.Float32)
        
        # ReLU
        zero = cutlass.Float32(0.0)
        acc_loaded = cute.where(acc_loaded > zero, acc_loaded, zero)
        
        # Store result
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc_loaded.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


@cute.jit
def _linear_bias_relu_host(
    mA:    cute.Tensor,
    mBT:   cute.Tensor,
    mBias: cute.Tensor,
    mC:    cute.Tensor,
    V:     cutlass.Constexpr,
    K:     cutlass.Constexpr
):
    B, O = mA.shape[0], mBT.shape[1]

    gBTv   = cute.zipped_divide(mBT,   (1, V))
    gBiasv = cute.zipped_divide(mBias, (V,))
    gCv    = cute.zipped_divide(mC,    (1, V))

    threads_per_block = 256
    O_groups = O // V
    
    grid_x = cute.ceil_div(O_groups, threads_per_block)
    grid_y = B

    # Pass K to the kernel wrapped in cutlass.Int32
    _linear_bias_relu_kernel(
        mA, gBTv, gBiasv, gCv, cutlass.Int32(K)
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


class ModelNew(nn.Module):
    """
    CuTe-accelerated: Y = ReLU(X @ W^T + bias)
    """
    def __init__(self, in_features: int, out_features: int, bias_shape):
        super().__init__()
        assert bias_shape == (out_features,), f"Expected bias_shape {(out_features,)}, got {bias_shape}"
        
        # Initialize exactly like reference Model
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        self._cache = {}

    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (O % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Input must be 2-D"
        
        # Ensure input is on CUDA and contiguous
        if not x.is_cuda:
            x = x.cuda()
        x = x.contiguous()
        
        B, K = x.shape
        O = self.gemm.out_features
        
        # Convert weights to match input dtype
        W = self.gemm.weight.detach().to(dtype=x.dtype, device='cuda').contiguous()
        b = self.bias.detach().to(dtype=x.dtype, device='cuda').contiguous()

        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, f"Vector width {V} must divide out_features {O}"

        # Allocate output
        y = torch.empty((B, O), dtype=x.dtype, device=x.device)
        
        # Transpose weight matrix
        WT = W.t().contiguous()

        # Create CuTe tensors from PyTorch tensors
        mA    = from_dlpack(x,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT   = from_dlpack(WT, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(b,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC    = from_dlpack(y,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype, V, K, O)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _linear_bias_relu_host,
                mA, mBT, mBias, mC,
                V,
                K
            )

        self._cache[key](mA, mBT, mBias, mC)
        
        return y


# Benchmark harness inputs
batch_size   = 128
in_features  = 1024
out_features = 512
bias_shape   = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]