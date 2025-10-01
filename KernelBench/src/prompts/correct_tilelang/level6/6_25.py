"""
Problem Name: 25_Matmul_LeakyReLU_Add_Mish_HardSwish
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.349 runtime_stats={'mean': 0.349, 'std': 0.00663, 'min': 0.338, 'max': 0.378, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.361, 'std': 0.0057, 'min': 0.353, 'max': 0.38, 'num_trials': 100}, 'speedup_ratio': 1.03}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------- #
#                    TileLang kernel factory (fused)                    #
# --------------------------------------------------------------------- #
def _build_fused_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Out = HardSwish( Mish( LeakyReLU( X @ W ) + Y ) )
    Shapes:
        X : (M, K)
        W : (K, N)
        Y : (M, N)
        Out : (M, N)
    All constants are baked in for maximal performance.
    """

    neg_slope = T.Cast(accum_dtype, 0.01)
    zero      = T.Cast(accum_dtype, 0.0)
    one       = T.Cast(accum_dtype, 1.0)
    three     = T.Cast(accum_dtype, 3.0)
    six       = T.Cast(accum_dtype, 6.0)
    inv6      = T.Cast(accum_dtype, 1.0 / 6.0)
    half      = T.Cast(accum_dtype, 0.5)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((M, K), in_dtype),
        W: T.Tensor((K, N), in_dtype),
        Y: T.Tensor((M, N), in_dtype),
        Out: T.Tensor((M, N), in_dtype),          # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),                # grid.x
            T.ceildiv(M, block_M),                # grid.y
            threads=128,                          # 4 warps
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_K, block_N), in_dtype)
            C   = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(
                    X[by * block_M : (by + 1) * block_M,
                      ko * block_K : (ko + 1) * block_K],
                    A_s,
                )
                T.copy(
                    W[ko * block_K : (ko + 1) * block_K,
                      bx * block_N : (bx + 1) * block_N],
                    B_s,
                )
                T.gemm(A_s, B_s, C)               # no-transpose GEMM

            # ---------------- epilogue ---------------- #
            for mi, ni in T.Parallel(block_M, block_N):
                g_m = by * block_M + mi
                g_n = bx * block_N + ni
                if (g_m < M) and (g_n < N):
                    val = C[mi, ni]                       # fp32 accumulator
                    # LeakyReLU
                    pos = T.max(val, zero)
                    neg = T.min(val, zero) * neg_slope
                    val = pos + neg
                    # residual add
                    val += T.Cast(accum_dtype, Y[g_m, g_n])
                    # Mish: x * tanh(softplus(x))
                    sp   = T.log(one + T.exp(val))
                    val  = val * T.tanh(sp)
                    # HardSwish: x * relu6(x+3)/6 + 0.5
                    relu6 = T.min(T.max(val + three, zero), six)
                    val   = val * relu6 * inv6 + half
                    Out[g_m, g_n] = T.Cast(in_dtype, val)

    return fused


# --------------------------------------------------------------------- #
#                           PyTorch wrapper                             #
# --------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Matmul → LeakyReLU → add(y) → Mish → HardSwish ( +0.5 ) fused in TileLang.
    """

    def __init__(self, dim1: int, dim2: int, dim3: int):
        super().__init__()
        self.dim1 = int(dim1)
        self.dim2 = int(dim2)
        self.dim3 = int(dim3)

        # weight initialised exactly like reference (torch.randn)
        self.weight = nn.Parameter(torch.randn(dim2, dim3))

        # kernel cache  :  key = (M, dtype)
        self._kern_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # ------------------------------------------------------------- #
    def _get_kernel(self, M: int, dtype: torch.dtype):
        key = (M, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                M=M,
                K=self.dim2,
                N=self.dim3,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B = x.shape[0]

        # -------- prepare / cast tensors -------- #
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        y_f16 = y.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)

        # flatten (B, dim1, *) → (M, *)
        M = B * self.dim1
        x_2d = x_f16.view(M, self.dim2)
        y_2d = y_f16.view(M, self.dim3)

        # -------- kernel -------- #
        kernel = self._get_kernel(M, x_f16.dtype)
        out_2d = kernel(x_2d, w_f16, y_2d)          # float16

        # reshape back
        out = out_2d.view(B, self.dim1, self.dim3)
        return out.to(orig_dtype)