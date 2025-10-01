"""
Problem Name: 39_Matmul_Hardtanh_HardSwish_LogSumExp_Sum_ResidualAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.163 runtime_stats={'mean': 0.163, 'std': 0.0274, 'min': 0.141, 'max': 0.388, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.251, 'std': 0.0699, 'min': 0.166, 'max': 0.846, 'num_trials': 100}, 'speedup_ratio': 1.54}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T

# --------------------------------------------------------------------------- #
# Kernel-1 :  GEMM  + HardTanh[-0.5,0.5] + HardSwish                        #
# --------------------------------------------------------------------------- #

def _build_gemm_tanh_hswish_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Y = hard_swish(hard_tanh(X @ Wᵀ))"""

    half   = T.Cast(accum_dtype, 0.5)
    nhalf  = T.Cast(accum_dtype, -0.5)
    zero   = T.Cast(accum_dtype, 0.0)
    three  = T.Cast(accum_dtype, 3.0)
    six    = T.Cast(accum_dtype, 6.0)
    inv6   = T.Cast(accum_dtype, 1.0 / 6.0)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),            # (B, feature_dim)
        W: T.Tensor((N, K), dtype),            # (hidden_dim, feature_dim)
        Y: T.Tensor((M, N), dtype),            # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),             # grid.x
            T.ceildiv(M, block_M),             # grid.y
            threads=128,                       # 4 warps
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C   = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C)
            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # load tiles
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
                # GEMM
                T.gemm(A_s, B_s, C)

            # epilogue – HardTanh then HardSwish
            for mi, ni in T.Parallel(block_M, block_N):
                g_m = by * block_M + mi
                g_n = bx * block_N + ni
                if (g_m < M) and (g_n < N):
                    v = C[mi, ni]
                    # HardTanh
                    v = T.max(nhalf, v)
                    v = T.min(half,  v)
                    # HardSwish  : v * relu6(v+3) / 6
                    t = v + three
                    t = T.max(zero, T.min(six, t))
                    h = v * t * inv6
                    Y[g_m, g_n] = T.Cast(dtype, h)

    return kernel

# --------------------------------------------------------------------------- #
# Kernel-2 :  row-wise LogSumExp  + broadcast residual add                    #
# --------------------------------------------------------------------------- #

def _build_lse_add_kernel(
    M: int,
    N: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        Y:   T.Tensor((M, N), dtype),      # activations from kernel-1
        Res: T.Tensor((M, N), dtype),      # residual to add
        Out: T.Tensor((M, N), dtype),      # auto-allocated
    ):
        with T.Kernel(M, threads=1) as bx:   # one thread per row (simple)
            max_val = T.alloc_local((1,), accum_dtype)
            max_val[0] = Y[bx, 0].astype(accum_dtype)
            for j in range(1, N):
                v = Y[bx, j].astype(accum_dtype)
                if v > max_val[0]:
                    max_val[0] = v

            sum_exp = T.alloc_local((1,), accum_dtype)
            sum_exp[0] = T.Cast(accum_dtype, 0.0)
            for j in range(N):
                sum_exp[0] += T.exp(Y[bx, j].astype(accum_dtype) - max_val[0])

            lse = max_val[0] + T.log(sum_exp[0])
            lse_fp16 = T.Cast(dtype, lse)

            for j in range(N):
                Out[bx, j] = Res[bx, j] + lse_fp16

    return kernel

# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """Optimised TileLang replacement of original model."""

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim  = int(hidden_dim)

        # weight initialised identical to nn.Linear (bias unused)
        self.weight = nn.Parameter(torch.empty(hidden_dim, feature_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # kernel caches
        self._k_lin_cache: Dict[Tuple[int, torch.dtype], callable] = {}
        self._k_lse_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # ---------------- kernel helpers ----------------
    def _get_lin_kernel(self, B: int, dtype: torch.dtype):
        key = (B, dtype)
        if key not in self._k_lin_cache:
            self._k_lin_cache[key] = _build_gemm_tanh_hswish_kernel(
                M=B,
                K=self.feature_dim,
                N=self.hidden_dim,
                dtype=str(dtype).split(".")[-1],
            )
        return self._k_lin_cache[key]

    def _get_lse_kernel(self, B: int, dtype: torch.dtype):
        key = (B, dtype)
        if key not in self._k_lse_cache:
            self._k_lse_cache[key] = _build_lse_add_kernel(
                M=B,
                N=self.hidden_dim,
                dtype=str(dtype).split(".")[-1],
            )
        return self._k_lse_cache[key]

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B = x.shape[0]

        x_fp16  = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16  = self.weight.to(device="cuda", dtype=torch.float16)
        res_fp16 = residual.to(device="cuda", dtype=torch.float16).contiguous()

        # kernel-1 : GEMM + HardTanh + HardSwish
        k1 = self._get_lin_kernel(B, x_fp16.dtype)
        act_fp16 = k1(x_fp16, w_fp16)           # (B, hidden_dim)

        # kernel-2 : LogSumExp row-wise + residual add
        k2 = self._get_lse_kernel(B, x_fp16.dtype)
        out_fp16 = k2(act_fp16, res_fp16)       # (B, hidden_dim)

        return out_fp16.to(orig_dtype)