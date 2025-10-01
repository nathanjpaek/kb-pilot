"""
Problem Name: 62_HardSwish_Matmul_Sum_LogSumExp_ResidualAdd_Hardtanh
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.165 runtime_stats={'mean': 0.165, 'std': 0.0138, 'min': 0.148, 'max': 0.206, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.164, 'std': 0.0153, 'min': 0.145, 'max': 0.212, 'num_trials': 100}, 'speedup_ratio': 0.994}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#               Kernel-factory :  Linear (bias)  +  HardSwish                 #
# --------------------------------------------------------------------------- #
def _build_linear_hswish_kernel(
    M: int,         # batch
    K: int,         # in_features
    N: int,         # hidden_size
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    six   = T.Cast(accum_dtype, 6.0)
    three = T.Cast(accum_dtype, 3.0)
    zero  = T.Cast(accum_dtype, 0.0)
    inv6  = T.Cast(accum_dtype, 1.0 / 6.0)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),          # (B, feature)
        W: T.Tensor((N, K), dtype),          # (hidden, feature)
        B: T.Tensor((N,),  dtype),           # (hidden,)
        Y: T.Tensor((M, N), dtype),          # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),           # grid.x
            T.ceildiv(M, block_M),           # grid.y
            threads=128,                     # 4 warps
        ) as (bx, by):
            A_s   = T.alloc_shared((block_M, block_K), dtype)
            B_s   = T.alloc_shared((block_K, block_N), dtype)
            Bias  = T.alloc_shared((block_N,),         dtype)
            Acc   = T.alloc_fragment((block_M, block_N), accum_dtype)

            # bias slice for this N-tile
            T.copy(B[bx * block_N : (bx + 1) * block_N], Bias)
            T.clear(Acc)

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
                T.gemm(A_s, B_s, Acc)

            # epilogue : +bias  &  HardSwish
            for mi, ni in T.Parallel(block_M, block_N):
                g_m = by * block_M + mi
                g_n = bx * block_N + ni
                if (g_m < M) and (g_n < N):
                    v = Acc[mi, ni] + Bias[ni].astype(accum_dtype)
                    t = v + three
                    t = T.min(six, T.max(zero, t))
                    h = v * t * inv6
                    Y[g_m, g_n] = T.Cast(dtype, h)

    return kernel


# --------------------------------------------------------------------------- #
#                          PyTorch  wrapper  module                           #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised version:
        kernel-1  :  Linear + HardSwish   (TileLang)
        host ops  :  row/col reductions, logsumexp, residual add, HardTanh
    """

    def __init__(self, feature_size: int, hidden_size: int):
        super().__init__()
        self.in_features  = int(feature_size)
        self.hidden_size  = int(hidden_size)

        # ---- parameters initialised identically to nn.Linear -------------
        self.weight = nn.Parameter(torch.empty(hidden_size, feature_size))
        self.bias   = nn.Parameter(torch.empty(hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(feature_size)
        nn.init.uniform_(self.bias, -bound, bound)

        # ---- kernel cache -------------------------------------------------
        self._k_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, batch: int, dtype: torch.dtype):
        key = (batch, dtype)
        if key not in self._k_cache:
            self._k_cache[key] = _build_linear_hswish_kernel(
                M=batch,
                K=self.in_features,
                N=self.hidden_size,
                dtype=str(dtype).split(".")[-1],
            )
        return self._k_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, residual: torch.Tensor):
        orig_dtype = x.dtype
        B = x.shape[0]

        # move to CUDA / fp16
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda",  dtype=torch.float16)
        res_f16 = residual.to(device="cuda", dtype=torch.float16)

        # ---- kernel  :  Linear + HardSwish ------------------------------
        ker = self._get_kernel(B, x_f16.dtype)
        y_f16 = ker(x_f16, w_f16, b_f16)          # (B, hidden)

        # ---- remaining lightweight ops ----------------------------------
        v = y_f16.sum(dim=0)                      # (hidden,)
        s = torch.matmul(y_f16, v)                # (B,)
        lse = torch.logsumexp(s.to(torch.float32), dim=0)  # scalar fp32
        res_sum = res_f16.sum().to(torch.float32)
        out = lse + res_sum
        out = torch.clamp(out, -3.0, 3.0)

        return out.to(orig_dtype)