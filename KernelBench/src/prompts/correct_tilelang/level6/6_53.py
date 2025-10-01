"""
Problem Name: 53_Linear_HardSwish_Matmul_Sum_LogSumExp_ResidualAdd_Hardtanh
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.14 runtime_stats={'mean': 0.14, 'std': 0.00404, 'min': 0.135, 'max': 0.162, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.148, 'std': 0.0407, 'min': 0.138, 'max': 0.547, 'num_trials': 100}, 'speedup_ratio': 1.06}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel 1 :  Linear (with bias) + HardSwish                                 #
# --------------------------------------------------------------------------- #
def _build_linear_hswish_kernel(
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
    """
    Y = hardswish( X @ Wᵀ + B )
    """

    six  = T.Cast(accum_dtype, 6.0)
    three= T.Cast(accum_dtype, 3.0)
    zero = T.Cast(accum_dtype, 0.0)
    inv6 = T.Cast(accum_dtype, 1.0 / 6.0)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,),  dtype),
        Out: T.Tensor((M, N), dtype),        # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),           # grid.x
            T.ceildiv(M, block_M),           # grid.y
            threads=128,                     # 4 warps
        ) as (bx, by):
            A_s   = T.alloc_shared((block_M, block_K), dtype)
            B_s   = T.alloc_shared((block_K, block_N), dtype)
            Bias_s= T.alloc_shared((block_N,),         dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # bias slice
            T.copy(B[bx * block_N : (bx + 1) * block_N], Bias_s)
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(
                    X[ by * block_M : (by + 1) * block_M,
                       ko * block_K : (ko + 1) * block_K],
                    A_s,
                )
                T.copy(
                    W[ ko * block_K : (ko + 1) * block_K,
                       bx * block_N : (bx + 1) * block_N],
                    B_s,
                )
                T.gemm(A_s, B_s, C_loc)      # (block_M, block_K) * (block_K, block_N)

            # epilogue: +bias and HardSwish
            for mi, ni in T.Parallel(block_M, block_N):
                g_m = by * block_M + mi
                g_n = bx * block_N + ni
                if (g_m < M) and (g_n < N):
                    v = C_loc[mi, ni] + Bias_s[ni].astype(accum_dtype)
                    t = v + three
                    t = T.min(six, T.max(zero, t))
                    h = v * t * inv6
                    Out[g_m, g_n] = T.Cast(dtype, h)

    return kernel


# --------------------------------------------------------------------------- #
# Kernel 2 : GEMM (hidden_dim → out_features)                                 #
# --------------------------------------------------------------------------- #
def _build_gemm_kernel(
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
    """
    Y = X @ W   ,  X:(M,K) , W:(K,N)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((K, N), dtype),
        Out: T.Tensor((M, N), dtype),        # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C   = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C)
            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(
                    X[ by * block_M : (by + 1) * block_M,
                       ko * block_K : (ko + 1) * block_K ],
                    A_s,
                )
                T.copy(
                    W[ ko * block_K : (ko + 1) * block_K,
                       bx * block_N : (bx + 1) * block_N ],
                    B_s,
                )
                T.gemm(A_s, B_s, C)          # no transpose

            for mi, ni in T.Parallel(block_M, block_N):
                g_m = by * block_M + mi
                g_n = bx * block_N + ni
                if (g_m < M) and (g_n < N):
                    Out[g_m, g_n] = T.Cast(dtype, C[mi, ni])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Optimised implementation replacing the two matrix multiplications with
    TileLang kernels. Remaining light-weight ops are kept in PyTorch.
    """

    def __init__(self, in_features: int, hidden_dim: int, out_features: int):
        super().__init__()

        # ---- first Linear ( identical initialisation ) -------------------
        self.weight1 = nn.Parameter(torch.empty(hidden_dim, in_features))
        self.bias1   = nn.Parameter(torch.empty(hidden_dim))
        nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.bias1, -bound, bound)

        # ---- second weight  ( matches torch.randn of original ) ----------
        self.weight = nn.Parameter(torch.randn(hidden_dim, out_features))

        # ---- kernel caches ----------------------------------------------
        self._k1_cache: Dict[Tuple[int, torch.dtype], callable] = {}
        self._k2_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_k1(self, B: int, dtype: torch.dtype):
        key = (B, dtype)
        if key not in self._k1_cache:
            self._k1_cache[key] = _build_linear_hswish_kernel(
                M=B,
                K=self.weight1.shape[1],
                N=self.weight1.shape[0],
            )
        return self._k1_cache[key]

    def _get_k2(self, B: int, dtype: torch.dtype):
        key = (B, dtype)
        if key not in self._k2_cache:
            self._k2_cache[key] = _build_gemm_kernel(
                M=B,
                K=self.weight.shape[0],
                N=self.weight.shape[1],
            )
        return self._k2_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B = x.shape[0]

        # ---- prepare tensors (fp16 on CUDA) -----------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w1_fp16 = self.weight1.to(device="cuda", dtype=torch.float16)
        b1_fp16 = self.bias1.to(device="cuda",  dtype=torch.float16)
        w2_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        res_fp16 = residual.to(device="cuda", dtype=torch.float16)

        # ---- kernel 1 : Linear + HardSwish ------------------------------
        k1 = self._get_k1(B, x_fp16.dtype)
        act_fp16 = k1(x_fp16, w1_fp16, b1_fp16)       # (B, hidden_dim)

        # ---- kernel 2 : hidden_dim → out_features -----------------------
        k2 = self._get_k2(B, x_fp16.dtype)
        y_fp16 = k2(act_fp16, w2_fp16)                # (B, out_features)

        # ---- remaining lightweight ops ----------------------------------
        # sum → logsumexp (same value because shape=(B,1)) ----------------
        s = y_fp16.sum(dim=1, keepdim=True)
        lse = torch.logsumexp(s, dim=1, keepdim=True)
        out = lse + res_fp16                          # broadcast add
        out = torch.clamp(out, -0.5, 0.5)             # HardTanh

        return out.to(orig_dtype)