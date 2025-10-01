"""
Problem Name: 67_Linear_Matmul_Sum_LogSumExp_HardSwish_ResidualAdd_Hardtanh
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0884 runtime_stats={'mean': 0.0884, 'std': 0.00922, 'min': 0.0783, 'max': 0.127, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.137, 'std': 0.013, 'min': 0.123, 'max': 0.178, 'num_trials': 100}, 'speedup_ratio': 1.55}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T

# ------------------------------------------------------------------ #
#  Kernel-1 :  fp16 Linear  (X @ Wᵀ + B)                             #
# ------------------------------------------------------------------ #
def _build_linear_kernel(
    M: int,                # batch
    K: int,                # feature_size
    N: int,                # hidden_size
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),          # (B, F)
        W: T.Tensor((N, K), dtype),          # (H, F)
        B: T.Tensor((N,),   dtype),          # (H,)
        Out: T.Tensor((M, N), dtype),        # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),           # grid.x
            T.ceildiv(M, block_M),           # grid.y
            threads=128,                     # 4 warps
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_N, block_K), dtype)   # note: (Ntile, Ktile)
            C   = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                # load X-tile
                T.copy(
                    X[by * block_M : (by + 1) * block_M,
                      ko * block_K : (ko + 1) * block_K],
                    A_s,
                )
                # load W-tile (no transpose, we‘ll transpose in GEMM)
                T.copy(
                    W[bx * block_N : (bx + 1) * block_N,
                      ko * block_K : (ko + 1) * block_K],
                    B_s,
                )
                # GEMM : A_s (M,K)  ·  B_sᵀ (K,N)
                T.gemm(A_s, B_s, C, transpose_B=True)

            # epilogue – add bias, store
            for mi, ni in T.Parallel(block_M, block_N):
                g_m = by * block_M + mi
                g_n = bx * block_N + ni
                if (g_m < M) and (g_n < N):
                    v = C[mi, ni] + B[g_n].astype(accum_dtype)
                    Out[g_m, g_n] = T.Cast(dtype, v)

    return kernel


# ------------------------------------------------------------------ #
#  Kernel-2 :  hardswish( X · vec )   →  (B,)                        #
# ------------------------------------------------------------------ #
def _build_dot_hswish_kernel(
    M: int,
    K: int,
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
        X:   T.Tensor((M, K), dtype),        # (B, F)
        V:   T.Tensor((K,),   dtype),        # (F,)
        Out: T.Tensor((M,),   dtype),        # auto-allocated  (B,)
    ):
        with T.Kernel(M, threads=1) as b:    # one thread per row (F=128 → tiny)
            acc = T.alloc_local((1,), accum_dtype)
            T.clear(acc)
            for k in T.serial(K):
                acc[0] += X[b, k].astype(accum_dtype) * V[k].astype(accum_dtype)

            v = acc[0]
            t = v + three
            t = T.min(six, T.max(zero, t))
            hs = v * t * inv6
            Out[b] = T.Cast(dtype, hs)

    return kernel


# ------------------------------------------------------------------ #
#  Kernel-3 :  add  +  hardtanh[-3,3]                                #
# ------------------------------------------------------------------ #
def _build_add_clamp_kernel(
    M: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    lo = T.Cast(accum_dtype, -3.0)
    hi = T.Cast(accum_dtype,  3.0)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A:   T.Tensor((M, N), dtype),        # residual   (B,H)
        v:   T.Tensor((M,),   dtype),        # scalar per row (B,)
        Out: T.Tensor((M, N), dtype),        # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            for mi, ni in T.Parallel(block_M, block_N):
                g_m = by * block_M + mi
                g_n = bx * block_N + ni
                if (g_m < M) and (g_n < N):
                    val = (
                        A[g_m, g_n].astype(accum_dtype)
                        + v[g_m].astype(accum_dtype)
                    )
                    val = T.min(hi, T.max(lo, val))
                    Out[g_m, g_n] = T.Cast(dtype, val)

    return kernel


# ------------------------------------------------------------------ #
#                     PyTorch  wrapper  module                       #
# ------------------------------------------------------------------ #
class ModelNew(nn.Module):
    """
    TileLang implementation of:
        residual = Linear(x)
        v        = hardswish( (x·sum(y)) )
        out      = hardtanh( residual + v )
    """

    def __init__(self, feature_size: int, hidden_size: int):
        super().__init__()

        # ---- identical parameter initialisation -----------------------
        self.weight = nn.Parameter(torch.empty(hidden_size, feature_size))
        self.bias   = nn.Parameter(torch.empty(hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(feature_size)
        nn.init.uniform_(self.bias, -bound, bound)

        self.feature_size = int(feature_size)
        self.hidden_size  = int(hidden_size)

        # kernel caches
        self._k_lin : Dict[Tuple[int, torch.dtype], callable] = {}
        self._k_vec : Dict[Tuple[int, torch.dtype], callable] = {}
        self._k_add : Dict[Tuple[int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernels(self, B: int, dtype: torch.dtype):
        key = (B, dtype)

        if key not in self._k_lin:
            self._k_lin[key] = _build_linear_kernel(
                M=B,
                K=self.feature_size,
                N=self.hidden_size,
                dtype=str(dtype).split(".")[-1],
            )
        if key not in self._k_vec:
            self._k_vec[key] = _build_dot_hswish_kernel(
                M=B,
                K=self.feature_size,
                dtype=str(dtype).split(".")[-1],
            )
        if key not in self._k_add:
            self._k_add[key] = _build_add_clamp_kernel(
                M=B,
                N=self.hidden_size,
                dtype=str(dtype).split(".")[-1],
            )
        return self._k_lin[key], self._k_vec[key], self._k_add[key]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        B = x.shape[0]

        # --- move / cast to GPU-fp16 ------------------------------------
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        y_f16 = y.to(device="cuda", dtype=torch.float16)
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda",  dtype=torch.float16)

        # --- compile / fetch kernels ------------------------------------
        k_lin, k_vec, k_add = self._get_kernels(B, x_f16.dtype)

        # --- 1) Linear ---------------------------------------------------
        residual_fp16 = k_lin(x_f16, w_f16, b_f16)          # (B,H)

        # --- 2) vec = hardswish( X · Σy ) -------------------------------
        vec_f16 = y_f16.sum(dim=0)                          # (F,)
        vec_f16 = k_vec(x_f16, vec_f16.contiguous())        # (B,)

        # --- 3) add + clamp[-3,3] ---------------------------------------
        out_fp16 = k_add(residual_fp16, vec_f16)            # (B,H)

        return out_fp16.to(orig_dtype)