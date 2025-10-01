"""
Problem Name: 80_Gemm_Max_Subtract_GELU
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.121 runtime_stats={'mean': 0.121, 'std': 0.0468, 'min': 0.0956, 'max': 0.483, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.082, 'std': 0.00859, 'min': 0.0752, 'max': 0.117, 'num_trials': 100}, 'speedup_ratio': 0.678}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ------------------------------------------------------------
# Kernel factory
# ------------------------------------------------------------
def _build_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        X: T.Tensor((M, K), dtype),          # inputs
        W: T.Tensor((N, K), dtype),          # weights (row-major)
        B: T.Tensor((N,), dtype),            # bias
        Y: T.Tensor((M, N), dtype),          # output created by TileLang
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            # Shared and fragment buffers
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)
            B_s = T.alloc_shared((block_N,), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Load bias slice once per block
            T.copy(
                B[bx * block_N:(bx + 1) * block_N],
                B_s,
            )

            # Clear accumulators
            T.clear(C_loc)

            # Main reduction loop over K
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(
                    X[by * block_M:(by + 1) * block_M,
                      ko * block_K:(ko + 1) * block_K],
                    X_s,
                )
                T.copy(
                    W[bx * block_N:(bx + 1) * block_N,
                      ko * block_K:(ko + 1) * block_K],
                    W_s,
                )
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Epilogue: add bias and store
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = C_loc[i, j] + T.Cast(accum_dtype, B_s[j])
                    Y[gm, gn] = T.Cast(dtype, val)

    return linear_kernel


# ------------------------------------------------------------
# PyTorch wrapper module
# ------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimised version of the original model using a TileLang kernel
    for the Linear (GEMM + bias) stage.
    """

    def __init__(self, in_features: int, out_features: int, max_dim: int):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.max_dim = int(max_dim)  # 0 or 1

        # ---- Parameters (identical initialisation to nn.Linear) ----
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---- Kernel cache ----
        self._kernel_cache: Dict[Tuple[int, torch.dtype], tilelang.PrimFunc] = {}

    # --------------------------------------------------------
    # Kernel retrieval / compilation
    # --------------------------------------------------------
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kern = _build_linear_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
            )
            self._kernel_cache[key] = kern
        return self._kernel_cache[key]

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move to CUDA / fp16 for kernel
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        batch_size = x_f16.shape[0]
        kernel = self._get_kernel(batch_size, x_f16.dtype)

        # GEMM (returns fp16 tensor)
        y_f16 = kernel(x_f16, w_f16, b_f16)

        # Remaining ops in PyTorch (cast to fp32 for better accuracy)
        y = y_f16.to(torch.float32)

        # Max over specified dim with keepdim=True
        y = torch.max(y, dim=self.max_dim, keepdim=True).values

        # Subtract mean across dim=1 (keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)

        # GELU activation
        y = 0.5 * y * (1.0 + torch.erf(y / math.sqrt(2.0)))

        return y.to(orig_dtype)