"""
Problem Name: 12_Gemm_Multiply_LeakyReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0422 runtime_stats={'mean': 0.0422, 'std': 0.00107, 'min': 0.0406, 'max': 0.0484, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0539, 'std': 0.00247, 'min': 0.0518, 'max': 0.0673, 'num_trials': 100}, 'speedup_ratio': 1.28}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_gemm_mul_leaky_kernel(
    M: int,
    N: int,
    K: int,
    multiplier: float,
    negative_slope: float,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    mul_const = float(multiplier)
    neg_slope_const = float(negative_slope)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),      # row-major [N, K]
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),    # created by TileLang
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            # Shared memory tiles
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)

            # Accumulator
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            # Reduction over K
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], X_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Bias + multiply + LeakyReLU + store
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    v = C_loc[i, j] + T.Cast(accum_dtype, B[gn])
                    v = v * mul_const
                    v_neg = v * neg_slope_const
                    v = T.max(v, v_neg)          # LeakyReLU
                    Out[gm, gn] = T.Cast(dtype, v)

    return kernel


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 multiplier: float, negative_slope: float):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)

        # Parameters identical to nn.Linear defaults
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}

    # -------- kernel cache -------- #
    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_gemm_mul_leaky_kernel(
                M=batch_size,
                N=self.out_features,
                K=self.in_features,
                multiplier=self.multiplier,
                negative_slope=self.negative_slope,
            )
        return self._kernel_cache[key]

    # -------- forward -------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16)
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel(x_f16.shape[0], x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16)

        return y_f16.to(orig_dtype)