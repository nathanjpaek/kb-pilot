"""
Problem Name: 64_Gemm_LogSumExp_LeakyReLU_LeakyReLU_GELU_GELU
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.252 runtime_stats={'mean': 0.252, 'std': 0.0468, 'min': 0.214, 'max': 0.642, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.184, 'std': 0.0312, 'min': 0.157, 'max': 0.467, 'num_trials': 100}, 'speedup_ratio': 0.73}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 64,
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
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            # Shared / fragment allocations
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)
            Y_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            B_s = T.alloc_shared((block_N,), dtype)

            # Layout optimizations (optional swizzle to improve bank conflicts)
            T.annotate_layout({X_s: tilelang.layout.make_swizzled_layout(X_s),
                               W_s: tilelang.layout.make_swizzled_layout(W_s)})

            # Copy bias slice once per block
            T.copy(B[bx * block_N:(bx + 1) * block_N], B_s)

            # Clear local accumulator
            T.clear(Y_loc)

            # Pipeline reduction loop across K dimension
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Load tiles from global memory
                T.copy(X[by * block_M:(by + 1) * block_M,
                         ko * block_K:(ko + 1) * block_K], X_s)
                T.copy(W[bx * block_N:(bx + 1) * block_N,
                         ko * block_K:(ko + 1) * block_K], W_s)

                # GEMM: X_s (M x K) @ W_s^T (K x N)  -> Y_loc
                T.gemm(X_s, W_s, Y_loc, transpose_B=True)

            # Add bias and store results
            for i, j in T.Parallel(block_M, block_N):
                Y_loc[i, j] += B_s[j]

            # Write back to global memory
            T.copy(Y_loc, Y[by * block_M:(by + 1) * block_M,
                            bx * block_N:(bx + 1) * block_N])

    return linear_kernel


class ModelNew(nn.Module):
    """
    Optimized version of the provided model using TileLang for the Linear layer.
    Subsequent operations (LogSumExp, LeakyReLU x2, GELU x2) are performed with
    standard PyTorch tensor semantics (no torch.nn.functional calls).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Parameter initialization identical to torch.nn.Linear defaults
        weight_fp32 = torch.empty(out_features, in_features, dtype=torch.float32)
        nn.init.kaiming_uniform_(weight_fp32, a=math.sqrt(5))
        self.weight = nn.Parameter(weight_fp32.to(torch.float16))

        if bias:
            bound = 1 / math.sqrt(in_features)
            bias_fp32 = torch.empty(out_features, dtype=torch.float32)
            nn.init.uniform_(bias_fp32, -bound, bound)
            self.bias = nn.Parameter(bias_fp32.to(torch.float16))
        else:
            self.register_parameter("bias", None)

        # Kernel cache keyed by (batch_size, dtype)
        self._kernel_cache: Dict[Tuple[int, torch.dtype], tilelang.PrimFunc] = {}

    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kernel = _build_linear_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure inputs and parameters are on CUDA and float16
        x = x.to(device="cuda", dtype=torch.float16)
        w = self.weight.to(device="cuda")
        b = self.bias.to(device="cuda") if self.use_bias else torch.zeros(
            self.out_features, device="cuda", dtype=torch.float16
        )

        batch_size = x.shape[0]

        # Retrieve or compile kernel for current batch size
        kernel = self._get_kernel(batch_size, x.dtype)

        # Invoke TileLang kernel (returns float16 tensor)
        y = kernel(x, w, b)

        # LogSumExp across features (dim=1)
        y = torch.logsumexp(y, dim=1, keepdim=True)

        # Two consecutive LeakyReLU (negative_slope=0.01)
        y = torch.where(y > 0, y, y * 0.01)
        y = torch.where(y > 0, y, y * 0.01)

        # Two consecutive GELU using erf formulation
        sqrt_2 = math.sqrt(2.0)
        y = 0.5 * y * (1.0 + torch.erf(y / sqrt_2))
        y = 0.5 * y * (1.0 + torch.erf(y / sqrt_2))

        return y