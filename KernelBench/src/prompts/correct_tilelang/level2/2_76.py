"""
Problem Name: 76_Gemm_Add_ReLU
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0428 runtime_stats={'mean': 0.0428, 'std': 0.00125, 'min': 0.0412, 'max': 0.0487, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.051, 'std': 0.0021, 'min': 0.0487, 'max': 0.0611, 'num_trials': 100}, 'speedup_ratio': 1.19}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _linear_bias_relu_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):
            X_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Copy tiles for current K-slice
                T.copy(X[by * block_M, ko * block_K], X_shared)
                T.copy(W[bx * block_N, ko * block_K], W_shared)

                # GEMM: X_shared (block_M x block_K) *
                #       W_shared^T (block_K x block_N)
                T.gemm(X_shared, W_shared, C_local, transpose_B=True)

            # Fuse bias add + ReLU and write back
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < M) and (gj < N):
                    val = C_local[i, j] + T.Cast(accum_dtype, B[gj])
                    val = T.max(val, T.Cast(accum_dtype, 0))
                    Out[gi, gj] = T.Cast(dtype, val)

    return main


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias_shape):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Parameter initialization identical to nn.Linear defaults
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Bias parameter same behaviour as original Model
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Cache for compiled TileLang kernels
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, dtype: str):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            kernel = _linear_bias_relu_kernel(
                M=batch_size,
                K=self.in_features,
                N=self.out_features,
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prepare tensors for TileLang kernel
        x_f16 = x.to(device="cuda", dtype=torch.float16)
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        # Retrieve or compile kernel
        kernel = self._get_kernel(x_f16.shape[0], "float16")

        # Execute kernel
        out_f16 = kernel(x_f16, w_f16, b_f16)

        # Return in original dtype
        return out_f16.to(x.dtype)