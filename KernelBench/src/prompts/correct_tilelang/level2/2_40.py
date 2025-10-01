"""
Problem Name: 40_Matmul_Scaling_ResidualAdd
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0391 runtime_stats={'mean': 0.0391, 'std': 0.0136, 'min': 0.0308, 'max': 0.125, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0944, 'std': 0.0642, 'min': 0.0564, 'max': 0.262, 'num_trials': 100}, 'speedup_ratio': 2.41}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_linear_scale_kernel(M, N, K, block_M=128, block_N=128, block_K=32, thread_num=128, scale_const=1.0, dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_scale_residual(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=thread_num,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], W_shared)
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = C_local[i, j] + T.Cast(accum_dtype, B[gn])
                    val *= scale_const
                    Out[gm, gn] = T.Cast(dtype, val)

    return linear_scale_residual


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_plus_one = 1.0 + float(scaling_factor)

        # Parameters initialized exactly like nn.Linear
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Cache for compiled kernels keyed by (batch_size, dtype)
        self._cached_kernels = {}

    def _get_kernel(self, batch_size: int, dtype: str):
        key = (batch_size, dtype)
        if key in self._cached_kernels:
            return self._cached_kernels[key]

        M, K, N = batch_size, self.in_features, self.out_features
        kernel = build_linear_scale_kernel(
            M=M, N=N, K=K,
            scale_const=self.scale_plus_one,
            dtype=dtype
        )
        self._cached_kernels[key] = kernel
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        kernel = self._get_kernel(x_fp16.shape[0], "float16")
        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)
        return out_fp16.to(x.dtype)