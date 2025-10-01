"""
Problem Name: 33_Gemm_Scale_BatchNorm
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.167 runtime_stats={'mean': 0.167, 'std': 0.0246, 'min': 0.151, 'max': 0.389, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.118, 'std': 0.0191, 'min': 0.108, 'max': 0.291, 'num_trials': 100}, 'speedup_ratio': 0.707}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _gemm_scale_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),          # row-major [N, K]
        B: T.Tensor((N,), dtype),            # bias
        S: T.Tensor((N,), dtype),            # scale
        O: T.Tensor((M, N), accum_dtype),    # output (fp32)
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            # Shared memory tiles
            X_s = T.alloc_shared((block_M, block_K), dtype)
            W_s = T.alloc_shared((block_N, block_K), dtype)
            # Accumulator in registers / fragments
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear accumulator
            T.clear(C_loc)

            # Pipeline over K dimension
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Load tiles into shared memory
                T.copy(X[by * block_M, ko * block_K], X_s)
                T.copy(W[bx * block_N, ko * block_K], W_s)

                # GEMM:  (block_M, block_K) x (block_N, block_K)^T
                T.gemm(X_s, W_s, C_loc, transpose_B=True)

            # Apply bias and scale, then write back
            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                in_bound = (global_m < M) and (global_n < N)
                if in_bound:
                    val = C_loc[i, j] + T.Cast(accum_dtype, B[global_n])
                    val *= T.Cast(accum_dtype, S[global_n])
                    O[global_m, global_n] = val

    return main


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super().__init__()
        # Linear parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_features)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Scale parameter (element-wise)
        self.scale = nn.Parameter(torch.randn(scale_shape))

        # BatchNorm affine parameters
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.eps = eps

        # Kernel cache
        self._cached_kernels = {}

    def _get_kernel(self, batch_size: int, dtype: str):
        key = (batch_size, dtype)
        if key not in self._cached_kernels:
            kernel = _gemm_scale_kernel(batch_size, self.weight.shape[0], self.weight.shape[1])
            self._cached_kernels[key] = kernel
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        W_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)
        s_fp16 = self.scale.to(device="cuda", dtype=torch.float16)

        M = x_fp16.shape[0]
        kernel = self._get_kernel(M, "float16")

        out_fp32 = kernel(x_fp16, W_fp16, b_fp16, s_fp16)

        # BatchNorm (training-mode semantics, per-batch stats)
        mean = out_fp32.mean(dim=0, keepdim=True)
        var = out_fp32.var(dim=0, unbiased=False, keepdim=True)
        out_norm = (out_fp32 - mean) / torch.sqrt(var + self.eps)

        gamma = self.bn_weight.to(device="cuda", dtype=out_norm.dtype)
        beta = self.bn_bias.to(device="cuda", dtype=out_norm.dtype)
        out_norm = out_norm * gamma + beta
        return out_norm