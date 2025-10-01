"""
Problem Name: 18_Matmul_with_transposed_both
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.163 runtime_stats={'mean': 0.163, 'std': 0.00346, 'min': 0.159, 'max': 0.178, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0601, 'std': 0.0231, 'min': 0.053, 'max': 0.276, 'num_trials': 100}, 'speedup_ratio': 0.369}}
"""

import torch
import torch.nn as nn
import math

import tilelang
import tilelang.language as T


def _build_kernel(M: int, N: int, K: int, dtype: str = "float16"):
    block_M, block_N, block_K = 128, 128, 32
    accum_dtype = "float32"

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm_kernel


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def _get_kernel(self, M: int, N: int, K: int, dtype: str = "float16"):
        key = (M, N, K, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _build_kernel(M, N, K, dtype)
        return self._cached_kernels[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_t = A.to(device="cuda", dtype=torch.float16).t().contiguous()
        B_t = B.to(device="cuda", dtype=torch.float16).t().contiguous()

        M, K = A_t.shape
        _, N = B_t.shape

        kernel = self._get_kernel(M, N, K, "float16")
        C = kernel(A_t, B_t)
        return C