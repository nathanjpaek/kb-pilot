"""
Problem Name: 15_Matmul_for_lower_triangular_matrices
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.363 runtime_stats={'mean': 0.363, 'std': 0.00151, 'min': 0.36, 'max': 0.368, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.247, 'std': 0.00574, 'min': 0.237, 'max': 0.261, 'num_trials': 100}, 'speedup_ratio': 0.68}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _matmul_tril_kernel(N, dtype="float16", block_M=128, block_N=128, block_K=32, num_stages=3):
    accum_dtype = "float32"

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((N, N), dtype),
        B: T.Tensor((N, N), dtype),
        C: T.Tensor((N, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(N, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(N, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < N) and (gj < N) and (gi >= gj):
                    C[gi, gj] = C_local[i, j]

    return kernel


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

    def _get_kernel(self, N, dtype):
        key = (N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _matmul_tril_kernel(N, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.to(device="cuda", dtype=torch.float16, copy=False)
        B = B.to(device="cuda", dtype=torch.float16, copy=False)
        N = A.shape[0]
        kernel = self._get_kernel(N, "float16")
        out = kernel(A, B)
        return out