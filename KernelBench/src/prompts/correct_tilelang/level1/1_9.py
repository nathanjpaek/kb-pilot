"""
Problem Name: 9_Tall_skinny_matrix_multiplication_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.628 runtime_stats={'mean': 0.628, 'std': 0.00961, 'min': 0.625, 'max': 0.723, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.215, 'std': 0.00113, 'min': 0.214, 'max': 0.222, 'num_trials': 100}, 'speedup_ratio': 0.342}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_ts_matmul_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 16,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def _get_kernel(self, M: int, N: int, K: int, dtype: str = "float16"):
        key = (M, N, K, dtype)
        if key not in self._cached_kernels:
            # Choose tiling heuristically for tall-skinny multiplications
            block_M = 128
            block_N = 16 if N <= 32 else 128
            block_K = 32
            self._cached_kernels[key] = _build_ts_matmul_kernel(
                M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype="float"
            )
        return self._cached_kernels[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "Inner dimensions must match for matrix multiplication"

        kernel = self._get_kernel(M, N, K, "float16")
        C = kernel(A, B)
        return C