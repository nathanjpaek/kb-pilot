"""
Problem Name: 11_4D_tensor_matrix_multiplication
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.34 runtime_stats={'mean': 4.34, 'std': 0.146, 'min': 4.26, 'max': 4.98, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.9, 'std': 0.108, 'min': 1.73, 'max': 2.06, 'num_trials': 100}, 'speedup_ratio': 0.438}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _tensor4d_matmul(
    b: int,
    i: int,
    j: int,
    l: int,
    k: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Creates a TileLang kernel that computes
        C[b, i, j, k] = sum_l A[b, i, j, l] * B[l, k]
    """

    M = b * i * j  # flatten (b, i, j) dims

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((b, i, j, l), dtype),
        B: T.Tensor((l, k), dtype),
        C: T.Tensor((b, i, j, k), dtype),
    ):
        # 2-D flattened views of A and C
        A_flat = T.Tensor((M, l), dtype, A.data)
        C_flat = T.Tensor((M, k), dtype, C.data)

        with T.Kernel(
            T.ceildiv(k, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(l, block_K), num_stages=num_stages):
                # Load tiles into shared memory
                T.copy(A_flat[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # Tile-level GEMM
                T.gemm(A_shared, B_shared, C_local)

            # Write results back
            T.copy(C_local, C_flat[by * block_M, bx * block_N])

    return kernel


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def _get_kernel(self, b, i, j, l, k, dtype="float16"):
        key = (b, i, j, l, k, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _tensor4d_matmul(b, i, j, l, k)
        return self._cached_kernels[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        b, i, j, l = A.shape
        k = B.shape[1]

        kernel = self._get_kernel(b, i, j, l, k)
        C = kernel(A, B)

        return C