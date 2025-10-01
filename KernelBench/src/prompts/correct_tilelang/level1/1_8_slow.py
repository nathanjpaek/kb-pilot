"""
Problem Name: 8_Matmul_with_irregular_shapes_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=83.3 runtime_stats={'mean': 83.3, 'std': 0.0651, 'min': 83.2, 'max': 83.5, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.14, 'std': 0.0233, 'min': 2.1, 'max': 2.26, 'num_trials': 100}, 'speedup_ratio': 0.0257}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_matmul_kernel(
    M,
    N,
    K,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        grid_x = T.ceildiv(N, block_N)
        grid_y = T.ceildiv(M, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            # Shared and local buffers
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear accumulator
            T.clear(C_local)

            # Number of K tiles
            num_k_tiles = T.ceildiv(K, block_K)

            for ko in T.Pipelined(num_k_tiles, num_stages=3):
                # Load A tile
                for i, k in T.Parallel(block_M, block_K):
                    gi = by * block_M + i
                    gk = ko * block_K + k
                    if gi < M and gk < K:
                        A_shared[i, k] = A[gi, gk]
                    else:
                        A_shared[i, k] = T.cast(0, dtype)

                # Load B tile
                for k, j in T.Parallel(block_K, block_N):
                    gk = ko * block_K + k
                    gj = bx * block_N + j
                    if gk < K and gj < N:
                        B_shared[k, j] = B[gk, gj]
                    else:
                        B_shared[k, j] = T.cast(0, dtype)

                # GEMM on the current tiles
                T.gemm(A_shared, B_shared, C_local)

            # Write results back to global memory
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if gi < M and gj < N:
                    C[gi, gj] = C_local[i, j]

    return matmul_kernel


class ModelNew(nn.Module):
    """
    Optimized model using TileLang for matrix multiplication with irregular shapes.
    """

    def __init__(self):
        super(ModelNew, self).__init__()

        # Fixed matrix sizes for this model
        self.M = 8205
        self.K = 2949
        self.N = 5921

        # Build and cache the kernel at initialization
        self._matmul_kernel = build_matmul_kernel(self.M, self.N, self.K)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication using the TileLang kernel.

        Args:
            A: Tensor of shape (M, K)
            B: Tensor of shape (K, N)

        Returns:
            Tensor of shape (M, N)
        """
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        C = self._matmul_kernel(A, B)
        return C