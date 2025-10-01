"""
Problem Name: 10_3D_tensor_matrix_multiplication
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.219 runtime_stats={'mean': 0.219, 'std': 0.0399, 'min': 0.164, 'max': 0.287, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.167, 'std': 0.0263, 'min': 0.0943, 'max': 0.28, 'num_trials': 100}, 'speedup_ratio': 0.763}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def matmul_kernel(M, N, K, *, block_M=128, block_N=128, block_K=32,
                  dtype="float16", accum_dtype="float", num_stages=3):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Load A and B tiles into shared memory
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)

                # Tile-level GEMM
                T.gemm(A_shared, B_shared, C_local)

            # Write result back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    """
    Optimized 3D tensor-matrix multiplication using TileLang.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def _get_kernel(self, M, N, K, dtype="float16"):
        key = (M, N, K, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = matmul_kernel(M, N, K, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A: Tensor of shape (N, M, K)
            B: Tensor of shape (K, L)

        Returns:
            Tensor of shape (N, M, L)
        """
        # Move to CUDA and convert to fp16 for TileLang
        A = A.to(device="cuda", dtype=torch.float16, non_blocking=True)
        B = B.to(device="cuda", dtype=torch.float16, non_blocking=True)

        N_dim, M_dim, K_dim = A.shape
        K_dim_B, L_dim = B.shape
        assert K_dim == K_dim_B, "Incompatible K dimensions"

        # Reshape A to 2D matrix
        A_2d = A.reshape(N_dim * M_dim, K_dim).contiguous()

        M_total = N_dim * M_dim
        kernel = self._get_kernel(M_total, L_dim, K_dim, dtype="float16")

        C_2d = kernel(A_2d, B)
        C = C_2d.view(N_dim, M_dim, L_dim)
        return C