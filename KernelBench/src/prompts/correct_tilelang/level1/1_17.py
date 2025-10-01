"""
Problem Name: 17_Matmul_with_transposed_B
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0704 runtime_stats={'mean': 0.0704, 'std': 0.0203, 'min': 0.0661, 'max': 0.271, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0493, 'std': 0.00107, 'min': 0.0479, 'max': 0.0559, 'num_trials': 100}, 'speedup_ratio': 0.7}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def matmul_transpose(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 3,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),  # B is not transposed yet
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Load A tile
                T.copy(A[by * block_M, ko * block_K], A_shared)
                # Load B tile (later transposed by GEMM)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                # Compute partial GEMM, transposing B inside the operation
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            # Store result
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang to compute C = A @ B.T
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int):
        key = (M, N, K)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = matmul_transpose(M, N, K)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computes torch.matmul(A, B.T) using a TileLang kernel.
        A: (M, K)
        B: (N, K)
        Returns: (M, N)
        """
        A = A.to(device="cuda", dtype=torch.float16)
        B = B.to(device="cuda", dtype=torch.float16)

        M, K = A.shape
        N = B.shape[0]  # since B is (N, K)

        kernel = self._get_kernel(M, N, K)
        C = kernel(A, B)
        return C