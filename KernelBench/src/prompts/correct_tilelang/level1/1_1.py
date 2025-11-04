"""
Problem Name: 1_Square_matrix_multiplication_
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0675 runtime_stats={'mean': 0.0675, 'std': 0.00138, 'min': 0.0652, 'max': 0.074, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0449, 'std': 0.000823, 'min': 0.0438, 'max': 0.0504, 'num_trials': 100}, 'speedup_ratio': 0.665}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_matmul_kernel(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """Return compiled TileLang kernel C = A @ B with shapes (M,K)*(K,N)."""

    @T.prim_func
    def matmul_kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        grid_x = T.ceildiv(N, block_N)
        grid_y = T.ceildiv(M, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            # Shared tiles
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            # Register accumulator
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_frag)

            # Loop over K dimension with pipeline
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                k_base = ko * block_K
                m_base = by * block_M
                n_base = bx * block_N

                # Load tiles to shared memory
                T.copy(A[m_base, k_base], A_s)
                T.copy(B[k_base, n_base], B_s)

                # Compute
                T.gemm(A_s, B_s, C_frag)

            # Write back
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return tilelang.compile(matmul_kernel, out_idx=[2], target="cuda")


class ModelNew(nn.Module):
    """TileLang-accelerated square matrix multiplication (C = A @ B)."""

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, tl_dtype: str):
        key = (M, N, K, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_matmul_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        A_c = A.contiguous()
        B_c = B.contiguous()

        M, K = A_c.shape
        Kb, N = B_c.shape
        assert K == Kb, "Inner dimensions must match"

        kernel = self._get_kernel(M, N, K, "float16")
        C = kernel(A_c, B_c)

        return C