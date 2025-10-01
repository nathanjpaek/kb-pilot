"""
Problem Name: 6_Matmul_with_large_K_dimension_
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.79 runtime_stats={'mean': 0.79, 'std': 0.00684, 'min': 0.78, 'max': 0.811, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0754, 'std': 0.0125, 'min': 0.072, 'max': 0.197, 'num_trials': 100}, 'speedup_ratio': 0.0954}}
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
    block_K: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Build a TileLang kernel that computes C = A @ B
      A : (M, K)
      B : (K, N)
      C : (M, N)
    """

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
            # Shared-memory tiles
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            # Register fragment for the accumulate tile
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)

            # Number of K-dimension tiles
            k_tiles = T.ceildiv(K, block_K)

            for ko in T.Pipelined(k_tiles, num_stages=4):
                # Global offsets
                m_base = by * block_M
                n_base = bx * block_N
                k_base = ko * block_K

                # Load the next A and B tiles into shared memory
                T.copy(A[m_base, k_base], A_s)
                T.copy(B[k_base, n_base], B_s)

                # Multiply-accumulate into the C fragment
                T.gemm(A_s, B_s, C_frag)

            # Write the accumulator back to global memory
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return matmul_kernel


class ModelNew(nn.Module):
    """
    Optimized implementation of torch.matmul(A, B) using TileLang.
    Handles arbitrary M,N but is tuned for very large K.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def _get_kernel(self, M: int, N: int, K: int, torch_dtype: torch.dtype):
        key = (M, N, K, torch_dtype)
        if key not in self._kernel_cache:
            tl_dtype = "float16" if torch_dtype == torch.float16 else "bfloat16"
            self._kernel_cache[key] = _build_matmul_kernel(M, N, K, dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A : (M, K)
            B : (K, N)
        Returns:
            C : (M, N) where C = A @ B
        """
        orig_dtype = A.dtype

        # Move to CUDA & cast to fp16 for Tensor Core execution
        A_fp16 = A.to(device="cuda", dtype=torch.float16)
        B_fp16 = B.to(device="cuda", dtype=torch.float16)

        M, K = A_fp16.shape
        Kb, N = B_fp16.shape
        assert K == Kb, "Inner dimensions must match for matmul"

        kernel = self._get_kernel(M, N, K, A_fp16.dtype)
        C_fp16 = kernel(A_fp16, B_fp16)

        return C_fp16.to(orig_dtype)