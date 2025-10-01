"""
Problem Name: 10_3D_tensor_matrix_multiplication
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.26 runtime_stats={'mean': 0.26, 'std': 0.039, 'min': 0.225, 'max': 0.384, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.154, 'std': 0.0378, 'min': 0.0897, 'max': 0.285, 'num_trials': 100}, 'speedup_ratio': 0.592}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ------------------------------------------------------------------
# TileLang kernel factory
# ------------------------------------------------------------------
def _build_tensor_matmul_kernel(
    M_total: int,  # N * M
    K: int,
    L: int,
    *,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((M_total, K), dtype),   # flattened (N*M, K)
        B: T.Tensor((K, L), dtype),        # (K, L)
        Out: T.Tensor((M_total, L), dtype) # created by TileLang
    ):
        with T.Kernel(
            T.ceildiv(L, block_N),          # grid.x
            T.ceildiv(M_total, block_M),    # grid.y
            threads=threads,
        ) as (bx, by):
            # Shared memory tiles
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)

            # Fragment accumulator
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            # Reduction over K dimension
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                # Load A tile
                T.copy(
                    A[by * block_M, ko * block_K],  # starting address
                    A_s
                )

                # Load B tile
                T.copy(
                    B[ko * block_K, bx * block_N],
                    B_s
                )

                # Compute partial GEMM
                T.gemm(A_s, B_s, C_loc)

            # Write results with boundary checks
            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                if (global_m < M_total) and (global_n < L):
                    Out[global_m, global_n] = T.Cast(dtype, C_loc[i, j])

    return kernel


# ------------------------------------------------------------------
# PyTorch wrapper module
# ------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized 3D tensor–matrix multiplication using TileLang.
    Computes C[n, m, l] = sum_k A[n, m, k] * B[k, l]
    """

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}  # keyed by (M_total, K, L, dtype)

    # --------------------------------------------------------------
    # Kernel cache helper
    # --------------------------------------------------------------
    def _get_kernel(self, M_total: int, K: int, L: int, dtype: torch.dtype):
        key = (M_total, K, L, dtype)
        if key not in self._kernel_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kernel_cache[key] = _build_tensor_matmul_kernel(
                M_total, K, L, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    # --------------------------------------------------------------
    # Forward
    # --------------------------------------------------------------
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A: (N, M, K)
            B: (K, L)
        Returns:
            (N, M, L)
        """
        assert A.dim() == 3 and B.dim() == 2, "Input ranks must be 3 and 2"
        N, M, K = A.shape
        K2, L = B.shape
        assert K2 == K, "Inner dimensions must match"

        orig_dtype = A.dtype

        A_f16 = A.to(dtype=torch.float16, device="cuda").contiguous()
        B_f16 = B.to(dtype=torch.float16, device="cuda").contiguous()

        M_total = N * M
        A_flat = A_f16.view(M_total, K)

        kernel = self._get_kernel(M_total, K, L, A_f16.dtype)
        C_flat = kernel(A_flat, B_f16)  # (M_total, L)

        C = C_flat.view(N, M, L)
        return C.to(orig_dtype)