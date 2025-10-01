"""
Problem Name: 7_Batched_matrix_multiplication
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.102 runtime_stats={'mean': 0.102, 'std': 0.0187, 'min': 0.0977, 'max': 0.287, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0443, 'std': 0.00258, 'min': 0.0423, 'max': 0.0568, 'num_trials': 100}, 'speedup_ratio': 0.434}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# -----------------------------------------------------------------------------
# Kernel factory ----------------------------------------------------------------
# -----------------------------------------------------------------------------
def _build_bmm_kernel(
    B: int,
    M: int,
    K: int,
    N: int,
    *,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Returns a TileLang kernel that computes batched matrix multiplication:
        for b in range(B):  C[b] = A[b] @ B[b]
    A: (B, M, K)   row-major
    B: (B, K, N)   row-major
    C: (B, M, N)   row-major (created by TileLang)
    """

    blocks_M = (M + block_M - 1) // block_M  # tiles in the M dimension
    grid_x   = (N + block_N - 1) // block_N  # tiles in the N dimension
    grid_y   = B * blocks_M                  # fuse (batch, M-tiles)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A_t: T.Tensor((B, M, K), dtype),
        B_t: T.Tensor((B, K, N), dtype),
        C_t: T.Tensor((B, M, N), dtype),
    ):
        with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
            # Decode fused y-index into (batch, m-tile)
            batch_id = by // blocks_M
            tile_m   = by %  blocks_M

            m_base = tile_m * block_M
            n_base = bx    * block_N

            # Shared memory tiles -------------------------------------------
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)

            # Accumulator fragment -----------------------------------------
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_loc)

            # Reduction loop over K dimension -------------------------------
            for ko in T.Pipelined(
                T.ceildiv(K, block_K), num_stages=num_stages
            ):
                k_base = ko * block_K

                # Load A tile
                T.copy(
                    A_t[batch_id, m_base, k_base],
                    A_s,
                )

                # Load B tile
                T.copy(
                    B_t[batch_id, k_base, n_base],
                    B_s,
                )

                # GEMM  (A_s: [M x K] , B_s: [K x N])
                T.gemm(A_s, B_s, C_loc)

            # Write results --------------------------------------------------
            for i, j in T.Parallel(block_M, block_N):
                gm = m_base + i
                gn = n_base + j
                if (gm < M) and (gn < N):
                    C_t[batch_id, gm, gn] = T.Cast(dtype, C_loc[i, j])

    return kernel


# -----------------------------------------------------------------------------
# PyTorch wrapper module -------------------------------------------------------
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    TileLang-accelerated batched matrix multiplication (torch.bmm replacement)
    """

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}  # keyed by (B, M, K, N, dtype)

    # ----------------------------------------------------------------------
    def _get_kernel(self, B: int, M: int, K: int, N: int, dtype: torch.dtype):
        key = (B, M, K, N, dtype)
        if key not in self._kernel_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kernel_cache[key] = _build_bmm_kernel(
                B, M, K, N, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    # ----------------------------------------------------------------------
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A : (B, M, K)
            B : (B, K, N)
        Returns:
            C : (B, M, N)
        """
        assert A.ndim == 3 and B.ndim == 3, "Inputs must be 3-D tensors"
        assert A.shape[0] == B.shape[0] and A.shape[2] == B.shape[1], "Shape mismatch"

        Bsz, M, K = A.shape
        _,  _, N  = B.shape

        orig_dtype = A.dtype

        A_f16 = A.to(device="cuda", dtype=torch.float16).contiguous()
        B_f16 = B.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(Bsz, M, K, N, A_f16.dtype)
        C_f16 = kernel(A_f16, B_f16)

        return C_f16.to(orig_dtype)