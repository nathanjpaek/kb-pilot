"""
Problem Name: 12_Matmul_with_diagonal_matrices_
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0562 runtime_stats={'mean': 0.0562, 'std': 0.00861, 'min': 0.0513, 'max': 0.133, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.212, 'std': 0.0316, 'min': 0.204, 'max': 0.525, 'num_trials': 100}, 'speedup_ratio': 3.77}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# -----------------------------------------------------------------------------
# TileLang kernel factory
# -----------------------------------------------------------------------------

def _build_scale_kernel(
    N: int,
    M: int,
    *,
    block_rows: int = 128,
    block_cols: int = 128,
    dtype: str = "float16",
):
    """Return a compiled kernel that computes C = diag(A) @ B  (row–wise scale)."""

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def scale_kernel(
        A: T.Tensor((N,), dtype),          # scale vector
        B: T.Tensor((N, M), dtype),        # input matrix
        C: T.Tensor((N, M), dtype),        # output matrix
    ):
        grid_x = T.ceildiv(M, block_cols)   # columns   → bx
        grid_y = T.ceildiv(N, block_rows)   # rows      → by

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            row_base = by * block_rows      # top-left element of the tile in B/C
            col_base = bx * block_cols

            scale_sh = T.alloc_shared((block_rows,), dtype)
            tile_B  = T.alloc_shared((block_rows, block_cols), dtype)

            # --- load scale vector tile into shared memory ---------------------------------
            T.copy(A[row_base], scale_sh)            # 1-D copy; handles boundaries

            # --- load B tile --------------------------------------------------------------
            T.copy(B[row_base, col_base], tile_B)    # 2-D copy; handles boundaries

            T.tvm_storage_sync("shared")             # make sure data visible to all threads

            # --- element-wise scaling -----------------------------------------------------
            for i, j in T.Parallel(block_rows, block_cols):
                tile_B[i, j] = tile_B[i, j] * scale_sh[i]

            # --- store results ------------------------------------------------------------
            T.copy(tile_B, C[row_base, col_base])

    return scale_kernel


# -----------------------------------------------------------------------------
# PyTorch wrapper module
# -----------------------------------------------------------------------------

class ModelNew(nn.Module):
    """Optimized module computing C = diag(A) @ B via TileLang."""

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}  # keyed by (N, M, dtype)

    # ------------------------------------------------------------------
    # kernel cache ------------------------------------------------------
    # ------------------------------------------------------------------
    def _get_kernel(self, N: int, M: int, tl_dtype: str):
        key = (N, M, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_scale_kernel(N, M, dtype=tl_dtype)
        return self._kernel_cache[key]

    # ------------------------------------------------------------------
    # forward -----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A : (N,)   scale vector – diagonal entries
            B : (N, M) matrix to be scaled
        Returns:
            C : (N, M) where C[i, j] = A[i] * B[i, j]
        """
        assert A.ndim == 1 and B.ndim == 2, "Input shapes must be (N,) and (N, M)"
        N, M = B.shape
        assert A.shape[0] == N, "Mismatch on leading dimension"

        orig_dtype = B.dtype
        A_fp16 = A.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        B_fp16 = B.to(device="cuda", dtype=torch.float16, copy=False).contiguous()

        kernel = self._get_kernel(N, M, "float16")
        C_fp16 = kernel(A_fp16, B_fp16)
        return C_fp16.to(orig_dtype)