"""
Problem Name: 5_cumprod
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.421 runtime_stats={'mean': 0.421, 'std': 0.0385, 'min': 0.359, 'max': 0.57, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.205, 'std': 0.018, 'min': 0.168, 'max': 0.268, 'num_trials': 100}, 'speedup_ratio': 0.487}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# TileLang kernel factory
# ---------------------------------------------------------------------------

def _build_cumprod_kernel(
    rows: int,
    cols: int,
    dim: int,
    dtype: str = "float32",
):
    """
    Generate a TileLang kernel that computes the cumulative product along the
    specified dimension of a 2-D tensor of shape (rows, cols).

    dim == 1  → prefix product along the second dimension (per-row scan)
    dim == 0  → prefix product along the first  dimension (per-col scan)
    """
    assert dim in (0, 1), "kernel only supports dim 0 or 1"

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def cumprod_kernel(
        X: T.Tensor((rows, cols), dtype),
        Y: T.Tensor((rows, cols), dtype),  # created at call-time by TileLang
    ):
        one_const = T.Cast(dtype, 1)

        if dim == 1:
            # ---------------------------------------------------------------
            # cumulative product along columns (axis = 1)
            # ---------------------------------------------------------------
            with T.Kernel(rows, threads=1) as by:
                prod = T.alloc_local((1,), dtype)
                prod[0] = one_const
                for j in range(cols):
                    prod[0] = prod[0] * X[by, j]
                    Y[by, j] = prod[0]
        else:
            # ---------------------------------------------------------------
            # cumulative product along rows (axis = 0)
            # ---------------------------------------------------------------
            with T.Kernel(cols, threads=1) as bx:
                prod = T.alloc_local((1,), dtype)
                prod[0] = one_const
                for i in range(rows):
                    prod[0] = prod[0] * X[i, bx]
                    Y[i, bx] = prod[0]

    return cumprod_kernel


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------

class ModelNew(nn.Module):
    """
    TileLang-accelerated cumulative product along a specified dimension.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._kernel_cache = {}  # keyed by (rows, cols, dim, dtype)

    # ---------------------------------------------------------------------
    # kernel cache --------------------------------------------------------
    # ---------------------------------------------------------------------
    def _get_kernel(self, rows: int, cols: int, dim: int, dtype: torch.dtype):
        tl_dtype = "float32" if dtype == torch.float32 else "float16"
        key = (rows, cols, dim, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_cumprod_kernel(
                rows, cols, dim, dtype=tl_dtype
            )
        return self._kernel_cache[key]

    # ---------------------------------------------------------------------
    # forward -------------------------------------------------------------
    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute cumulative product along `self.dim`, matching torch.cumprod.
        Currently supports 2-D inputs with dim ∈ {0, 1, -1}.
        """
        assert x.dim() == 2, "ModelNew currently supports 2-D tensors only"

        dim = self.dim
        if dim < 0:
            dim += x.dim()
        assert dim in (0, 1), "dim must be 0 or 1 for 2-D tensor"

        orig_dtype = x.dtype
        x_fp32 = x.to(device="cuda", dtype=torch.float32, copy=False).contiguous()

        rows, cols = x_fp32.shape
        kernel = self._get_kernel(rows, cols, dim, x_fp32.dtype)
        y_fp32 = kernel(x_fp32)

        return y_fp32.to(orig_dtype)