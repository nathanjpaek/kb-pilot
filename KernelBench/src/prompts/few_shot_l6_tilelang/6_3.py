"""
Problem Name: 3_Swish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0581 runtime_stats={'mean': 0.0581, 'std': 0.0024, 'min': 0.0544, 'max': 0.0716, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0419, 'std': 0.0152, 'min': 0.0376, 'max': 0.19, 'num_trials': 100}, 'speedup_ratio': 0.721}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------
# TileLang kernel factory
# ---------------------------------------------------------------------
def build_swish_kernel(numel: int,
                       block_size: int = 256,
                       dtype: str = "float16"):
    """
    Returns a compiled TileLang kernel implementing Swish:
        Y[i] = X[i] * sigmoid(X[i])  for i in [0, numel)
    """

    @tilelang.jit(out_idx=-1)           # last tensor (Y) allocated at call site
    @T.prim_func
    def kernel(
        X: T.Tensor((numel,), dtype),    # input
        Y: T.Tensor((numel,), dtype),    # output (auto-allocated)
    ):
        one_const = T.Cast(dtype, 1.0)

        # gridDim.x = ceil(numel / block_size)
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx

            if idx < numel:
                val    = X[idx]
                sig    = one_const / (one_const + T.exp(-val))
                Y[idx] = val * sig

    return kernel


# ---------------------------------------------------------------------
# PyTorch wrapper module
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    TileLang-accelerated Swish activation:  Y = X * sigmoid(X)
    """

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}   # keyed by (numel, dtype)

    # --------------------------------------------------------------
    # kernel cache --------------------------------------------------
    # --------------------------------------------------------------
    def _get_kernel(self, numel: int, tl_dtype: str):
        key = (numel, tl_dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_swish_kernel(numel, dtype=tl_dtype)
        return self._kernel_cache[key]

    # --------------------------------------------------------------
    # forward -------------------------------------------------------
    # --------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation element-wise.
        """
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        numel = x_f16.numel()

        kernel = self._get_kernel(numel, "float16")
        y_f16 = kernel(x_f16.view(-1)).view_as(x_f16)

        return y_f16.to(orig_dtype)