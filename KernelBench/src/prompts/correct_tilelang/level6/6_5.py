"""
Problem Name: 5_cumprod
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.288 runtime_stats={'mean': 0.288, 'std': 0.00674, 'min': 0.282, 'max': 0.33, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.175, 'std': 0.013, 'min': 0.171, 'max': 0.302, 'num_trials': 100}, 'speedup_ratio': 0.608}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ------------------------------------------------------------ #
# Kernel factory : cumulative product along axis-1 for (B, N)
# ------------------------------------------------------------ #
def _build_cumprod_kernel(B: int, N: int, dtype: str = "float16"):
    accum_dtype = "float32"

    @tilelang.jit(out_idx=-1)        # allocate output on the fly
    @T.prim_func
    def cumprod_kernel(
        X: T.Tensor((B, N), dtype),
        Y: T.Tensor((B, N), dtype),
    ):
        # one CUDA block per input row, 1 thread per block
        with T.Kernel(B, threads=1) as bx:
            running = T.alloc_local((1,), accum_dtype)
            running[0] = T.Cast(accum_dtype, 1)
            for j in T.serial(N):
                running[0] *= T.Cast(accum_dtype, X[bx, j])
                Y[bx, j] = T.Cast(dtype, running[0])

    return cumprod_kernel


# ------------------------------------------------------------ #
# PyTorch wrapper
# ------------------------------------------------------------ #
class ModelNew(nn.Module):
    """
    TileLang-accelerated cumulative product for 2-D tensors along dim==1.
    Falls back to PyTorch otherwise.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._kernel_cache = {}            # (B, N, dtype) -> kernel

    # ---------- kernel cache ----------
    def _get_kernel(self, B: int, N: int, dtype: str):
        key = (B, N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_cumprod_kernel(B, N, dtype)
        return self._kernel_cache[key]

    # ---------------- forward ----------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # optimised path: 2-D tensor & dim == 1 / ‑1
        if x.dim() != 2 or self.dim not in (1, -1):
            return torch.cumprod(x, dim=self.dim)

        # prepare data
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        B, N = x_fp16.shape

        # get / compile kernel
        kernel = self._get_kernel(B, N, "float16")

        # launch kernel
        y_fp16 = kernel(x_fp16)

        return y_fp16.to(x.dtype)