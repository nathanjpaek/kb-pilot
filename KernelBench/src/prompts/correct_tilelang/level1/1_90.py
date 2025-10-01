"""
Problem Name: 90_cumprod
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0811 runtime_stats={'mean': 0.0811, 'std': 0.00323, 'min': 0.0783, 'max': 0.0961, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0285, 'std': 0.00615, 'min': 0.0261, 'max': 0.0859, 'num_trials': 100}, 'speedup_ratio': 0.351}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_cumprod_kernel(B: int, N: int, dtype: str = "float16"):
    accum_dtype = "float32"

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def cumprod_kernel(
        X: T.Tensor((B, N), dtype),
        Y: T.Tensor((B, N), dtype),
    ):
        # one block per batch row, single thread
        with T.Kernel(B, threads=1) as bx:
            running = T.alloc_local((1,), accum_dtype)
            running[0] = T.Cast(accum_dtype, 1)
            for j in T.serial(N):
                running[0] *= T.Cast(accum_dtype, X[bx, j])
                Y[bx, j] = T.Cast(dtype, running[0])

    return cumprod_kernel


class ModelNew(nn.Module):
    """
    TileLang-accelerated cumulative product along dim==1 for 2-D inputs.
    Falls back to PyTorch for other cases.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self._kernel_cache = {}  # (B, N, dtype) -> kernel

    # ---------- kernel cache ----------
    def _get_kernel(self, B: int, N: int, dtype: str):
        key = (B, N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_cumprod_kernel(B, N, dtype)
        return self._kernel_cache[key]

    # ------------- forward ------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # optimise only 2-D cumprod over dim==1
        if x.dim() != 2 or self.dim not in (1, -1):
            return torch.cumprod(x, dim=self.dim)

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        B, N = x_fp16.shape
        kernel = self._get_kernel(B, N, "float16")
        y_fp16 = kernel(x_fp16)
        return y_fp16.to(x.dtype)