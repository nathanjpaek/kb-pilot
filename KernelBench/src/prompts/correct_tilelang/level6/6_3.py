"""
Problem Name: 3_Swish
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0452 runtime_stats={'mean': 0.0452, 'std': 0.0107, 'min': 0.0408, 'max': 0.146, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0338, 'std': 0.0068, 'min': 0.0308, 'max': 0.0946, 'num_trials': 100}, 'speedup_ratio': 0.748}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------------------- #

def _build_swish_kernel(numel: int, block_size: int = 256, dtype: str = "float16"):
    """Return a compiled TileLang kernel that applies Swish to a 1-D tensor."""

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def swish_kernel(
        X: T.Tensor((numel,), dtype),
        Y: T.Tensor((numel,), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                val = X[idx]
                Y[idx] = val * T.sigmoid(val)

    return swish_kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """Swish activation implemented with a TileLang kernel."""

    def __init__(self):
        super().__init__()
        # Cache compiled kernels: key = (numel, dtype_str)
        self._cached_kernels = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, numel: int, dtype_str: str):
        key = (numel, dtype_str)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _build_swish_kernel(numel, dtype=dtype_str)
        return self._cached_kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        flat_x = x_fp16.view(-1)
        numel = flat_x.numel()

        kernel = self._get_kernel(numel, "float16")
        flat_y = kernel(flat_x)
        y_fp16 = flat_y.view_as(x_fp16)
        return y_fp16.to(orig_dtype)