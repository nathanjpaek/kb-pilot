"""
Problem Name: 26_GELU_
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0351 runtime_stats={'mean': 0.0351, 'std': 0.00145, 'min': 0.0332, 'max': 0.0428, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0165, 'std': 0.00108, 'min': 0.0154, 'max': 0.0215, 'num_trials': 100}, 'speedup_ratio': 0.47}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_gelu_kernel(numel: int, block_size: int = 256,
                       in_dtype: str = "float16", accum_dtype: str = "float"):
    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)
    half_const = 0.5

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gelu_kernel(
        X: T.Tensor((numel,), in_dtype),
        Y: T.Tensor((numel,), in_dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            for tx in T.Parallel(block_size):
                idx: T.int32 = bx * block_size + tx
                if idx < numel:
                    x_val_f32 = T.Cast(accum_dtype, X[idx])
                    gelu_f32 = (
                        half_const
                        * x_val_f32
                        * (1.0 + T.erf(x_val_f32 * inv_sqrt2))
                    )
                    Y[idx] = T.Cast(in_dtype, gelu_f32)

    return gelu_kernel


class ModelNew(nn.Module):
    """
    GELU activation accelerated with TileLang.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}  # (numel, dtype) -> kernel

    def _get_kernel(self, numel: int, dtype: torch.dtype):
        key = (numel, dtype)
        if key not in self._kernel_cache:
            tl_dtype = "float16" if dtype == torch.float16 else "float"
            self._kernel_cache[key] = _build_gelu_kernel(numel, in_dtype=tl_dtype)
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        numel = x_fp16.numel()

        kernel = self._get_kernel(numel, x_fp16.dtype)
        y_fp16 = kernel(x_fp16.view(-1)).view_as(x_fp16)

        return y_fp16.to(orig_dtype)