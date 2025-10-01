"""
Problem Name: 14_Gemm_Divide_Sum_Scaling
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.126 runtime_stats={'mean': 0.126, 'std': 0.0575, 'min': 0.0401, 'max': 0.227, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.134, 'std': 0.0711, 'min': 0.0444, 'max': 0.33, 'num_trials': 100}, 'speedup_ratio': 1.06}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_kernel(
    batch_size: int,
    input_size: int,
    scale_const: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, input_size), dtype),
        Wsum: T.Tensor((input_size,), dtype),
        Out: T.Tensor((batch_size, 1), dtype),
    ):
        with T.Kernel(T.ceildiv(batch_size, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            row = bx * block_size + tx
            if row < batch_size:
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)
                for k in range(input_size):
                    acc[0] += (
                        T.Cast(accum_dtype, X[row, k])
                        * T.Cast(accum_dtype, Wsum[k])
                    )
                val = acc[0] * scale_const
                Out[row, 0] = T.Cast(dtype, val)

    return fused_kernel


class ModelNew(nn.Module):
    """
    Optimized model using TileLang kernels.
    Performs: Out = scaling_factor / 2 * sum_j ((X @ W.T)_ij)  (kept as shape (B,1))
    """

    def __init__(self, input_size, hidden_size, scaling_factor):
        super().__init__()
        # Parameters
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.scale_const = float(scaling_factor) / 2.0  # combine divide by 2

        # Kernel cache keyed by (batch_size, dtype)
        self._kernels = {}

    def _get_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_kernel(
                batch_size,
                self.input_size,
                self.scale_const,
                dtype=dtype,
            )
        return self._kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)

        # Precompute column sums of the weight matrix (shape: (input_size,))
        w_col_sum = w_fp16.sum(dim=0).contiguous()

        batch_size = x_fp16.shape[0]
        kernel = self._get_kernel(batch_size, "float16")

        out_fp16 = kernel(x_fp16, w_col_sum)
        return out_fp16.to(orig_dtype)