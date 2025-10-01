"""
Problem Name: 28_HardSigmoid
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0452 runtime_stats={'mean': 0.0452, 'std': 0.0035, 'min': 0.0413, 'max': 0.0578, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0229, 'std': 0.00272, 'min': 0.0187, 'max': 0.0338, 'num_trials': 100}, 'speedup_ratio': 0.507}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_hardsigmoid_kernel(N: int, block_size: int = 256, dtype: str = "float16"):
    """
    Factory that returns a compiled TileLang kernel which applies the HardSigmoid
    activation to a 1-D tensor of length N.
    """

    @tilelang.jit(out_idx=-1)  # create the output tensor during runtime
    @T.prim_func
    def hardsigmoid_kernel(
        X: T.Tensor((N,), dtype),
        Y: T.Tensor((N,), dtype),
    ):
        # Launch 1-D grid of blocks; each block has `block_size` threads
        with T.Kernel(T.ceildiv(N, block_size), threads=block_size) as bx:
            # Parallel loop over threads within the block
            for tx in T.Parallel(block_size):
                idx: T.int32 = bx * block_size + tx
                if idx < N:
                    # HardSigmoid: clamp((x + 3) / 6, 0, 1)
                    val = T.Cast(dtype, X[idx]) * T.Cast(dtype, 0.1666666716337204) + T.Cast(
                        dtype, 0.5
                    )
                    val = T.if_then_else(val < T.Cast(dtype, 0.0), T.Cast(dtype, 0.0), val)
                    val = T.if_then_else(val > T.Cast(dtype, 1.0), T.Cast(dtype, 1.0), val)
                    Y[idx] = val

    return hardsigmoid_kernel


class ModelNew(nn.Module):
    """
    Optimized model using TileLang for HardSigmoid activation.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Cache kernels keyed by (numel, dtype) to avoid recompilation
        self._cached_kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies HardSigmoid activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with HardSigmoid applied, same shape as input.
        """
        # Move to CUDA and fp16 for the TileLang kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        num_elements = x_fp16.numel()

        # Retrieve or compile the appropriate kernel
        key = (num_elements, x_fp16.dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = build_hardsigmoid_kernel(num_elements)

        kernel = self._cached_kernels[key]

        # Flatten input for the 1-D kernel, run it, then reshape back
        y_fp16 = kernel(x_fp16.contiguous().view(-1))
        y = y_fp16.view_as(x_fp16).to(dtype=x.dtype)

        return y