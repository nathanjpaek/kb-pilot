"""
Problem Name: 32_HardTanh
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.036 runtime_stats={'mean': 0.036, 'std': 0.00128, 'min': 0.0342, 'max': 0.0436, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0204, 'std': 0.00465, 'min': 0.0186, 'max': 0.065, 'num_trials': 100}, 'speedup_ratio': 0.567}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _hardtanh_kernel(numel: int, block_size: int = 256, dtype: str = "float16"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def hardtanh(
        X: T.Tensor((numel,), dtype),
        Y: T.Tensor((numel,), dtype),
    ):
        grid_dim = T.ceildiv(numel, block_size)

        # 1-D launch configuration (grid_dim blocks, block_size threads)
        with T.Kernel(grid_dim, threads=block_size) as bx:
            # Parallel loop over threads within the CUDA block
            for tx in T.Parallel(block_size):
                idx: T.int32 = bx * block_size + tx
                if idx < numel:
                    val = X[idx]
                    val = T.max(val, -1.0)
                    val = T.min(val, 1.0)
                    Y[idx] = val

    return hardtanh


class ModelNew(nn.Module):
    """
    Optimized HardTanh using TileLang.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self.block_size = 256
        self._cached_kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to CUDA and fp16 for kernel execution
        x_device = x.to(device="cuda", dtype=torch.float16)
        orig_shape = x_device.shape
        x_flat = x_device.contiguous().view(-1)

        numel = x_flat.numel()
        key = (numel, x_flat.dtype)

        if key not in self._cached_kernels:
            self._cached_kernels[key] = _hardtanh_kernel(
                numel=numel,
                block_size=self.block_size,
                dtype="float16",
            )

        kernel = self._cached_kernels[key]
        y_flat = kernel(x_flat)
        y = y_flat.view(orig_shape).to(x.dtype)
        return y