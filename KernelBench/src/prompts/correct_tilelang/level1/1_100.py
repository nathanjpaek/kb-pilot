"""
Problem Name: 100_HingeLoss
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0481 runtime_stats={'mean': 0.0481, 'std': 0.00162, 'min': 0.0458, 'max': 0.0579, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0426, 'std': 0.00187, 'min': 0.0404, 'max': 0.0542, 'num_trials': 100}, 'speedup_ratio': 0.886}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _hinge_kernel(M: int, block: int = 256, in_dtype: str = "float16", accum_dtype: str = "float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        pred: T.Tensor((M,), in_dtype),
        target: T.Tensor((M,), in_dtype),
        out: T.Tensor((M,), in_dtype),
    ):
        # Each block processes `block` elements
        with T.Kernel(T.ceildiv(M, block), threads=block) as bx:
            # Parallel over threads inside a block
            for tx in T.Parallel(block):
                idx: T.int32 = bx * block + tx
                if idx < M:
                    val = T.Cast(accum_dtype, 1) - T.Cast(accum_dtype, pred[idx] * target[idx])
                    val = T.max(val, T.Cast(accum_dtype, 0))
                    out[idx] = T.Cast(in_dtype, val)

    return main


class ModelNew(nn.Module):
    """
    Optimized model that computes Hinge Loss for binary classification using TileLang kernels.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def _get_kernel(self, numel: int, dtype: str = "float16"):
        key = (numel, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _hinge_kernel(numel, in_dtype=dtype)
        return self._cached_kernels[key]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # Move tensors to GPU and cast to float16 (TileLang currently focuses on fp16 I/O)
        preds = predictions.to(device="cuda", dtype=torch.float16).reshape(-1)
        targs = targets.to(device="cuda", dtype=torch.float16).reshape(-1)

        # Compile/retrieve the cached kernel for current tensor size
        kernel = self._get_kernel(preds.numel())

        # Execute the kernel to compute clamped hinge components
        hinge_components = kernel(preds, targs)
        return hinge_components.mean()