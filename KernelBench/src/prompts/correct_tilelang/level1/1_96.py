"""
Problem Name: 96_HuberLoss
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0447 runtime_stats={'mean': 0.0447, 'std': 0.00109, 'min': 0.0428, 'max': 0.0489, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0312, 'std': 0.0017, 'min': 0.0296, 'max': 0.042, 'num_trials': 100}, 'speedup_ratio': 0.698}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def smooth_l1_kernel(M, N, block_size=256, dtype="float16"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        Pred: T.Tensor((M, N), dtype),
        Tgt: T.Tensor((M, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        num_elem: T.int32 = M * N
        with T.Kernel(T.ceildiv(num_elem, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx: T.int32 = bx * block_size + tx
            if idx < num_elem:
                row: T.int32 = idx // N
                col: T.int32 = idx % N
                p_val_f32: T.float32 = T.Cast("float32", Pred[row, col])
                t_val_f32: T.float32 = T.Cast("float32", Tgt[row, col])
                diff: T.float32 = p_val_f32 - t_val_f32
                abs_diff: T.float32 = T.if_then_else(diff >= 0, diff, -diff)
                val: T.float32 = T.if_then_else(
                    abs_diff < 1.0,
                    0.5 * diff * diff,      # 0.5 * diff^2 when |diff| < 1
                    abs_diff - 0.5,         # |diff| - 0.5 otherwise
                )
                Out[row, col] = T.Cast(dtype, val)

    return main


class ModelNew(nn.Module):
    """
    Optimized Smooth L1 (Huber) Loss using TileLang.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move inputs to CUDA and convert to float16 for TileLang
        preds_f16 = predictions.to(device="cuda", dtype=torch.float16)
        targs_f16 = targets.to(device="cuda", dtype=torch.float16)

        M, N = preds_f16.shape
        key = (M, N)

        if key not in self._kernel_cache:
            self._kernel_cache[key] = smooth_l1_kernel(M, N)

        kernel = self._kernel_cache[key]
        element_losses = kernel(preds_f16, targs_f16)
        return element_losses.mean()