"""
Problem Name: 98_KLDivLoss
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0829 runtime_stats={'mean': 0.0829, 'std': 0.00441, 'min': 0.0772, 'max': 0.101, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0622, 'std': 0.0112, 'min': 0.0564, 'max': 0.121, 'num_trials': 100}, 'speedup_ratio': 0.75}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_kl_kernel(B, N, block_M=8, block_N=128, dtype="float32"):
    threads = block_M * block_N  # must be <= 1024 on CUDA

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kl_kernel(
        pred: T.Tensor((B, N), dtype),
        targ: T.Tensor((B, N), dtype),
        out: T.Tensor((B, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(B, block_M), T.ceildiv(N, block_N), threads=threads) as (bx, by):
            for i, j in T.Parallel(block_M, block_N):
                gi = bx * block_M + i
                gj = by * block_N + j
                if (gi < B) and (gj < N):
                    t_val = T.Cast("float32", targ[gi, gj])
                    p_val = T.Cast("float32", pred[gi, gj])
                    out[gi, gj] = t_val * (T.log(t_val) - T.log(p_val))

    return kl_kernel


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._cached_kernels = {}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.to(device="cuda", dtype=torch.float32)
        targets = targets.to(device="cuda", dtype=torch.float32)

        B, N = predictions.shape
        key = (B, N, predictions.dtype)

        if key not in self._cached_kernels:
            self._cached_kernels[key] = build_kl_kernel(B, N)

        kl_kernel = self._cached_kernels[key]
        contribution = kl_kernel(predictions, targets)  # (B, N) float32 tensor
        loss = contribution.sum() / B  # batchmean reduction
        return loss