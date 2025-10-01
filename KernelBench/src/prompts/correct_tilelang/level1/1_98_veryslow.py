"""
Problem Name: 98_KLDivLoss
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0684 runtime_stats={'mean': 0.0684, 'std': 0.00261, 'min': 0.0652, 'max': 0.08, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0549, 'std': 0.00284, 'min': 0.0521, 'max': 0.0672, 'num_trials': 100}, 'speedup_ratio': 0.803}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_kl_kernel(
    B,
    N,
    block_M: int = 32,
    block_N: int = 128,
    dtype: str = "float32",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kl_div_kernel(
        P: T.Tensor((B, N), dtype),  # predictions (probabilities)
        Q: T.Tensor((B, N), dtype),  # targets (probabilities)
        Out: T.Tensor((B,), accum_dtype),  # per-sample KL result
    ):
        with T.Kernel(T.ceildiv(B, block_M), threads=128) as by:
            P_sh = T.alloc_shared((block_M, block_N), dtype)
            Q_sh = T.alloc_shared((block_M, block_N), dtype)
            acc = T.alloc_fragment((block_M,), accum_dtype)

            T.clear(acc)

            num_steps = T.ceildiv(N, block_N)
            for k in T.Pipelined(num_steps, num_stages=3):
                T.copy(P[by * block_M, k * block_N], P_sh)
                T.copy(Q[by * block_M, k * block_N], Q_sh)

                for i, j in T.Parallel(block_M, block_N):
                    g_row = by * block_M + i
                    g_col = k * block_N + j
                    if (g_row < B) and (g_col < N):
                        p_val = P_sh[i, j]
                        q_val = Q_sh[i, j]
                        acc[i] += q_val * (T.log(q_val) - T.log(p_val))

            for i in T.Parallel(block_M):
                g_row = by * block_M + i
                if g_row < B:
                    Out[g_row] = acc[i]

    return kl_div_kernel


class ModelNew(nn.Module):
    """
    Optimized model computing KL divergence between two distributions using TileLang.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        self._kernel_cache = {}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = predictions.to(device="cuda", dtype=torch.float32)
        targets = targets.to(device="cuda", dtype=torch.float32)

        B, N = predictions.shape
        cache_key = (B, N, predictions.dtype)

        if cache_key not in self._kernel_cache:
            self._kernel_cache[cache_key] = build_kl_kernel(B, N)

        kl_kernel = self._kernel_cache[cache_key]
        per_sample = kl_kernel(predictions, targets)  # (B,) tensor
        return per_sample.sum() / B