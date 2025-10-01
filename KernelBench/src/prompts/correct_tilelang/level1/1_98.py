"""
Problem Name: 98_KLDivLoss
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0494 runtime_stats={'mean': 0.0494, 'std': 0.00162, 'min': 0.048, 'max': 0.0629, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0501, 'std': 0.00222, 'min': 0.0475, 'max': 0.0647, 'num_trials': 100}, 'speedup_ratio': 1.01}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def build_kl_kernel(
    B: int,
    F: int,
    block_M: int = 128,
    block_K: int = 64,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    TileLang kernel factory for KL divergence (per-sample).
    Computes  sum_j  tgt * (log tgt – log pred)  for each batch row.
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        Pred: T.Tensor((B, F), in_dtype),
        Tgt: T.Tensor((B, F), in_dtype),
        Out: T.Tensor((B,), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(B, block_M), threads=256) as bx:
            # Shared tiles
            P_s = T.alloc_shared((block_M, block_K), in_dtype)
            T_s = T.alloc_shared((block_M, block_K), in_dtype)

            # Per-sample accumulator
            kl_loc = T.alloc_fragment((block_M,), accum_dtype)
            T.clear(kl_loc)

            k_tiles = T.ceildiv(F, block_K)
            eps = T.Cast(accum_dtype, 1e-8)

            for ko in T.Pipelined(k_tiles, num_stages=2):
                # Tile origin indices
                row_off = bx * block_M
                col_off = ko * block_K

                # Load current tiles
                T.copy(Pred[row_off, col_off], P_s)
                T.copy(Tgt[row_off, col_off], T_s)

                # Parallel accumulation over the tile
                for i, j in T.Parallel(block_M, block_K):
                    # Bounds check inside Parallel
                    gi = row_off + i
                    gj = col_off + j
                    if (gi < B) and (gj < F):
                        p_val = T.Cast(accum_dtype, P_s[i, j])
                        t_val = T.Cast(accum_dtype, T_s[i, j])

                        # KL contribution:  t * (log t - log p)
                        kl_loc[i] += t_val * (
                            T.log(t_val + eps) - T.log(p_val + eps)
                        )

            # Write per-sample result
            for i in T.Parallel(block_M):
                gi = bx * block_M + i
                if gi < B:
                    Out[gi] = kl_loc[i]

    return kernel


class ModelNew(nn.Module):
    """
    TileLang-accelerated KL-Divergence (reduction='batchmean')
    """

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

    def _get_kernel(self, B: int, F: int, dtype: torch.dtype):
        key = (B, F, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_kl_kernel(B, F)
        return self._kernel_cache[key]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # Move inputs to CUDA fp16 (I/O dtype) and make contiguous
        preds_fp16 = predictions.to(device="cuda", dtype=torch.float16).contiguous()
        targs_fp16 = targets.to(device="cuda", dtype=torch.float16).contiguous()

        B, F = preds_fp16.shape
        kernel = self._get_kernel(B, F, preds_fp16.dtype)

        per_sample_kl = kernel(preds_fp16, targs_fp16)  # fp32 tensor of shape (B,)

        # batchmean reduction: mean over batch dimension
        return per_sample_kl.mean()