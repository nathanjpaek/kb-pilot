"""
Problem Name: 97_CosineSimilarityLoss
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0527 runtime_stats={'mean': 0.0527, 'std': 0.00118, 'min': 0.0508, 'max': 0.059, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0817, 'std': 0.00228, 'min': 0.0796, 'max': 0.0922, 'num_trials': 100}, 'speedup_ratio': 1.55}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def cosine_similarity_kernel(
    B: int,
    F: int,
    block_M: int = 128,
    block_K: int = 64,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        P: T.Tensor((B, F), in_dtype),  # predictions
        Tgt: T.Tensor((B, F), in_dtype),  # targets
        Out: T.Tensor((B,), "float32"),  # cosine similarity per sample
    ):
        with T.Kernel(T.ceildiv(B, block_M), threads=256) as bx:
            P_shared = T.alloc_shared((block_M, block_K), in_dtype)
            T_shared = T.alloc_shared((block_M, block_K), in_dtype)

            dot_loc = T.alloc_fragment((block_M,), accum_dtype)
            norm_p_loc = T.alloc_fragment((block_M,), accum_dtype)
            norm_t_loc = T.alloc_fragment((block_M,), accum_dtype)

            T.clear(dot_loc)
            T.clear(norm_p_loc)
            T.clear(norm_t_loc)

            for ko in T.Pipelined(T.ceildiv(F, block_K), num_stages=2):
                T.copy(P[bx * block_M, ko * block_K], P_shared)
                T.copy(Tgt[bx * block_M, ko * block_K], T_shared)

                for i, j in T.Parallel(block_M, block_K):
                    val_p = T.Cast(accum_dtype, P_shared[i, j])
                    val_t = T.Cast(accum_dtype, T_shared[i, j])
                    dot_loc[i] += val_p * val_t
                    norm_p_loc[i] += val_p * val_p
                    norm_t_loc[i] += val_t * val_t

            eps = T.Cast(accum_dtype, 1e-8)
            for i in T.Parallel(block_M):
                r = bx * block_M + i
                if r < B:
                    inv_norm = T.rsqrt(norm_p_loc[i] * norm_t_loc[i] + eps)
                    cos_val = dot_loc[i] * inv_norm
                    Out[r] = T.Cast("float32", cos_val)

    return kernel


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._cached_kernels = {}

    def _get_kernel(self, batch: int, features: int, dtype: torch.dtype):
        key = (batch, features, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = cosine_similarity_kernel(
                batch, features, in_dtype="float16", accum_dtype="float"
            )
        return self._cached_kernels[key]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        device = torch.device("cuda")
        predictions = predictions.to(device=device, dtype=torch.float16)
        targets = targets.to(device=device, dtype=torch.float16)

        B, F = predictions.shape
        kernel = self._get_kernel(B, F, predictions.dtype)
        cos_sim = kernel(predictions, targets)
        return torch.mean(1.0 - cos_sim)