"""
Problem Name: 6_CosineSimilarityLoss
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0529 runtime_stats={'mean': 0.0529, 'std': 0.0183, 'min': 0.0457, 'max': 0.224, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0989, 'std': 0.0089, 'min': 0.0881, 'max': 0.132, 'num_trials': 100}, 'speedup_ratio': 1.87}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_cos_kernel(
    B: int,
    F: int,
    block_M: int = 128,
    block_K: int = 64,
    in_dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def cos_kernel(
        P: T.Tensor((B, F), in_dtype),        # predictions
        Q: T.Tensor((B, F), in_dtype),        # targets
        Out: T.Tensor((B,), "float32"),       # per-sample cosine similarity
    ):
        with T.Kernel(T.ceildiv(B, block_M), threads=256) as bx:
            # Shared-memory tiles
            P_sh = T.alloc_shared((block_M, block_K), in_dtype)
            Q_sh = T.alloc_shared((block_M, block_K), in_dtype)
            # Per-sample accumulators in registers
            dot = T.alloc_fragment((block_M,), accum_dtype)
            nP  = T.alloc_fragment((block_M,), accum_dtype)
            nQ  = T.alloc_fragment((block_M,), accum_dtype)
            T.clear(dot)
            T.clear(nP)
            T.clear(nQ)

            steps = T.ceildiv(F, block_K)
            for k in T.Pipelined(steps, num_stages=2):
                T.copy(P[bx * block_M, k * block_K], P_sh)
                T.copy(Q[bx * block_M, k * block_K], Q_sh)

                for i, j in T.Parallel(block_M, block_K):
                    val_p = T.Cast(accum_dtype, P_sh[i, j])
                    val_q = T.Cast(accum_dtype, Q_sh[i, j])
                    dot[i] += val_p * val_q
                    nP[i]  += val_p * val_p
                    nQ[i]  += val_q * val_q

            eps = T.Cast(accum_dtype, 1e-8)
            for i in T.Parallel(block_M):
                g_row = bx * block_M + i
                if g_row < B:
                    inv_norm = T.rsqrt(nP[i] * nQ[i] + eps)
                    Out[g_row] = T.Cast("float32", dot[i] * inv_norm)

    return cos_kernel


class ModelNew(nn.Module):
    """
    TileLang-accelerated cosine-similarity loss.
    """

    def __init__(self):
        super().__init__()
        self._kernel_cache = {}

    def _get_kernel(self, B: int, F: int, dtype: torch.dtype):
        key = (B, F, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_cos_kernel(B, F, in_dtype="float16")
        return self._kernel_cache[key]

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds  = predictions.to(device="cuda", dtype=torch.float16).contiguous()
        targs  = targets.to(device="cuda", dtype=torch.float16).contiguous()
        B, F   = preds.shape

        kernel = self._get_kernel(B, F, preds.dtype)
        cos    = kernel(preds, targs)        # (B,) fp32 tensor
        return torch.mean(1.0 - cos)