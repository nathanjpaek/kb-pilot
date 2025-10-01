"""
Problem Name: 38_Matmul_Clamp_Max_GlobalAvgPool_Sum
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=18.4 runtime_stats={'mean': 18.4, 'std': 0.0338, 'min': 18.3, 'max': 18.4, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.123, 'std': 0.0121, 'min': 0.118, 'max': 0.238, 'num_trials': 100}, 'speedup_ratio': 0.00668}}
"""

import math
from typing import Dict, Tuple

import torch
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                               Kernel factory                                #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    B: int,
    L: int,
    K: int,
    F: int,
    threads_per_block: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Computes, for every batch element b:

        S[b] = (1 / L) * sum_{l=0}^{L-1} max_f  clamp( dot( X[b,l,:], Y[b,:,f] ), -1, 1 )

    Shapes
        X : (B, L, K)   in_dtype
        Y : (B, K, F)   in_dtype
        S : (B,)        in_dtype       (auto-allocated by TileLang)
    """

    inv_L   = T.Cast(accum_dtype, 1.0 / L)
    clamp_lo = T.Cast(accum_dtype, -1.0)
    clamp_hi = T.Cast(accum_dtype,  1.0)
    neg_inf  = T.Cast(accum_dtype, -3.4e38)   # for max init

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((B, L, K), in_dtype),
        Y: T.Tensor((B, K, F), in_dtype),
        S: T.Tensor((B,),       in_dtype),          # auto-allocated
    ):
        with T.Kernel(B, threads=threads_per_block) as b:   # grid.x = B
            tid = T.get_thread_binding(0)

            partial = T.alloc_local((1,), accum_dtype)
            T.clear(partial)

            l_tiles = T.ceildiv(L, threads_per_block)
            for lt in T.serial(l_tiles):
                l = lt * threads_per_block + tid
                if l < L:
                    max_val = T.alloc_local((1,), accum_dtype)
                    max_val[0] = neg_inf

                    for f in T.serial(F):
                        dot = T.alloc_local((1,), accum_dtype)
                        T.clear(dot)
                        for k in T.serial(K):
                            dot[0] += (
                                X[b, l, k].astype(accum_dtype)
                                * Y[b, k, f].astype(accum_dtype)
                            )

                        # clamp to [-1,1]
                        dot[0] = T.max(dot[0], clamp_lo)
                        dot[0] = T.min(dot[0], clamp_hi)

                        # track maximum over f
                        max_val[0] = T.max(max_val[0], dot[0])

                    partial[0] += max_val[0]

            # ---- block reduction ----
            reduced = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, c: a + c, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        partial[0],
                        True,
                        reduced[0],
                        tid,
                        dtype="handle",
                    )
                )

            if tid == 0:
                S[b] = T.Cast(in_dtype, reduced[0] * inv_L)

    return fused


# --------------------------------------------------------------------------- #
#                           PyTorch wrapper module                            #
# --------------------------------------------------------------------------- #
class ModelNew(torch.nn.Module):
    """
    Fused implementation of:
        Matmul → clamp[-1,1] → max(feature) → global-avg-pool(seq) → sum
    Produces a (batch_size,) tensor identical to the reference Model.
    """

    def __init__(self):
        super().__init__()
        # cache : {(B,L,K,F,dtype) : compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self, B: int, L: int, K: int, F: int, dtype: torch.dtype
    ):
        key = (B, L, K, F, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(B, L, K, F)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args
            x : (B, L, K)
            y : (B, K, F)
        Returns
            (B,) tensor – same dtype as input
        """
        orig_dtype = x.dtype
        device = torch.device("cuda")

        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        y_f16 = y.to(device=device, dtype=torch.float16).contiguous()

        B, L, K = x_f16.shape
        _, _, F = y_f16.shape

        kernel = self._get_kernel(B, L, K, F, x_f16.dtype)

        out_fp16 = kernel(x_f16, y_f16)   # (B,)

        return out_fp16.to(orig_dtype)