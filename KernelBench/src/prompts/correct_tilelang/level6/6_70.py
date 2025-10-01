"""
Problem Name: 70_Matmul_Clamp_GlobalAvgPool_Sum_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=36.0 runtime_stats={'mean': 36.0, 'std': 0.0415, 'min': 35.9, 'max': 36.1, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.167, 'std': 0.0108, 'min': 0.161, 'max': 0.265, 'num_trials': 100}, 'speedup_ratio': 0.00464}}
"""

import math
from typing import Dict, Tuple

import torch
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                         TileLang fused kernel factory                       #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    B: int,
    L: int,
    F: int,
    H: int,
    threads_per_block: int = 128,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Computes, for each batch element b:

        S[b] = (1 / L) * sum_{l,h} ReLU( dot( X[b,l,:], Y[b,:,h] ) )

    X : (B, L, F)  fp16
    Y : (B, F, H)  fp16
    S : (B,)       fp16   (allocated by TileLang, returned)
    """

    inv_L = 1.0 / L
    zero  = T.Cast(accum_dtype, 0.0)
    inv_L_c = T.Cast(accum_dtype, inv_L)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((B, L, F), in_dtype),
        Y: T.Tensor((B, F, H), in_dtype),
        S: T.Tensor((B,),        in_dtype),      # auto-allocated
    ):
        with T.Kernel(B, threads=threads_per_block) as b:
            tid = T.get_thread_binding(0)

            # ---- per-thread partial accumulator ------------------------- #
            partial = T.alloc_local((1,), accum_dtype)
            T.clear(partial)

            # Each thread iterates over rows l it owns (stride = threads)
            for l_blk in T.serial(T.ceildiv(L, threads_per_block)):
                l_idx = l_blk * threads_per_block + tid
                if l_idx < L:
                    for h in T.serial(H):
                        dot = T.alloc_local((1,), accum_dtype)
                        T.clear(dot)
                        for f in T.serial(F):
                            dot[0] += (
                                X[b, l_idx, f].astype(accum_dtype)
                                * Y[b, f, h].astype(accum_dtype)
                            )
                        # ReLU and accumulate
                        if dot[0] > zero:
                            partial[0] += dot[0]

            # ---- block-level reduction to obtain S[b] -------------------- #
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
                S[b] = T.Cast(in_dtype, reduced[0] * inv_L_c)

    return fused


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(torch.nn.Module):
    """
    Fuses:   Matmul → ReLU → GlobalAvgPool1d → Sum → (device) → max(batch)
    into a single TileLang kernel that outputs the per-batch scalar vector,
    after which only the final max across batch is done in PyTorch.
    """

    def __init__(self):
        super().__init__()
        # kernel cache : key = (B, L, F, H, dtype)
        self._kern_cache: Dict[Tuple[int, int, int, int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self,
        B: int,
        L: int,
        F: int,
        H: int,
        dtype: torch.dtype,
    ):
        key = (B, L, F, H, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(B, L, F, H)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Args:
            x : (B, L, F)
            y : (B, F, H)
        Returns:
            scalar 0-D tensor – identical to reference computation
        """
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # Move to CUDA / fp16
        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        y_f16 = y.to(device=device, dtype=torch.float16).contiguous()

        B, L, F = x_f16.shape
        _, _, H = y_f16.shape

        kernel = self._get_kernel(B, L, F, H, x_f16.dtype)

        # Kernel returns (B,) fp16 tensor of per-batch values
        s_fp16 = kernel(x_f16, y_f16)

        # Final max over batch (cheap on host)
        out = s_fp16.to(orig_dtype).max(dim=0).values
        return out