"""
Problem Name: 68_Matmul_Sum_HardSwish_LogSumExp_ResidualAdd_Hardtanh
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0747 runtime_stats={'mean': 0.0747, 'std': 0.0166, 'min': 0.0642, 'max': 0.209, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.151, 'std': 0.0313, 'min': 0.131, 'max': 0.426, 'num_trials': 100}, 'speedup_ratio': 2.02}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------- #
#                     Kernel-1 :   S[k] = Σᵢ Y[i,k]                     #
# --------------------------------------------------------------------- #
def _build_col_sum_kernel(M: int, K: int, dtype: str = "float16", accum_dtype: str = "float32"):
    threads = 128  # per block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        Y:   T.Tensor((M, K), dtype),          # input
        Out: T.Tensor((K,),    dtype),         # auto-allocated
    ):
        with T.Kernel(K, threads=threads) as bk:        # one column per block
            tid = T.get_thread_binding(0)

            # ---------- partial reduction over rows -------------------- #
            part = T.alloc_local((1,), accum_dtype)
            T.clear(part)

            m_tiles = T.ceildiv(M, threads)
            for mt in T.serial(m_tiles):
                m = mt * threads + tid
                if m < M:
                    part[0] += Y[m, bk].astype(accum_dtype)

            # ---------- intra-block reduction -------------------------- #
            total = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, c: a + c, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        part[0],
                        True,
                        total[0],
                        tid,
                        dtype="handle",
                    )
                )

            if tid == 0:
                Out[bk] = T.Cast(dtype, total[0])

    return kernel


# --------------------------------------------------------------------- #
#     Kernel-2 :  dot  +  HardSwish  +  residual  +  HardTanh           #
# --------------------------------------------------------------------- #
def _build_dot_act_kernel(M: int, K: int, dtype: str = "float16", accum_dtype: str = "float32"):
    threads = 128

    three  = T.Cast(accum_dtype, 3.0)
    six    = T.Cast(accum_dtype, 6.0)
    zero   = T.Cast(accum_dtype, 0.0)
    inv6   = T.Cast(accum_dtype, 1.0 / 6.0)
    lo_ht  = T.Cast(accum_dtype, -0.5)
    hi_ht  = T.Cast(accum_dtype,  0.5)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:    T.Tensor((M, K), dtype),         # (batch, feature)
        S:    T.Tensor((K,),    dtype),        # summed Y
        R:    T.Tensor((M,),    dtype),        # residual noise
        Out:  T.Tensor((M,),    dtype),        # auto-allocated
    ):
        with T.Kernel(M, threads=threads) as bm:        # one row per block
            tid = T.get_thread_binding(0)

            # ---------- partial dot product ---------------------------- #
            part = T.alloc_local((1,), accum_dtype)
            T.clear(part)

            k_tiles = T.ceildiv(K, threads)
            for kt in T.serial(k_tiles):
                k = kt * threads + tid
                if k < K:
                    part[0] += (
                        X[bm, k].astype(accum_dtype)
                        * S[k].astype(accum_dtype)
                    )

            # ---------- reduce to a scalar ----------------------------- #
            dot = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, c: a + c, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        part[0],
                        True,
                        dot[0],
                        tid,
                        dtype="handle",
                    )
                )

            if tid == 0:
                v = dot[0]

                # HardSwish :  v * relu6(v+3)/6
                t = v + three
                t = T.min(six, T.max(zero, t))
                v = v * t * inv6

                # + residual
                v += R[bm].astype(accum_dtype)

                # HardTanh [-0.5,0.5]
                v = T.max(lo_ht, v)
                v = T.min(hi_ht, v)

                Out[bm] = T.Cast(dtype, v)

    return kernel


# --------------------------------------------------------------------- #
#                           PyTorch wrapper                             #
# --------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Re-implementation of the reference model using two TileLang kernels:
        1) column-wise sum of Y
        2) per-row dot + HardSwish + random residual + HardTanh
    """

    def __init__(self, feature_size: int, hidden_size: int):
        super().__init__()
        self.feature_size = int(feature_size)

        # kernel caches
        self._sum_kern_cache: Dict[Tuple[int, torch.dtype], callable] = {}
        self._dot_kern_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    # --------------------------------------------------------------- #
    def _get_sum_kernel(self, M: int, dtype: torch.dtype):
        key = (M, dtype)
        if key not in self._sum_kern_cache:
            self._sum_kern_cache[key] = _build_col_sum_kernel(
                M, self.feature_size, dtype=str(dtype).split(".")[-1]
            )
        return self._sum_kern_cache[key]

    def _get_dot_kernel(self, M: int, dtype: torch.dtype):
        key = (M, dtype)
        if key not in self._dot_kern_cache:
            self._dot_kern_cache[key] = _build_dot_act_kernel(
                M, self.feature_size, dtype=str(dtype).split(".")[-1]
            )
        return self._dot_kern_cache[key]

    # --------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = torch.device("cuda")

        # ---- prepare tensors ---------------------------------------
        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        y_f16 = y.to(device=device, dtype=torch.float16).contiguous()

        B, _ = x_f16.shape

        # ---- kernel-1 :  column sum of Y ----------------------------
        sum_kernel = self._get_sum_kernel(B, x_f16.dtype)
        s_f16 = sum_kernel(y_f16)                       # (feature_size,)

        # ---- random residual (identical to reference) --------------
        noise_f16 = (0.1 * torch.randn(B, device=device, dtype=torch.float16))

        # ---- kernel-2 :  dot + activations --------------------------
        dot_kernel = self._get_dot_kernel(B, x_f16.dtype)
        out_f16 = dot_kernel(x_f16, s_f16, noise_f16)   # (B,)

        return out_f16.unsqueeze(1).to(orig_dtype)