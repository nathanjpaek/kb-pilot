"""
Problem Name: 4_Matrix_vector_multiplication_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.23 runtime_stats={'mean': 1.23, 'std': 0.0138, 'min': 1.22, 'max': 1.3, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0449, 'std': 0.000971, 'min': 0.0434, 'max': 0.0502, 'num_trials': 100}, 'speedup_ratio': 0.0365}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def splitk_gemv_vectorized(
    M: int,
    K: int,
    BLOCK_M: int = 128,       # rows per block   (≤1024 / REDUCE_THREADS)
    REDUCE_THREADS: int = 8,  # threads.y  (split-K factor)
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """Return a prim_func that computes C = A @ B with split-K reduction."""

    dtype_bits = 16 if dtype == "float16" else 32
    TILE_K     = 128 // dtype_bits            # 8 fp16 or 4 fp32 elements per lane
    BLOCK_K    = REDUCE_THREADS * TILE_K      # cols each split-K slice owns

    @tilelang.jit(out_idx=-1)                 # last arg (C) is the output
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, 1), dtype),
        C: T.Tensor((M, 1), dtype),           # auto-allocated by TileLang
    ):
        with T.Kernel(T.ceildiv(M, BLOCK_M),
                      threads=(BLOCK_M, REDUCE_THREADS)) as bm:

            tm = T.get_thread_binding(0)      # row id inside the tile
            tk = T.get_thread_binding(1)      # split-K thread id

            # Shared & local storage
            C_shared = T.alloc_shared((BLOCK_M,), accum_dtype)
            A_local  = T.alloc_local((TILE_K,), dtype)
            B_local  = T.alloc_local((TILE_K,), dtype)
            C_accum  = T.alloc_local((1,),      accum_dtype)

            # 1 ▸ clear shared accumulator (one thread per row)
            if tk == 0:
                C_shared[tm] = T.Cast(accum_dtype, 0)
            T.tvm_storage_sync("shared")      # barrier ─ ensures everyone sees 0

            T.clear(C_accum)

            # 2 ▸ main split-K loop
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    a_idx = bk * BLOCK_K + tk * TILE_K + k
                    m_idx = bm * BLOCK_M + tm
                    if (a_idx < K) and (m_idx < M):
                        A_local[k] = A[m_idx, a_idx]
                        B_local[k] = B[a_idx, 0]
                    else:                      # ragged tail
                        A_local[k] = 0.0
                        B_local[k] = 0.0

                for k in T.serial(TILE_K):
                    C_accum[0] += (
                        A_local[k].astype(accum_dtype)
                        * B_local[k].astype(accum_dtype)
                    )

            # 3 ▸ cooperative reduction
            T.atomic_add(C_shared[tm], C_accum[0])
            T.tvm_storage_sync("shared")      # barrier ─ wait for all atomics

            # 4 ▸ single thread writes final value
            if (tk == 0) and (bm * BLOCK_M + tm < M):
                C[bm * BLOCK_M + tm, 0] = T.Cast(dtype, C_shared[tm])

    return main

class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self._kern_cache = {}

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        assert A.dim() == 2 and B.shape == (A.shape[1], 1), "shape mismatch"

        M, K = A.shape
        key  = (M, K)
        if key not in self._kern_cache:
            self._kern_cache[key] = splitk_gemv_vectorized(M, K)

        kernel = self._kern_cache[key]

        A_fp16 = A.to(dtype=torch.float16, device="cuda").contiguous()
        B_fp16 = B.to(dtype=torch.float16, device="cuda").contiguous()

        C_fp16 = kernel(A_fp16, B_fp16)
        return C_fp16