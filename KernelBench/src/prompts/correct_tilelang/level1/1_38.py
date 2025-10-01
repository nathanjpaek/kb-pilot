"""
Problem Name: 38_L1Norm_
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.106 runtime_stats={'mean': 0.106, 'std': 0.0066, 'min': 0.101, 'max': 0.151, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0636, 'std': 0.00765, 'min': 0.0576, 'max': 0.104, 'num_trials': 100}, 'speedup_ratio': 0.6}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """Optimized L1-normalization along dim=1 using TileLang."""

    def __init__(self):
        super(ModelNew, self).__init__()
        # cache: (M, N, dtype) -> compiled kernel
        self._cached_kernels = {}

    def _get_kernel(self, M: int, N: int, dtype: str = "float16"):
        key = (M, N, dtype)
        if key in self._cached_kernels:
            return self._cached_kernels[key]

        # Tiling parameters – chosen to balance shared-memory usage and occupancy
        block_M = 16
        block_K = 256
        accum_dtype = "float"

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def l1norm(
            A: T.Tensor((M, N), dtype),   # input
            Out: T.Tensor((M, N), dtype),  # output
        ):
            num_k_tiles = T.ceildiv(N, block_K)

            with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
                # scratch buffers
                A_s   = T.alloc_shared((block_M, block_K), dtype)
                acc   = T.alloc_fragment((block_M, block_K), accum_dtype)
                row_s = T.alloc_fragment((block_M,), accum_dtype)

                T.clear(acc)

                # ---------------- first pass: accumulate |x| -----------------
                for kt in range(num_k_tiles):
                    # load tile into shared mem (with bounds check)
                    for i, j in T.Parallel(block_M, block_K):
                        gi = bx * block_M + i
                        gj = kt * block_K + j
                        valid = (gi < M) and (gj < N)
                        A_s[i, j] = T.if_then_else(
                            valid,
                            A[gi, gj],
                            T.Cast(dtype, 0),
                        )

                    # take abs(x) and accumulate per-element in acc buffer
                    for i, j in T.Parallel(block_M, block_K):
                        v = T.Cast(accum_dtype, A_s[i, j])
                        abs_v = T.if_then_else(v >= 0.0, v, -v)
                        acc[i, j] += abs_v

                # reduce K dimension -> per-row sums
                T.reduce_sum(acc, row_s, dim=1)

                # compute inverse sums (1 / L1)
                for i in T.Parallel(block_M):
                    row_s[i] = 1.0 / row_s[i]

                # ---------------- second pass: scale and write ---------------
                for kt in range(num_k_tiles):
                    # reload tile
                    for i, j in T.Parallel(block_M, block_K):
                        gi = bx * block_M + i
                        gj = kt * block_K + j
                        valid = (gi < M) and (gj < N)
                        A_s[i, j] = T.if_then_else(
                            valid,
                            A[gi, gj],
                            T.Cast(dtype, 0),
                        )

                    # scale and store
                    for i, j in T.Parallel(block_M, block_K):
                        gi = bx * block_M + i
                        gj = kt * block_K + j
                        if (gi < M) and (gj < N):
                            Out[gi, gj] = T.Cast(
                                dtype,
                                T.Cast(accum_dtype, A_s[i, j]) * row_s[i],
                            )

        self._cached_kernels[key] = l1norm
        return l1norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # move to CUDA & fp16 for the TileLang kernel
        x_half = x.to(device="cuda", dtype=torch.float16).contiguous()
        M, N = x_half.shape

        # fetch / compile kernel
        kernel = self._get_kernel(M, N)

        # execute and cast back to original dtype
        y_half = kernel(x_half)
        return y_half.to(x.dtype)