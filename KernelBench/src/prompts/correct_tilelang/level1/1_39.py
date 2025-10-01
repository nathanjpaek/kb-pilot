"""
Problem Name: 39_L2Norm_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0959 runtime_stats={'mean': 0.0959, 'std': 0.0352, 'min': 0.0807, 'max': 0.295, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.034, 'std': 0.0241, 'min': 0.0213, 'max': 0.142, 'num_trials': 100}, 'speedup_ratio': 0.355}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def l2norm_kernel_factory(M, N, block_M=16, block_K=256, dtype="float16", accum_dtype="float"):
    """
    Generates a TileLang kernel that performs per-row L2 normalisation for
    an (M, N) matrix.
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def l2norm(
        A: T.Tensor((M, N), dtype),
        Out: T.Tensor((M, N), dtype),
    ):
        num_k_step = T.ceildiv(N, block_K)

        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            # Shared buffers and fragments
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            acc_sq = T.alloc_fragment((block_M, block_K), accum_dtype)
            row_sum = T.alloc_fragment((block_M,), accum_dtype)

            # --- First pass: accumulate sum of squares ------------------------
            T.clear(acc_sq)
            for k in range(num_k_step):
                # Load a tile of A into shared memory with boundary checks
                for i, j in T.Parallel(block_M, block_K):
                    gi = bx * block_M + i
                    gj = k * block_K + j
                    inb = (gi < M) and (gj < N)
                    A_shared[i, j] = T.if_then_else(
                        inb,
                        A[gi, gj],
                        T.Cast(dtype, 0),
                    )

                # Square and accumulate
                for i, j in T.Parallel(block_M, block_K):
                    val = T.Cast(accum_dtype, A_shared[i, j])
                    acc_sq[i, j] += val * val

            # Reduce across K dimension to get per-row sum of squares
            T.reduce_sum(acc_sq, row_sum, dim=1)

            # Compute 1 / sqrt(sum)  (inverse L2 norm)
            for i in T.Parallel(block_M):
                row_sum[i] = T.rsqrt(row_sum[i])

            # --- Second pass: apply normalisation and write output ------------
            for k in range(num_k_step):
                # Reload the tile
                for i, j in T.Parallel(block_M, block_K):
                    gi = bx * block_M + i
                    gj = k * block_K + j
                    inb = (gi < M) and (gj < N)
                    A_shared[i, j] = T.if_then_else(
                        inb,
                        A[gi, gj],
                        T.Cast(dtype, 0),
                    )

                # Multiply by inverse norm and store
                for i, j in T.Parallel(block_M, block_K):
                    gi = bx * block_M + i
                    gj = k * block_K + j
                    if (gi < M) and (gj < N):
                        Out[gi, gj] = T.Cast(
                            dtype,
                            T.Cast(accum_dtype, A_shared[i, j]) * row_sum[i],
                        )

    return l2norm


class ModelNew(nn.Module):
    """
    Optimised L2 normalisation module using TileLang.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Cache compiled kernels keyed by (M, N, dtype)
        self._cached_kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TileLang currently targets CUDA float16 I/O with float32 accumulation
        x_half = x.to(device="cuda", dtype=torch.float16)

        M, N = x_half.shape
        key = (M, N, x_half.dtype)

        # Compile (or retrieve) kernel for current shape
        if key not in self._cached_kernels:
            kernel = l2norm_kernel_factory(M, N)
            self._cached_kernels[key] = kernel
        else:
            kernel = self._cached_kernels[key]

        # Invoke kernel and cast back to original dtype
        y_half = kernel(x_half)
        return y_half.to(x.dtype)