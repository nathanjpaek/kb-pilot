"""
Problem Name: 48_Mean_reduction_over_a_dimension
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0582 runtime_stats={'mean': 0.0582, 'std': 0.0408, 'min': 0.037, 'max': 0.326, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0198, 'std': 0.00819, 'min': 0.0158, 'max': 0.0945, 'num_trials': 100}, 'speedup_ratio': 0.34}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def reduce_mean_kernel(M, K, block_M=128, block_K=32, dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        Out: T.Tensor((M,), dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), threads=128) as bx:
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            acc_frag = T.alloc_fragment((block_M,), accum_dtype)

            T.clear(acc_frag)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[bx * block_M, ko * block_K], A_shared)
                for i, j in T.Parallel(block_M, block_K):
                    acc_frag[i] += T.Cast(accum_dtype, A_shared[i, j])

            for i in T.Parallel(block_M):
                idx = bx * block_M + i
                if idx < M:
                    Out[idx] = T.Cast(dtype, acc_frag[i] / K)

    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang to perform mean reduction along a specific dimension.
    """

    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        self._cached_kernels = {}

    def _get_kernel(self, M: int, K: int):
        key = (M, K)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = reduce_mean_kernel(M, K)
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        dim = self.dim if self.dim >= 0 else self.dim + x.dim()
        K = x.shape[dim]

        # Move reduced dimension to the end and flatten remaining ones
        order = [i for i in range(x.dim()) if i != dim] + [dim]
        x_fp16 = x.to(device="cuda", dtype=torch.float16).permute(order).contiguous()

        M = x_fp16.numel() // K
        x_flat = x_fp16.view(M, K)

        kernel = self._get_kernel(M, K)
        out_flat = kernel(x_flat)

        out_shape = x_fp16.shape[:-1]  # shape without the reduced dimension
        out = out_flat.view(out_shape)

        return out.to(orig_dtype)