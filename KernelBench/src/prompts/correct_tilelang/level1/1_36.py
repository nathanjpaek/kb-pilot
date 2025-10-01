"""
Problem Name: 36_RMSNorm_
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.527 runtime_stats={'mean': 0.527, 'std': 0.0011, 'min': 0.525, 'max': 0.533, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.365, 'std': 0.000895, 'min': 0.362, 'max': 0.369, 'num_trials': 100}, 'speedup_ratio': 0.693}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def rms_norm_kernel(M, N, blk_m=64, eps: float = 1e-5, dtype: str = "float16"):
    """
    Factory that returns a compiled TileLang kernel performing RMSNorm over the second
    dimension (length = N) for a 2-D tensor of shape (M, N).
    """

    @tilelang.jit(out_idx=-1)  # allocate the output tensor automatically
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),  # input
        B: T.Tensor((M, N), dtype),  # output
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=128) as bx:
            # Shared / local buffers
            A_shared = T.alloc_shared((blk_m, N), dtype)
            A_local = T.alloc_fragment((blk_m, N), dtype)
            A_pow_local = T.alloc_fragment((blk_m, N), dtype)
            A_powsum = T.alloc_fragment((blk_m,), dtype)

            # Copy a block of rows from global to shared
            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            # Move to registers
            T.copy(A_shared, A_local)

            # Square the values
            for i, j in T.Parallel(blk_m, N):
                A_pow_local[i, j] = A_local[i, j] * A_local[i, j]

            # Sum over feature dimension
            T.reduce_sum(A_pow_local, A_powsum, dim=1)

            # Compute 1 / sqrt(mean + eps)
            for i in T.Parallel(blk_m):
                A_powsum[i] = T.rsqrt(A_powsum[i] / N + eps)

            # Normalize
            for i, j in T.Parallel(blk_m, N):
                A_local[i, j] *= A_powsum[i]

            # Write back
            T.copy(A_local, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main


class ModelNew(nn.Module):
    """
    Optimized RMSNorm implementation using TileLang.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, block_m: int = 64):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = float(eps)
        self.block_m = block_m
        self._cached_kernels = {}

    def _get_kernel(self, M: int, dtype_str: str = "float16"):
        key = (M, dtype_str)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = rms_norm_kernel(
                M, self.num_features, self.block_m, self.eps, dtype_str
            )
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to input tensor `x`.
        """
        orig_dtype = x.dtype
        x = x.to(device="cuda", dtype=torch.float16)

        # Move feature dimension to the end for contiguous 2-D view
        permuted = x.permute(0, *range(2, x.ndim), 1).contiguous()
        M = permuted.numel() // self.num_features
        x_2d = permuted.view(M, self.num_features)

        # Compile / fetch kernel and run
        kernel = self._get_kernel(M)
        out_2d = kernel(x_2d)

        # Restore original shape and permute back
        out_permuted = out_2d.view(permuted.shape)
        out = out_permuted.permute(0, out_permuted.ndim - 1, *range(1, out_permuted.ndim - 1))
        return out.to(orig_dtype)