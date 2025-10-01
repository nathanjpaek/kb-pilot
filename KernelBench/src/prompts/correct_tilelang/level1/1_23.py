"""
Problem Name: 23_Softmax
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.621 runtime_stats={'mean': 0.621, 'std': 0.0195, 'min': 0.615, 'max': 0.806, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0208, 'std': 0.00239, 'min': 0.0192, 'max': 0.0391, 'num_trials': 100}, 'speedup_ratio': 0.0335}}
"""

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """
    Optimized Softmax model using TileLang.
    """

    def __init__(self):
        super(ModelNew, self).__init__()
        # Cache compiled kernels keyed by (batch, dim, dtype)
        self._kernel_cache = {}

    def _softmax_kernel(self, B: int, N: int, dtype: str = "float32"):
        """
        Compile (or fetch from cache) a TileLang kernel that performs
        Softmax along the second dimension of a (B, N) tensor.
        """
        key = (B, N, dtype)
        if key in self._kernel_cache:
            return self._kernel_cache[key]

        @tilelang.jit(out_idx=-1)  # output tensor created at runtime
        @T.prim_func
        def softmax_kernel(
            X: T.Tensor((B, N), dtype),
            Y: T.Tensor((B, N), dtype),
        ):
            # Launch one thread-block per batch element, single thread per block
            with T.Kernel(B, 1, threads=1) as (bx, _):
                # Allocate local scalars
                max_val = T.alloc_local((1,), dtype)
                sum_val = T.alloc_local((1,), dtype)

                # First pass: find max value in the row for numerical stability
                max_val[0] = X[bx, 0]
                for j in range(1, N):
                    max_val[0] = T.max(max_val[0], X[bx, j])

                # Second pass: compute exponentials and running sum
                sum_val[0] = 0.0
                for j in range(N):
                    exp_val = T.exp(X[bx, j] - max_val[0])
                    Y[bx, j] = exp_val  # temporarily store unnormalized exp
                    sum_val[0] += exp_val

                inv_sum = 1.0 / sum_val[0]

                # Third pass: normalize
                for j in range(N):
                    Y[bx, j] = Y[bx, j] * inv_sum

        self._kernel_cache[key] = softmax_kernel
        return softmax_kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move tensor to CUDA and use float32 for better numerical accuracy
        x = x.to(device="cuda", dtype=torch.float32)

        B, N = x.shape
        kernel = self._softmax_kernel(B, N, dtype="float32")
        return kernel(x)