"""
Problem Name: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.148 runtime_stats={'mean': 0.148, 'std': 0.00392, 'min': 0.144, 'max': 0.168, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.142, 'std': 0.00347, 'min': 0.138, 'max': 0.16, 'num_trials': 100}, 'speedup_ratio': 0.959}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def matmul_bias(
    M: int,
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),      # (batch, in_features)
        B: T.Tensor((K, N), dtype),      # (in_features, hidden)  -- weight.T
        bias: T.Tensor((N,), dtype),     # (hidden,)
        C: T.Tensor((M, N), dtype),      # output
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Load tiles of A and B into shared memory
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                # Perform GEMM on the tiles
                T.gemm(A_shared, B_shared, C_local)

            # Add bias
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if gi < M and gj < N:
                    C_local[i, j] += T.Cast(accum_dtype, bias[gj])

            # Write back to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang for the linear layer. Remaining lightweight
    element-wise operations are performed with PyTorch.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        scale_factor: float,
        clamp_min: float,
        clamp_max: float,
    ):
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize parameters identically to nn.Linear defaults
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(input_size)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Cache compiled kernels keyed by (batch_size, dtype)
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, dtype: torch.dtype):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = matmul_bias(
                batch_size, self.hidden_size, self.input_size
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        W_t = self.weight.t().contiguous().to(device="cuda", dtype=torch.float16)
        bias_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        batch_size = x_fp16.shape[0]
        kernel = self._get_kernel(batch_size, x_fp16.dtype)

        y = kernel(x_fp16, W_t, bias_fp16)

        # Remaining element-wise operations
        y = y * self.scale_factor
        y = y + y  # residual addition
        y = torch.clamp(y, self.clamp_min, self.clamp_max)
        y = torch.logsumexp(y, dim=1, keepdim=True)
        y = y * torch.nn.functional.mish(y)
        return y