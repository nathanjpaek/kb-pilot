"""
Problem Name: 45_Gemm_Sigmoid_Sum_LogSumExp
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0639 runtime_stats={'mean': 0.0639, 'std': 0.0245, 'min': 0.0583, 'max': 0.307, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.107, 'std': 0.00353, 'min': 0.103, 'max': 0.122, 'num_trials': 100}, 'speedup_ratio': 1.67}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_fused_gemm_sigmoid_sum_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    block_M: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    threads = block_M  # one CUDA thread per row in the tile

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        RowSum: T.Tensor((batch_size,), accum_dtype),
    ):
        num_blocks = T.ceildiv(batch_size, block_M)

        with T.Kernel(num_blocks, threads=threads) as bx:
            tx = T.get_thread_binding(0)  # row id inside the block
            row_global = bx * block_M + tx

            if row_global < batch_size:
                partial_sum = T.alloc_local((1,), accum_dtype)
                partial_sum[0] = T.Cast(accum_dtype, 0)

                for j in range(out_features):
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = B[j].astype(accum_dtype)
                    for k in range(in_features):
                        acc[0] += (
                            X[row_global, k].astype(accum_dtype)
                            * W[j, k].astype(accum_dtype)
                        )
                    sig_val = T.Cast(
                        accum_dtype, 1.0
                    ) / (T.Cast(accum_dtype, 1.0) + T.exp(-acc[0]))
                    partial_sum[0] += sig_val

                RowSum[row_global] = partial_sum[0]

    return fused_kernel


def _build_logsumexp_kernel(batch_size: int, dtype: str = "float32"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def logsumexp_kernel(
        X: T.Tensor((batch_size,), dtype),
        Out: T.Tensor((1,), dtype),
    ):
        with T.Kernel(1, threads=1):
            max_val = T.alloc_local((1,), dtype)
            sum_exp = T.alloc_local((1,), dtype)

            # Compute max
            max_val[0] = X[0]
            for i in range(1, batch_size):
                max_val[0] = T.max(max_val[0], X[i])

            # Compute sum of exp(x - max)
            sum_exp[0] = T.Cast(dtype, 0)
            for i in range(batch_size):
                sum_exp[0] += T.exp(X[i] - max_val[0])

            Out[0] = T.log(sum_exp[0]) + max_val[0]

    return logsumexp_kernel


class ModelNew(nn.Module):
    """
    Optimized implementation of the original model using TileLang kernels.
    Performs: Linear → Sigmoid → row-wise Sum → LogSumExp (over batch).
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)  # unused but kept for parity

        # Linear1 parameters (used)
        self.weight1 = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias1 = nn.Parameter(torch.empty(hidden_size))

        # Linear2 parameters (initialized for completeness, unused)
        self.weight2 = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias2 = nn.Parameter(torch.empty(output_size))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))
            bound1 = 1 / math.sqrt(input_size)
            torch.nn.init.uniform_(self.bias1, -bound1, bound1)

            torch.nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))
            bound2 = 1 / math.sqrt(hidden_size)
            torch.nn.init.uniform_(self.bias2, -bound2, bound2)

        # Kernel caches
        self._sum_kernel_cache = {}
        self._logsumexp_kernel_cache = {}

    def _get_sum_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._sum_kernel_cache:
            self._sum_kernel_cache[key] = _build_fused_gemm_sigmoid_sum_kernel(
                batch_size,
                self.input_size,
                self.hidden_size,
                dtype=dtype,
                accum_dtype="float",
            )
        return self._sum_kernel_cache[key]

    def _get_logsumexp_kernel(self, batch_size: int, dtype: str = "float32"):
        key = (batch_size, dtype)
        if key not in self._logsumexp_kernel_cache:
            self._logsumexp_kernel_cache[key] = _build_logsumexp_kernel(
                batch_size, dtype=dtype
            )
        return self._logsumexp_kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        # Prepare inputs
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w1_fp16 = self.weight1.to(device="cuda", dtype=torch.float16).contiguous()
        b1_fp16 = self.bias1.to(device="cuda", dtype=torch.float16).contiguous()

        # Kernel 1: fused GEMM + Sigmoid + row-wise Sum
        sum_kernel = self._get_sum_kernel(batch_size, dtype="float16")
        row_sums_fp32 = sum_kernel(x_fp16, w1_fp16, b1_fp16)  # float32 output

        # Kernel 2: LogSumExp over batch
        lse_kernel = self._get_logsumexp_kernel(batch_size, dtype="float32")
        result_fp32 = lse_kernel(row_sums_fp32)

        return result_fp32.squeeze(0)