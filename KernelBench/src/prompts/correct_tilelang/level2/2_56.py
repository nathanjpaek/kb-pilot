"""
Problem Name: 56_Matmul_Sigmoid_Sum
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0411 runtime_stats={'mean': 0.0411, 'std': 0.00271, 'min': 0.0378, 'max': 0.0517, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0583, 'std': 0.0173, 'min': 0.0474, 'max': 0.214, 'num_trials': 100}, 'speedup_ratio': 1.42}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_kernel(
    batch_size: int,
    in_features: int,
    hidden_size: int,
    block_M: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_linear_sigmoid_sum(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((hidden_size, in_features), dtype),
        B: T.Tensor((hidden_size,), dtype),
        Out: T.Tensor((batch_size, 1), dtype),  # auto-allocated (out_idx = -1)
    ):
        one_f32 = T.Cast(accum_dtype, 1.0)
        with T.Kernel(T.ceildiv(batch_size, block_M), threads=block_M) as blk:
            tx = T.get_thread_binding(0)
            m = blk * block_M + tx

            if m < batch_size:
                acc_vec = T.alloc_local((hidden_size,), accum_dtype)
                row_sum = T.alloc_local((1,), accum_dtype)

                # initialize accumulators
                for j in range(hidden_size):
                    acc_vec[j] = 0.0
                row_sum[0] = 0.0

                # GEMM row computation
                for k in range(in_features):
                    x_val = T.Cast(accum_dtype, X[m, k])
                    for j in range(hidden_size):
                        w_val = T.Cast(accum_dtype, W[j, k])
                        acc_vec[j] += x_val * w_val

                # bias add, sigmoid, accumulate row sum
                for j in range(hidden_size):
                    val = acc_vec[j] + T.Cast(accum_dtype, B[j])
                    sig = one_f32 / (one_f32 + T.exp(-val))
                    row_sum[0] += sig

                Out[m, 0] = T.Cast(dtype, row_sum[0])

    return fused_linear_sigmoid_sum


class ModelNew(nn.Module):
    """
    Optimized model using a fused TileLang kernel for
    Linear → Sigmoid → Row-wise Sum
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Parameters matching nn.Linear initialization
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(input_size)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache keyed by (batch_size, dtype)
        self._kernels = {}

    def _get_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_kernel(
                batch_size,
                self.input_size,
                self.hidden_size,
                dtype=dtype,
            )
        return self._kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(x_fp16.shape[0])
        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        return out_fp16.to(x.dtype)