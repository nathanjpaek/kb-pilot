"""
Problem Name: 51_Gemm_Subtract_GlobalAvgPool_LogSumExp_GELU_ResidualAdd
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.31 runtime_stats={'mean': 0.31, 'std': 0.00245, 'min': 0.306, 'max': 0.319, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.141, 'std': 0.00401, 'min': 0.136, 'max': 0.16, 'num_trials': 100}, 'speedup_ratio': 0.455}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _make_fused_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    block_threads: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    inv_out_feat = 1.0 / out_features
    sqrt_half = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_kernel(
        X: T.Tensor((batch_size, in_features), dtype),          # original input
        Wsum: T.Tensor((in_features,), dtype),                 # sum_j W[j, :]
        offset: T.Tensor((1,), dtype),                         # bias_sum - subtract_sum
        Out: T.Tensor((batch_size, in_features), dtype),
    ):
        half_const = T.Cast(accum_dtype, 0.5)
        one_const = T.Cast(accum_dtype, 1.0)
        inv_out = T.Cast(accum_dtype, inv_out_feat)
        sqrt_half_const = T.Cast(accum_dtype, sqrt_half)

        grid = T.ceildiv(batch_size, block_threads)

        with T.Kernel(grid, threads=block_threads) as bx:
            tx = T.get_thread_binding(0)
            row = bx * block_threads + tx

            if row < batch_size:
                acc = T.Cast(accum_dtype, 0)

                # Dot product X_row · Wsum
                for k in range(in_features):
                    acc += (
                        T.Cast(accum_dtype, X[row, k])
                        * T.Cast(accum_dtype, Wsum[k])
                    )

                acc = (acc + T.Cast(accum_dtype, offset[0])) * inv_out

                gelu_val = half_const * acc * (
                    one_const + T.erf(acc * sqrt_half_const)
                )

                gelu_fp16 = T.Cast(dtype, gelu_val)

                # Broadcast addition with original X
                for k in range(in_features):
                    Out[row, k] = X[row, k] + gelu_fp16

    return fused_kernel


class ModelNew(nn.Module):
    """
    Optimized model using TileLang that fuses:
    Linear -> Subtract -> GlobalAvgPool -> LogSumExp -> GELU -> ResidualAdd
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias

        # Parameters mirroring original PyTorch layer initializations
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            with torch.no_grad():
                bound = 1 / math.sqrt(in_features)
                torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # subtract parameter
        self.subtract = nn.Parameter(torch.randn(out_features))

        # Kernel cache
        self._kernel_cache = {}

    def _get_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _make_fused_kernel(
                batch_size, self.in_features, self.out_features
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        # Prepare dynamic tensors
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        w_sum = w_fp16.sum(dim=0, keepdim=False).contiguous()

        if self.bias is not None:
            bias_sum = self.bias.to(device="cuda", dtype=torch.float16).sum()
        else:
            bias_sum = torch.tensor(0.0, dtype=torch.float16, device="cuda")

        sub_sum = self.subtract.to(device="cuda", dtype=torch.float16).sum()
        offset_val = (bias_sum - sub_sum).unsqueeze(0).contiguous()

        kernel = self._get_kernel(x_fp16.shape[0])
        out_fp16 = kernel(x_fp16, w_sum, offset_val)

        return out_fp16.to(orig_dtype)