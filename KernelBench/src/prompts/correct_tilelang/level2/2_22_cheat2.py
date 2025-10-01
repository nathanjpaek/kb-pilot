"""
Problem Name: 22_Matmul_Scale_ResidualAdd_Clamp_LogSumExp_Mish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.269 runtime_stats={'mean': 0.269, 'std': 0.0786, 'min': 0.128, 'max': 0.401, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.309, 'std': 0.0733, 'min': 0.132, 'max': 0.441, 'num_trials': 100}, 'speedup_ratio': 1.15}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def fused_linear_kernel(
    M: int,
    in_features: int,
    out_features: int,
    factor: float,
    clamp_min: float,
    clamp_max: float,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((M, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Y: T.Tensor((M, out_features), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_features, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            Bias_shared = T.alloc_shared((block_N,), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            Out_shared = T.alloc_shared((block_M, block_N), dtype)

            T.clear(C_local)

            # Load bias for this block
            for j in T.Parallel(block_N):
                gcol = bx * block_N + j
                Bias_shared[j] = T.if_then_else(
                    gcol < out_features, B[gcol], T.Cast(dtype, 0)
                )

            # K loop
            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            # Post-processing
            for i, j in T.Parallel(block_M, block_N):
                grow = by * block_M + i
                gcol = bx * block_N + j
                if (grow < M) and (gcol < out_features):
                    val = C_local[i, j] + Bias_shared[j]
                    val *= T.Cast(accum_dtype, factor)
                    val = T.min(T.max(val, clamp_min), clamp_max)
                    Out_shared[i, j] = T.Cast(dtype, val)

            T.copy(Out_shared, Y[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        scale_factor: float,
        clamp_min: float,
        clamp_max: float,
    ):
        super(ModelNew, self).__init__()
        self.in_features = input_size
        self.out_features = hidden_size
        self.scale_factor = scale_factor
        self.factor = 2.0 * scale_factor  # accounts for x = x*scale; x = x + x
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Parameters
        self.weight = nn.Parameter(torch.empty(hidden_size, input_size))
        self.bias = nn.Parameter(torch.empty(hidden_size))

        # Initialization identical to nn.Linear defaults
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(input_size)
        nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache
        self._kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        W = self.weight.to(device="cuda", dtype=torch.float16)
        B = self.bias.to(device="cuda", dtype=torch.float16)

        M = x.shape[0]
        key = (M, x.dtype)

        if key not in self._kernels:
            self._kernels[key] = fused_linear_kernel(
                M,
                self.in_features,
                self.out_features,
                self.factor,
                self.clamp_min,
                self.clamp_max,
            )

        y = self._kernels[key](x, W, B).to(torch.float32)

        # Remaining operations in PyTorch
        y = torch.logsumexp(y, dim=1, keepdim=True)

        softplus = torch.log1p(torch.exp(y))
        mish = y * torch.tanh(softplus)
        y = y * mish
        return y