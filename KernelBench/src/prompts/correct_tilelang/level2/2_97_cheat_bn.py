"""
Problem Name: 97_Matmul_BatchNorm_BiasAdd_Divide_Swish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.295 runtime_stats={'mean': 0.295, 'std': 0.0103, 'min': 0.279, 'max': 0.333, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.127, 'std': 0.0317, 'min': 0.116, 'max': 0.428, 'num_trials': 100}, 'speedup_ratio': 0.431}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _gemm_fused_kernel(M: int,
                       K: int,
                       N: int,
                       divide_value: float,
                       block_M: int = 128,
                       block_N: int = 128,
                       block_K: int = 32,
                       num_stages: int = 2,
                       dtype: str = "float16",
                       accum_dtype: str = "float32"):
    """
    Build a fused kernel that performs:
        Y = Swish( (X @ W^T) * scale + shift )
    scale, shift already contain BatchNorm, bias, divide value fusion in host code.
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((M, K), dtype),                 # (batch, in_features)
        WT: T.Tensor((K, N), dtype),                # (in_features, out_features)
        scale: T.Tensor((N,), accum_dtype),         # fused scale per feature
        shift: T.Tensor((N,), accum_dtype),         # fused shift per feature
        Out: T.Tensor((M, N), dtype),               # (batch, out_features)
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),                  # grid dim x
            T.ceildiv(M, block_M),                  # grid dim y
            threads=128,                            # threads per block
        ) as (bx, by):
            # Shared / fragment allocations
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear accumulators
            T.clear(C_loc)

            k_iter = T.ceildiv(K, block_K)

            # Pipelined GEMM over K dimension
            for ko in T.Pipelined(k_iter, num_stages=num_stages):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(WT[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_loc)

            # Element-wise post-processing & write-back
            for i, j in T.Parallel(block_M, block_N):
                global_row = by * block_M + i
                global_col = bx * block_N + j
                if (global_row < M) and (global_col < N):
                    val = C_loc[i, j] * scale[global_col] + shift[global_col]
                    val = val * T.sigmoid(val)
                    Out[global_row, global_col] = T.Cast(dtype, val)

    return fused


class ModelNew(nn.Module):
    """
    Optimized implementation of the reference model using TileLang.
    Includes Linear, BatchNorm1d, bias addition, division, and Swish activation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bn_eps: float = 1e-5,
        bn_momentum: float = 0.1,
        bias_shape: Tuple[int] = (1,),
        divide_value: float = 1.0,
    ):
        super().__init__()

        # ---------- Linear parameters ----------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.bias_lin = nn.Parameter(torch.empty(out_features))
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.bias_lin, -bound, bound)

        # ---------- BatchNorm parameters ----------
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        # ---------- Extra bias / scalar ----------
        self.bias_extra = nn.Parameter(torch.randn(bias_shape))
        self.divide_value = float(divide_value)

        # ---------- Kernel cache ----------
        self._kernel_cache: Dict[Tuple[int, torch.dtype], callable] = {}

    def _get_kernel(self, M: int, dtype: torch.dtype):
        key = (M, dtype)
        if key not in self._kernel_cache:
            K = self.weight.shape[1]
            N = self.weight.shape[0]
            kernel = _gemm_fused_kernel(
                M=M,
                K=K,
                N=N,
                divide_value=self.divide_value,
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, in_features) – assumed float32
        Returns:
            Tensor of shape (batch, out_features)
        """
        # Move inputs & parameters to CUDA / fp16 where appropriate
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        weight_t_fp16 = self.weight.t().contiguous().to(dtype=torch.float16, device="cuda")
        bias_lin_fp32 = self.bias_lin.to(device="cuda", dtype=torch.float32)

        M, K = x.shape
        N = self.weight.shape[0]

        # --------- BatchNorm statistics ------------
        if self.training:
            with torch.no_grad():
                # compute pre-BN activation: gemm + bias via torch for statistics
                act_tmp = (x.to(self.weight.dtype) @ self.weight.t()) + self.bias_lin
                mean = act_tmp.mean(dim=0)
                var = act_tmp.var(dim=0, unbiased=False)

                # Update running stats
                self.running_mean.mul_(1 - self.bn_momentum).add_(self.bn_momentum * mean)
                unbiased_var = var * (M / (M - 1)) if M > 1 else var
                self.running_var.mul_(1 - self.bn_momentum).add_(self.bn_momentum * unbiased_var)
        else:
            mean = self.running_mean
            var = self.running_var

        # --------- Build fused scale & shift ------------
        rstd = torch.rsqrt(var + self.bn_eps)
        scale = (self.bn_weight * rstd) / self.divide_value
        shift = (self.bn_bias - mean * self.bn_weight * rstd + self.bias_extra) / self.divide_value

        scale_fp32 = scale.to(device="cuda", dtype=torch.float32).contiguous()
        shift_fp32 = shift.to(device="cuda", dtype=torch.float32).contiguous()

        # --------- Kernel invocation ------------
        kernel = self._get_kernel(M, x_fp16.dtype)
        out_fp16 = kernel(x_fp16, weight_t_fp16, scale_fp32, shift_fp32)

        return out_fp16.to(dtype=x.dtype, device=x.device)