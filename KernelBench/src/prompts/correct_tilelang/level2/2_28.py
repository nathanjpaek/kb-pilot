"""
Problem Name: 28_BMM_InstanceNorm_Sum_ResidualAdd_Multiply
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0642 runtime_stats={'mean': 0.0642, 'std': 0.00184, 'min': 0.0614, 'max': 0.0705, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.106, 'std': 0.00403, 'min': 0.101, 'max': 0.129, 'num_trials': 100}, 'speedup_ratio': 1.65}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _make_linear_kernel(
    batch_size: int,
    in_features: int,
    out_features: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        X: T.Tensor((batch_size, in_features), dtype),
        W: T.Tensor((out_features, in_features), dtype),
        B: T.Tensor((out_features,), dtype),
        Out: T.Tensor((batch_size, out_features), dtype),
    ):
        with T.Kernel(
            T.ceildiv(out_features, block_N),
            T.ceildiv(batch_size, block_M),
            threads=128,
        ) as (bx, by):
            A_sh = T.alloc_shared((block_M, block_K), dtype)
            W_sh = T.alloc_shared((block_N, block_K), dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_loc)

            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_sh)
                T.copy(W[bx * block_N, ko * block_K], W_sh)
                T.gemm(A_sh, W_sh, C_loc, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < batch_size) and (gn < out_features):
                    val = C_loc[i, j] + B[gn].astype(accum_dtype)
                    Out[gm, gn] = T.Cast(dtype, val)

    return linear_kernel


def _make_norm_res_kernel(
    batch_size: int,
    features: int,
    eps_val: float,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def norm_res_kernel(
        Z: T.Tensor((batch_size, features), dtype),
        Y: T.Tensor((batch_size, features), dtype),
        Out: T.Tensor((batch_size, features), dtype),
    ):
        eps_const = T.Cast(accum_dtype, eps_val)
        with T.Kernel(batch_size, threads=1) as bx:
            mean = T.alloc_local((1,), accum_dtype)
            var = T.alloc_local((1,), accum_dtype)

            mean[0] = T.Cast(accum_dtype, 0)
            var[0] = T.Cast(accum_dtype, 0)

            for j in range(features):
                v = Z[bx, j].astype(accum_dtype)
                mean[0] += v
                var[0] += v * v

            inv_feat = T.Cast(accum_dtype, 1.0 / features)
            mean[0] = mean[0] * inv_feat
            var[0] = var[0] * inv_feat - mean[0] * mean[0]
            inv_std = T.rsqrt(var[0] + eps_const)

            for j in range(features):
                z_val = Z[bx, j].astype(accum_dtype)
                y_val = Y[bx, j].astype(accum_dtype)
                norm = (z_val - mean[0]) * inv_std
                res = (norm + y_val) * y_val
                Out[bx, j] = T.Cast(dtype, res)

    return norm_res_kernel


class ModelNew(nn.Module):
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.eps = float(eps)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self._linear_kernels = {}
        self._norm_kernels = {}

    def _get_linear_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._linear_kernels:
            self._linear_kernels[key] = _make_linear_kernel(
                batch_size,
                self.in_features,
                self.out_features,
                dtype=dtype,
            )
        return self._linear_kernels[key]

    def _get_norm_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype, self.eps)
        if key not in self._norm_kernels:
            self._norm_kernels[key] = _make_norm_res_kernel(
                batch_size,
                self.out_features,
                self.eps,
                dtype=dtype,
            )
        return self._norm_kernels[key]

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        y_fp16 = y.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        batch_size = x_fp16.shape[0]

        linear_kernel = self._get_linear_kernel(batch_size)
        z_fp16 = linear_kernel(x_fp16, w_fp16, b_fp16)

        norm_kernel = self._get_norm_kernel(batch_size)
        out_fp16 = norm_kernel(z_fp16, y_fp16)

        return out_fp16