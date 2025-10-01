"""
Problem Name: 30_Gemm_GroupNorm_Hardtanh
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.09 runtime_stats={'mean': 0.09, 'std': 0.00145, 'min': 0.0876, 'max': 0.0964, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0829, 'std': 0.00339, 'min': 0.0793, 'max': 0.099, 'num_trials': 100}, 'speedup_ratio': 0.921}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------- GEMM (Linear) Kernel --------------------- #
def _make_gemm_kernel(
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
            # Shared and local tiles
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Clear accumulator
            T.clear(C_local)

            # Main reduction loop (K dimension)
            for ko in T.Pipelined(T.ceildiv(in_features, block_K), num_stages=2):
                T.copy(
                    X[by * block_M, ko * block_K],
                    A_shared,
                )
                T.copy(
                    W[bx * block_N, ko * block_K],
                    W_shared,
                )
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            # Write results with bias addition
            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                if (global_m < batch_size) and (global_n < out_features):
                    val = C_local[i, j] + T.Cast(accum_dtype, B[global_n])
                    Out[global_m, global_n] = T.Cast(dtype, val)

    return linear_kernel


# --------------------- GroupNorm + HardTanh Kernel --------------------- #
def _make_groupnorm_kernel(
    batch_size: int,
    num_channels: int,
    num_groups: int,
    eps: float,
    hardtanh_min: float,
    hardtanh_max: float,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    group_size = num_channels // num_groups
    eps_const = eps
    min_const = hardtanh_min
    max_const = hardtanh_max

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def groupnorm_kernel(
        X: T.Tensor((batch_size, num_channels), dtype),
        gamma: T.Tensor((num_channels,), dtype),
        beta: T.Tensor((num_channels,), dtype),
        Y: T.Tensor((batch_size, num_channels), dtype),
    ):
        with T.Kernel(batch_size, threads=1) as bx:
            for g in range(num_groups):
                # Local accumulators for mean and variance
                mean_val = T.alloc_local((1,), accum_dtype)
                var_val = T.alloc_local((1,), accum_dtype)
                mean_val[0] = T.Cast(accum_dtype, 0)
                var_val[0] = T.Cast(accum_dtype, 0)

                # First pass: compute mean and variance
                for c in range(group_size):
                    ch_idx = g * group_size + c
                    v = T.Cast(accum_dtype, X[bx, ch_idx])
                    mean_val[0] += v
                    var_val[0] += v * v

                denom = T.Cast(accum_dtype, group_size)
                mean_val[0] = mean_val[0] / denom
                var_val[0] = var_val[0] / denom - mean_val[0] * mean_val[0]
                inv_std = T.Cast(accum_dtype, 1.0) / T.sqrt(
                    var_val[0] + T.Cast(accum_dtype, eps_const)
                )

                # Second pass: normalize, scale, bias, hardtanh
                for c in range(group_size):
                    ch_idx = g * group_size + c
                    v = T.Cast(accum_dtype, X[bx, ch_idx])
                    v = (v - mean_val[0]) * inv_std
                    v = (
                        v * T.Cast(accum_dtype, gamma[ch_idx])
                        + T.Cast(accum_dtype, beta[ch_idx])
                    )
                    v = T.min(
                        T.max(v, T.Cast(accum_dtype, min_const)),
                        T.Cast(accum_dtype, max_const),
                    )
                    Y[bx, ch_idx] = T.Cast(dtype, v)

    return groupnorm_kernel


# --------------------- PyTorch Wrapper Module --------------------- #
class ModelNew(nn.Module):
    """
    Optimized model using TileLang kernels for:
        - Linear (GEMM) with bias
        - Group Normalization
        - HardTanh activation
    """

    def __init__(self, in_features, out_features, num_groups, hardtanh_min, hardtanh_max):
        super().__init__()

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_groups = int(num_groups)
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        self.eps = 1e-5

        # Linear parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # GroupNorm parameters
        self.gn_weight = nn.Parameter(torch.ones(out_features))
        self.gn_bias = nn.Parameter(torch.zeros(out_features))

        # Kernel caches
        self._gemm_kernels = {}
        self._gn_kernels = {}

    # -------- Kernel Getters -------- #
    def _get_gemm_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._gemm_kernels:
            self._gemm_kernels[key] = _make_gemm_kernel(
                batch_size,
                self.in_features,
                self.out_features,
            )
        return self._gemm_kernels[key]

    def _get_gn_kernel(self, batch_size: int, dtype: str = "float16"):
        key = (batch_size, dtype)
        if key not in self._gn_kernels:
            self._gn_kernels[key] = _make_groupnorm_kernel(
                batch_size,
                self.out_features,
                self.num_groups,
                self.eps,
                self.hardtanh_min,
                self.hardtanh_max,
            )
        return self._gn_kernels[key]

    # -------- Forward Pass -------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move tensors to CUDA in fp16 for compute
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()
        gamma_fp16 = self.gn_weight.to(device="cuda", dtype=torch.float16).contiguous()
        beta_fp16 = self.gn_bias.to(device="cuda", dtype=torch.float16).contiguous()

        batch_size = x_fp16.shape[0]

        # GEMM
        gemm_kernel = self._get_gemm_kernel(batch_size)
        y_fp16 = gemm_kernel(x_fp16, w_fp16, b_fp16)

        # GroupNorm + HardTanh
        gn_kernel = self._get_gn_kernel(batch_size)
        out_fp16 = gn_kernel(y_fp16, gamma_fp16, beta_fp16)

        return out_fp16.to(orig_dtype)