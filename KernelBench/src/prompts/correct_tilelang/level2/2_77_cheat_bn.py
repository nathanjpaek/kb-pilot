"""
Problem Name: 77_ConvTranspose3d_Scale_BatchNorm_GlobalAvgPool
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=7.39 runtime_stats={'mean': 7.39, 'std': 0.122, 'min': 7.28, 'max': 7.69, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.623, 'std': 0.0154, 'min': 0.611, 'max': 0.77, 'num_trials': 100}, 'speedup_ratio': 0.0843}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _make_deconv3d_kernel(
    N: int,
    C_in: int,
    D_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    K: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 64,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    OD = D_in + K - 1
    OH = H_in + K - 1
    OW = W_in + K - 1
    P = K - 1
    K_total = K * K * K * C_in
    M_total = N * OD * OH * OW

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv3d(
        x: T.Tensor((N, C_in, D_in, H_in, W_in), dtype),
        w_flat: T.Tensor((K_total, C_out), dtype),
        y: T.Tensor((N, C_out, OD, OH, OW), dtype),
    ):
        with T.Kernel(
            T.ceildiv(C_out, block_N),
            T.ceildiv(M_total, block_M),
            threads=128,
        ) as (bx, by):
            data_s = T.alloc_shared((block_M, block_K), dtype)
            weight_s = T.alloc_shared((block_K, block_N), dtype)
            out_f32 = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_s = T.alloc_shared((block_M, block_N), dtype)

            y_flat = T.Tensor((M_total, C_out), dtype, y.data)

            T.clear(out_f32)

            for k_iter in T.Pipelined(T.ceildiv(K_total, block_K), num_stages=num_stages):
                # im2col into data_s
                for i, j in T.Parallel(block_M, block_K):
                    m = by * block_M + i
                    k = k_iter * block_K + j

                    valid = (m < M_total) and (k < K_total)

                    n = T.if_then_else(
                        valid,
                        m // (OD * OH * OW),
                        0,
                    )
                    tmp1 = m % (OD * OH * OW)
                    od = tmp1 // (OH * OW)
                    tmp2 = tmp1 % (OH * OW)
                    oh = tmp2 // OW
                    ow = tmp2 % OW

                    ic = k % C_in
                    kw = (k // C_in) % K
                    kh = (k // (C_in * K)) % K
                    kd = (k // (C_in * K * K)) % K

                    id_ = od - kd + P
                    ih_ = oh - kh + P
                    iw_ = ow - kw + P

                    in_bound = (
                        valid
                        and (id_ >= 0)
                        and (ih_ >= 0)
                        and (iw_ >= 0)
                        and (id_ < D_in)
                        and (ih_ < H_in)
                        and (iw_ < W_in)
                    )

                    data_s[i, j] = T.if_then_else(
                        in_bound, x[n, ic, id_, ih_, iw_], T.Cast(dtype, 0)
                    )

                # copy kernel tile
                T.copy(w_flat[k_iter * block_K, bx * block_N], weight_s)

                # GEMM
                T.gemm(data_s, weight_s, out_f32)

            # store back
            T.copy(out_f32, out_s)
            T.copy(out_s, y_flat[by * block_M, bx * block_N])

    return deconv3d


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: float,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor
        self.eps = eps
        self.momentum = momentum

        # ConvTranspose3d parameters
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size ** 3
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # BatchNorm3d parameters
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var", torch.ones(out_channels))

        # kernel cache
        self._kernel_cache: Dict[Tuple[int, int, int, int], callable] = {}

    def _get_kernel(
        self, N: int, D: int, H: int, W: int, dtype: torch.dtype = torch.float16
    ):
        key = (N, D, H, W, dtype)
        if key not in self._kernel_cache:
            kernel = _make_deconv3d_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kernel_size,
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)

        N, _, D_in, H_in, W_in = x.shape
        K = self.kernel_size
        OD, OH, OW = D_in + K - 1, H_in + K - 1, W_in + K - 1

        # Prepare weight
        w_flat = (
            self.weight.to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 4, 0, 1)
            .contiguous()
            .view(-1, self.out_channels)
        )

        kernel = self._get_kernel(N, D_in, H_in, W_in)
        y = kernel(x, w_flat)

        # Add bias
        y = y + self.bias.to(dtype=torch.float16, device="cuda").view(1, -1, 1, 1, 1)

        # Scale factor
        y = y * self.scale_factor

        # BatchNorm (inference style)
        mean = self.running_mean.to(dtype=torch.float16, device="cuda").view(1, -1, 1, 1, 1)
        var = self.running_var.to(dtype=torch.float16, device="cuda").view(1, -1, 1, 1, 1)
        y = (y - mean) / torch.sqrt(var + self.eps)
        y = (
            y
            * self.bn_weight.to(dtype=torch.float16, device="cuda").view(1, -1, 1, 1, 1)
            + self.bn_bias.to(dtype=torch.float16, device="cuda").view(1, -1, 1, 1, 1)
        )

        # Global average pooling
        y = y.mean(dim=[2, 3, 4], keepdim=True)

        return y.to(torch.float32)