"""
Problem Name: 54_conv_standard_3D__square_input__square_kernel
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=9.1 runtime_stats={'mean': 9.1, 'std': 0.0571, 'min': 9.0, 'max': 9.31, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.09, 'std': 0.0309, 'min': 2.03, 'max': 2.25, 'num_trials': 100}, 'speedup_ratio': 0.23}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def conv3d_kernel(
    N,
    C,
    D,
    H,
    W,
    F,
    K,
    stride,
    padding,
    dilation,
    block_M=128,
    block_N=128,
    block_K=32,
    threads=128,
    dtype="float16",
    accum_dtype="float",
):
    OD = (D + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    OH = (H + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    OW = (W + 2 * padding - dilation * (K - 1) - 1) // stride + 1
    KK = K * K * K * C

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        inp: T.Tensor((N, D, H, W, C), dtype),
        ker: T.Tensor((K, K, K, C, F), dtype),
        out: T.Tensor((N, OD, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(N * OD * OH * OW, block_M),
            threads=threads,
        ) as (bx, by):
            a_shared = T.alloc_shared((block_M, block_K), dtype)
            b_shared = T.alloc_shared((block_K, block_N), dtype)
            c_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            ker_flat = T.Tensor((KK, F), dtype, ker.data)
            out_flat = T.Tensor((N * OD * OH * OW, F), dtype, out.data)

            T.clear(c_local)

            num_k_tiles = T.ceildiv(KK, block_K)
            for k_iter in T.Pipelined(num_k_tiles, num_stages=3):
                for i, j in T.Parallel(block_M, block_K):
                    k_idx = k_iter * block_K + j
                    m_idx = by * block_M + i

                    if m_idx < N * OD * OH * OW and k_idx < KK:
                        n_idx = m_idx // (OD * OH * OW)
                        rem_1 = m_idx % (OD * OH * OW)
                        od_idx = rem_1 // (OH * OW)
                        rem_2 = rem_1 % (OH * OW)
                        oh_idx = rem_2 // OW
                        ow_idx = rem_2 % OW

                        c_idx = k_idx % C
                        kw_idx = (k_idx // C) % K
                        kh_idx = (k_idx // (C * K)) % K
                        kd_idx = k_idx // (C * K * K)

                        in_d = od_idx * stride - padding + kd_idx * dilation
                        in_h = oh_idx * stride - padding + kh_idx * dilation
                        in_w = ow_idx * stride - padding + kw_idx * dilation

                        in_bounds = (
                            (in_d >= 0)
                            and (in_d < D)
                            and (in_h >= 0)
                            and (in_h < H)
                            and (in_w >= 0)
                            and (in_w < W)
                        )
                        a_shared[i, j] = T.if_then_else(
                            in_bounds,
                            inp[n_idx, in_d, in_h, in_w, c_idx],
                            T.cast(0, dtype),
                        )
                    else:
                        a_shared[i, j] = T.cast(0, dtype)

                T.copy(ker_flat[k_iter * block_K, bx * block_N], b_shared)

                T.gemm(a_shared, b_shared, c_local)

            T.copy(c_local, out_flat[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        assert groups == 1, "Grouped convolution not supported in this TileLang implementation."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

        weight_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * kernel_size * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        self._kernel_cache = {}

    def _get_kernel(self, shapes, dtype):
        key = (*shapes, dtype)
        if key not in self._kernel_cache:
            N, C, D, H, W, F, K, stride, padding, dilation = shapes
            kernel = conv3d_kernel(
                N,
                C,
                D,
                H,
                W,
                F,
                K,
                stride,
                padding,
                dilation,
            )
            self._kernel_cache[key] = kernel
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        N, C, D, H, W = x.shape
        K = self.kernel_size
        F = self.out_channels

        inp_t = x.permute(0, 2, 3, 4, 1).contiguous()
        ker_t = (
            self.weight.to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 4, 1, 0)
            .contiguous()
        )

        kernel_fn = self._get_kernel(
            (N, C, D, H, W, F, K, self.stride, self.padding, self.dilation),
            inp_t.dtype,
        )

        out_t = kernel_fn(inp_t, ker_t)
        out_t = out_t.permute(0, 4, 1, 2, 3)

        if self.bias is not None:
            bias_cuda = self.bias.to(device="cuda", dtype=torch.float16)
            out_t = out_t + bias_cuda.view(1, -1, 1, 1, 1)

        return out_t.to(torch.float32)