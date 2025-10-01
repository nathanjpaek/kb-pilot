"""
Problem Name: 80_conv_standard_2D_square_input_asymmetric_kernel___dilated____padded__
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.902 runtime_stats={'mean': 0.902, 'std': 0.0384, 'min': 0.793, 'max': 1.04, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.452, 'std': 0.0331, 'min': 0.385, 'max': 0.624, 'num_trials': 100}, 'speedup_ratio': 0.501}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def conv2d_kernel(
    N,
    C,
    H,
    W,
    F,
    KH,
    KW,
    stride_h,
    stride_w,
    dil_h,
    dil_w,
    pad_h,
    pad_w,
    block_M=128,
    block_N=64,
    block_K=32,
    num_stages=2,
    dtype="float16",
    accum_dtype="float",
):
    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1
    K_TOTAL = KH * KW * C

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        data: T.Tensor((N, H, W, C), dtype),
        kernel: T.Tensor((KH, KW, C, F), dtype),
        out: T.Tensor((N, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N), T.ceildiv(N * OH * OW, block_M), threads=128
        ) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            acc_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((K_TOTAL, F), dtype, kernel.data)
            out_flat = T.Tensor((N * OH * OW, F), dtype, out.data)

            T.clear(acc_local)

            for k_iter in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # Load im2col tile into shared memory
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i

                    batch_idx = m // (OH * OW)
                    oh_idx = (m % (OH * OW)) // OW
                    ow_idx = m % OW

                    kh_idx = (k // (KW * C))
                    kw_idx = (k // C) % KW
                    c_idx = k % C

                    access_h = oh_idx * stride_h + kh_idx * dil_h - pad_h
                    access_w = ow_idx * stride_w + kw_idx * dil_w - pad_w

                    in_bound = (
                        (k < K_TOTAL)
                        and (m < N * OH * OW)
                        and (access_h >= 0)
                        and (access_w >= 0)
                        and (access_h < H)
                        and (access_w < W)
                    )

                    data_shared[i, j] = T.if_then_else(
                        in_bound,
                        data[batch_idx, access_h, access_w, c_idx],
                        T.cast(0, dtype),
                    )

                # Load kernel tile into shared memory
                T.copy(kernel_flat[k_iter * block_K, bx * block_N], kernel_shared)

                # GEMM
                T.gemm(data_shared, kernel_shared, acc_local)

            # Write results back
            T.copy(acc_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

    return main


class ModelNew(nn.Module):
    """
    Optimized Conv2D model using TileLang kernels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        bias: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_height, self.kernel_width = kernel_size
        self.stride_h = stride if isinstance(stride, int) else stride[0]
        self.stride_w = stride if isinstance(stride, int) else stride[1]
        self.pad_h, self.pad_w = padding
        self.dil_h, self.dil_w = dilation
        self.bias_flag = bias

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels, self.kernel_height, self.kernel_width
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias_flag:
            bound = 1 / math.sqrt(in_channels)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # Kernel cache
        self._kernels = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        w = self.weight.to(device="cuda", dtype=torch.float16)

        N, C, H, W = x.shape
        F = self.out_channels
        KH = self.kernel_height
        KW = self.kernel_width

        key = (N, C, H, W, F, KH, KW, self.stride_h, self.stride_w,
               self.dil_h, self.dil_w, self.pad_h, self.pad_w, x.dtype)

        if key not in self._kernels:
            self._kernels[key] = conv2d_kernel(
                N,
                C,
                H,
                W,
                F,
                KH,
                KW,
                self.stride_h,
                self.stride_w,
                self.dil_h,
                self.dil_w,
                self.pad_h,
                self.pad_w,
            )

        kernel_fn = self._kernels[key]

        # Reorder tensors to NHWC / HWCF
        x_perm = x.permute(0, 2, 3, 1).contiguous()
        w_perm = w.permute(2, 3, 1, 0).contiguous()

        out = kernel_fn(x_perm, w_perm)
        out = out.permute(0, 3, 1, 2).contiguous()

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1)

        return out