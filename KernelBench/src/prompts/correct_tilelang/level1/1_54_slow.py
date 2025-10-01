"""
Problem Name: 54_conv_standard_3D__square_input__square_kernel
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=10.2 runtime_stats={'mean': 10.2, 'std': 0.0947, 'min': 10.1, 'max': 10.5, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.12, 'std': 0.0293, 'min': 2.07, 'max': 2.35, 'num_trials': 100}, 'speedup_ratio': 0.208}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def conv3d_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    F: int,
    KD: int,
    KH: int,
    KW: int,
    stride_d: int,
    stride_h: int,
    stride_w: int,
    pad_d: int,
    pad_h: int,
    pad_w: int,
    dil_d: int = 1,
    dil_h: int = 1,
    dil_w: int = 1,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    OD = (D + 2 * pad_d - dil_d * (KD - 1) - 1) // stride_d + 1
    OH = (H + 2 * pad_h - dil_h * (KH - 1) - 1) // stride_h + 1
    OW = (W + 2 * pad_w - dil_w * (KW - 1) - 1) // stride_w + 1

    K_TOTAL = KD * KH * KW * C
    M_TOTAL = N * OD * OH * OW

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        data: T.Tensor((N, D, H, W, C), dtype),
        kernel: T.Tensor((KD, KH, KW, C, F), dtype),
        out: T.Tensor((N, OD, OH, OW, F), dtype),
    ):
        with T.Kernel(
            T.ceildiv(F, block_N),
            T.ceildiv(M_TOTAL, block_M),
            threads=128,
        ) as (bx, by):
            data_shared = T.alloc_shared((block_M, block_K), dtype)
            kernel_shared = T.alloc_shared((block_K, block_N), dtype)
            acc_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            out_shared = T.alloc_shared((block_M, block_N), dtype)

            kernel_flat = T.Tensor((K_TOTAL, F), dtype, kernel.data)
            out_flat = T.Tensor((M_TOTAL, F), dtype, out.data)

            T.clear(acc_local)

            for ko in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # Load data tile into shared memory with im2col indexing
                for i, j in T.Parallel(block_M, block_K):
                    m = by * block_M + i
                    k = ko * block_K + j

                    batch_idx = m // (OD * OH * OW)
                    rem_m = m % (OD * OH * OW)
                    od_idx = rem_m // (OH * OW)
                    rem_m2 = rem_m % (OH * OW)
                    oh_idx = rem_m2 // OW
                    ow_idx = rem_m2 % OW

                    kd_idx = k // (KH * KW * C)
                    rem_k1 = k % (KH * KW * C)
                    kh_idx = rem_k1 // (KW * C)
                    rem_k2 = rem_k1 % (KW * C)
                    kw_idx = rem_k2 // C
                    c_idx = rem_k2 % C

                    d_in = od_idx * stride_d + kd_idx * dil_d - pad_d
                    h_in = oh_idx * stride_h + kh_idx * dil_h - pad_h
                    w_in = ow_idx * stride_w + kw_idx * dil_w - pad_w

                    in_bound = (
                        (m < M_TOTAL)
                        and (k < K_TOTAL)
                        and (d_in >= 0)
                        and (d_in < D)
                        and (h_in >= 0)
                        and (h_in < H)
                        and (w_in >= 0)
                        and (w_in < W)
                    )

                    data_shared[i, j] = T.if_then_else(
                        in_bound,
                        data[batch_idx, d_in, h_in, w_in, c_idx],
                        T.cast(0, dtype),
                    )

                # Load kernel tile into shared memory
                T.copy(kernel_flat[ko * block_K, bx * block_N], kernel_shared)

                # GEMM
                T.gemm(data_shared, kernel_shared, acc_local)

            # Store results
            T.copy(acc_local, out_shared)
            T.copy(out_shared, out_flat[by * block_M, bx * block_N])

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
        super().__init__()
        assert (
            groups == 1
        ), "Grouped convolution is not supported in this optimized version."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_flag = bias

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                in_channels,
                kernel_size,
                kernel_size,
                kernel_size,
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            bound = 1 / math.sqrt(in_channels * kernel_size ** 3)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # Kernel cache
        self._kernels = {}

    def _get_kernel(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = conv3d_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
                self.stride,
                self.stride,
                self.stride,
                self.padding,
                self.padding,
                self.padding,
                self.dilation,
                self.dilation,
                self.dilation,
            )
        return self._kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(device="cuda", dtype=torch.float16)
        w = self.weight.to(device="cuda", dtype=torch.float16)

        N, C, D, H, W = x.shape
        kernel_fn = self._get_kernel(N, D, H, W, x.dtype)

        # Reorder tensors
        x_perm = x.permute(0, 2, 3, 4, 1).contiguous()  # N D H W C
        w_perm = w.permute(2, 3, 4, 1, 0).contiguous()  # KD KH KW C F

        out = kernel_fn(x_perm, w_perm)
        out = out.permute(0, 4, 1, 2, 3).contiguous()  # N F D_out H_out W_out

        if self.bias is not None:
            out = out + self.bias.view(1, -1, 1, 1, 1)

        return out