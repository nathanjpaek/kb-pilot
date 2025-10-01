"""
Problem Name: 65_conv_transposed_2D__square_input__asymmetric_kernel
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.422 runtime_stats={'mean': 0.422, 'std': 0.00135, 'min': 0.419, 'max': 0.426, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.138, 'std': 0.0226, 'min': 0.132, 'max': 0.362, 'num_trials': 100}, 'speedup_ratio': 0.327}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_deconv2d_kernel(
    N: int,
    CI: int,
    HI: int,
    WI: int,
    CO: int,
    KH: int,
    KW: int,
    block_M: int = 128,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    HO = HI + KH - 1
    WO = WI + KW - 1
    K_TOTAL = CI * KH * KW

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv2d_kernel(
        X: T.Tensor((N, HI, WI, CI), dtype),
        W: T.Tensor((KH, KW, CI, CO), dtype),
        Out: T.Tensor((N, HO, WO, CO), dtype),
    ):
        with T.Kernel(
            T.ceildiv(CO, block_N), T.ceildiv(N * HO * WO, block_M), threads=128
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_K, block_N), dtype)
            Acc_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(Acc_local)

            W_flat = T.Tensor((K_TOTAL, CO), dtype, W.data)
            for k_iter in T.Pipelined(T.ceildiv(K_TOTAL, block_K), num_stages=num_stages):
                # Load input tile into shared memory
                for i, j in T.Parallel(block_M, block_K):
                    k = k_iter * block_K + j
                    m = by * block_M + i

                    if (k < K_TOTAL) and (m < N * HO * WO):
                        n_idx = m // (HO * WO)
                        rem = m % (HO * WO)
                        h_out = rem // WO
                        w_out = rem % WO

                        kh_idx = k // (KW * CI)
                        kw_idx = (k // CI) % KW
                        c_idx = k % CI

                        h_in = h_out - kh_idx
                        w_in = w_out - kw_idx

                        in_bounds = (
                            (h_in >= 0)
                            and (h_in < HI)
                            and (w_in >= 0)
                            and (w_in < WI)
                        )

                        A_shared[i, j] = T.if_then_else(
                            in_bounds,
                            X[n_idx, h_in, w_in, c_idx],
                            T.Cast(dtype, 0),
                        )
                    else:
                        A_shared[i, j] = T.Cast(dtype, 0)

                # Load weight tile into shared memory
                T.copy(W_flat[k_iter * block_K, bx * block_N], W_shared)

                # GEMM
                T.gemm(A_shared, W_shared, Acc_local)

            # Write results
            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                if (global_m < N * HO * WO) and (global_n < CO):
                    n_idx = global_m // (HO * WO)
                    rem = global_m % (HO * WO)
                    h_out = rem // WO
                    w_out = rem % WO
                    Out[n_idx, h_out, w_out, global_n] = T.Cast(dtype, Acc_local[i, j])

    return deconv2d_kernel


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert stride == 1 and padding == 0 and output_padding == 0 and groups == 1, (
            "This optimized implementation currently supports stride=1, "
            "padding=0, output_padding=0, groups=1."
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KH, self.KW = kernel_size
        self.use_bias = bias

        # Parameters
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, self.KH, self.KW)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = in_channels * self.KH * self.KW
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # Cache for compiled kernels
        self._kernel_cache = {}

    def _get_kernel(self, N: int, HI: int, WI: int, dtype: str):
        key = (N, HI, WI, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_deconv2d_kernel(
                N,
                self.in_channels,
                HI,
                WI,
                self.out_channels,
                self.KH,
                self.KW,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, CI, HI, WI = x_fp16.shape
        kernel = self._get_kernel(N, HI, WI, "float16")

        # Reorder to NHWC / HWCI layout
        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()
        w_hwci_co = (
            self.weight.to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 0, 1)
            .contiguous()
        )

        out_nhwc = kernel(x_nhwc, w_hwci_co)
        out_tensor = out_nhwc.permute(0, 3, 1, 2).contiguous()

        if self.use_bias:
            out_tensor += self.bias.view(1, -1, 1, 1).to(out_tensor.dtype)

        return out_tensor.to(orig_dtype)