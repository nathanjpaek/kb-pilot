"""
Problem Name: 2_ConvTranspose2d_BiasAdd_Clamp_Scaling_Clamp_Divide
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.187 runtime_stats={'mean': 0.187, 'std': 0.0481, 'min': 0.139, 'max': 0.308, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.254, 'std': 0.0539, 'min': 0.208, 'max': 0.506, 'num_trials': 100}, 'speedup_ratio': 1.36}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_deconv_kernel(
    N: int,
    IC: int,
    OC: int,
    H_in: int,
    W_in: int,
    K: int,
    stride: int,
    padding: int,
    output_padding: int,
    scaling_factor: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    H_out = (H_in - 1) * stride - 2 * padding + K + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K + output_padding
    TOT = N * OC * H_out * W_out

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv_kernel(
        X:  T.Tensor((N, IC, H_in, W_in), dtype),
        Wt: T.Tensor((IC, OC, K, K),      dtype),   # transpose-conv weights
        Bc: T.Tensor((OC,),               dtype),   # conv-bias
        B2: T.Tensor((OC,),               dtype),   # extra bias (broadcast)
        Out: T.Tensor((N, OC, H_out, W_out), dtype),
    ):
        zero_f  = T.Cast(accum_dtype, 0)
        one_f   = T.Cast(accum_dtype, 1)
        scale_f = T.Cast(accum_dtype, scaling_factor)

        with T.Kernel(T.ceildiv(TOT, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOT:
                # Decode flat index -> n, oc, oh, ow
                tmp = idx
                ow  = tmp % W_out
                tmp //= W_out
                oh  = tmp % H_out
                tmp //= H_out
                oc  = tmp % OC
                n   = tmp // OC

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = zero_f

                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        h_nom = oh + padding - kh
                        if (h_nom % stride) == 0:
                            ih = h_nom // stride
                            if (0 <= ih) and (ih < H_in):
                                for kw in T.serial(K):
                                    w_nom = ow + padding - kw
                                    if (w_nom % stride) == 0:
                                        iw = w_nom // stride
                                        if (0 <= iw) and (iw < W_in):
                                            acc[0] += (
                                                X[n, ic, ih, iw].astype(accum_dtype)
                                                * Wt[ic, oc, kh, kw].astype(accum_dtype)
                                            )

                val = acc[0] + Bc[oc].astype(accum_dtype) + B2[oc].astype(accum_dtype)

                # First clamp
                val = T.max(val, zero_f)
                val = T.min(val, one_f)

                # Scale, second clamp, un-scale
                val = val * scale_f
                val = T.max(val, zero_f)
                val = T.min(val, one_f)
                val = val / scale_f

                Out[n, oc, oh, ow] = T.Cast(dtype, val)

    return deconv_kernel


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        bias_shape: tuple,
        scaling_factor: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.scaling_factor = float(scaling_factor)

        # Conv-transpose parameters
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        self.bias_conv = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.bias_conv, -bound, bound)

        # Extra bias (added post-conv)
        self.bias2 = nn.Parameter(torch.randn(bias_shape))

        # Kernel cache
        self._kernels = {}

    def _get_kernel(self, N, H_in, W_in, dtype: str = "float16"):
        key = (N, H_in, W_in, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_deconv_kernel(
                N,
                self.in_channels,
                self.out_channels,
                H_in,
                W_in,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.scaling_factor,
                dtype=dtype,
            )
        return self._kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, IC, H_in, W_in = x_fp16.shape
        kernel = self._get_kernel(N, H_in, W_in)

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bc_fp16 = self.bias_conv.to(device="cuda", dtype=torch.float16).contiguous()
        b2_fp16 = self.bias2.to(device="cuda", dtype=torch.float16).contiguous().view(-1)

        out_fp16 = kernel(x_fp16, w_fp16, bc_fp16, b2_fp16)
        return out_fp16.to(orig_dtype)