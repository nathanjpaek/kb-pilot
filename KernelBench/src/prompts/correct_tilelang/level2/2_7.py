"""
Problem Name: 7_Conv3d_ReLU_LeakyReLU_GELU_Sigmoid_BiasAdd
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.102 runtime_stats={'mean': 0.102, 'std': 0.00105, 'min': 0.101, 'max': 0.106, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.688, 'std': 0.00222, 'min': 0.684, 'max': 0.7, 'num_trials': 100}, 'speedup_ratio': 6.75}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_fused_conv3d_kernel(
    N: int,
    C_IN: int,
    D_IN: int,
    H_IN: int,
    W_IN: int,
    C_OUT: int,
    K: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    D_OUT = D_IN - K + 1
    H_OUT = H_IN - K + 1
    W_OUT = W_IN - K + 1
    NUMEL = N * C_OUT * D_OUT * H_OUT * W_OUT

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused_kernel(
        X:     T.Tensor((N, C_IN, D_IN, H_IN, W_IN), dtype),
        Wt:    T.Tensor((C_OUT, C_IN, K, K, K),       dtype),
        Bconv: T.Tensor((C_OUT,),                      dtype),
        Badd:  T.Tensor((C_OUT,),                      dtype),
        Y:     T.Tensor((N, C_OUT, D_OUT, H_OUT, W_OUT), dtype),
    ):
        half  = T.Cast(accum_dtype, 0.5)
        one   = T.Cast(accum_dtype, 1.0)
        zero  = T.Cast(accum_dtype, 0.0)
        slope = T.Cast(accum_dtype, 0.01)
        sqrt2_pi = T.Cast(accum_dtype, 0.7978845608028654)   # √(2/π)
        c0 = T.Cast(accum_dtype, 0.044715)

        with T.Kernel(T.ceildiv(NUMEL, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx

            if idx < NUMEL:
                w_idx = idx % W_OUT
                tmp   = idx // W_OUT
                h_idx = tmp % H_OUT
                tmp   //= H_OUT
                d_idx = tmp % D_OUT
                tmp   //= D_OUT
                co    = tmp % C_OUT
                n     = tmp // C_OUT

                acc = T.Cast(accum_dtype, Bconv[co])

                for ci in range(C_IN):
                    for kz in range(K):
                        for ky in range(K):
                            for kx in range(K):
                                inp  = T.Cast(
                                    accum_dtype,
                                    X[n, ci, d_idx + kz, h_idx + ky, w_idx + kx],
                                )
                                wt   = T.Cast(
                                    accum_dtype,
                                    Wt[co, ci, kz, ky, kx],
                                )
                                acc += inp * wt

                # ReLU
                acc = T.max(acc, zero)
                # LeakyReLU (redundant after ReLU but kept for completeness)
                acc = T.if_then_else(acc > zero, acc, acc * slope)

                # GELU (approximate)
                gelu_inner = acc + c0 * acc * acc * acc
                acc = half * acc * (one + T.tanh(sqrt2_pi * gelu_inner))

                # Sigmoid
                acc = one / (one + T.exp(-acc))

                # Final bias add (broadcast)
                acc += T.Cast(accum_dtype, Badd[co])

                Y[n, co, d_idx, h_idx, w_idx] = T.Cast(dtype, acc)

    return fused_kernel


class ModelNew(nn.Module):
    """
    Optimized model replacing 3D convolution + activations with a fused TileLang kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        if isinstance(kernel_size, int):
            self.kernel_size = int(kernel_size)
        else:
            # enforce cubic kernel for simplicity in this benchmark
            assert (
                kernel_size[0] == kernel_size[1] == kernel_size[2]
            ), "Only cubic kernels supported"
            self.kernel_size = int(kernel_size[0])

        # Conv3d parameters
        weight_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(weight_shape))
        self.bias_conv = nn.Parameter(torch.empty(self.out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = self.in_channels * self.kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias_conv, -bound, bound)

        # Extra bias added after sigmoid
        self.bias_out = nn.Parameter(torch.randn(bias_shape))

        # TileLang kernel cache
        self._kern_cache = {}

    def _get_kernel(self, N, D, H, W, dtype):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_conv3d_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                dtype=dtype,
            )
        return self._kern_cache[key]

    def forward(self, x: torch.Tensor):
        # Ensure CUDA + FP16 tensors
        x_fp16 = x.to(dtype=torch.float16, device="cuda").contiguous()
        w_fp16 = self.weight.to(dtype=torch.float16, device="cuda").contiguous()
        bconv_fp16 = self.bias_conv.to(dtype=torch.float16, device="cuda").contiguous()
        bout_fp16 = self.bias_out.view(-1).to(dtype=torch.float16, device="cuda").contiguous()

        N, C_in, D_in, H_in, W_in = x_fp16.shape
        kernel = self._get_kernel(N, D_in, H_in, W_in, "float16")

        y_fp16 = kernel(x_fp16, w_fp16, bconv_fp16, bout_fp16)
        return y_fp16