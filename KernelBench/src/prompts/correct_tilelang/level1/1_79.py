"""
Problem Name: 79_conv_transposed_1D_asymmetric_input_square_kernel___padded____strided____dilated__
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.046 runtime_stats={'mean': 0.046, 'std': 0.0109, 'min': 0.0415, 'max': 0.143, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0427, 'std': 0.00724, 'min': 0.0377, 'max': 0.0882, 'num_trials': 100}, 'speedup_ratio': 0.928}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
def _conv_transpose1d_kernel(
    N: int,
    C_in: int,
    C_out: int,
    L_in: int,
    K: int,
    stride: int,
    padding: int,
    dilation: int,
    bias_enabled: bool,
    block_OC: int = 16,
    block_L: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """Build a TileLang kernel for 1-D transposed convolution."""
    L_out = (L_in - 1) * stride - 2 * padding + dilation * (K - 1) + 1

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((N, C_in, L_in), dtype),
        W: T.Tensor((C_in, C_out, K), dtype),
        B: T.Tensor((C_out,), dtype),
        Y: T.Tensor((N, C_out, L_out), dtype),
    ):
        grid_x = T.ceildiv(L_out, block_L)
        grid_y = T.ceildiv(N * C_out, block_OC)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            Acc = T.alloc_fragment((block_OC, block_L), accum_dtype)
            T.clear(Acc)

            # ---- bias initialisation ------------------------------------------------
            for oc, l in T.Parallel(block_OC, block_L):
                g_oc = by * block_OC + oc
                n_idx = g_oc // C_out
                c_out_idx = g_oc % C_out
                l_out_idx = bx * block_L + l

                valid = (
                    (n_idx < N) and (c_out_idx < C_out) and (l_out_idx < L_out)
                )
                bias_val = T.if_then_else(
                    valid & bias_enabled, B[c_out_idx], T.Cast(dtype, 0)
                )
                Acc[oc, l] = T.Cast(accum_dtype, bias_val)

            # ---- main accumulation loop -------------------------------------------
            for c_in in range(C_in):
                for k in range(K):
                    for oc, l in T.Parallel(block_OC, block_L):
                        g_oc = by * block_OC + oc
                        n_idx = g_oc // C_out
                        c_out_idx = g_oc % C_out
                        l_out_idx = bx * block_L + l

                        in_range = (
                            (n_idx < N)
                            and (c_out_idx < C_out)
                            and (l_out_idx < L_out)
                        )

                        num = l_out_idx + padding - k * dilation
                        cond = in_range & ((num % stride) == 0)
                        l_in_idx = num // stride
                        inside_x = cond & (l_in_idx >= 0) & (l_in_idx < L_in)

                        x_val = T.if_then_else(
                            inside_x, X[n_idx, c_in, l_in_idx], T.Cast(dtype, 0)
                        )
                        w_val = W[c_in, c_out_idx, k]

                        Acc[oc, l] += (
                            T.Cast(accum_dtype, x_val) * T.Cast(accum_dtype, w_val)
                        )

            # ---- write back --------------------------------------------------------
            for oc, l in T.Parallel(block_OC, block_L):
                g_oc = by * block_OC + oc
                n_idx = g_oc // C_out
                c_out_idx = g_oc % C_out
                l_out_idx = bx * block_L + l

                if (n_idx < N) and (c_out_idx < C_out) and (l_out_idx < L_out):
                    Y[n_idx, c_out_idx, l_out_idx] = T.Cast(dtype, Acc[oc, l])

    return main


# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    TileLang-optimised ConvTranspose1d supporting stride, padding and dilation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias_enabled = bias

        # ---- parameter initialisation (identical to nn.ConvTranspose1d) ----------
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * kernel_size
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # kernel cache
        self._cached_kernels = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, L_in: int, dtype: torch.dtype):
        key = (N, L_in, dtype)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = _conv_transpose1d_kernel(
                N=N,
                C_in=self.in_channels,
                C_out=self.out_channels,
                L_in=L_in,
                K=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                bias_enabled=self.bias_enabled,
            )
        return self._cached_kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)

        if self.bias_enabled:
            b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)
        else:
            b_fp16 = torch.zeros(
                self.out_channels, device="cuda", dtype=torch.float16
            )

        N, _, L_in = x_fp16.shape
        kernel = self._get_kernel(N, L_in, x_fp16.dtype)

        y_fp16 = kernel(x_fp16, w_fp16, b_fp16)
        return y_fp16.to(x.dtype)