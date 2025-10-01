"""
Problem Name: 5_ConvTranspose2d_Subtract_Tanh
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.702 runtime_stats={'mean': 0.702, 'std': 0.00174, 'min': 0.699, 'max': 0.708, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0888, 'std': 0.00168, 'min': 0.0864, 'max': 0.0989, 'num_trials': 100}, 'speedup_ratio': 0.126}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
# Kernel factory
# ----------------------------------------------------------------------------- #
def _build_fused_deconv_tanh_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    stride: int,
    padding: int,
    output_padding: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Hout = (Hin - 1) * stride - 2 * padding + K + output_padding
    Wout = (Win - 1) * stride - 2 * padding + K + output_padding
    total_elems = N * Cout * Hout * Wout
    grid = (total_elems + threads_per_block - 1) // threads_per_block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv_fused_tanh(
        X: T.Tensor((N, Cin, Hin, Win), dtype),
        Wt: T.Tensor((Cin, Cout, K, K), dtype),      # weight as stored by nn.ConvTranspose2d
        B_conv: T.Tensor((Cout,), dtype),            # conv-transpose bias
        B_sub: T.Tensor((Cout,), dtype),             # bias to subtract after conv
        Out: T.Tensor((N, Cout, Hout, Wout), dtype),
    ):
        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total_elems:
                # Decode flat index -> (n, oc, oh, ow)
                ow = idx % Wout
                tmp = idx // Wout
                oh = tmp % Hout
                tmp //= Hout
                oc = tmp % Cout
                n  = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                # Transposed-conv accumulation
                for ic in T.serial(Cin):
                    for kh in T.serial(K):
                        h_nom = oh + padding - kh
                        if (h_nom % stride) == 0:
                            ih = h_nom // stride
                            if (ih >= 0) and (ih < Hin):
                                for kw in T.serial(K):
                                    w_nom = ow + padding - kw
                                    if (w_nom % stride) == 0:
                                        iw = w_nom // stride
                                        if (iw >= 0) and (iw < Win):
                                            acc[0] += (
                                                X[n, ic, ih, iw].astype(accum_dtype)
                                                * Wt[ic, oc, kh, kw].astype(accum_dtype)
                                            )

                # Add conv bias, subtract extra bias, apply tanh
                val = acc[0] + B_conv[oc].astype(accum_dtype)
                val = val - B_sub[oc].astype(accum_dtype)
                val = T.tanh(val)

                Out[n, oc, oh, ow] = T.Cast(dtype, val)

    return deconv_fused_tanh


# ----------------------------------------------------------------------------- #
# PyTorch wrapper
# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    TileLang-accelerated replacement for the original model:
        y = tanh( ConvTranspose2d(x) − bias2 )
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias_shape: tuple,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
    ):
        super().__init__()
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.kernel_size   = kernel_size
        self.stride        = stride
        self.padding       = padding
        self.output_pad    = output_padding

        # ----- weights & conv-bias (same init as nn.ConvTranspose2d) ----------
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.bias_conv = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.bias_conv, -bound, bound)

        # ----- extra bias that is subtracted after the convolution ------------
        self.bias2 = nn.Parameter(torch.randn(bias_shape))

        # Kernel cache  {(N,Hin,Win,dtype) : compiled_kernel}
        self._kernels = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, Hin: int, Win: int, dtype: str = "float16"):
        key = (N, Hin, Win, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_fused_deconv_tanh_kernel(
                N,
                self.in_channels,
                Hin,
                Win,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_pad,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, Cin, Hin, Win = x_fp16.shape

        w_fp16  = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bc_fp16 = self.bias_conv.to(device="cuda", dtype=torch.float16).contiguous()
        b2_fp16 = self.bias2.view(-1).to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(N, Hin, Win, "float16")
        y_fp16 = kernel(x_fp16, w_fp16, bc_fp16, b2_fp16)

        return y_fp16.to(orig_dtype)