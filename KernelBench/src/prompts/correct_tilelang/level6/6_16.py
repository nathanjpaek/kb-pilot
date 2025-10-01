"""
Problem Name: 16_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=15.7 runtime_stats={'mean': 15.7, 'std': 0.0655, 'min': 15.7, 'max': 16.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.94, 'std': 0.00968, 'min': 1.93, 'max': 2.03, 'num_trials': 100}, 'speedup_ratio': 0.124}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory                                                              #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    Cin: int,
    Din: int,
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
    Dout = (Din - 1) * stride - 2 * padding + K + output_padding
    Hout = (Hin - 1) * stride - 2 * padding + K + output_padding
    Wout = (Win - 1) * stride - 2 * padding + K + output_padding
    TOT  = N * Cout * Dout * Hout * Wout
    GRID = (TOT + threads_per_block - 1) // threads_per_block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:          T.Tensor((N, Cin, Din, Hin, Win), dtype),
        Wt:         T.Tensor((Cin, Cout, K, K, K),    dtype),   # conv-transpose weights
        B_conv:     T.Tensor((Cout,),                 dtype),   # conv bias
        B_extra:    T.Tensor((Cout,),                 dtype),   # extra bias
        Out:        T.Tensor((N, Cout, Dout, Hout, Wout), dtype),
    ):
        zero_f = T.Cast(accum_dtype, 0)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                # unravel flat index
                ow  = idx % Wout
                tmp = idx // Wout
                oh  = tmp % Hout
                tmp //= Hout
                od  = tmp % Dout
                tmp //= Dout
                oc  = tmp % Cout
                n   = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = zero_f

                # ConvTranspose3d accumulate
                for ic in T.serial(Cin):
                    for kd in T.serial(K):
                        d_nom = od + padding - kd
                        if (d_nom % stride) == 0:
                            id_ = d_nom // stride
                            if (id_ >= 0) and (id_ < Din):
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
                                                            T.Cast(accum_dtype, X[n, ic, id_, ih, iw])
                                                            * T.Cast(accum_dtype, Wt[ic, oc, kd, kh, kw])
                                                        )

                O_val = acc[0] + T.Cast(accum_dtype, B_conv[oc])    # conv output

                # fused post-ops
                t1   = O_val + T.Cast(accum_dtype, B_extra[oc])     # + bias
                t2   = t1 + O_val                                   # residual add
                t3   = t2 * O_val                                   # multiply
                res  = t3 + O_val                                   # final residual add

                Out[n, oc, od, oh, ow] = T.Cast(dtype, res)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d fused with (+bias → +residual → *residual → +residual)
    implemented entirely in one TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        bias_shape: tuple,
    ):
        super().__init__()
        self.in_channels   = int(in_channels)
        self.out_channels  = int(out_channels)
        self.kernel_size   = int(kernel_size)
        self.stride        = int(stride)
        self.padding       = int(padding)
        self.output_pad    = int(output_padding)

        # ConvTranspose3d parameters with identical init
        w_shape = (
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kernel_size ** 3
        bound  = 1 / math.sqrt(fan_in)
        self.conv_bias = nn.Parameter(torch.empty(self.out_channels))
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # extra bias (added after conv)
        self.extra_bias = nn.Parameter(torch.randn(bias_shape))

        # kernel cache  {(N,Din,Hin,Win,dtype):kernel}
        self._kernels = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, Din: int, Hin: int, Win: int, dtype: str):
        key = (N, Din, Hin, Win, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_fused_kernel(
                N,
                self.in_channels,
                Din,
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

        N, Cin, Din, Hin, Win = x_fp16.shape

        w_fp16  = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bc_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()
        be_fp16 = self.extra_bias.view(-1).to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(N, Din, Hin, Win, "float16")
        y_fp16 = kernel(x_fp16, w_fp16, bc_fp16, be_fp16)

        return y_fp16.to(orig_dtype)