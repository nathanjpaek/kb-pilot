"""
Problem Name: 42_ConvTranspose2d_GlobalAvgPool_BiasAdd_LogSumExp_Sum_Multiply
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.283 runtime_stats={'mean': 0.283, 'std': 0.00539, 'min': 0.277, 'max': 0.304, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.182, 'std': 0.0152, 'min': 0.159, 'max': 0.232, 'num_trials': 100}, 'speedup_ratio': 0.643}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    threads_per_block: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    stride = 1
    padding = 0
    output_padding = 0

    Hout = (Hin - 1) * stride - 2 * padding + K + output_padding
    Wout = (Win - 1) * stride - 2 * padding + K + output_padding
    HW   = Hout * Wout
    HW_f = float(HW)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((N, Cin, Hin, Win), dtype),
        Wt:     T.Tensor((Cin, Cout, K, K),  dtype),
        Bconv:  T.Tensor((Cout,),            dtype),   # Conv-transpose bias
        Bextra: T.Tensor((Cout,),            dtype),   # Added afterwards
        Out:    T.Tensor((N, 1),             dtype),   # y[n,0]
    ):
        hw_inv   = T.Cast(accum_dtype, 1.0 / HW_f)
        ten_const= T.Cast(accum_dtype, 10.0)

        with T.Kernel(N, threads=threads_per_block) as bn:
            # bn ≡ batch-index
            acc = T.alloc_fragment((Cout,), accum_dtype)
            T.clear(acc)

            # -------------------- accumulate summed deconv -----------------
            for ic in range(Cin):
                for ih in range(Hin):
                    for iw in range(Win):
                        x_val = T.Cast(accum_dtype, X[bn, ic, ih, iw])

                        for kh in range(K):
                            oh = ih * stride - padding + kh
                            if (oh >= 0) and (oh < Hout):
                                for kw in range(K):
                                    ow = iw * stride - padding + kw
                                    if (ow >= 0) and (ow < Wout):
                                        for tc in T.Parallel(Cout):
                                            w_val = T.Cast(
                                                accum_dtype, Wt[ic, tc, kh, kw]
                                            )
                                            acc[tc] += x_val * w_val

            # --------------- channel-wise mean + both biases ----------------
            for tc in T.Parallel(Cout):
                acc[tc] = acc[tc] * hw_inv \
                          + T.Cast(accum_dtype, Bconv[tc]) \
                          + T.Cast(accum_dtype, Bextra[tc])

            # ---------------------- log-sum-exp across C --------------------
            max_val = T.alloc_local((1,), accum_dtype)
            max_val[0] = acc[0]
            for c in range(1, Cout):
                max_val[0] = T.max(max_val[0], acc[c])

            sum_exp = T.alloc_local((1,), accum_dtype)
            sum_exp[0] = T.Cast(accum_dtype, 0)
            for c in range(Cout):
                sum_exp[0] += T.exp(acc[c] - max_val[0])

            lse = max_val[0] + T.log(sum_exp[0])

            # --------------------------- scale & store ----------------------
            res = lse * ten_const
            Out[bn, 0] = T.Cast(dtype, res)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d -> GlobalAvgPool -> +bias -> LogSumExp(C) -> *10
    implemented in a single TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias_shape: tuple,
    ):
        super().__init__()

        self.in_c  = in_channels
        self.out_c = out_channels
        self.k     = kernel_size

        # ---- parameters mirroring nn.ConvTranspose2d ----------------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # Extra bias added after global pooling
        self.extra_bias = nn.Parameter(torch.randn(bias_shape).view(-1))

        # Kernel cache  {(N,H,W,dtype): compiled_kernel}
        self._kern_cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H_in: int, W_in: int, dtype: str = "float16"):
        key = (N, H_in, W_in, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N,
                self.in_c,
                H_in,
                W_in,
                self.out_c,
                self.k,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, _, H_in, W_in = x_fp16.shape
        kernel = self._get_kernel(N, H_in, W_in, "float16")

        w_fp16  = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bc_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()
        be_fp16 = self.extra_bias.to(device="cuda", dtype=torch.float16).contiguous()

        out_fp16 = kernel(x_fp16, w_fp16, bc_fp16, be_fp16)
        return out_fp16.to(orig_dtype)  # (N,1)