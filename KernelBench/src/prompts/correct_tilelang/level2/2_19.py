"""
Problem Name: 19_ConvTranspose2d_GELU_GroupNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=10.2 runtime_stats={'mean': 10.2, 'std': 0.0169, 'min': 10.2, 'max': 10.3, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.526, 'std': 0.00988, 'min': 0.52, 'max': 0.616, 'num_trials': 100}, 'speedup_ratio': 0.0516}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
# TileLang kernel factory: ConvTranspose2d + GELU
# ----------------------------------------------------------------------------- #
def _build_deconv_gelu_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    stride: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Hout = (Hin - 1) * stride + K      # pad=0, dil=1, out_pad=0
    Wout = (Win - 1) * stride + K
    total_elems = N * Cout * Hout * Wout
    grid = (total_elems + threads_per_block - 1) // threads_per_block

    inv_sqrt2 = 0.7071067811865476  # 1 / sqrt(2)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv_gelu(
        X:   T.Tensor((N, Cin, Hin, Win), dtype),
        Wt:  T.Tensor((Cin, Cout, K, K),  dtype),   # weight as stored by nn.ConvTranspose2d
        B:   T.Tensor((Cout,),            dtype),   # conv-transpose bias
        Out: T.Tensor((N, Cout, Hout, Wout), dtype),
    ):
        zero_f  = T.Cast(accum_dtype, 0)
        half_f  = T.Cast(accum_dtype, 0.5)
        inv_s2  = T.Cast(accum_dtype, inv_sqrt2)

        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total_elems:
                # unravel index -> (n, oc, oh, ow)
                ow = idx % Wout
                t1 = idx // Wout
                oh = t1 % Hout
                t2 = t1 // Hout
                oc = t2 % Cout
                n  = t2 // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = zero_f

                # main transposed-conv accumulation
                for ic in T.serial(Cin):
                    for kh in T.serial(K):
                        ih_nom = oh - kh
                        if (ih_nom % stride) == 0:
                            ih = ih_nom // stride
                            if (ih >= 0) and (ih < Hin):
                                for kw in T.serial(K):
                                    iw_nom = ow - kw
                                    if (iw_nom % stride) == 0:
                                        iw = iw_nom // stride
                                        if (iw >= 0) and (iw < Win):
                                            acc[0] += (
                                                X[n, ic, ih, iw].astype(accum_dtype)
                                                * Wt[ic, oc, kh, kw].astype(accum_dtype)
                                            )

                val = acc[0] + B[oc].astype(accum_dtype)
                # GELU
                val = half_f * val * (T.Cast(accum_dtype, 1.0) + T.erf(val * inv_s2))

                Out[n, oc, oh, ow] = T.Cast(dtype, val)

    return deconv_gelu


# ----------------------------------------------------------------------------- #
# PyTorch wrapper module
# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    y = GroupNorm( GELU( ConvTranspose2d(x) ) )
    The ConvTranspose2d+GELU is replaced by a fused TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        num_groups: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.num_groups   = num_groups
        self.eps          = eps

        # Conv-transpose parameters (same init as nn.ConvTranspose2d)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.bias, -bound, bound)

        # GroupNorm parameters (gamma / beta)
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias   = nn.Parameter(torch.zeros(out_channels))

        # kernel cache
        self._kernels: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, Hin: int, Win: int, dtype: str = "float16"):
        key = (N, Hin, Win, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_deconv_gelu_kernel(
                N,
                self.in_channels,
                Hin,
                Win,
                self.out_channels,
                self.kernel_size,
                self.stride,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Prepare inputs
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, _, Hin, Win = x_fp16.shape

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(N, Hin, Win, "float16")
        y = kernel(x_fp16, w_fp16, b_fp16)  # (N,Cout,Hout,Wout), fp16

        # ---------------- GroupNorm ---------------- #
        C = self.out_channels
        G = self.num_groups
        y = y.to(dtype=torch.float32)  # improve precision for statistics
        B, _, Hout, Wout = y.shape
        y = y.view(B, G, C // G, Hout, Wout)

        mean = y.mean(dim=[2, 3, 4], keepdim=True)
        var  = y.var(dim=[2, 3, 4], unbiased=False, keepdim=True)
        y_norm = (y - mean) / torch.sqrt(var + self.eps)

        y_norm = y_norm.view(B, C, Hout, Wout)
        y_norm = (
            y_norm
            * self.gn_weight.view(1, C, 1, 1).to(dtype=torch.float32, device="cuda")
            + self.gn_bias.view(1, C, 1, 1).to(dtype=torch.float32, device="cuda")
        )

        return y_norm.to(orig_dtype)