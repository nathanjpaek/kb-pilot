"""
Problem Name: 61_ConvTranspose3d_ReLU_GroupNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.84 runtime_stats={'mean': 3.84, 'std': 0.00759, 'min': 3.83, 'max': 3.87, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.155, 'std': 0.00143, 'min': 0.152, 'max': 0.162, 'num_trials': 100}, 'speedup_ratio': 0.0404}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : ConvTranspose3d + ReLU                            #
# --------------------------------------------------------------------------- #

def _build_deconv_relu_kernel(
    N: int,
    Cin: int,
    Din: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Dout = Din + K - 1  # stride=1, pad=0, out_pad=0
    Hout = Hin + K - 1
    Wout = Win + K - 1
    TOT = N * Cout * Dout * Hout * Wout
    GRID = (TOT + threads_per_block - 1) // threads_per_block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv_relu(
        X:  T.Tensor((N, Cin, Din, Hin, Win), dtype),
        Wt: T.Tensor((Cin, Cout, K, K, K),     dtype),
        Y:  T.Tensor((N, Cout, Dout, Hout, Wout), dtype),
    ):
        zero_f = T.Cast(accum_dtype, 0)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                # unravel linear index -> (n, oc, od, oh, ow)
                ow = idx % Wout
                t1 = idx // Wout
                oh = t1 % Hout
                t2 = t1 // Hout
                od = t2 % Dout
                t3 = t2 // Dout
                oc = t3 % Cout
                n  = t3 // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = zero_f

                for ic in T.serial(Cin):
                    for kd in T.serial(K):
                        id_ = od - kd
                        if (id_ >= 0) and (id_ < Din):
                            for kh in T.serial(K):
                                ih_ = oh - kh
                                if (ih_ >= 0) and (ih_ < Hin):
                                    for kw in T.serial(K):
                                        iw_ = ow - kw
                                        if (iw_ >= 0) and (iw_ < Win):
                                            acc[0] += (
                                                X[n, ic, id_, ih_, iw_].astype(accum_dtype)
                                                * Wt[ic, oc, kd, kh, kw].astype(accum_dtype)
                                            )

                val = T.max(acc[0], zero_f)  # ReLU
                Y[n, oc, od, oh, ow] = T.Cast(dtype, val)

    return deconv_relu


# --------------------------------------------------------------------------- #
# Optimised PyTorch module                                                    #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  ReLU (TileLang fused)  →  GroupNorm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int,
        bias: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps

        # ConvTranspose3d parameters (same init as PyTorch)
        w_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in = in_channels * kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # GroupNorm gamma / beta
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias   = nn.Parameter(torch.zeros(out_channels))

        # kernel cache  (N, D, H, W, dtype) -> kernel
        self._kernels: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: str = "float16"):
        key = (N, D, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_deconv_relu_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, _, D_in, H_in, W_in = x_fp16.shape

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(N, D_in, H_in, W_in, "float16")
        y_fp16 = kernel(x_fp16, w_fp16)  # (N,Cout,Dout,Hout,Wout)

        # ---------------- GroupNorm (fp32) ---------------- #
        y = y_fp16.to(dtype=torch.float32)
        C = self.out_channels
        G = self.groups
        N_, _, D_out, H_out, W_out = y.shape
        y = y.view(N_, G, C // G, D_out, H_out, W_out)

        mean = y.mean(dim=[2, 3, 4, 5], keepdim=True)
        var  = y.var(dim=[2, 3, 4, 5], unbiased=False, keepdim=True)
        y_norm = (y - mean) / torch.sqrt(var + self.eps)
        y_norm = y_norm.view(N_, C, D_out, H_out, W_out)

        y_norm = (
            y_norm * self.gn_weight.view(1, C, 1, 1, 1).to(dtype=torch.float32, device="cuda")
            + self.gn_bias.view(1, C, 1, 1, 1).to(dtype=torch.float32, device="cuda")
        )

        return y_norm.to(orig_dtype)