"""
Problem Name: 17_Conv2d_InstanceNorm_Divide
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.125 runtime_stats={'mean': 0.125, 'std': 0.0229, 'min': 0.11, 'max': 0.313, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0836, 'std': 0.0189, 'min': 0.0722, 'max': 0.247, 'num_trials': 100}, 'speedup_ratio': 0.669}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                      TileLang kernel factory :  Conv2d                      #
# --------------------------------------------------------------------------- #
def _build_conv2d_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    block_size: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Hout = Hin - K + 1
    Wout = Win - K + 1
    TOT  = N * Cout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv2d_kernel(
        X: T.Tensor((N, Cin, Hin, Win), in_dtype),
        Wt: T.Tensor((Cout, Cin, K, K),   in_dtype),
        B:  T.Tensor((Cout,),             in_dtype),
        Y:  T.Tensor((N, Cout, Hout, Wout), in_dtype),   # created by TileLang
    ):
        with T.Kernel(T.ceildiv(TOT, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOT:
                ow = idx % Wout
                tmp = idx // Wout
                oh = tmp % Hout
                tmp //= Hout
                oc = tmp % Cout
                n  = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, B[oc])

                for ic in T.serial(Cin):
                    for kh in T.serial(K):
                        ih = oh + kh
                        for kw in T.serial(K):
                            iw = ow + kw
                            acc[0] += (
                                T.Cast(accum_dtype, X[n, ic, ih, iw])
                                * T.Cast(accum_dtype, Wt[oc, ic, kh, kw])
                            )

                Y[n, oc, oh, ow] = T.Cast(in_dtype, acc[0])

    return conv2d_kernel


# --------------------------------------------------------------------------- #
#          TileLang kernel :  Instance-Norm  +  divide-by-constant            #
# --------------------------------------------------------------------------- #
def _build_norm_div_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    divide_const: float,
    block_size: int = 256,
    in_dtype: str = "float16",
    accum_dtype: str = "float32",
):
    numel = N * C * H * W
    inv_div = 1.0 / float(divide_const)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def norm_div_kernel(
        X:      T.Tensor((N, C, H, W), in_dtype),   # conv output
        MEAN:   T.Tensor((N, C),       in_dtype),
        INVSTD: T.Tensor((N, C),       in_dtype),
        Y:      T.Tensor((N, C, H, W), in_dtype),   # output created by TL
    ):
        inv_div_f = T.Cast(accum_dtype, inv_div)

        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                w  = idx % W
                idx //= W
                h  = idx % H
                idx //= H
                c  = idx % C
                n  = idx // C

                x_val = X[n, c, h, w].astype(accum_dtype)
                m_val = MEAN[n, c].astype(accum_dtype)
                inv_s = INVSTD[n, c].astype(accum_dtype)

                norm = (x_val - m_val) * inv_s
                outv = norm * inv_div_f

                Y[n, c, h, w] = T.Cast(in_dtype, outv)

    return norm_div_kernel


# --------------------------------------------------------------------------- #
#                               PyTorch wrapper                               #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → InstanceNorm2d → divide(by const)  — all ops executed with
    TileLang kernels (conv kernel, then fused norm÷ kernel).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        divide_by: float,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_c   = int(in_channels)
        self.out_c  = int(out_channels)
        self.k      = int(kernel_size)
        self.div_by = float(divide_by)
        self.eps    = float(eps)

        # --- Conv2d parameters (same initialisation as torch) -------------- #
        self.weight = nn.Parameter(
            torch.empty(self.out_c, self.in_c, self.k, self.k)
        )
        self.bias   = nn.Parameter(torch.empty(self.out_c))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = self.in_c * self.k * self.k
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel caches
        self._conv_kernels: Dict[Tuple[int, int, int, str], callable] = {}
        self._norm_kernels: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_conv_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._conv_kernels:
            self._conv_kernels[key] = _build_conv2d_kernel(
                N,
                self.in_c,
                H,
                W,
                self.out_c,
                self.k,
                in_dtype=dtype,
            )
        return self._conv_kernels[key]

    # ------------------------------------------------------------------ #
    def _get_norm_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._norm_kernels:
            self._norm_kernels[key] = _build_norm_div_kernel(
                N,
                self.out_c,
                H,
                W,
                self.div_by,
                in_dtype=dtype,
            )
        return self._norm_kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, H_in, W_in = x_fp16.shape
        assert Cin == self.in_c, "in_channels mismatch"

        # ------------------- Conv2d ------------------- #
        conv_kernel = self._get_conv_kernel(N, H_in, W_in, "float16")
        y_fp16 = conv_kernel(x_fp16, w_fp16, b_fp16)

        # ---------------- mean / inv-std --------------- #
        y_f32   = y_fp16.to(torch.float32)
        mean_f  = y_f32.mean(dim=(2, 3))                                # (N, C)
        var_f   = y_f32.var(dim=(2, 3), unbiased=False)
        invstd_f = torch.rsqrt(var_f + self.eps)

        mean_fp16   = mean_f.to(device="cuda", dtype=torch.float16)
        invstd_fp16 = invstd_f.to(device="cuda", dtype=torch.float16)

        # ------------- norm + divide kernel ------------ #
        norm_kernel = self._get_norm_kernel(N, y_fp16.shape[2], y_fp16.shape[3], "float16")
        out_fp16    = norm_kernel(y_fp16, mean_fp16, invstd_fp16)

        return out_fp16.to(orig_dtype)