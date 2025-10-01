"""
Problem Name: 29_ConvTranspose2d_InstanceNorm_Hardtanh_AvgPool_Clamp
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.53 runtime_stats={'mean': 1.53, 'std': 0.00328, 'min': 1.52, 'max': 1.54, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.14, 'std': 0.00823, 'min': 1.14, 'max': 1.21, 'num_trials': 100}, 'speedup_ratio': 0.745}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory : HardTanh → AvgPool2d(k=2,s=2) → Clamp                      #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    H_in: int,
    W_in: int,
    clamp_lo: float,
    clamp_hi: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    pool_k = 2
    pool_s = 2
    H_out = H_in // 2
    W_out = W_in // 2
    total = N * C * H_out * W_out

    hard_lo = -1.0
    hard_hi = 1.0
    inv4 = 0.25

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, H_in, W_in), dtype),
        Out: T.Tensor((N, C, H_out, W_out), dtype),
    ):
        h_lo = T.Cast(accum_dtype, hard_lo)
        h_hi = T.Cast(accum_dtype, hard_hi)
        c_lo = T.Cast(accum_dtype, clamp_lo)
        c_hi = T.Cast(accum_dtype, clamp_hi)
        inv4c = T.Cast(accum_dtype, inv4)

        with T.Kernel(T.ceildiv(total, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total:
                wo  = idx % W_out
                tmp = idx // W_out
                ho  = tmp % H_out
                tmp //= H_out
                c   = tmp % C
                n   = tmp // C

                base_h = ho * pool_s
                base_w = wo * pool_s

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for kh in T.serial(pool_k):
                    for kw in T.serial(pool_k):
                        val = T.Cast(accum_dtype, X[n, c, base_h + kh, base_w + kw])
                        val = T.max(val, h_lo)
                        val = T.min(val, h_hi)
                        acc[0] += val

                avg = acc[0] * inv4c
                avg = T.max(avg, c_lo)
                avg = T.min(avg, c_hi)

                Out[n, c, ho, wo] = T.Cast(dtype, avg)

    return kernel


# --------------------------------------------------------------------------- #
#                           PyTorch wrapper                                   #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → InstanceNorm2d → HardTanh → AvgPool2d(k=2,s=2) → Clamp
    All post-InstanceNorm steps are fused into one TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        instance_norm_features: int,
        min_val: float,
        max_val: float,
        eps: float = 1e-5,
    ):
        super().__init__()

        # ---------- ConvTranspose2d parameters (identical init) ------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # InstanceNorm hyper-parameter
        self.eps = float(eps)

        # Clamp range after pooling
        self.clamp_lo = float(min_val)
        self.clamp_hi = float(max_val)

        # Conv params
        self.stride  = int(stride)
        self.padding = int(padding)

        # kernel cache : {(N,C,H,W,dtype,lo,hi)} -> kernel
        self._kern_cache: Dict[Tuple, callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self, N: int, C: int, H: int, W: int, dtype: str
    ) -> callable:
        key = (N, C, H, W, dtype, self.clamp_lo, self.clamp_hi)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N,
                C,
                H,
                W,
                self.clamp_lo,
                self.clamp_hi,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # 1. ConvTranspose2d
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose2d(
            x_fp16,
            w_fp16,
            b_fp16,
            stride=self.stride,
            padding=self.padding,
        )  # (N,C,H,W)

        # 2. InstanceNorm2d (per-sample-channel)
        y_fp32 = y.to(torch.float32)
        mean = y_fp32.mean(dim=(2, 3), keepdim=True)
        var  = y_fp32.var(dim=(2, 3), unbiased=False, keepdim=True)
        y_norm = (y_fp32 - mean) / torch.sqrt(var + self.eps)
        y_norm_fp16 = y_norm.to(torch.float16).contiguous()

        # 3. Fused TileLang kernel
        N, C, H_in, W_in = y_norm_fp16.shape
        kernel = self._get_kernel(N, C, H_in, W_in, "float16")
        out_fp16 = kernel(y_norm_fp16)

        return out_fp16.to(orig_dtype)