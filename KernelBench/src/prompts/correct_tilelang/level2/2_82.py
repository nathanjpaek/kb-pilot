"""
Problem Name: 82_Conv2d_Tanh_Scaling_BiasAdd_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0675 runtime_stats={'mean': 0.0675, 'std': 0.0225, 'min': 0.0568, 'max': 0.269, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0736, 'std': 0.0188, 'min': 0.0646, 'max': 0.236, 'num_trials': 100}, 'speedup_ratio': 1.09}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory                                                               #
# --------------------------------------------------------------------------- #
def _build_fused_pool_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    pool_k: int,
    scale_val: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    H_out = (H - pool_k) // pool_k + 1
    W_out = (W - pool_k) // pool_k + 1
    TOTAL = N * C * H_out * W_out

    scale_c = float(scale_val)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),            # conv output
        B: T.Tensor((C,), dtype),                    # bias to add
        Y: T.Tensor((N, C, H_out, W_out), dtype),    # result
    ):
        scale_const = T.Cast(accum_dtype, scale_c)
        minus_inf   = T.Cast(accum_dtype, -3.4e38)

        with T.Kernel(T.ceildiv(TOTAL, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOTAL:
                ow  = idx % W_out
                tmp = idx // W_out
                oh  = tmp % H_out
                tmp //= H_out
                c   = tmp % C
                n   = tmp // C

                base_h = oh * pool_k
                base_w = ow * pool_k

                mval = T.alloc_local((1,), accum_dtype)
                mval[0] = minus_inf

                for kh in range(pool_k):
                    h_idx = base_h + kh
                    if h_idx < H:
                        for kw in range(pool_k):
                            w_idx = base_w + kw
                            if w_idx < W:
                                v = T.Cast(accum_dtype, X[n, c, h_idx, w_idx])
                                v = T.tanh(v)                      # tanh
                                v = v * scale_const               # scaling
                                v = v + T.Cast(accum_dtype, B[c]) # bias add
                                mval[0] = T.max(mval[0], v)       # max

                Y[n, c, oh, ow] = T.Cast(dtype, mval[0])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                              #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → tanh → scale → bias add → MaxPool2d
    The post-convolution pipeline is fused into a single TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scaling_factor: float,
        bias_shape: tuple,
        pool_kernel_size: int,
    ):
        super().__init__()

        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size  = int(kernel_size)
        self.pool_k       = int(pool_kernel_size)
        self.scale        = float(scaling_factor)

        # ---- Conv2d parameters (identical init to nn.Conv2d) --------------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # ---- extra bias added after scaling ------------------------------
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # ---- kernel cache -------------------------------------------------
        self._kernels: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_fused_pool_kernel(
                N,
                self.out_channels,
                H,
                W,
                self.pool_k,
                self.scale,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ----- move & cast to fp16 ----------------------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_conv_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()

        # ----- convolution (cuDNN) ----------------------------------------
        y_conv = F.conv2d(x_fp16, w_fp16, b_conv_fp16, stride=1, padding=0)

        # ----- fused TileLang kernel --------------------------------------
        N, C, Hc, Wc = y_conv.shape
        kernel = self._get_kernel(N, Hc, Wc, "float16")

        bias_fp16 = self.bias.view(-1).to(device="cuda", dtype=torch.float16).contiguous()
        y_fp16 = kernel(y_conv.contiguous(), bias_fp16)

        return y_fp16.to(orig_dtype)