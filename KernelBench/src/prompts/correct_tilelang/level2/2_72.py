"""
Problem Name: 72_ConvTranspose3d_BatchNorm_AvgPool_AvgPool
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=69.0 runtime_stats={'mean': 69.0, 'std': 0.0476, 'min': 69.0, 'max': 69.2, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 70.3, 'std': 0.0474, 'min': 70.2, 'max': 70.4, 'num_trials': 100}, 'speedup_ratio': 1.02}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : two consecutive AvgPool3d(k=2,s=2)              #
# --------------------------------------------------------------------------- #
def _make_double_avgpool_kernel(
    N: int,
    C: int,
    D0: int,
    H0: int,
    W0: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    # sizes after first and second pooling (PyTorch formula: floor((x-2)/2)+1)
    D1 = (D0 - 2) // 2 + 1
    H1 = (H0 - 2) // 2 + 1
    W1 = (W0 - 2) // 2 + 1

    D2 = (D1 - 2) // 2 + 1
    H2 = (H1 - 2) // 2 + 1
    W2 = (W1 - 2) // 2 + 1

    TOT = N * C * D2 * H2 * W2
    GRID = (TOT + threads_per_block - 1) // threads_per_block
    inv_64 = 1.0 / 64.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def pool64(
        X: T.Tensor((N, C, D0, H0, W0), dtype),
        Y: T.Tensor((N, C, D2, H2, W2), dtype),
    ):
        inv64_f = T.Cast(accum_dtype, inv_64)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w2 = idx % W2
                t1 = idx // W2
                h2 = t1 % H2
                t2 = t1 // H2
                d2 = t2 % D2
                t3 = t2 // D2
                c = t3 % C
                n = t3 // C

                base_d = d2 * 4
                base_h = h2 * 4
                base_w = w2 * 4

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for kd in T.serial(4):
                    d0 = base_d + kd
                    in_d_ok = d0 < D0
                    if in_d_ok:
                        for kh in T.serial(4):
                            h0 = base_h + kh
                            in_h_ok = h0 < H0
                            if in_h_ok:
                                for kw in T.serial(4):
                                    w0 = base_w + kw
                                    if w0 < W0:
                                        acc[0] += T.Cast(
                                            accum_dtype, X[n, c, d0, h0, w0]
                                        )

                Y[n, c, d2, h2, w2] = T.Cast(dtype, acc[0] * inv64_f)

    return pool64


# --------------------------------------------------------------------------- #
# Optimised PyTorch module                                                   #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d → BatchNorm3d → fused double AvgPool3d (TileLang)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias_shape,  # unused but kept for signature compatibility
    ):
        super().__init__()

        # --- ConvTranspose3d ------------------------------------------------
        self.weight = nn.Parameter(
            torch.empty(
                in_channels,
                out_channels,
                kernel_size,
                kernel_size,
                kernel_size,
            )
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size ** 3
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.stride = int(stride)
        self.padding = int(padding)

        # BatchNorm3d parameters / buffers
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var", torch.ones(out_channels))
        self.bn_eps = 1e-5
        self.bn_momentum = 0.1  # kept for interface; not used

        # kernel cache
        self._kern_cache: Dict[Tuple[int, int, int, int], callable] = {}

    # ------------------------------------------------------------------ #
    # helper to obtain (or compile) kernel
    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D0: int, H0: int, W0: int, dtype: str):
        key = (N, D0, H0, W0, dtype)
        if key not in self._kern_cache:
            C = self.weight.shape[1]  # out_channels
            self._kern_cache[key] = _make_double_avgpool_kernel(
                N, C, D0, H0, W0, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = "cuda"

        # -------- ConvTranspose3d (cuDNN) ---------------------------------
        w = self.weight.to(device=device, dtype=orig_dtype)
        b = self.bias.to(device=device, dtype=orig_dtype)
        x = x.to(device=device, dtype=orig_dtype)
        x = F.conv_transpose3d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
        )

        # -------- BatchNorm3d (cuDNN) -------------------------------------
        weight = self.bn_weight.to(device=device, dtype=orig_dtype)
        bias = self.bn_bias.to(device=device, dtype=orig_dtype)
        running_mean = self.running_mean.to(device=device, dtype=orig_dtype)
        running_var = self.running_var.to(device=device, dtype=orig_dtype)
        x = F.batch_norm(
            x,
            running_mean,
            running_var,
            weight,
            bias,
            training=self.training,
            momentum=self.bn_momentum,
            eps=self.bn_eps,
        )

        # -------- fused double AvgPool (TileLang) --------------------------
        x_fp16 = x.to(dtype=torch.float16).contiguous()
        N, C, D0, H0, W0 = x_fp16.shape
        kernel = self._get_kernel(N, D0, H0, W0, "float16")
        y_fp16 = kernel(x_fp16)

        return y_fp16.to(orig_dtype)