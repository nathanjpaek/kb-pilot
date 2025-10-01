"""
Problem Name: 93_ConvTranspose3d_Tanh_AvgPool_BiasAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.66 runtime_stats={'mean': 1.66, 'std': 0.0292, 'min': 1.65, 'max': 1.91, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.0, 'std': 0.012, 'min': 1.99, 'max': 2.11, 'num_trials': 100}, 'speedup_ratio': 1.2}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------- #
#                  TileLang kernel factory: tanh → AvgPool → +bias      #
# --------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    Din: int,
    Hin: int,
    Win: int,
    pool_k: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    # Spatial sizes after pooling (stride = kernel_size = pool_k)
    Dout = (Din - pool_k) // pool_k + 1
    Hout = (Hin - pool_k) // pool_k + 1
    Wout = (Win - pool_k) // pool_k + 1

    TOT = N * C * Dout * Hout * Wout
    inv_k3 = 1.0 / float(pool_k ** 3)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def fused(
        X: T.Tensor((N, C, Din, Hin, Win), dtype),
        B: T.Tensor((C,), dtype),
        Y: T.Tensor((N, C, Dout, Hout, Wout), dtype),
    ):
        inv_c = T.Cast(accum_dtype, inv_k3)
        one_c = T.Cast(accum_dtype, 1.0)
        two_c = T.Cast(accum_dtype, 2.0)

        with T.Kernel(T.ceildiv(TOT, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w_out  = idx % Wout
                t1     = idx // Wout
                h_out  = t1 % Hout
                t2     = t1 // Hout
                d_out  = t2 % Dout
                t3     = t2 // Dout
                c      = t3 % C
                n      = t3 // C

                base_d = d_out * pool_k
                base_h = h_out * pool_k
                base_w = w_out * pool_k

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for kd in T.serial(pool_k):
                    d_in = base_d + kd
                    if d_in < Din:
                        for kh in T.serial(pool_k):
                            h_in = base_h + kh
                            if h_in < Hin:
                                for kw in T.serial(pool_k):
                                    w_in = base_w + kw
                                    if w_in < Win:
                                        x_val = T.Cast(
                                            accum_dtype,
                                            X[n, c, d_in, h_in, w_in],
                                        )
                                        # tanh(x)  = (1 - e^{-2x}) / (1 + e^{-2x})
                                        e = T.exp(-two_c * x_val)
                                        tanh_x = (one_c - e) / (one_c + e)
                                        acc[0] += tanh_x

                avg = acc[0] * inv_c + T.Cast(accum_dtype, B[c])
                Y[n, c, d_out, h_out, w_out] = T.Cast(dtype, avg)

    return fused


# --------------------------------------------------------------------- #
#                           Optimised PyTorch module                    #
# --------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  tanh  →  AvgPool3d(pool_size)  →  +bias
    tanh, pooling and bias-add are fused into a single TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias_shape: tuple,
        pool_size: int,
    ):
        super().__init__()
        # ConvTranspose3d (kept as cuDNN op; default init preserved)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        # learnable bias
        self.bias = nn.Parameter(torch.randn(bias_shape))

        self.pool_k = int(pool_size)

        # kernel cache  {(N,D,H,W,dtype): kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self,
        N: int,
        D: int,
        H: int,
        W: int,
        dtype: str = "float16",
    ) -> callable:
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            C = self.conv_transpose.out_channels
            self._kern_cache[key] = _build_fused_kernel(
                N,
                C,
                D,
                H,
                W,
                self.pool_k,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # 1) ConvTranspose3d
        x = self.conv_transpose(x)

        # 2) Prepare for TileLang kernel
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = x_fp16.shape

        kernel = self._get_kernel(N, D, H, W, "float16")
        bias_flat = self.bias.to(device="cuda", dtype=torch.float16).contiguous().view(-1)

        # 3) Launch fused kernel
        y_fp16 = kernel(x_fp16, bias_flat)

        return y_fp16.to(orig_dtype)