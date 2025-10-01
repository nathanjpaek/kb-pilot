"""
Problem Name: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.23 runtime_stats={'mean': 0.23, 'std': 0.00112, 'min': 0.228, 'max': 0.236, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.139, 'std': 0.00218, 'min': 0.136, 'max': 0.153, 'num_trials': 100}, 'speedup_ratio': 0.604}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_fused_pool_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    pool_k: int,
    pool_s: int,
    hard_min: float,
    hard_max: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    H_out = (H - pool_k) // pool_s + 1
    W_out = (W - pool_k) // pool_s + 1
    n_pool = H_out * W_out
    total_nc = N * C

    one_f = 1.0
    two_f = 2.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Out: T.Tensor((N, C, 1, 1), dtype),
    ):
        hard_min_c = T.Cast(accum_dtype, hard_min)
        hard_max_c = T.Cast(accum_dtype, hard_max)
        inv_one_c = T.Cast(accum_dtype, one_f)
        two_c = T.Cast(accum_dtype, two_f)
        pool_cnt_c = T.Cast(accum_dtype, n_pool)

        with T.Kernel(T.ceildiv(total_nc, threads_per_block), threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total_nc:
                n = idx // C
                c = idx % C

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for ho in range(H_out):
                    h_start = ho * pool_s
                    for wo in range(W_out):
                        w_start = wo * pool_s

                        mval = T.alloc_local((1,), accum_dtype)
                        mval[0] = T.Cast(accum_dtype, -3.4e38)

                        for kh in range(pool_k):
                            for kw in range(pool_k):
                                h_idx = h_start + kh
                                w_idx = w_start + kw
                                val = T.Cast(accum_dtype, X[n, c, h_idx, w_idx])
                                mval[0] = T.max(mval[0], val)

                        clp = T.max(mval[0], hard_min_c)
                        clp = T.min(clp, hard_max_c)
                        acc[0] += clp

                mean_val = acc[0] / pool_cnt_c

                # tanh via exp: tanh(x) = (1 - e^{-2x}) / (1 + e^{-2x})
                exp_val = T.exp(-two_c * mean_val)
                tanh_val = (inv_one_c - exp_val) / (inv_one_c + exp_val)

                Out[n, c, 0, 0] = T.Cast(dtype, tanh_val)

    return kernel


class ModelNew(nn.Module):
    """
    ConvTranspose2d followed by fused MaxPool2d → HardTanh → mean → tanh
    implemented with TileLang.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        maxpool_kernel_size: int,
        maxpool_stride: int,
        hardtanh_min: float,
        hardtanh_max: float,
    ):
        super().__init__()

        # --- ConvTranspose2d parameters -----------------------------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Store hyper-params
        self.stride = int(stride)
        self.padding = int(padding)
        self.pool_k = int(maxpool_kernel_size)
        self.pool_s = int(maxpool_stride)
        self.hard_min = float(hardtanh_min)
        self.hard_max = float(hardtanh_max)

        # Kernel cache
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str) -> callable:
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_pool_kernel(
                N,
                C,
                H,
                W,
                self.pool_k,
                self.pool_s,
                self.hard_min,
                self.hard_max,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        # Transposed convolution
        y = F.conv_transpose2d(
            x_fp16,
            w_fp16,
            b_fp16,
            stride=self.stride,
            padding=self.padding,
        )

        N, C, H, W = y.shape
        kernel = self._get_kernel(N, C, H, W, "float16")

        out_fp16 = kernel(y)

        return out_fp16.to(orig_dtype)