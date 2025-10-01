"""
Problem Name: 10_ConvTranspose2d_MaxPool_Hardtanh_Mean_Tanh
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.262 runtime_stats={'mean': 0.262, 'std': 0.0444, 'min': 0.244, 'max': 0.639, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.14, 'std': 0.02, 'min': 0.132, 'max': 0.331, 'num_trials': 100}, 'speedup_ratio': 0.534}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn

import tilelang
import tilelang.language as T


# ------------------------- Kernel Factories -------------------------------- #
def _build_pool_sum_kernel(
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
    total_elems = N * C * H_out * W_out

    hard_min_c = float(hard_min)
    hard_max_c = float(hard_max)

    @tilelang.jit
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Sum: T.Tensor((N, C, 1, 1), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(total_elems, threads_per_block), threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            gidx = bx * threads_per_block + tx
            if gidx < total_elems:
                # Decompose flattened index -> (n,c,ho,wo)
                wc = H_out * W_out
                nc = C * wc
                n = gidx // nc
                rem = gidx - n * nc
                c = rem // wc
                pos = rem - c * wc
                ho = pos // W_out
                wo = pos - ho * W_out

                h_start = ho * pool_s
                w_start = wo * pool_s

                # local register for running max
                mval = T.Cast(accum_dtype, -3.4e38)

                for kh in T.unroll(pool_k):
                    for kw in T.unroll(pool_k):
                        val = T.Cast(
                            accum_dtype, X[n, c, h_start + kh, w_start + kw]
                        )
                        mval = T.max(mval, val)

                # clamp (HardTanh)
                mval = T.max(mval, T.Cast(accum_dtype, hard_min_c))
                mval = T.min(mval, T.Cast(accum_dtype, hard_max_c))

                # Accumulate spatial sum with atomic add
                T.atomic_add(Sum[n, c, 0, 0], mval)

    return kernel, n_pool


def _build_finalize_kernel(
    N: int,
    C: int,
    n_pool: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    inv_one = 1.0
    two_f = 2.0
    total_nc = N * C

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        Sum: T.Tensor((N, C, 1, 1), accum_dtype),
        Out: T.Tensor((N, C, 1, 1), dtype),
    ):
        pool_cnt_c = T.Cast(accum_dtype, n_pool)
        one_c = T.Cast(accum_dtype, inv_one)
        two_c = T.Cast(accum_dtype, two_f)

        with T.Kernel(T.ceildiv(total_nc, threads_per_block), threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < total_nc:
                n = idx // C
                c = idx - n * C

                mean_val = Sum[n, c, 0, 0] / pool_cnt_c

                # tanh via exp for numerical stability
                exp_val = T.exp(-two_c * mean_val)
                tanh_val = (one_c - exp_val) / (one_c + exp_val)

                Out[n, c, 0, 0] = T.Cast(dtype, tanh_val)

    return kernel


# ---------------------------- PyTorch Module ------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d followed by fused MaxPool2d → HardTanh → mean → tanh,
    where pooling + HardTanh + mean + tanh are implemented by two
    high-performance TileLang kernels.
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

        # Transposed-conv parameters
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Store hyper-parameters
        self.stride = int(stride)
        self.padding = int(padding)
        self.pool_k = int(maxpool_kernel_size)
        self.pool_s = int(maxpool_stride)
        self.hard_min = float(hardtanh_min)
        self.hard_max = float(hardtanh_max)

        # Kernel caches
        self._sum_kernel_cache: Dict[Tuple[int, int, int, int, str], Tuple[callable, int]] = {}
        self._fin_kernel_cache: Dict[Tuple[int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_sum_kernel(
        self, N: int, C: int, H: int, W: int, dtype: str
    ) -> Tuple[callable, int]:
        key = (N, C, H, W, dtype)
        if key not in self._sum_kernel_cache:
            self._sum_kernel_cache[key] = _build_pool_sum_kernel(
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
        return self._sum_kernel_cache[key]

    def _get_finalize_kernel(self, N: int, C: int, n_pool: int, dtype: str) -> callable:
        key = (N, C, dtype)
        if key not in self._fin_kernel_cache:
            self._fin_kernel_cache[key] = _build_finalize_kernel(
                N,
                C,
                n_pool,
                dtype=dtype,
            )
        return self._fin_kernel_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Convert tensors to fp16 on CUDA for the custom kernels
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        # 1. Transposed convolution
        y = F.conv_transpose2d(
            x_fp16,
            w_fp16,
            b_fp16,
            stride=self.stride,
            padding=self.padding,
        )

        N, C, H, W = y.shape

        # 2. Pooled-sum kernel (atomic add accumulation)
        sum_kernel, n_pool = self._get_sum_kernel(N, C, H, W, "float16")

        # Accumulator buffer in FP32, zero-inited
        sum_buf = torch.zeros((N, C, 1, 1), dtype=torch.float32, device="cuda")
        sum_kernel(y, sum_buf)

        # 3. Finalize kernel: mean + tanh  → FP16 output
        finalize_kernel = self._get_finalize_kernel(N, C, n_pool, "float16")
        out_fp16 = finalize_kernel(sum_buf)

        return out_fp16.to(orig_dtype)