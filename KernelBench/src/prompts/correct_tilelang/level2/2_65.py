"""
Problem Name: 65_Conv2d_AvgPool_Sigmoid_Sum
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0587 runtime_stats={'mean': 0.0587, 'std': 0.0332, 'min': 0.0488, 'max': 0.379, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0731, 'std': 0.0624, 'min': 0.0612, 'max': 0.691, 'num_trials': 100}, 'speedup_ratio': 1.25}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : AvgPool2d(k=2) + Sigmoid + Sum→scalar per sample #
# --------------------------------------------------------------------------- #
def _build_pool_sigmoid_sum_kernel(
    N: int,
    C: int,
    H_in: int,
    W_in: int,
    pool_k: int = 2,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    H_out = H_in // pool_k
    W_out = W_in // pool_k
    TOT_PER_SAMPLE = C * H_out * W_out
    pool_cnt_f = float(pool_k * pool_k)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H_in, W_in), dtype),   # conv output, pre-pool
        Y: T.Tensor((N,), dtype),                 # scalar per sample
    ):
        inv_pool = T.Cast(accum_dtype, 1.0 / pool_cnt_f)
        one_c    = T.Cast(accum_dtype, 1.0)

        with T.Kernel(N, threads=threads_per_block) as bn:   # one block == one sample
            tx   = T.get_thread_binding(0)

            part = T.alloc_local((1,), accum_dtype)
            part[0] = T.Cast(accum_dtype, 0)

            for it in T.serial(T.ceildiv(TOT_PER_SAMPLE, threads_per_block)):
                idx = it * threads_per_block + tx
                if idx < TOT_PER_SAMPLE:
                    w_out = idx % W_out
                    tmp   = idx // W_out
                    h_out = tmp % H_out
                    c     = tmp // H_out

                    base_h = h_out * pool_k
                    base_w = w_out * pool_k

                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, 0)

                    for kh in range(pool_k):
                        for kw in range(pool_k):
                            acc[0] += T.Cast(
                                accum_dtype,
                                X[bn, c, base_h + kh, base_w + kw],
                            )

                    avg = acc[0] * inv_pool
                    sig = one_c / (one_c + T.exp(-avg))    # sigmoid
                    part[0] += sig

            total = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        part[0],
                        True,
                        total[0],
                        tx,
                        dtype="handle",
                    )
                )

            if tx == 0:
                Y[bn] = T.Cast(dtype, total[0])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang kernel                                 #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → fused (AvgPool2d(k=2) → Sigmoid → Sum over C,H,W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_kernel_size: int,
    ):
        super().__init__()

        # ---------------- Conv2d parameters (identical init) ----------------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Hyper-params
        self.kernel_size = int(kernel_size)
        self.pool_k = int(pool_kernel_size)

        # Kernel cache  : (N,H_in,W_in,dtype) -> compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self,
        N: int,
        C: int,
        H_in: int,
        W_in: int,
        dtype: str = "float16",
    ):
        key = (N, H_in, W_in, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_pool_sigmoid_sum_kernel(
                N, C, H_in, W_in, self.pool_k, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move to CUDA + fp16
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16)

        # ---------------------- convolution (cuDNN) ------------------------
        y = F.conv2d(x_fp16, w_fp16, b_fp16, stride=1, padding=0)
        N, C, H_conv, W_conv = y.shape

        # ------------------- fused TileLang kernel ------------------------
        kernel = self._get_kernel(N, C, H_conv, W_conv, dtype="float16")
        out_fp16 = kernel(y.contiguous())          # (N,)

        return out_fp16.to(orig_dtype)