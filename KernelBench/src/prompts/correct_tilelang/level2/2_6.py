"""
Problem Name: 6_Conv3d_Softmax_MaxPool_MaxPool
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.497 runtime_stats={'mean': 0.497, 'std': 0.0102, 'min': 0.491, 'max': 0.591, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.802, 'std': 0.00856, 'min': 0.794, 'max': 0.878, 'num_trials': 100}, 'speedup_ratio': 1.61}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factories
# --------------------------------------------------------------------------- #
def _make_softmax_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    spatial = N * D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def softmax_ch(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(spatial, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < spatial:
                w = idx % W
                tmp = idx // W
                h = tmp % H
                tmp //= H
                d = tmp % D
                tmp //= D
                n = tmp

                # First pass : sum(exp)
                sum_exp = T.alloc_local((1,), accum_dtype)
                sum_exp[0] = T.Cast(accum_dtype, 0)

                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sum_exp[0] += T.exp(val)

                inv_sum = T.Cast(accum_dtype, 1.0) / sum_exp[0]

                # Second pass : write normalised values
                for c in T.serial(C):
                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    out_val = T.exp(val) * inv_sum
                    Y[n, c, d, h, w] = T.Cast(dtype, out_val)

    return softmax_ch


def _make_pool_kernel(
    N: int,
    C: int,
    D_in: int,
    H_in: int,
    W_in: int,
    pool_k: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    stride = pool_k * pool_k  # effective after two successive pools
    D_out = D_in // stride
    H_out = H_in // stride
    W_out = W_in // stride
    total_out = N * C * D_out * H_out * W_out

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def maxpool_stride4(
        X: T.Tensor((N, C, D_in, H_in, W_in), dtype),
        Y: T.Tensor((N, C, D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(T.ceildiv(total_out, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total_out:
                ow = idx % W_out
                tmp = idx // W_out
                oh = tmp % H_out
                tmp //= H_out
                od = tmp % D_out
                tmp //= D_out
                c = tmp % C
                n = tmp // C

                base_d = od * stride
                base_h = oh * stride
                base_w = ow * stride

                local_max = T.alloc_local((1,), accum_dtype)
                local_max[0] = T.Cast(accum_dtype, -1.0e30)

                for kd2 in T.serial(pool_k):
                    for kd0 in T.serial(pool_k):
                        for kh2 in T.serial(pool_k):
                            for kh0 in T.serial(pool_k):
                                for kw2 in T.serial(pool_k):
                                    for kw0 in T.serial(pool_k):
                                        id_ = base_d + kd2 * pool_k + kd0
                                        ih_ = base_h + kh2 * pool_k + kh0
                                        iw_ = base_w + kw2 * pool_k + kw0
                                        val = T.Cast(
                                            accum_dtype,
                                            X[n, c, id_, ih_, iw_],
                                        )
                                        if val > local_max[0]:
                                            local_max[0] = val

                Y[n, c, od, oh, ow] = T.Cast(dtype, local_max[0])

    return maxpool_stride4


# --------------------------------------------------------------------------- #
# Optimised PyTorch Module
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d → Softmax(dim=1) → two MaxPool3d(kernel=2,stride=2)  ⇒  TileLang.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_kernel_size: int,
    ):
        super().__init__()

        # ---------------- Conv3d parameters (identical init) ----------------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # Hyper-params
        self.kernel_size = int(kernel_size)
        self.pool_k = int(pool_kernel_size)

        # Kernel caches
        self._softmax_cache: Dict[Tuple, callable] = {}
        self._pool_cache: Dict[Tuple, callable] = {}

    # --------------------------------------------------------------------- #
    # Helper : kernel retrieval / compilation
    # --------------------------------------------------------------------- #
    def _get_softmax_kernel(self, N, C, D, H, W, dtype):
        key = (N, C, D, H, W, dtype)
        if key not in self._softmax_cache:
            self._softmax_cache[key] = _make_softmax_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._softmax_cache[key]

    def _get_pool_kernel(self, N, C, D, H, W, dtype):
        key = (N, C, D, H, W, dtype)
        if key not in self._pool_cache:
            self._pool_cache[key] = _make_pool_kernel(
                N, C, D, H, W, self.pool_k, dtype=dtype
            )
        return self._pool_cache[key]

    # --------------------------------------------------------------------- #
    # Forward
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------------------- Conv (cuDNN) -------------------------------
        weight = self.weight
        bias = self.bias
        x = torch.nn.functional.conv3d(
            x, weight, bias, stride=1, padding=0, dilation=1
        )

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D1, H1, W1 = x_fp16.shape

        # ------------------- channel-wise softmax ---------------------------
        softmax_k = self._get_softmax_kernel(N, C, D1, H1, W1, "float16")
        x_soft = softmax_k(x_fp16)

        # -------------------- fused 2× max-pool -----------------------------
        pool_k = self._get_pool_kernel(N, C, D1, H1, W1, "float16")
        y_fp16 = pool_k(x_soft)

        return y_fp16.to(x.dtype)