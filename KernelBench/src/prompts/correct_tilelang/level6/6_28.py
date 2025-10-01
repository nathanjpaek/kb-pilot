"""
Problem Name: 28_Conv2d_Softmax_AvgPool_AvgPool_ResidualAdd_Tanh
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.967 runtime_stats={'mean': 0.967, 'std': 0.0293, 'min': 0.957, 'max': 1.23, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.61, 'std': 0.0322, 'min': 1.6, 'max': 1.85, 'num_trials': 100}, 'speedup_ratio': 1.66}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                      Kernel-factory : channel-wise Softmax                  #
# --------------------------------------------------------------------------- #
def _build_softmax_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    NUM_PIX = N * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, H, W), dtype),
        Out: T.Tensor((N, C, H, W), dtype),
    ):
        neg_inf = T.Cast(accum_dtype, -3.4028234663852886e38)
        with T.Kernel(T.ceildiv(NUM_PIX, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < NUM_PIX:
                w  = idx % W
                tmp = idx // W
                h  = tmp % H
                n  = tmp // H

                # 1. maximum
                m = T.alloc_local((1,), accum_dtype)
                m[0] = neg_inf
                for c in T.serial(C):
                    v = T.Cast(accum_dtype, X[n, c, h, w])
                    if v > m[0]:
                        m[0] = v

                # 2. exp ‑ m and sum
                exps = T.alloc_local((C,), accum_dtype)
                s = T.alloc_local((1,), accum_dtype)
                s[0] = T.Cast(accum_dtype, 0)
                for c in T.serial(C):
                    e = T.exp(T.Cast(accum_dtype, X[n, c, h, w]) - m[0])
                    exps[c] = e
                    s[0] += e

                inv_s = T.Cast(accum_dtype, 1.0) / s[0]

                # 3. normalise
                for c in T.serial(C):
                    Out[n, c, h, w] = T.Cast(dtype, exps[c] * inv_s)

    return kernel


# --------------------------------------------------------------------------- #
#          Kernel-factory : (AvgPool2 → AvgPool2) + Residual + tanh           #
# --------------------------------------------------------------------------- #
def _build_pool_add_tanh_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    pool_k: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    block_size: int = 256,
):
    # After two successive k×k, s=k average pools (no padding)
    H1 = (H // pool_k)
    W1 = (W // pool_k)
    H2 = (H1 // pool_k)
    W2 = (W1 // pool_k)

    patch = pool_k * pool_k          # dimension of fused patch (k²)
    denom = float(patch * patch)     # k⁴ elements in 2-stage pooling
    TOT = N * C * H2 * W2

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, H, W), dtype),      # softmax output
        R:   T.Tensor((N, C), dtype),            # residual vector
        Out: T.Tensor((N, C, H2, W2), dtype),
    ):
        denom_c = T.Cast(accum_dtype, denom)
        one  = T.Cast(accum_dtype, 1.0)
        two  = T.Cast(accum_dtype, 2.0)

        with T.Kernel(T.ceildiv(TOT, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOT:
                wo2 = idx % W2
                tmp = idx // W2
                ho2 = tmp % H2
                tmp //= H2
                c   = tmp % C
                n   = tmp // C

                base_h = ho2 * patch
                base_w = wo2 * patch

                s = T.alloc_local((1,), accum_dtype)
                s[0] = T.Cast(accum_dtype, 0)

                for kh in T.serial(patch):
                    for kw in T.serial(patch):
                        v = T.Cast(accum_dtype, X[n, c, base_h + kh, base_w + kw])
                        s[0] += v

                avg = s[0] / denom_c
                avg += T.Cast(accum_dtype, R[n, c])

                exp_val = T.exp(-two * avg)
                tanh_val = (one - exp_val) / (one + exp_val)

                Out[n, c, ho2, wo2] = T.Cast(dtype, tanh_val)

    return kernel, H2, W2


# --------------------------------------------------------------------------- #
#                              PyTorch wrapper                                #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d  →  Softmax(dim=1)  →  AvgPool2d(k)  →  AvgPool2d(k)
           →  +Linear-derived residual  →  tanh
    Softmax and the pooled-add-tanh stages are realised with TileLang kernels.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int, in_features: int):
        super().__init__()

        # -------- Conv2d (manual parameter initialisation) ----------------
        w_shape = (out_channels, in_channels, kernel_size, kernel_size)
        self.conv_weight = nn.Parameter(torch.empty(w_shape))
        self.conv_bias   = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # -------- Linear (manual parameter initialisation) ----------------
        self.linear_weight = nn.Parameter(torch.empty(out_channels, in_features))
        self.linear_bias   = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.linear_weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_features)
            torch.nn.init.uniform_(self.linear_bias, -bound, bound)

        # Hyper-parameters
        self.kernel_size = int(kernel_size)
        self.pool_k      = int(pool_size)

        # Kernel caches
        self._softmax_cache: Dict[Tuple[int, int, int, int, str], callable] = {}
        self._pool_cache:    Dict[Tuple[int, int, int, int, str], Tuple[callable, int, int]] = {}

    # ------------------------------------------------------------------ #
    def _get_softmax_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._softmax_cache:
            self._softmax_cache[key] = _build_softmax_kernel(N, C, H, W, dtype=dtype)
        return self._softmax_cache[key]

    def _get_pool_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._pool_cache:
            self._pool_cache[key] = _build_pool_add_tanh_kernel(
                N, C, H, W, self.pool_k, dtype=dtype
            )
        return self._pool_cache[key]  # returns (kernel, H2, W2)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------- Conv2d ------------------------------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        w_fp16 = self.conv_weight.to(device="cuda", dtype=torch.float16)
        b_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16)

        y = F.conv2d(x_fp16, w_fp16, b_fp16)          # (N,C,H,W)
        N, C, H, W = y.shape

        # ---------- Softmax kernel ---------------------------------------
        softmax_kernel = self._get_softmax_kernel(N, C, H, W, "float16")
        y_sm = softmax_kernel(y.contiguous())

        # ---------- Residual (PyTorch GEMM) ------------------------------
        inp_flat = x_fp16.view(N, -1)                # (N, in_features)
        lin_w = self.linear_weight.to(device="cuda", dtype=torch.float16)
        lin_b = self.linear_bias.to(device="cuda", dtype=torch.float16)
        res = torch.addmm(lin_b, inp_flat, lin_w.t()).to(torch.float16)  # (N,C)

        # ---------- Fused pool + add + tanh ------------------------------
        pool_kernel, H2, W2 = self._get_pool_kernel(N, C, H, W, "float16")
        out_fp16 = pool_kernel(y_sm, res.contiguous())

        return out_fp16.to(orig_dtype)