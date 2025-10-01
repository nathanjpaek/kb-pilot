"""
Problem Name: 78_ConvTranspose3d_Max_Max_Sum
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.516 runtime_stats={'mean': 0.516, 'std': 0.00149, 'min': 0.513, 'max': 0.522, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.53, 'std': 0.00181, 'min': 0.527, 'max': 0.541, 'num_trials': 100}, 'speedup_ratio': 1.03}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : fused (max-pool k=6,s=6) + channel-sum            #
# --------------------------------------------------------------------------- #
def _build_fused_pool_sum_kernel(
    N: int,
    C: int,
    D_in: int,
    H_in: int,
    W_in: int,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    K = 6  # effective kernel & stride
    D_out = (D_in - K) // K + 1
    H_out = (H_in - K) // K + 1
    W_out = (W_in - K) // K + 1
    TOT   = N * D_out * H_out * W_out

    neg_inf = -3.4028234663852886e38  # float32 −inf

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D_in, H_in, W_in), dtype),
        Y: T.Tensor((N, 1, D_out, H_out, W_out), dtype),
    ):
        neg_f = T.Cast(accum_dtype, neg_inf)

        with T.Kernel(T.ceildiv(TOT, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                w_out  = idx % W_out
                tmp1   = idx // W_out
                h_out  = tmp1 % H_out
                tmp2   = tmp1 // H_out
                d_out  = tmp2 % D_out
                n      = tmp2 // D_out

                base_d = d_out * K
                base_h = h_out * K
                base_w = w_out * K

                sum_acc = T.alloc_local((1,), accum_dtype)
                sum_acc[0] = T.Cast(accum_dtype, 0)

                for c in T.serial(C):
                    max_val = T.alloc_local((1,), accum_dtype)
                    max_val[0] = neg_f
                    for kd in T.serial(K):
                        for kh in T.serial(K):
                            for kw in T.serial(K):
                                val = T.Cast(
                                    accum_dtype,
                                    X[n, c, base_d + kd, base_h + kh, base_w + kw],
                                )
                                max_val[0] = T.max(max_val[0], val)
                    sum_acc[0] += max_val[0]

                Y[n, 0, d_out, h_out, w_out] = T.Cast(dtype, sum_acc[0])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with TileLang kernel                                        #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d → MaxPool3d(k=2,s=2) → MaxPool3d(k=3,s=3) → sum(dim=1,keepdim=True)
    The two pools + sum are fused into one TileLang kernel (k=6,s=6 + channel-sum).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()

        # ---- ConvTranspose3d parameters (identical init) ------------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.bias   = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size ** 3
        bound  = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # store hyper-params
        self.stride  = int(stride)
        self.padding = int(padding)

        # kernel cache : {(N,D,H,W,dtype) : kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            C = self.weight.shape[1]
            self._kern_cache[key] = _build_fused_pool_sum_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------------- ConvTranspose3d (cuDNN) ------------------------
        w = self.weight.to(device="cuda", dtype=torch.float16)
        b = self.bias.to(device="cuda", dtype=torch.float16)
        x = x.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose3d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
        )  # (N,C,D,H,W)

        N, C, D_in, H_in, W_in = y.shape

        # ---------------- Fused pool+sum TileLang kernel -----------------
        kernel = self._get_kernel(N, D_in, H_in, W_in, "float16")
        out_fp16 = kernel(y.contiguous())

        return out_fp16.to(orig_dtype)