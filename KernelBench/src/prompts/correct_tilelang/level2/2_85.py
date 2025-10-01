"""
Problem Name: 85_Conv2d_GroupNorm_Scale_MaxPool_Clamp
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.105 runtime_stats={'mean': 0.105, 'std': 0.00314, 'min': 0.0996, 'max': 0.119, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.1, 'std': 0.00344, 'min': 0.0968, 'max': 0.121, 'num_trials': 100}, 'speedup_ratio': 0.952}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :  scale ×,   max-pool(k=k,stride=k),   clamp       #
# --------------------------------------------------------------------------- #
def _build_scale_pool_clamp_kernel(
    N: int,
    C: int,
    H_in: int,
    W_in: int,
    pool_k: int,
    clamp_min: float,
    clamp_max: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    H_out = (H_in - pool_k) // pool_k + 1
    W_out = (W_in - pool_k) // pool_k + 1
    TOT   = N * C * H_out * W_out

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, H_in, W_in), dtype),      # normalized feature-map
        S: T.Tensor((C,), dtype),                    # per-channel scale
        Y: T.Tensor((N, C, H_out, W_out), dtype),    # output
    ):
        cmin = T.Cast(accum_dtype, clamp_min)
        cmax = T.Cast(accum_dtype, clamp_max)

        with T.Kernel(T.ceildiv(TOT, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                wo  = idx %   W_out
                tmp1 = idx // W_out
                ho  = tmp1 %  H_out
                tmp2 = tmp1 // H_out
                c   = tmp2 %  C
                n   = tmp2 // C

                s_val = S[c].astype(accum_dtype)

                h0 = ho * pool_k
                w0 = wo * pool_k

                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = T.Cast(accum_dtype, -3.4e38)

                for kh in T.serial(pool_k):
                    for kw in T.serial(pool_k):
                        val = (
                            X[n, c, h0 + kh, w0 + kw].astype(accum_dtype)
                            * s_val
                        )
                        max_val[0] = T.max(max_val[0], val)

                clp = T.max(max_val[0], cmin)
                clp = T.min(clp, cmax)

                Y[n, c, ho, wo] = T.Cast(dtype, clp)

    return kernel


# --------------------------------------------------------------------------- #
#                     PyTorch wrapper using TileLang                          #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv2d → GroupNorm → (fused TileLang)  scale → MaxPool2d → clamp
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        num_groups: int,
        scale_shape: tuple,
        maxpool_kernel_size: int,
        clamp_min: float,
        clamp_max: float,
    ):
        super().__init__()

        # ---------------- Conv2d parameters ------------------------------- #
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

        # identical weight/bias initialisation
        torch.nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.conv.bias, -bound, bound)

        # ---------------- GroupNorm (same as reference) ------------------- #
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

        # ---------------- Learnable scale --------------------------------- #
        self.scale = nn.Parameter(torch.ones(scale_shape))

        # Hyper-parameters
        self.pool_k     = int(maxpool_kernel_size)
        self.clamp_min  = float(clamp_min)
        self.clamp_max  = float(clamp_max)

        # Kernel cache
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self,
        N: int,
        C: int,
        H_in: int,
        W_in: int,
        dtype: str,
    ):
        key = (N, C, H_in, W_in, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_scale_pool_clamp_kernel(
                N,
                C,
                H_in,
                W_in,
                self.pool_k,
                self.clamp_min,
                self.clamp_max,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # -------- PyTorch ops (conv + groupnorm) ------------------------- #
        y = self.conv(x)
        y = self.group_norm(y)

        # -------- Prepare for TileLang kernel --------------------------- #
        y_fp16 = y.to(device="cuda", dtype=torch.float16).contiguous()
        scale_fp16 = self.scale.to(device="cuda", dtype=torch.float16).contiguous().view(-1)

        N, C, H_in, W_in = y_fp16.shape
        kernel = self._get_kernel(N, C, H_in, W_in, "float16")

        out_fp16 = kernel(y_fp16, scale_fp16)

        return out_fp16.to(orig_dtype)