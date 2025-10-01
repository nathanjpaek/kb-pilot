"""
Problem Name: 79_Conv3d_Multiply_InstanceNorm_Clamp_Multiply_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.689 runtime_stats={'mean': 0.689, 'std': 0.0021, 'min': 0.686, 'max': 0.698, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.828, 'std': 0.00173, 'min': 0.825, 'max': 0.837, 'num_trials': 100}, 'speedup_ratio': 1.2}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :   clamp → mul → channel-max                       #
# --------------------------------------------------------------------------- #
def _build_clamp_mul_max_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    clamp_min: float,
    clamp_max: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    voxels = N * D * H * W
    neg_inf = -3.4028234663852886e38  # −inf in fp32

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), dtype),   # input after InstNorm
        M: T.Tensor((C,), dtype),              # per-channel multiplier
        Y: T.Tensor((N, D, H, W), dtype),      # output (channel-max)
    ):
        cmin = T.Cast(accum_dtype, clamp_min)
        cmax = T.Cast(accum_dtype, clamp_max)
        ninf = T.Cast(accum_dtype, neg_inf)

        with T.Kernel(T.ceildiv(voxels, threads_per_block), threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < voxels:
                w  = idx % W
                tmp1 = idx // W
                h  = tmp1 % H
                tmp2 = tmp1 // H
                d  = tmp2 % D
                n  = tmp2 // D

                max_val = T.alloc_local((1,), accum_dtype)
                max_val[0] = ninf

                for c in T.serial(C):
                    val = X[n, c, d, h, w].astype(accum_dtype)
                    val = T.max(val, cmin)
                    val = T.min(val, cmax)
                    val *= M[c].astype(accum_dtype)
                    max_val[0] = T.max(max_val[0], val)

                Y[n, d, h, w] = T.Cast(dtype, max_val[0])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper using TileLang                                              #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d → ×multiplier → InstanceNorm3d → (fused TileLang) clamp → ×multiplier → channel-max
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        multiplier_shape: tuple,
        clamp_min: float,
        clamp_max: float,
    ):
        super().__init__()

        # ---------------- Conv3d parameters (identical init) --------------- #
        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size, kernel_size, kernel_size
            )
        )
        self.bias = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size ** 3
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---------------- Learnable multiplier ----------------------------- #
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

        # ---------------- InstanceNorm3d ----------------------------------- #
        self.instance_norm = nn.InstanceNorm3d(out_channels, affine=False)

        # Hyper-params
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Kernel cache
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ---------------------------------------------------------------------- #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_clamp_mul_max_kernel(
                N,
                C,
                D,
                H,
                W,
                self.clamp_min,
                self.clamp_max,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ---------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # Move parameters & input to CUDA fp16 (once)
        self.weight.data = self.weight.data.to(device="cuda", dtype=torch.float16)
        self.bias.data = self.bias.data.to(device="cuda", dtype=torch.float16)
        self.multiplier.data = self.multiplier.data.to(device="cuda", dtype=torch.float16)
        self.instance_norm.to(device="cuda", dtype=torch.float16)

        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # ---------------- Conv3d ------------------------------------------ #
        y = F.conv3d(x_fp16, self.weight, self.bias)

        # ---------------- First multiply ---------------------------------- #
        y = y * self.multiplier  # broadcast over spatial dims

        # ---------------- InstanceNorm3d ---------------------------------- #
        y = self.instance_norm(y)

        # ---------------- Fused TileLang kernel --------------------------- #
        N, C, D, H, W = y.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        mult_flat = self.multiplier.view(-1)
        out_fp16 = kernel(y.contiguous(), mult_flat)

        return out_fp16.to(orig_dtype)