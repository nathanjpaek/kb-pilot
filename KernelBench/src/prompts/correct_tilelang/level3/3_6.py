"""
Problem Name: 6_GoogleNetInceptionModule
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=7.04 runtime_stats={'mean': 7.04, 'std': 0.0368, 'min': 7.02, 'max': 7.33, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 7.23, 'std': 0.0345, 'min': 7.19, 'max': 7.5, 'num_trials': 100}, 'speedup_ratio': 1.03}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : channel-concat of four tensors                    #
# --------------------------------------------------------------------------- #
def _build_concat_kernel(
    N: int,
    H: int,
    W: int,
    C1: int,
    C2: int,
    C3: int,
    C4: int,
    threads: int = 256,
    dtype: str = "float16",
):
    Ctot = C1 + C2 + C3 + C4
    spatial = N * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def concat4(
        A: T.Tensor((N, C1, H, W), dtype),
        B: T.Tensor((N, C2, H, W), dtype),
        C: T.Tensor((N, C3, H, W), dtype),
        D: T.Tensor((N, C4, H, W), dtype),
        Y: T.Tensor((N, Ctot, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(spatial, threads), threads=threads) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads + tx
            if idx < spatial:
                w = idx % W
                tmp = idx // W
                h = tmp % H
                n = tmp // H

                # ---- copy branch-1 ---------------------------------------
                for c in T.serial(C1):
                    Y[n, c, h, w] = A[n, c, h, w]

                # ---- copy branch-2 ---------------------------------------
                for c in T.serial(C2):
                    Y[n, C1 + c, h, w] = B[n, c, h, w]

                # ---- copy branch-3 ---------------------------------------
                for c in T.serial(C3):
                    Y[n, C1 + C2 + c, h, w] = C[n, c, h, w]

                # ---- copy branch-4 ---------------------------------------
                for c in T.serial(C4):
                    Y[n, C1 + C2 + C3 + c, h, w] = D[n, c, h, w]

    return concat4


# --------------------------------------------------------------------------- #
# PyTorch module with TileLang fusion                                         #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Inception-like block with a fused TileLang kernel for the final concatenation.
    """

    def __init__(
        self,
        in_channels: int,
        out_1x1: int,
        reduce_3x3: int,
        out_3x3: int,
        reduce_5x5: int,
        out_5x5: int,
        pool_proj: int,
    ):
        super().__init__()

        # identical sub-branches ------------------------------------------------
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)

        self.branch3r  = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3   = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)

        self.branch5r  = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5   = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)

        self.pool      = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool_proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)

        # channel sizes for concat kernel
        self.C1 = out_1x1
        self.C2 = out_3x3
        self.C3 = out_5x5
        self.C4 = pool_proj

        # kernel cache : keyed by (N,H,W,dtype)
        self._kern_cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_concat_kernel(
                N,
                H,
                W,
                self.C1,
                self.C2,
                self.C3,
                self.C4,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _conv2d(inp, weight, bias, stride=1, padding=0):
        return F.conv2d(inp, weight, bias, stride=stride, padding=padding)

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # move input to CUDA-fp16
        x_fp16 = x.to(device="cuda", dtype=torch.float16)

        # ----------------------- branch-1 (1×1) -----------------------------
        w1 = self.branch1x1.weight.to(device="cuda", dtype=torch.float16)
        b1 = self.branch1x1.bias.to(device="cuda", dtype=torch.float16)
        br1 = self._conv2d(x_fp16, w1, b1)

        # ----------------------- branch-2 (1×1 → 3×3) -----------------------
        w2r = self.branch3r.weight.to(device="cuda", dtype=torch.float16)
        b2r = self.branch3r.bias.to(device="cuda", dtype=torch.float16)
        t2  = self._conv2d(x_fp16, w2r, b2r)

        w2  = self.branch3.weight.to(device="cuda", dtype=torch.float16)
        b2  = self.branch3.bias.to(device="cuda", dtype=torch.float16)
        br2 = self._conv2d(t2, w2, b2, padding=1)

        # ----------------------- branch-3 (1×1 → 5×5) -----------------------
        w3r = self.branch5r.weight.to(device="cuda", dtype=torch.float16)
        b3r = self.branch5r.bias.to(device="cuda", dtype=torch.float16)
        t3  = self._conv2d(x_fp16, w3r, b3r)

        w3  = self.branch5.weight.to(device="cuda", dtype=torch.float16)
        b3  = self.branch5.bias.to(device="cuda", dtype=torch.float16)
        br3 = self._conv2d(t3, w3, b3, padding=2)

        # ----------------------- branch-4 (pool → 1×1) ----------------------
        p   = self.pool(x_fp16)
        wp4 = self.pool_proj.weight.to(device="cuda", dtype=torch.float16)
        bp4 = self.pool_proj.bias.to(device="cuda", dtype=torch.float16)
        br4 = self._conv2d(p, wp4, bp4)

        # ensure contiguous for kernel
        br1c = br1.contiguous()
        br2c = br2.contiguous()
        br3c = br3.contiguous()
        br4c = br4.contiguous()

        N, _, H, W = br1c.shape

        # ----------------------- TileLang concat ---------------------------
        kernel = self._get_kernel(N, H, W, "float16")
        out_fp16 = kernel(br1c, br2c, br3c, br4c)

        return out_fp16.to(orig_dtype)