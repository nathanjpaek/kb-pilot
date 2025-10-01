"""
Problem Name: 21_EfficientNetMBConv
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.46 runtime_stats={'mean': 4.46, 'std': 0.0116, 'min': 4.45, 'max': 4.56, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.47, 'std': 0.00988, 'min': 4.45, 'max': 4.52, 'num_trials': 100}, 'speedup_ratio': 1.0}}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from typing import Dict, Tuple


# --------------------------------------------------------------------------- #
# TileLang kernel factory : element-wise add                                  #
# --------------------------------------------------------------------------- #
def _build_add_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    threads: int = 256,
    dtype: str = "float16",
):
    spatial = N * H * W  # each thread handles one (n,h,w) tile

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def add2(
        A: T.Tensor((N, C, H, W), dtype),
        B: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(spatial, threads), threads=threads) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads + tx
            if idx < spatial:
                w = idx % W
                tmp = idx // W
                h = tmp % H
                n = tmp // H
                for c in T.Parallel(C):
                    Y[n, c, h, w] = A[n, c, h, w] + B[n, c, h, w]

    return add2


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused residual add                                     #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    MBConv block with TileLang-accelerated residual addition.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        super().__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )

        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # kernel cache :  (N,C,H,W,dtype) -> compiled kernel
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_add_kernel(N, C, H, W, dtype=dtype)
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        if hasattr(self, "expand_conv"):
            x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.project_conv(x)

        if not self.use_residual:
            return x

        # ------------------ fused residual add in TileLang -----------------
        orig_dtype = x.dtype
        out_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        id_fp16 = identity.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, H, W = out_fp16.shape
        kernel = self._get_kernel(N, C, H, W, "float16")
        y_fp16 = kernel(out_fp16, id_fp16)  # element-wise add

        return y_fp16.to(orig_dtype)