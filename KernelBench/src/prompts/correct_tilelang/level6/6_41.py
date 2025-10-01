"""
Problem Name: 41_ConvTranspose2d_BatchNorm_Sigmoid_ConvTranspose2d_BatchNorm_Sigmoid
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.565 runtime_stats={'mean': 0.565, 'std': 0.00952, 'min': 0.556, 'max': 0.649, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.06, 'std': 0.00919, 'min': 1.05, 'max': 1.14, 'num_trials': 100}, 'speedup_ratio': 1.88}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory :   BatchNorm2d (inference)  →  Sigmoid             #
# --------------------------------------------------------------------------- #

def _build_bn_sigmoid_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    eps: float = 1e-5,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT = N * C * H * W
    GRID = (TOT + block_size - 1) // block_size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:     T.Tensor((N, C, H, W), dtype),   # input
        Gamma: T.Tensor((C,), dtype),           # BN weight
        Beta:  T.Tensor((C,), dtype),           # BN bias
        RMean: T.Tensor((C,), dtype),           # running mean
        RVar:  T.Tensor((C,), dtype),           # running var
        Y:     T.Tensor((N, C, H, W), dtype),   # output
    ):
        eps_c  = T.Cast(accum_dtype, eps)
        one_c  = T.Cast(accum_dtype, 1.0)

        with T.Kernel(GRID, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOT:
                w = idx % W
                tmp = idx // W
                h = tmp % H
                tmp //= H
                c = tmp % C
                n = tmp // C

                val   = T.Cast(accum_dtype, X[n, c, h, w])
                mean  = T.Cast(accum_dtype, RMean[c])
                var   = T.Cast(accum_dtype, RVar[c])
                gamma = T.Cast(accum_dtype, Gamma[c])
                beta  = T.Cast(accum_dtype, Beta[c])

                norm = (val - mean) / T.sqrt(var + eps_c)
                bn   = norm * gamma + beta
                sig  = one_c / (one_c + T.exp(-bn))
                Y[n, c, h, w] = T.Cast(dtype, sig)

    return kernel


# --------------------------------------------------------------------------- #
#                             PyTorch wrapper                                 #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → BatchNorm2d → Sigmoid → ConvTranspose2d → BatchNorm2d → Sigmoid
    BatchNorm+Sigmoid pairs are fused into TileLang kernels.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        # ---------------- ConvTranspose2d #1 parameters -------------------
        w_shape1 = (in_channels, out_channels, kernel_size, kernel_size)
        self.w1 = nn.Parameter(torch.empty(w_shape1))
        torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        fan_in1 = in_channels * kernel_size * kernel_size
        bound1 = 1 / math.sqrt(fan_in1)
        self.b1 = nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.b1, -bound1, bound1)

        # ---------------- BatchNorm2d #1 parameters -----------------------
        self.gamma1 = nn.Parameter(torch.ones(out_channels))
        self.beta1  = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("rm1", torch.zeros(out_channels))
        self.register_buffer("rv1", torch.ones(out_channels))
        self.bn_eps = 1e-5  # default

        # ---------------- ConvTranspose2d #2 parameters -------------------
        w_shape2 = (out_channels, out_channels, kernel_size, kernel_size)
        self.w2 = nn.Parameter(torch.empty(w_shape2))
        torch.nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        fan_in2 = out_channels * kernel_size * kernel_size
        bound2 = 1 / math.sqrt(fan_in2)
        self.b2 = nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.b2, -bound2, bound2)

        # ---------------- BatchNorm2d #2 parameters -----------------------
        self.gamma2 = nn.Parameter(torch.ones(out_channels))
        self.beta2  = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("rm2", torch.zeros(out_channels))
        self.register_buffer("rv2", torch.ones(out_channels))

        # Kernel cache  : {(N,C,H,W,dtype) : kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_bn_sigmoid_kernel(
                N, C, H, W, eps=self.bn_eps, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # -------------------- First ConvTranspose2d ----------------------
        x = x.to(device="cuda", dtype=torch.float16)
        y1 = F.conv_transpose2d(x, self.w1.to("cuda", torch.float16), self.b1.to("cuda", torch.float16))
        N1, C1, H1, W1 = y1.shape

        # -------- fused (BN1 + Sigmoid) ----------------------------------
        k1 = self._get_kernel(N1, C1, H1, W1, "float16")
        y1 = k1(
            y1.contiguous(),
            self.gamma1.to("cuda", torch.float16).contiguous(),
            self.beta1.to("cuda", torch.float16).contiguous(),
            self.rm1.to("cuda", torch.float16).contiguous(),
            self.rv1.to("cuda", torch.float16).contiguous(),
        )

        # -------------------- Second ConvTranspose2d ---------------------
        y2 = F.conv_transpose2d(
            y1,
            self.w2.to("cuda", torch.float16),
            self.b2.to("cuda", torch.float16),
        )
        N2, C2, H2, W2 = y2.shape

        # -------- fused (BN2 + Sigmoid) ----------------------------------
        k2 = self._get_kernel(N2, C2, H2, W2, "float16")
        y2 = k2(
            y2.contiguous(),
            self.gamma2.to("cuda", torch.float16).contiguous(),
            self.beta2.to("cuda", torch.float16).contiguous(),
            self.rm2.to("cuda", torch.float16).contiguous(),
            self.rv2.to("cuda", torch.float16).contiguous(),
        )

        return y2.to(orig_dtype)