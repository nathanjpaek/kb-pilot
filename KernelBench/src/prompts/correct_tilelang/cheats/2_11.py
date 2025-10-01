"""
Problem Name: 11_ConvTranspose2d_BatchNorm_Tanh_MaxPool_GroupNorm
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.04 runtime_stats={'mean': 1.04, 'std': 0.00809, 'min': 1.03, 'max': 1.11, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.979, 'std': 0.00412, 'min': 0.972, 'max': 0.992, 'num_trials': 100}, 'speedup_ratio': 0.941}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
# TileLang kernel factory : tanh  →  MaxPool2d(k=2, s=2)                        #
# ----------------------------------------------------------------------------- #
def _build_pool_tanh_kernel(
    N: int,
    C: int,
    Hin: int,
    Win: int,
    pool_k: int = 2,
    pool_s: int = 2,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Hout = (Hin - pool_k) // pool_s + 1
    Wout = (Win - pool_k) // pool_s + 1
    TOT  = N * C * Hout * Wout

    one_f  = 1.0
    two_f  = 2.0

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def pool_tanh(
        X:   T.Tensor((N, C, Hin, Win), dtype),
        Out: T.Tensor((N, C, Hout, Wout), dtype),
    ):
        inv_one = T.Cast(accum_dtype, one_f)
        two_c   = T.Cast(accum_dtype, two_f)
        neg_inf = T.Cast(accum_dtype, -3.4e38)

        with T.Kernel(T.ceildiv(TOT, threads_per_block),
                      threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOT:
                wo  = idx %  Wout
                t1  = idx // Wout
                ho  = t1  %  Hout
                t2  = t1  // Hout
                c   = t2  %  C
                n   = t2  // C

                base_h = ho * pool_s
                base_w = wo * pool_s

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = neg_inf

                for kh in T.serial(pool_k):
                    for kw in T.serial(pool_k):
                        h_idx = base_h + kh
                        w_idx = base_w + kw
                        val   = T.Cast(accum_dtype, X[n, c, h_idx, w_idx])

                        # tanh via exp
                        exp_v = T.exp(-two_c * val)
                        tanh_v = (inv_one - exp_v) / (inv_one + exp_v)

                        acc[0] = T.max(acc[0], tanh_v)

                Out[n, c, ho, wo] = T.Cast(dtype, acc[0])

    return pool_tanh


# ----------------------------------------------------------------------------- #
#                           PyTorch wrapper module                              #
# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → BatchNorm2d → (fused TileLang) tanh → MaxPool2d → GroupNorm
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,       # unused but kept for API compatibility
        num_groups: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        # ---------------- ConvTranspose2d parameters ----------------------- #
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.bias   = nn.Parameter(torch.empty(out_channels))

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        self.stride  = int(stride)
        self.padding = int(padding)

        # ---------------- BatchNorm2d params / buffers --------------------- #
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias   = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var",  torch.ones(out_channels))
        self.bn_eps       = 1e-5
        self.bn_momentum  = 0.1

        # ---------------- GroupNorm learnables ----------------------------- #
        self.gn_weight = nn.Parameter(torch.ones(out_channels))
        self.gn_bias   = nn.Parameter(torch.zeros(out_channels))
        self.num_groups = num_groups
        self.gn_eps     = eps

        # ---------------- Kernel cache ------------------------------------ #
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_pool_kernel(
        self,
        N: int,
        C: int,
        H: int,
        W: int,
        dtype_str: str = "float16",
    ):
        key = (N, C, H, W, dtype_str)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_pool_tanh_kernel(
                N, C, H, W, dtype=dtype_str
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = "cuda"

        # -------- ConvTranspose2d (cuDNN) -------------------------------- #
        w = self.weight.to(device=device, dtype=orig_dtype)
        b = self.bias.to(device=device, dtype=orig_dtype)
        x = x.to(device=device, dtype=orig_dtype)

        x = F.conv_transpose2d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
        )

        # -------- BatchNorm2d ------------------------------------------- #
        bn_w = self.bn_weight.to(device=device, dtype=orig_dtype)
        bn_b = self.bn_bias.to(device=device, dtype=orig_dtype)
        run_m = self.running_mean.to(device=device, dtype=orig_dtype)
        run_v = self.running_var.to(device=device, dtype=orig_dtype)

        x = F.batch_norm(
            x,
            run_m,
            run_v,
            weight=bn_w,
            bias=bn_b,
            training=self.training,
            momentum=self.bn_momentum,
            eps=self.bn_eps,
        )

        # -------- fused tanh + MaxPool2d (TileLang) ---------------------- #
        x_fp16 = x.to(dtype=torch.float16).contiguous()
        N, C, H, W = x_fp16.shape
        kernel = self._get_pool_kernel(N, C, H, W, "float16")
        y_fp16 = kernel(x_fp16)      # shape (N,C,H/2,W/2)

        # -------- GroupNorm (fp32 for accuracy) ------------------------- #
        y = y_fp16.to(dtype=torch.float32)
        G = self.num_groups
        C_out = C
        B, _, H_out, W_out = y.shape
        y = y.view(B, G, C_out // G, H_out, W_out)

        mean = y.mean(dim=[2, 3, 4], keepdim=True)
        var  = y.var(dim=[2, 3, 4], unbiased=False, keepdim=True)
        y_norm = (y - mean) / torch.sqrt(var + self.gn_eps)
        y_norm = y_norm.view(B, C_out, H_out, W_out)

        y_norm = (
            y_norm * self.gn_weight.view(1, C_out, 1, 1).to(dtype=torch.float32, device=device)
            + self.gn_bias.view(1, C_out, 1, 1).to(dtype=torch.float32, device=device)
        )

        return y_norm.to(orig_dtype)