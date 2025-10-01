"""
Problem Name: 54_ConvTranspose2d_BatchNorm_Softmax_MaxPool
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.36 runtime_stats={'mean': 1.36, 'std': 0.172, 'min': 1.33, 'max': 3.03, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.12, 'std': 0.0332, 'min': 2.11, 'max': 2.44, 'num_trials': 100}, 'speedup_ratio': 1.56}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory:   BatchNorm2d (inference) → Softmax(dim=1) → MaxPool2d k=2  #
# --------------------------------------------------------------------------- #
def _build_bn_softmax_pool_kernel(
    N: int,
    C: int,
    H_in: int,
    W_in: int,
    eps: float = 1e-5,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    pool_k = 2
    pool_s = 2
    H_out = (H_in - pool_k) // pool_s + 1
    W_out = (W_in - pool_k) // pool_s + 1
    spatial_out = N * H_out * W_out

    neg_inf = -3.4028234663852886e38  # smallest fp32

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((N, C, H_in, W_in), dtype),
        Gamma:  T.Tensor((C,), dtype),
        Beta:   T.Tensor((C,), dtype),
        RMean:  T.Tensor((C,), dtype),
        RVar:   T.Tensor((C,), dtype),
        Y:      T.Tensor((N, C, H_out, W_out), dtype),   # output
    ):
        eps_c    = T.Cast(accum_dtype, eps)
        neg_inf_c = T.Cast(accum_dtype, neg_inf)

        grid = T.ceildiv(spatial_out, threads_per_block)
        with T.Kernel(grid, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < spatial_out:
                wo  = idx % W_out
                tmp = idx // W_out
                ho  = tmp % H_out
                n   = tmp // H_out

                base_h = ho * pool_s
                base_w = wo * pool_s

                # Local buffers
                max_buf  = T.alloc_local((C,), accum_dtype)
                temp_buf = T.alloc_local((C,), accum_dtype)

                # initialise maxima
                for c in T.serial(C):
                    max_buf[c] = neg_inf_c

                # iterate over the 2x2 pool window
                for kh in T.serial(pool_k):
                    for kw in T.serial(pool_k):
                        h = base_h + kh
                        w = base_w + kw

                        # pass 1 : compute exp(BN(x)) and sum_exp
                        sum_exp = T.alloc_local((1,), accum_dtype)
                        sum_exp[0] = T.Cast(accum_dtype, 0)

                        for c in T.serial(C):
                            x_val = T.Cast(accum_dtype, X[n, c, h, w])

                            mean_c  = T.Cast(accum_dtype, RMean[c])
                            var_c   = T.Cast(accum_dtype, RVar[c])
                            gamma_c = T.Cast(accum_dtype, Gamma[c])
                            beta_c  = T.Cast(accum_dtype, Beta[c])

                            bn = (x_val - mean_c) / T.sqrt(var_c + eps_c)
                            bn = bn * gamma_c + beta_c

                            e = T.exp(bn)
                            temp_buf[c] = e
                            sum_exp[0] += e

                        inv_sum = T.Cast(accum_dtype, 1.0) / sum_exp[0]

                        # pass 2 : normalise & update maxima
                        for c in T.serial(C):
                            sm_val = temp_buf[c] * inv_sum
                            if sm_val > max_buf[c]:
                                max_buf[c] = sm_val

                # write results
                for c in T.serial(C):
                    Y[n, c, ho, wo] = T.Cast(dtype, max_buf[c])

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper with fused TileLang kernel                                  #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d  →  BatchNorm2d (inference) → Softmax(dim=1) → MaxPool2d(k=2,s=2)
    BatchNorm, softmax and pooling are fused into a single TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        batchnorm_features: int,
        pool_kernel_size: int,
        pool_stride: int,
        eps: float = 1e-5,
    ):
        super().__init__()

        # ---------------- ConvTranspose2d parameters ----------------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        self.conv_bias = nn.Parameter(torch.empty(out_channels))
        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound  = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # ---------------- BatchNorm2d parameters --------------------------
        self.gamma = nn.Parameter(torch.ones(batchnorm_features))
        self.beta  = nn.Parameter(torch.zeros(batchnorm_features))
        self.register_buffer("running_mean", torch.zeros(batchnorm_features))
        self.register_buffer("running_var",  torch.ones(batchnorm_features))
        self.eps = float(eps)

        # Pool params (fixed to k=2,s=2 in kernel factory)
        assert pool_kernel_size == 2 and pool_stride == 2, "Kernel supports k=s=2 only"

        # Kernel cache : {(N,C,H,W,dtype): kernel}
        self._kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}

        # Store conv hyper-params
        self.stride = int(stride)
        self.padding = int(padding)

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: str):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_bn_softmax_pool_kernel(
                N, C, H, W, eps=self.eps, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ------------ ConvTranspose2d (cuDNN) ----------------------------
        w = self.weight.to(device="cuda", dtype=torch.float16)
        b = self.conv_bias.to(device="cuda", dtype=torch.float16)
        x_fp16 = x.to(device="cuda", dtype=torch.float16)
        y = F.conv_transpose2d(
            x_fp16,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
        )  # (N,C,H,W)

        N, C, H_in, W_in = y.shape
        kernel = self._get_kernel(N, C, H_in, W_in, "float16")

        # Prepare BN params
        gamma_fp16 = self.gamma.to(device="cuda", dtype=torch.float16).contiguous()
        beta_fp16  = self.beta.to(device="cuda", dtype=torch.float16).contiguous()
        rm_fp16    = self.running_mean.to(device="cuda", dtype=torch.float16).contiguous()
        rv_fp16    = self.running_var.to(device="cuda", dtype=torch.float16).contiguous()

        out_fp16 = kernel(y.contiguous(), gamma_fp16, beta_fp16, rm_fp16, rv_fp16)

        return out_fp16.to(orig_dtype)