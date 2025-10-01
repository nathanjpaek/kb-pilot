"""
Problem Name: 15_ConvTranspose3d_BatchNorm_Subtract
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.46 runtime_stats={'mean': 2.46, 'std': 0.017, 'min': 2.45, 'max': 2.59, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.42, 'std': 0.0257, 'min': 2.41, 'max': 2.66, 'num_trials': 100}, 'speedup_ratio': 0.984}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory: per-(n,c) block computes mean over (D,H,W) then subtracts   #
# --------------------------------------------------------------------------- #
def _make_mean_sub_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    threads: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    DHW = D * H * W
    grid = N * C
    inv_DHW = 1.0 / float(DHW)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Y: T.Tensor((N, C, D, H, W), dtype),
    ):
        inv_val = T.Cast(accum_dtype, inv_DHW)

        with T.Kernel(grid, threads=threads) as bc:  # one block per (n,c)
            tid = T.get_thread_binding(0)
            n = bc // C
            c = bc % C

            # ------------- first pass: parallel sum ------------------------
            partial = T.alloc_local((1,), accum_dtype)
            partial[0] = T.Cast(accum_dtype, 0)

            steps = T.ceildiv(DHW, threads)
            for k in T.serial(steps):
                idx = k * threads + tid
                if idx < DHW:
                    d = idx // (H * W)
                    hw = idx % (H * W)
                    h = hw // W
                    w = hw % W
                    partial[0] += T.Cast(accum_dtype, X[n, c, d, h, w])

            total = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        partial[0],
                        True,
                        total[0],
                        tid,
                        dtype="handle",
                    )
                )

            mean_val = total[0] * inv_val

            # ------------- second pass: subtract mean ----------------------
            for k in T.serial(steps):
                idx = k * threads + tid
                if idx < DHW:
                    d = idx // (H * W)
                    hw = idx % (H * W)
                    h = hw // W
                    w = hw % W
                    val = T.Cast(accum_dtype, X[n, c, d, h, w]) - mean_val
                    Y[n, c, d, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  BatchNorm3d  →  (TileLang) subtract spatial mean
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        bias: bool = True,
    ):
        super().__init__()

        # ---------- ConvTranspose3d params (identical init) ----------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size ** 3
        bound = 1.0 / math.sqrt(fan_in)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

        # ---------- BatchNorm3d params ------------------------------------
        self.bn_weight = nn.Parameter(torch.ones(out_channels))
        self.bn_bias = nn.Parameter(torch.zeros(out_channels))
        self.register_buffer("running_mean", torch.zeros(out_channels))
        self.register_buffer("running_var", torch.ones(out_channels))
        self.bn_eps = 1e-5
        self.bn_momentum = 0.1

        # Kernel cache {(N,C,D,H,W,dtype): compiled kernel}
        self._kernels: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(
        self, N: int, C: int, D: int, H: int, W: int, dtype: str = "float16"
    ):
        key = (N, C, D, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _make_mean_sub_kernel(N, C, D, H, W, dtype=dtype)
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = "cuda"

        # -------- ConvTranspose3d ----------------------------------------
        w = self.weight.to(device=device, dtype=orig_dtype)
        b = self.bias.to(device=device, dtype=orig_dtype) if self.bias is not None else None
        x = x.to(device=device, dtype=orig_dtype)
        x = F.conv_transpose3d(
            x,
            w,
            b,
            stride=self.stride,
            padding=self.padding,
        )

        # -------- BatchNorm3d -------------------------------------------
        x = F.batch_norm(
            x,
            self.running_mean.to(device=device, dtype=orig_dtype),
            self.running_var.to(device=device, dtype=orig_dtype),
            self.bn_weight.to(device=device, dtype=orig_dtype),
            self.bn_bias.to(device=device, dtype=orig_dtype),
            training=self.training,
            momentum=self.bn_momentum,
            eps=self.bn_eps,
        )

        # -------- TileLang mean-subtraction -----------------------------
        x_fp16 = x.contiguous().to(dtype=torch.float16)
        N, C, D, H, W = x_fp16.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        y_fp16 = kernel(x_fp16)

        return y_fp16.to(orig_dtype)