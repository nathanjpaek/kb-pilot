"""
Problem Name: 60_Conv3d_Mean_Subtract
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.68 runtime_stats={'mean': 1.68, 'std': 0.00788, 'min': 1.68, 'max': 1.76, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.74, 'std': 0.00906, 'min': 1.74, 'max': 1.83, 'num_trials': 100}, 'speedup_ratio': 1.04}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : spatial mean per (n,c) then subtract              #
# --------------------------------------------------------------------------- #
def _build_mean_sub_kernel(
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
        Y: T.Tensor((N, C, D, H, W), dtype),  # auto-allocated
    ):
        inv_val = T.Cast(accum_dtype, inv_DHW)

        with T.Kernel(grid, threads=threads) as bc:  # 1 block per (n,c)
            tid = T.get_thread_binding(0)
            n = bc // C
            c = bc % C

            # ------------- first pass : sum over DHW -----------------------
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

            # ------------- second pass : subtract mean ---------------------
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
# Optimised PyTorch wrapper                                                   #
# --------------------------------------------------------------------------- #
class ModelNew(torch.nn.Module):
    """
    Conv3d  →  (TileLang) subtract spatial mean per (n,c)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        # ---- create Conv3d parameters with identical initialisation -------
        w_shape = (out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        self.weight = torch.nn.Parameter(torch.empty(w_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size ** 3
        bound = 1 / math.sqrt(fan_in)
        self.bias = torch.nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # kernel cache : (N,C,D,H,W,dtype) → kernel
        self._kernels: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_mean_sub_kernel(N, C, D, H, W, dtype=dtype)
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device = "cuda"

        # ------------------ Conv3d in fp16 for speed -----------------------
        x_fp16 = x.to(device=device, dtype=torch.float16)
        w_fp16 = self.weight.to(device=device, dtype=torch.float16)
        b_fp16 = self.bias.to(device=device, dtype=torch.float16)
        x_fp16 = F.conv3d(x_fp16, w_fp16, b_fp16, stride=1, padding=0)

        # ------------------ TileLang mean-subtract ------------------------
        N, C, D, H, W = x_fp16.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        y_fp16 = kernel(x_fp16.contiguous())

        return y_fp16.to(orig_dtype)