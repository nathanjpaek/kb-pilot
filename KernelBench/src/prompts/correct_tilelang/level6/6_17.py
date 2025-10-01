"""
Problem Name: 17_ConvTranspose3d_Swish_GroupNorm_HardSwish
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=23.6 runtime_stats={'mean': 23.6, 'std': 0.00829, 'min': 23.6, 'max': 23.7, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 23.4, 'std': 0.00855, 'min': 23.4, 'max': 23.4, 'num_trials': 100}, 'speedup_ratio': 0.992}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """
    ConvTranspose3d → Swish → GroupNorm → HardSwish
    Swish & HardSwish are executed via high-performance TileLang kernels.
    """

    # ------------------------------------------------------------------ #
    # ----------------------  kernel factories  ------------------------ #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_swish_kernel(
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        block: int = 256,
        dtype: str = "float16",
        accum_dtype: str = "float32",
    ):
        total = N * C * D * H * W

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def swish(
            X: T.Tensor((N, C, D, H, W), dtype),
            Y: T.Tensor((N, C, D, H, W), dtype),
        ):
            one_f = T.Cast(accum_dtype, 1.0)

            with T.Kernel(T.ceildiv(total, block), threads=block) as bx:
                tx = T.get_thread_binding(0)
                idx = bx * block + tx
                if idx < total:
                    w = idx % W
                    t1 = idx // W
                    h = t1 % H
                    t2 = t1 // H
                    d = t2 % D
                    t3 = t2 // D
                    c = t3 % C
                    n = t3 // C

                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    sig = one_f / (one_f + T.exp(-val))
                    out = val * sig
                    Y[n, c, d, h, w] = T.Cast(dtype, out)

        return swish

    @staticmethod
    def _build_hardswish_kernel(
        N: int,
        C: int,
        D: int,
        H: int,
        W: int,
        block: int = 256,
        dtype: str = "float16",
        accum_dtype: str = "float32",
    ):
        total = N * C * D * H * W
        six_val = 6.0
        three_val = 3.0

        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def hswish(
            X: T.Tensor((N, C, D, H, W), dtype),
            Y: T.Tensor((N, C, D, H, W), dtype),
        ):
            six = T.Cast(accum_dtype, six_val)
            three = T.Cast(accum_dtype, three_val)
            one_sixth = T.Cast(accum_dtype, 1.0 / six_val)

            with T.Kernel(T.ceildiv(total, block), threads=block) as bx:
                tx = T.get_thread_binding(0)
                idx = bx * block + tx
                if idx < total:
                    w = idx % W
                    t1 = idx // W
                    h = t1 % H
                    t2 = t1 // H
                    d = t2 % D
                    t3 = t2 // D
                    c = t3 % C
                    n = t3 // C

                    val = T.Cast(accum_dtype, X[n, c, d, h, w])
                    hsig = T.max(T.min(val + three, six), T.Cast(accum_dtype, 0.0)) * one_sixth
                    out = val * hsig
                    Y[n, c, d, h, w] = T.Cast(dtype, out)

        return hswish

    # ------------------------------------------------------------------ #
    # ---------------------------  ctor  -------------------------------- #
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        groups: int,
        eps: float,
        bias: bool = True,
    ):
        super().__init__()

        # keep original PyTorch layers for parameter correctness
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.group_norm = nn.GroupNorm(
            num_groups=groups,
            num_channels=out_channels,
            eps=eps,
        )

        # kernel caches : { key -> compiled_kernel }
        self._swish_cache = {}   # key: (N,C,D,H,W,dtype)
        self._hswish_cache = {}

    # ------------------------------------------------------------------ #
    # ----------------------- kernel getters --------------------------- #
    # ------------------------------------------------------------------ #
    def _get_swish_kernel(self, shape, dtype_str: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype_str)
        if key not in self._swish_cache:
            self._swish_cache[key] = self._build_swish_kernel(
                N, C, D, H, W, dtype=dtype_str
            )
        return self._swish_cache[key]

    def _get_hswish_kernel(self, shape, dtype_str: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype_str)
        if key not in self._hswish_cache:
            self._hswish_cache[key] = self._build_hardswish_kernel(
                N, C, D, H, W, dtype=dtype_str
            )
        return self._hswish_cache[key]

    # ------------------------------------------------------------------ #
    # --------------------------- forward ------------------------------ #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # 1) ConvTranspose3d (stay in fp32 for accuracy)
        x = self.conv_transpose(x)

        # 2) Swish in fp16 via TileLang
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        swish_kernel = self._get_swish_kernel(x_fp16.shape, "float16")
        x_fp16 = swish_kernel(x_fp16)

        # 3) GroupNorm back in fp32
        x = x_fp16.to(dtype=orig_dtype)
        x = self.group_norm(x)

        # 4) HardSwish in fp16 via TileLang
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        hswish_kernel = self._get_hswish_kernel(x_fp16.shape, "float16")
        x_fp16 = hswish_kernel(x_fp16)

        # 5) cast back to original dtype and return
        return x_fp16.to(orig_dtype)