"""
Problem Name: 75_conv_transposed_2D_asymmetric_input_asymmetric_kernel_strided__grouped____padded____dilated__
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=279.0 runtime_stats={'mean': 279.0, 'std': 0.0679, 'min': 279.0, 'max': 279.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.22, 'std': 0.0286, 'min': 1.21, 'max': 1.5, 'num_trials': 100}, 'speedup_ratio': 0.00437}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
def _build_deconv2d_kernel(
    N: int,
    CI: int,
    HI: int,
    WI: int,
    CO: int,
    KH: int,
    KW: int,
    stride_h: int,
    stride_w: int,
    pad_h: int,
    pad_w: int,
    dil_h: int,
    dil_w: int,
    groups: int,
    threads_per_block: int = 64,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    # Output spatial size (output_padding == 0)
    HO = (HI - 1) * stride_h - 2 * pad_h + dil_h * (KH - 1) + 1
    WO = (WI - 1) * stride_w - 2 * pad_w + dil_w * (KW - 1) + 1

    CI_g = CI // groups
    CO_g = CO // groups

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv2d_kernel(
        X: T.Tensor((N, CI, HI, WI), dtype),                   # N C H W
        Wt: T.Tensor((CI, CO_g, KH, KW), dtype),               # C  Co/G  KH KW
        Out: T.Tensor((N, CO, HO, WO), dtype),                 # N Co H  W
    ):
        with T.Kernel(
            HO * WO,       # bx : flattened spatial position
            CO,            # by : output channel
            N,             # bz : batch
            threads=threads_per_block,
        ) as (bx, by, bz):
            # Decode spatial coordinates
            h_out = bx // WO
            w_out = bx % WO

            n_idx  = bz
            co_idx = by

            # --- grouping info ---
            g_id   = co_idx // CO_g           # which group
            co_loc = co_idx % CO_g            # local output ch
            ci_start = g_id * CI_g
            ci_end   = ci_start + CI_g

            acc = T.alloc_local((1,), accum_dtype)
            T.clear(acc)

            # --------  main accumulation  --------
            for ci in T.serial(ci_start, ci_end):          # input channels of this group
                for kh in T.serial(KH):
                    h_in_pre = h_out + pad_h - dil_h * kh
                    if (h_in_pre % stride_h == 0):
                        h_in = h_in_pre // stride_h
                        if (h_in >= 0) and (h_in < HI):
                            for kw in T.serial(KW):
                                w_in_pre = w_out + pad_w - dil_w * kw
                                if (w_in_pre % stride_w == 0):
                                    w_in = w_in_pre // stride_w
                                    if (w_in >= 0) and (w_in < WI):
                                        x_val = X[n_idx, ci, h_in, w_in]
                                        w_val = Wt[ci, co_loc, kh, kw]
                                        acc[0] += (
                                            T.Cast(accum_dtype, x_val)
                                            * T.Cast(accum_dtype, w_val)
                                        )

            Out[n_idx, co_idx, h_out, w_out] = T.Cast(dtype, acc[0])

    return deconv2d_kernel


# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    TileLang-optimised ConvTranspose2d supporting:
        • arbitrary stride, padding, dilation (tuple of 2 ints)
        • arbitrary groups
        • output_padding = (0,0)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1),
        padding: tuple = (0, 0),
        dilation: tuple = (1, 1),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()

        # ---------- parameter book-keeping ----------
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.KH, self.KW = kernel_size
        self.stride_h, self.stride_w = stride
        self.pad_h, self.pad_w = padding
        self.dil_h, self.dil_w = dilation
        self.groups = groups
        self.use_bias = bias

        # ---------- parameters (PyTorch-identical init) ----------
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels // groups, self.KH, self.KW)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            fan_in = in_channels * self.KH * self.KW
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # compiled kernels cache
        self._kern_cache = {}

    # ------------------------------------------------------------ #
    def _get_kernel(self, N: int, HI: int, WI: int, dtype: torch.dtype):
        key = (N, HI, WI, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_deconv2d_kernel(
                N,
                self.in_channels,
                HI,
                WI,
                self.out_channels,
                self.KH,
                self.KW,
                self.stride_h,
                self.stride_w,
                self.pad_h,
                self.pad_w,
                self.dil_h,
                self.dil_w,
                self.groups,
                dtype=str(dtype).split(".")[-1],
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.contiguous().to(device="cuda", dtype=torch.float16)

        N, CI, HI, WI = x_fp16.shape
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(N, HI, WI, x_fp16.dtype)
        out_fp16 = kernel(x_fp16, w_fp16)

        if self.use_bias:
            out_fp16 = out_fp16 + self.bias.view(1, -1, 1, 1).to(out_fp16.dtype)

        return out_fp16.to(orig_dtype)