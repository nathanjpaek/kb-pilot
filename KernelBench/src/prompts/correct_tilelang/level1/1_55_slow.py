"""
Problem Name: 55_conv_standard_2D__asymmetric_input__square_kernel
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.92 runtime_stats={'mean': 0.92, 'std': 0.00138, 'min': 0.917, 'max': 0.924, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.169, 'std': 0.000958, 'min': 0.167, 'max': 0.175, 'num_trials': 100}, 'speedup_ratio': 0.184}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _make_conv2d_kernel(
    B: int,
    C: int,
    OC: int,
    H: int,
    W: int,
    K: int,
    dtype: str = "float16",
    accum_dtype: str = "float32",
    threads: int = 128,
):
    OH = H - K + 1  # stride=1, padding=0, dilation=1
    OW = W - K + 1

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv2d_kernel(
        X: T.Tensor((B, C, H, W), dtype),
        Wt: T.Tensor((OC, C, K, K), dtype),
        Y: T.Tensor((B, OC, OH, OW), dtype),
    ):
        grid_x = OW
        grid_y = B * OH
        num_oc_tiles = T.ceildiv(OC, threads)

        with T.Kernel(grid_x, grid_y, threads=threads) as (bx, by):
            tx = T.get_thread_binding(0)

            b = by // OH
            oh = by % OH
            ow = bx

            acc = T.alloc_local((1,), accum_dtype)

            for oc_tile in T.serial(num_oc_tiles):
                oc = oc_tile * threads + tx
                if (oc < OC) and (b < B) and (ow < OW) and (oh < OH):
                    acc[0] = T.Cast(accum_dtype, 0)

                    for ic in T.serial(C):
                        for kh in T.serial(K):
                            for kw in T.serial(K):
                                acc[0] += (
                                    T.Cast(accum_dtype, X[b, ic, oh + kh, ow + kw])
                                    * T.Cast(accum_dtype, Wt[oc, ic, kh, kw])
                                )

                    Y[b, oc, oh, ow] = T.Cast(dtype, acc[0])

    return conv2d_kernel


class ModelNew(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self._kernel_cache = {}

    def _get_kernel(self, B: int, H: int, W: int, dtype: str):
        key = (B, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _make_conv2d_kernel(
                B,
                self.in_channels,
                self.out_channels,
                H,
                W,
                self.kernel_size,
                dtype=dtype,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16)

        B, _, H, W = x_fp16.shape
        kernel = self._get_kernel(B, H, W, "float16")
        y_fp16 = kernel(x_fp16, w_fp16)

        return y_fp16.to(orig_dtype)