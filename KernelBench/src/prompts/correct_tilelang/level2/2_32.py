"""
Problem Name: 32_Conv2d_Scaling_Min
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0622 runtime_stats={'mean': 0.0622, 'std': 0.0144, 'min': 0.0544, 'max': 0.188, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0606, 'std': 0.0138, 'min': 0.0559, 'max': 0.192, 'num_trials': 100}, 'speedup_ratio': 0.974}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_conv_min_kernel(
    N: int,
    Cin: int,
    H: int,
    W: int,
    Cout: int,
    K: int,
    scale_factor: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    OH = H - K + 1
    OW = W - K + 1
    tot = N * OH * OW
    grid = (tot + block_size - 1) // block_size
    scl = float(scale_factor)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((N, H, W, Cin), dtype),          # NHWC
        Wt: T.Tensor((K, K, Cin, Cout), dtype),       # HWCF
        B:  T.Tensor((Cout,), dtype),
        Y:  T.Tensor((N, OH, OW, 1), dtype),          # NHW1
    ):
        scl_c = T.Cast(accum_dtype, scl)

        with T.Kernel(grid, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < tot:
                n  = idx // (OH * OW)
                rem = idx % (OH * OW)
                oh  = rem // OW
                ow  = rem % OW

                min_val = T.alloc_local((1,), accum_dtype)
                min_val[0] = T.Cast(accum_dtype, 3.4e38)   # large positive

                for oc in range(Cout):
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, 0)

                    for kh in range(K):
                        for kw in range(K):
                            for ci in range(Cin):
                                a_val = X[n, oh + kh, ow + kw, ci].astype(accum_dtype)
                                w_val = Wt[kh, kw, ci, oc].astype(accum_dtype)
                                acc[0] += a_val * w_val

                    acc[0] += B[oc].astype(accum_dtype)
                    acc[0] = acc[0] * scl_c
                    min_val[0] = T.min(min_val[0], acc[0])

                Y[n, oh, ow, 0] = T.Cast(dtype, min_val[0])

    return kernel


class ModelNew(nn.Module):
    """
    Fused Conv2d → scaling → channel-minimum implemented with TileLang.
    Output shape: (N, 1, OH, OW)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scale_factor: float,
    ):
        super().__init__()
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size  = int(kernel_size)
        self.scale_factor = float(scale_factor)

        # Parameters identical to nn.Conv2d defaults
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        self._kern_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str):
        key = (N, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_conv_min_kernel(
                N,
                self.in_channels,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.scale_factor,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, H, W = x_fp16.shape
        assert C == self.in_channels

        # NHWC for better coalesced access
        x_nhwc = x_fp16.permute(0, 2, 3, 1).contiguous()

        w_fp16 = (
            self.weight.to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 1, 0)    # HWCF
            .contiguous()
        )
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(N, H, W, "float16")
        y_nhw1 = kernel(x_nhwc, w_fp16, b_fp16)   # (N, OH, OW, 1)

        OH = H - self.kernel_size + 1
        OW = W - self.kernel_size + 1
        y = y_nhw1.permute(0, 3, 1, 2).contiguous()  # (N,1,OH,OW)

        return y.to(orig_dtype)