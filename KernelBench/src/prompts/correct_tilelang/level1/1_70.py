"""
Problem Name: 70_conv_transposed_3D__asymmetric_input__square_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=177.0 runtime_stats={'mean': 177.0, 'std': 0.0562, 'min': 177.0, 'max': 178.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.451, 'std': 0.00411, 'min': 0.445, 'max': 0.468, 'num_trials': 100}, 'speedup_ratio': 0.00255}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_deconv3d_kernel(N: int,
                           CI: int,
                           CO: int,
                           Din: int,
                           Hin: int,
                           Win: int,
                           K: int = 3,
                           threads_per_block: int = 64,
                           dtype: str = "float16",
                           accum_dtype: str = "float"):
    Dout: int = Din + K - 1  # stride = 1, padding = 0, output_padding = 0
    Hout: int = Hin + K - 1
    Wout: int = Win + K - 1
    num_spatial: int = Dout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv3d_kernel(
        X: T.Tensor((N, CI, Din, Hin, Win), dtype),
        W: T.Tensor((CI, CO, K, K, K), dtype),
        Out: T.Tensor((N, CO, Dout, Hout, Wout), dtype),
    ):
        with T.Kernel(num_spatial, CO, N, threads=threads_per_block) as (bx, by, bz):
            # Decode spatial coordinates from flat index bx
            d_out = bx // (Hout * Wout)
            temp = bx % (Hout * Wout)
            h_out = temp // Wout
            w_out = temp % Wout

            c_out = by
            n_idx = bz

            acc = T.alloc_local((1,), accum_dtype)
            T.clear(acc)

            for c_in in T.serial(CI):
                for kd in T.serial(K):
                    din = d_out - kd
                    valid_d = (din >= 0) and (din < Din)
                    for kh in T.serial(K):
                        hin = h_out - kh
                        valid_h = (hin >= 0) and (hin < Hin)
                        for kw in T.serial(K):
                            win = w_out - kw
                            in_bounds = (
                                valid_d and valid_h and (win >= 0) and (win < Win)
                            )
                            inp_val = T.if_then_else(
                                in_bounds,
                                X[n_idx, c_in, din, hin, win],
                                T.Cast(dtype, 0),
                            )
                            w_val = W[c_in, c_out, kd, kh, kw]
                            acc[0] += (
                                T.Cast(accum_dtype, inp_val)
                                * T.Cast(accum_dtype, w_val)
                            )

            # Write result
            Out[n_idx, c_out, d_out, h_out, w_out] = T.Cast(dtype, acc[0])

    return deconv3d_kernel


class ModelNew(nn.Module):
    """TileLang-optimised ConvTranspose3d for stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=False."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert (
            stride == 1
            and padding == 0
            and output_padding == 0
            and dilation == 1
            and groups == 1
            and bias is False
        ), "Only stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=False supported."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = kernel_size

        # Weight parameter initialised identically to nn.ConvTranspose3d
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, self.K, self.K, self.K)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Kernel cache keyed by input shape & dtype
        self._kernel_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, Din: int, Hin: int, Win: int, dtype: torch.dtype):
        key = (N, Din, Hin, Win, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_deconv3d_kernel(
                N,
                self.in_channels,
                self.out_channels,
                Din,
                Hin,
                Win,
                self.K,
                dtype=str(dtype).split(".")[-1],
            )
        return self._kernel_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        N, CI, Din, Hin, Win = x_fp16.shape
        kernel = self._get_kernel(N, Din, Hin, Win, x_fp16.dtype)

        out_fp16 = kernel(x_fp16, w_fp16)
        return out_fp16.to(orig_dtype)