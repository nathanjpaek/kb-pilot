"""
Problem Name: 74_conv_transposed_1D_dilated
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.046 runtime_stats={'mean': 0.046, 'std': 0.0126, 'min': 0.0394, 'max': 0.153, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0566, 'std': 0.0156, 'min': 0.0491, 'max': 0.196, 'num_trials': 100}, 'speedup_ratio': 1.23}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------
# Kernel factory -------------------------------------------------------
# ----------------------------------------------------------------------
def build_deconv1d_kernel(
    N: int,
    Cin: int,
    Lin: int,
    Cout: int,
    K: int,
    stride: int,
    padding: int,
    dilation: int,
    block_L: int = 32,
    block_Co: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """
    Build a CUDA TileLang kernel performing ConvTranspose1d.

    X :  (N, Cin, Lin)
    W :  (Cin, Cout, K)
    B :  (Cout,)
    Out:  (N, Cout, Lout)
    """
    Lout = (Lin - 1) * stride - 2 * padding + dilation * (K - 1) + 1
    n_len_tiles = T.ceildiv(Lout, block_L)

    grid_x = T.ceildiv(Cout, block_Co)
    grid_y = N * n_len_tiles
    threads_per_block = block_L * block_Co  # 32*32 = 1024

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv1d_kernel(
        X: T.Tensor((N, Cin, Lin), dtype),          # N C_in L_in
        W: T.Tensor((Cin, Cout, K), dtype),         # C_in C_out K
        B: T.Tensor((Cout,), dtype),                # C_out
        Out: T.Tensor((N, Cout, Lout), dtype),      # N C_out L_out
    ):
        with T.Kernel(grid_x, grid_y, threads=threads_per_block) as (bx, by):
            tid = T.get_thread_binding(0)
            l_local = tid % block_L          # index inside length tile
            co_local = tid // block_L        # index inside channel tile

            batch_idx = by // n_len_tiles
            l_tile_idx = by % n_len_tiles

            y_out = l_tile_idx * block_L + l_local
            co_out = bx * block_Co + co_local

            valid = (batch_idx < N) and (y_out < Lout) and (co_out < Cout)

            acc = T.alloc_local((1,), accum_dtype)
            if valid:
                acc[0] = T.Cast(accum_dtype, B[co_out])

                for ci in T.serial(Cin):
                    for k_idx in T.serial(K):
                        tmp = y_out + padding - k_idx * dilation
                        in_range = (tmp >= 0)
                        if in_range and ((tmp % stride) == 0):
                            x_pos = tmp // stride
                            if x_pos < Lin:
                                acc[0] += (
                                    T.Cast(accum_dtype, X[batch_idx, ci, x_pos])
                                    * T.Cast(accum_dtype, W[ci, co_out, k_idx])
                                )

                Out[batch_idx, co_out, y_out] = T.Cast(dtype, acc[0])

    return deconv1d_kernel


# ----------------------------------------------------------------------
# PyTorch wrapper ------------------------------------------------------
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    High-performance ConvTranspose1d implemented with TileLang.
    Supports arbitrary stride, padding and dilation (groups=1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.use_bias = bias

        # Parameters: identical initialisation to nn.ConvTranspose1d
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.use_bias:
            fan_in = in_channels * kernel_size
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(torch.empty(out_channels))
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter("bias", None)

        # kernel cache  {(N, Lin, dtype): compiled_kernel}
        self._kern_cache = {}

    # ------------------------------------------------------------------
    def _get_kernel(self, N: int, Lin: int, dtype: str):
        key = (N, Lin, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = build_deconv1d_kernel(
                N,
                self.in_channels,
                Lin,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()

        if self.bias is not None:
            b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()
        else:
            b_fp16 = torch.zeros(
                self.out_channels, dtype=torch.float16, device="cuda"
            )

        N, _, Lin = x_fp16.shape
        kernel = self._get_kernel(N, Lin, "float16")

        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        return out_fp16.to(orig_dtype)