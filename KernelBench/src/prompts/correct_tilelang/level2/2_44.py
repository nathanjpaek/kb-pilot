"""
Problem Name: 44_ConvTranspose2d_Multiply_GlobalAvgPool_GlobalAvgPool_Mean
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.156 runtime_stats={'mean': 0.156, 'std': 0.00133, 'min': 0.154, 'max': 0.166, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.175, 'std': 0.00173, 'min': 0.172, 'max': 0.184, 'num_trials': 100}, 'speedup_ratio': 1.12}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    stride: int,
    padding: int,
    output_padding: int,
    multiplier: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Hout = (Hin - 1) * stride - 2 * padding + K + output_padding
    Wout = (Win - 1) * stride - 2 * padding + K + output_padding
    HW   = Hout * Wout
    GRID = (N * Cout + threads_per_block - 1) // threads_per_block

    mul_cst = float(multiplier)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:    T.Tensor((N, Cin, Hin, Win), dtype),
        Wt:   T.Tensor((Cin, Cout, K, K),   dtype),
        Bias: T.Tensor((Cout,),             dtype),
        Out:  T.Tensor((N, Cout, 1, 1),     dtype),
    ):
        inv_hw   = T.Cast(accum_dtype, 1.0 / HW)
        mul_val  = T.Cast(accum_dtype, mul_cst)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < N * Cout:
                n  = idx // Cout
                oc = idx % Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                # ----------------- accumulate de-conv (no bias) -------------
                for ic in T.serial(Cin):
                    for ih in T.serial(Hin):
                        for iw in T.serial(Win):
                            x_val = T.Cast(accum_dtype, X[n, ic, ih, iw])
                            for kh in T.serial(K):
                                for kw in T.serial(K):
                                    w_val = T.Cast(accum_dtype, Wt[ic, oc, kh, kw])
                                    acc[0] += x_val * w_val

                mean_val = acc[0] * inv_hw + T.Cast(accum_dtype, Bias[oc])
                out_val  = mean_val * mul_val

                Out[n, oc, 0, 0] = T.Cast(dtype, out_val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose2d → scalar multiply → two global-average-pool stages
    fused into a single TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        multiplier: float,
    ):
        super().__init__()

        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.kernel_size   = kernel_size
        self.stride        = stride
        self.padding       = padding
        self.output_pad    = output_padding
        self.multiplier    = float(multiplier)

        # --- parameters identical to nn.ConvTranspose2d --------------------
        w_shape = (in_channels, out_channels, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * kernel_size * kernel_size
        bound  = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache  {(N,H,W,dtype) : kernel}
        self._kern_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, Hin: int, Win: int, dtype: str = "float16"):
        key = (N, Hin, Win, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N,
                self.in_channels,
                Hin,
                Win,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_pad,
                self.multiplier,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, Hin, Win = x_fp16.shape
        kernel = self._get_kernel(N, Hin, Win, "float16")

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        out_fp16 = kernel(x_fp16, w_fp16, b_fp16)

        # reshape to (N, C, 1, 1) to match PyTorch behaviour
        return out_fp16.to(orig_dtype)