"""
Problem Name: 25_Conv2d_Min_Tanh_Tanh
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0421 runtime_stats={'mean': 0.0421, 'std': 0.0111, 'min': 0.0379, 'max': 0.148, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0637, 'std': 0.0185, 'min': 0.0575, 'max': 0.235, 'num_trials': 100}, 'speedup_ratio': 1.51}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------- #
def _build_conv_min_tanh_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Hout = Hin - K + 1
    Wout = Win - K + 1
    total_elems = N * Hout * Wout
    grid_dim = (total_elems + block_size - 1) // block_size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv_min_tanh(
        X: T.Tensor((N, Cin, Hin, Win), dtype),
        Wt: T.Tensor((Cout, Cin, K, K), dtype),
        B: T.Tensor((Cout,), dtype),
        Y: T.Tensor((N, 1, Hout, Wout), dtype),
    ):
        with T.Kernel(grid_dim, threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total_elems:
                ow = idx % Wout
                tmp = idx // Wout
                oh = tmp % Hout
                n  = tmp // Hout

                min_val = T.alloc_local((1,), accum_dtype)

                for oc in T.serial(Cout):
                    acc = T.alloc_local((1,), accum_dtype)
                    acc[0] = T.Cast(accum_dtype, 0)

                    for ic in T.serial(Cin):
                        for kh in T.serial(K):
                            ih = oh + kh
                            for kw in T.serial(K):
                                iw = ow + kw
                                acc[0] += (
                                    X[n, ic, ih, iw].astype(accum_dtype)
                                    * Wt[oc, ic, kh, kw].astype(accum_dtype)
                                )

                    acc[0] += B[oc].astype(accum_dtype)

                    if oc == 0:
                        min_val[0] = acc[0]
                    else:
                        min_val[0] = T.min(min_val[0], acc[0])

                # Two successive tanh activations
                val = T.tanh(min_val[0])
                val = T.tanh(val)

                Y[n, 0, oh, ow] = T.Cast(dtype, val)

    return conv_min_tanh


# --------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    TileLang‐accelerated replacement of the reference model:
        y = tanh( tanh( reduce_min_1ch( Conv2d(x) ) ) )
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)

        # ----- parameters with identical initialization to nn.Conv2d --------
        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache  {(N,H,W,dtype) : compiled_kernel}
        self._kernels = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, H: int, W: int, dtype: str = "float16"):
        key = (N, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_conv_min_tanh_kernel(
                N,
                self.in_channels,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, Hin, Win = x_fp16.shape
        kernel = self._get_kernel(N, Hin, Win, "float16")

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        y_fp16 = kernel(x_fp16, w_fp16, b_fp16)
        return y_fp16.to(orig_dtype)