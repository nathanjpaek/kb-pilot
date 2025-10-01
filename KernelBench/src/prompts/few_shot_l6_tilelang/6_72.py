"""
Problem Name: 72_Conv3d_Tanh_Clamp_Sigmoid_Divide
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.45 runtime_stats={'mean': 3.45, 'std': 0.0385, 'min': 3.41, 'max': 3.59, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.22, 'std': 0.0236, 'min': 4.16, 'max': 4.25, 'num_trials': 100}, 'speedup_ratio': 1.22}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# -----------------------------------------------------------------------------#
# TileLang kernel factory ------------------------------------------------------#
# -----------------------------------------------------------------------------#
def _build_conv3d_fused_kernel(
    N: int,
    Cin: int,
    Cout: int,
    Din: int,
    Hin: int,
    Win: int,
    K: int,
    clamp_min: float,
    clamp_max: float,
    *,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    Dout = Din - K + 1
    Hout = Hin - K + 1
    Wout = Win - K + 1
    out_numel = N * Cout * Dout * Hout * Wout
    k_min = float(clamp_min)
    k_max = float(clamp_max)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv3d_fused(
        X: T.Tensor((N, Cin, Din, Hin, Win), dtype),       # Input
        Wt: T.Tensor((Cout, Cin, K, K, K), dtype),         # Weights
        B:  T.Tensor((Cout,), dtype),                      # Bias
        Out: T.Tensor((N, Cout, Dout, Hout, Wout), dtype), # Output (created by TileLang)
    ):
        with T.Kernel(T.ceildiv(out_numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < out_numel:
                w_idx  = idx % Wout
                tmp1   = idx // Wout
                h_idx  = tmp1 % Hout
                tmp2   = tmp1 // Hout
                d_idx  = tmp2 % Dout
                tmp3   = tmp2 // Dout
                oc     = tmp3 % Cout
                n_idx  = tmp3 // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for ic in T.serial(Cin):
                    for kz in T.serial(K):
                        for ky in T.serial(K):
                            for kx in T.serial(K):
                                in_val = X[
                                    n_idx,
                                    ic,
                                    d_idx + kz,
                                    h_idx + ky,
                                    w_idx + kx,
                                ]
                                w_val = Wt[oc, ic, kz, ky, kx]
                                acc[0] += (
                                    T.Cast(accum_dtype, in_val)
                                    * T.Cast(accum_dtype, w_val)
                                )

                acc[0] += T.Cast(accum_dtype, B[oc])

                val = T.tanh(acc[0])
                val = T.clamp(val, k_min, k_max)

                Out[n_idx, oc, d_idx, h_idx, w_idx] = T.Cast(dtype, val)

    return conv3d_fused


# -----------------------------------------------------------------------------#
# PyTorch wrapper module -------------------------------------------------------#
# -----------------------------------------------------------------------------#
class ModelNew(nn.Module):
    """
    TileLang-accelerated 3D Conv + Tanh + Clamp (+ fused cancelled ops).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, clamp_min: float, clamp_max: float):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Parameters (same init as nn.Conv3d)
        weight_shape = (out_channels, in_channels,
                        kernel_size, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.empty(weight_shape))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.empty(out_channels))
        fan_in = in_channels * kernel_size * kernel_size * kernel_size
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # Kernel cache, keyed by (N, Din, Hin, Win, dtype)
        self._kern_cache = {}

    # ---------------------------------------------------------------------#
    def _get_kernel(self, N: int, Din: int, Hin: int, Win: int,
                    dtype: torch.dtype):
        key = (N, Din, Hin, Win, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kern_cache[key] = _build_conv3d_fused_kernel(
                N, self.in_channels, self.out_channels,
                Din, Hin, Win, self.kernel_size,
                self.clamp_min, self.clamp_max,
                dtype=tl_dtype,
            )
        return self._kern_cache[key]

    # ---------------------------------------------------------------------#
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (N, Cin, D, H, W) input tensor
        Returns:
            out : (N, Cout, D-K+1, H-K+1, W-K+1)
        """
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16, copy=False)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16, copy=False)

        N, _, Din, Hin, Win = x_f16.shape
        kernel = self._get_kernel(N, Din, Hin, Win, x_f16.dtype)

        out_f16 = kernel(x_f16, w_f16, b_f16)
        return out_f16.to(orig_dtype)