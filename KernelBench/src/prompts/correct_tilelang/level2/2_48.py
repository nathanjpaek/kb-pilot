"""
Problem Name: 48_Conv3d_Scaling_Tanh_Multiply_Sigmoid
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.792 runtime_stats={'mean': 0.792, 'std': 0.021, 'min': 0.779, 'max': 0.986, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.687, 'std': 0.0201, 'min': 0.677, 'max': 0.88, 'num_trials': 100}, 'speedup_ratio': 0.867}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory                                                                
# --------------------------------------------------------------------------- #

def _build_fused_conv3d_kernel(
    N: int,
    Cin: int,
    Din: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Dout = Din - K + 1
    Hout = Hin - K + 1
    Wout = Win - K + 1
    TOT  = N * Cout * Dout * Hout * Wout
    GRID = (TOT + block_size - 1) // block_size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:      T.Tensor((N, Cin, Din, Hin, Win), dtype),
        Wt:     T.Tensor((Cout, Cin, K, K, K),    dtype),
        Bconv:  T.Tensor((Cout,),                 dtype),   # conv bias
        Scale:  T.Tensor((Cout,),                 dtype),   # first per-channel scale
        Bmul:   T.Tensor((Cout,),                 dtype),   # second per-channel scale
        Y:      T.Tensor((N, Cout, Dout, Hout, Wout), dtype),
    ):
        one  = T.Cast(accum_dtype, 1.0)
        zero = T.Cast(accum_dtype, 0.0)

        with T.Kernel(GRID, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < TOT:
                # unravel linear idx -> n, co, d, h, w
                w_out = idx % Wout
                tmp   = idx // Wout
                h_out = tmp % Hout
                tmp  //= Hout
                d_out = tmp % Dout
                tmp  //= Dout
                co    = tmp % Cout
                n     = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, Bconv[co])

                for ci in T.serial(Cin):
                    for kz in T.serial(K):
                        for ky in T.serial(K):
                            for kx in T.serial(K):
                                val_in = T.Cast(
                                    accum_dtype,
                                    X[n, ci, d_out + kz, h_out + ky, w_out + kx],
                                )
                                val_w  = T.Cast(
                                    accum_dtype,
                                    Wt[co, ci, kz, ky, kx],
                                )
                                acc[0] += val_in * val_w

                # scale, tanh, multiply, sigmoid
                val = acc[0] * T.Cast(accum_dtype, Scale[co])
                val = T.tanh(val)
                val = val * T.Cast(accum_dtype, Bmul[co])
                val = one / (one + T.exp(-val))

                Y[n, co, d_out, h_out, w_out] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper                                                               
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """Optimised replacement model: sigmoid( tanh( Conv3d(x) * scale ) * bias )."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        scaling_factor,   # kept for interface parity (unused, see original)
        bias_shape: tuple,
    ):
        super().__init__()

        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size  = int(kernel_size)

        # Conv3d weight & bias (same init as nn.Conv3d)
        weight_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_channels * self.kernel_size ** 3
        bound  = 1 / math.sqrt(fan_in)
        self.conv_bias = nn.Parameter(torch.empty(self.out_channels))
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # Learnable per-channel scaling and multiplicative bias (randn init)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.mul_bias       = nn.Parameter(torch.randn(bias_shape))

        # kernel cache
        self._kern_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_conv3d_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                dtype=dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, _, D_in, H_in, W_in = x_fp16.shape

        w_fp16   = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bc_fp16  = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()
        sf_fp16  = self.scaling_factor.view(-1).to(device="cuda", dtype=torch.float16).contiguous()
        mb_fp16  = self.mul_bias.view(-1).to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(N, D_in, H_in, W_in, "float16")
        y_fp16 = kernel(x_fp16, w_fp16, bc_fp16, sf_fp16, mb_fp16)

        return y_fp16.to(orig_dtype)