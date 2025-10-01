"""
Problem Name: 47_Conv3d_Mish_Tanh
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.127 runtime_stats={'mean': 0.127, 'std': 0.00109, 'min': 0.126, 'max': 0.134, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0963, 'std': 0.00123, 'min': 0.0947, 'max': 0.104, 'num_trials': 100}, 'speedup_ratio': 0.758}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory
# --------------------------------------------------------------------------- #

def _build_conv3d_mish_tanh_kernel(
    N: int,
    Cin: int,
    Din: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    stride: int,
    padding: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    Dout = (Din + 2 * padding - K) // stride + 1
    Hout = (Hin + 2 * padding - K) // stride + 1
    Wout = (Win + 2 * padding - K) // stride + 1
    NUMEL = N * Cout * Dout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:  T.Tensor((N, Cin, Din, Hin, Win), dtype),
        Wt: T.Tensor((Cout, Cin, K, K, K),      dtype),
        B:  T.Tensor((Cout,),                   dtype),
        Y:  T.Tensor((N, Cout, Dout, Hout, Wout), dtype),
    ):
        one  = T.Cast(accum_dtype, 1.0)
        with T.Kernel(T.ceildiv(NUMEL, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < NUMEL:
                wo = idx % Wout
                tmp = idx // Wout
                ho = tmp % Hout
                tmp //= Hout
                do = tmp % Dout
                tmp //= Dout
                co = tmp % Cout
                n  = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, B[co])

                for ci in T.serial(Cin):
                    for kd in T.serial(K):
                        di = do * stride + kd - padding
                        if (di >= 0) and (di < Din):
                            for kh in T.serial(K):
                                hi = ho * stride + kh - padding
                                if (hi >= 0) and (hi < Hin):
                                    for kw in T.serial(K):
                                        wi = wo * stride + kw - padding
                                        if (wi >= 0) and (wi < Win):
                                            acc[0] += (
                                                X[n, ci, di, hi, wi].astype(accum_dtype)
                                                * Wt[co, ci, kd, kh, kw].astype(accum_dtype)
                                            )

                # Mish activation: x * tanh(ln(1+e^x))
                sp  = T.log(one + T.exp(acc[0]))
                val = acc[0] * T.tanh(sp)
                # Subsequent tanh
                val = T.tanh(val)

                Y[n, co, do, ho, wo] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# PyTorch wrapper
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """TileLang-accelerated version of Conv3d → Mish → Tanh."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)

        # ---- parameters (identical init to nn.Conv3d) --------------------
        w_shape = (
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
            self.kernel_size,
        )
        self.weight = nn.Parameter(torch.empty(w_shape))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kernel_size ** 3
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        nn.init.uniform_(self.bias, -bound, bound)

        # kernel cache {(N,D,H,W,dtype): kernel}
        self._kernels = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: str = "float16"):
        key = (N, D, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_conv3d_mish_tanh_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                dtype=dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, D_in, H_in, W_in = x_fp16.shape
        kernel = self._get_kernel(N, D_in, H_in, W_in, "float16")

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        y_fp16 = kernel(x_fp16, w_fp16, b_fp16)
        return y_fp16.to(orig_dtype)