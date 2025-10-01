"""
Problem Name: 50_Conv3d_Tanh_Clamp_Swish_Divide
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.66 runtime_stats={'mean': 1.66, 'std': 0.0177, 'min': 1.63, 'max': 1.68, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.23, 'std': 0.0105, 'min': 2.22, 'max': 2.25, 'num_trials': 100}, 'speedup_ratio': 1.34}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------
# TileLang kernel factory
# ---------------------------------------------------------------------
def _build_conv3d_fused_kernel(
    N: int,
    IC: int,
    OC: int,
    D_in: int,
    H_in: int,
    W_in: int,
    KD: int,
    KH: int,
    KW: int,
    clamp_min: float,
    clamp_max: float,
    block_size: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    D_out = D_in - KD + 1
    H_out = H_in - KH + 1
    W_out = W_in - KW + 1
    numel = N * OC * D_out * H_out * W_out

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv3d_fused(
        X:     T.Tensor((N, IC, D_in, H_in, W_in), dtype),
        Wght:  T.Tensor((OC, IC, KD, KH, KW),      dtype),
        Bias:  T.Tensor((OC,),                     dtype),
        Out:   T.Tensor((N, OC, D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                # ------------------------------------------------------
                # unravel linear index → (n, oc, d, h, w)
                # ------------------------------------------------------
                w_out = idx % W_out
                tmp   = idx // W_out
                h_out = tmp % H_out
                tmp   //= H_out
                d_out = tmp % D_out
                tmp   //= D_out
                oc    = tmp % OC
                n     = tmp // OC

                # ------------------------------------------------------
                # convolution accumulation
                # ------------------------------------------------------
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, Bias[oc])

                for ic in range(IC):
                    for kd in range(KD):
                        for kh in range(KH):
                            for kw in range(KW):
                                inp = X[n, ic,
                                        d_out + kd,
                                        h_out + kh,
                                        w_out + kw]
                                wv  = Wght[oc, ic, kd, kh, kw]
                                acc[0] += (
                                    T.Cast(accum_dtype, inp)
                                    * T.Cast(accum_dtype, wv)
                                )

                val = acc[0]                           # float32

                # ------------------------------------------------------
                # fused activation chain
                #   tanh → clamp → *sigmoid → /sigmoid
                # ------------------------------------------------------
                # tanh(x)  = 2*sigmoid(2x) - 1
                sig2x = 1.0 / (1.0 + T.exp(-2.0 * val))
                val   = 2.0 * sig2x - 1.0

                # clamp
                val = T.clamp(val, clamp_min, clamp_max)

                # first sigmoid & multiply
                sig1 = 1.0 / (1.0 + T.exp(-val))
                val2 = val * sig1

                # second sigmoid & divide
                sig2 = 1.0 / (1.0 + T.exp(-val2))
                out_val = val2 / sig2

                # store
                Out[n, oc, d_out, h_out, w_out] = T.Cast(dtype, out_val)

    return conv3d_fused


# ---------------------------------------------------------------------
# PyTorch wrapper module
# ---------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    TileLang-accelerated model:

        y = conv3d(x)
        z = tanh(y)
        z = clamp(z, clamp_min, clamp_max)
        w = z * sigmoid(z)
        out = w / sigmoid(w)
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 clamp_min: float,
                 clamp_max: float):
        super().__init__()

        # --- parameters ------------------------------------------------
        self.in_channels  = int(in_channels)
        self.out_channels = int(out_channels)

        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size

        # weight & bias identical to nn.Conv3d defaults
        self.weight = nn.Parameter(torch.empty(
            out_channels, in_channels, self.kd, self.kh, self.kw))
        self.bias   = nn.Parameter(torch.empty(out_channels))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * self.kd * self.kh * self.kw
        bound  = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # clamp params
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # kernel cache: keyed by (N, D_in, H_in, W_in, dtype)
        self._kernels = {}

    # ------------------------------------------------------------------
    # kernel cache helper ----------------------------------------------
    # ------------------------------------------------------------------
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: torch.dtype):
        key = (N, D, H, W, dtype)
        if key not in self._kernels:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kernels[key] = _build_conv3d_fused_kernel(
                N, self.in_channels, self.out_channels,
                D, H, W,
                self.kd, self.kh, self.kw,
                self.clamp_min, self.clamp_max,
                dtype=tl_dtype,
            )
        return self._kernels[key]

    # ------------------------------------------------------------------
    # forward -----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16, copy=False).contiguous()

        N, _, D_in, H_in, W_in = x_f16.shape
        kernel = self._get_kernel(N, D_in, H_in, W_in, x_f16.dtype)

        y_f16 = kernel(x_f16, w_f16, b_f16)
        return y_f16.to(orig_dtype)