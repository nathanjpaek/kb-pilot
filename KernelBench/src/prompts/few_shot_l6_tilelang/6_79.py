"""
Problem Name: 79_Conv3d_Tanh_Sigmoid_Divide_Swish_Clamp
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.561 runtime_stats={'mean': 0.561, 'std': 0.000928, 'min': 0.56, 'max': 0.566, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.29, 'std': 0.0122, 'min': 2.27, 'max': 2.31, 'num_trials': 100}, 'speedup_ratio': 4.08}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# TileLang kernel factory ----------------------------------------------------
# ---------------------------------------------------------------------------


def _build_fused_conv3d_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    O: int,
    KD: int,
    KH: int,
    KW: int,
    *,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    D_out = D - KD + 1
    H_out = H - KH + 1
    W_out = W - KW + 1
    numel = N * O * D_out * H_out * W_out

    eps_const = 1e-5
    neg_half = -0.5
    pos_half = 0.5

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv3d_fused(
        X: T.Tensor((N, C, D, H, W), dtype),
        Wt: T.Tensor((O, C, KD, KH, KW), dtype),
        B: T.Tensor((O,), dtype),
        Y: T.Tensor((N, O, D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                # -----------------------------------------------------------
                # Unravel flat index -> (n, oc, od, oh, ow)
                # -----------------------------------------------------------
                ow = idx % W_out
                tmp1 = idx // W_out
                oh = tmp1 % H_out
                tmp2 = tmp1 // H_out
                od = tmp2 % D_out
                tmp3 = tmp2 // D_out
                oc = tmp3 % O
                n = tmp3 // O

                # -----------------------------------------------------------
                # Convolution accumulate (float32)
                # -----------------------------------------------------------
                acc = T.Cast(accum_dtype, 0)
                for ic in T.serial(C):
                    for kd in T.serial(KD):
                        for kh in T.serial(KH):
                            for kw in T.serial(KW):
                                xv = X[n, ic, od + kd, oh + kh, ow + kw]
                                wv = Wt[oc, ic, kd, kh, kw]
                                acc += (
                                    xv.astype(accum_dtype)
                                    * wv.astype(accum_dtype)
                                )

                acc += B[oc].astype(accum_dtype)

                # -----------------------------------------------------------
                # Fused non-linear chain
                # tanh -> sigmoid -> divide by sigmoid(sigmoid) + eps ->
                # swish -> clamp
                # -----------------------------------------------------------
                # tanh
                exp_pos = T.exp(acc)
                exp_neg = T.exp(-acc)
                tanh_val = (exp_pos - exp_neg) / (exp_pos + exp_neg)

                # first sigmoid
                s1 = T.Cast(accum_dtype, 1.0) / (
                    T.Cast(accum_dtype, 1.0) + T.exp(-tanh_val)
                )

                # second sigmoid
                s2 = T.Cast(accum_dtype, 1.0) / (
                    T.Cast(accum_dtype, 1.0) + T.exp(-s1)
                )

                # division
                x_div = s1 / (s2 + T.Cast(accum_dtype, eps_const))

                # swish = x * sigmoid(x)
                sig_x = T.Cast(accum_dtype, 1.0) / (
                    T.Cast(accum_dtype, 1.0) + T.exp(-x_div)
                )
                swish_val = x_div * sig_x

                # clamp to [-0.5, 0.5]
                clamped = T.max(
                    T.Cast(accum_dtype, neg_half),
                    T.min(swish_val, T.Cast(accum_dtype, pos_half)),
                )

                Y[n, oc, od, oh, ow] = T.Cast(dtype, clamped)

    return conv3d_fused


# ---------------------------------------------------------------------------
# PyTorch wrapper module -----------------------------------------------------
# ---------------------------------------------------------------------------


class ModelNew(nn.Module):
    """
    TileLang-accelerated version of the original model
    (Conv3d + tanh + sigmoid chain + swish + clamp)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size):
        super().__init__()

        # ------------------------------------------------------------------
        # Parameters (match nn.Conv3d defaults) -----------------------------
        # ------------------------------------------------------------------
        if isinstance(kernel_size, int):
            k_d = k_h = k_w = kernel_size
        else:
            k_d, k_h, k_w = kernel_size

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.k_d = int(k_d)
        self.k_h = int(k_h)
        self.k_w = int(k_w)

        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                self.k_d,
                self.k_h,
                self.k_w,
            )
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels))

        # Weight & bias init identical to PyTorch's nn.Conv3d
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_channels * self.k_d * self.k_h * self.k_w
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # kernel cache
        self._kern_cache = {}

    # ------------------------------------------------------------------
    # Kernel getter ----------------------------------------------------
    # ------------------------------------------------------------------
    def _get_kernel(self, N: int, D: int, H: int, W: int, dtype: torch.dtype):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype == torch.float16 else "bfloat16"
            self._kern_cache[key] = _build_fused_conv3d_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.k_d,
                self.k_h,
                self.k_w,
                dtype=tl_dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------
    # Forward ----------------------------------------------------------
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        N, C, D, H, W = x_f16.shape
        assert C == self.in_channels, "Input channel mismatch"

        kernel = self._get_kernel(N, D, H, W, x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, b_f16)

        return y_f16.to(orig_dtype)