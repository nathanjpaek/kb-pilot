"""
Problem Name: 58_Conv3d_Tanh_Clamp_Sigmoid_Swish_Divide
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.278 runtime_stats={'mean': 0.278, 'std': 0.0379, 'min': 0.218, 'max': 0.345, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.05, 'std': 0.0308, 'min': 1.99, 'max': 2.12, 'num_trials': 100}, 'speedup_ratio': 7.37}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ---------------------------------------------------------------------------
# TileLang kernel factory
# ---------------------------------------------------------------------------


def _build_conv3d_fused_kernel(
    N: int,
    C_in: int,
    D: int,
    H: int,
    W: int,
    C_out: int,
    Kd: int,
    Kh: int,
    Kw: int,
    clamp_min: float,
    clamp_max: float,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    D_out = D - Kd + 1
    H_out = H - Kh + 1
    W_out = W - Kw + 1
    total_elems = N * C_out * D_out * H_out * W_out
    cmin = float(clamp_min)
    cmax = float(clamp_max)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C_in, D, H, W), dtype),
        Wt: T.Tensor((C_out, C_in, Kd, Kh, Kw), dtype),
        B: T.Tensor((C_out,), dtype),
        Out: T.Tensor((N, C_out, D_out, H_out, W_out), dtype),
    ):
        one_acc = T.Cast(accum_dtype, 1.0)
        half_acc = T.Cast(accum_dtype, 0.5)

        with T.Kernel(T.ceildiv(total_elems, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total_elems:
                tmp = idx
                ow = tmp % W_out
                tmp //= W_out
                oh = tmp % H_out
                tmp //= H_out
                od = tmp % D_out
                tmp //= D_out
                co = tmp % C_out
                n = tmp // C_out

                acc = T.Cast(accum_dtype, 0)

                for ci in range(C_in):
                    for kd in range(Kd):
                        id_in = od + kd
                        for kh in range(Kh):
                            ih_in = oh + kh
                            for kw in range(Kw):
                                iw_in = ow + kw
                                acc += (
                                    T.Cast(
                                        accum_dtype, X[n, ci, id_in, ih_in, iw_in]
                                    )
                                    * T.Cast(
                                        accum_dtype, Wt[co, ci, kd, kh, kw]
                                    )
                                )

                acc += T.Cast(accum_dtype, B[co])

                # Tanh -> Clamp -> Sigmoid -> Swish variant -> divide by 2
                acc = T.tanh(acc)
                acc = T.clamp(acc, cmin, cmax)
                sig1 = one_acc / (one_acc + T.exp(-acc))
                sig2 = one_acc / (one_acc + T.exp(-sig1))
                out_val = half_acc * sig1 * sig2

                Out[n, co, od, oh, ow] = T.Cast(dtype, out_val)

    return kernel


# ---------------------------------------------------------------------------
# PyTorch wrapper
# ---------------------------------------------------------------------------


class ModelNew(nn.Module):
    """
    TileLang-accelerated replacement for the original 3D-conv model with fused
    tanh → clamp → sigmoid → swish → divide-by-2 post-processing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        clamp_min: float,
        clamp_max: float,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Parse kernel size -------------------------------------------------
        # ------------------------------------------------------------------
        if isinstance(kernel_size, int):
            Kd = Kh = Kw = int(kernel_size)
        else:
            assert len(kernel_size) == 3
            Kd, Kh, Kw = [int(k) for k in kernel_size]
        self.Kd, self.Kh, self.Kw = Kd, Kh, Kw

        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # ------------------------------------------------------------------
        # Parameters – same init as nn.Conv3d ------------------------------
        # ------------------------------------------------------------------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, Kd, Kh, Kw)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = in_channels * Kd * Kh * Kw
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # ------------------------------------------------------------------
        # Kernel cache ------------------------------------------------------
        # ------------------------------------------------------------------
        self._kern_cache = {}

    # ----------------------------------------------------------------------
    # Kernel retrieval / compilation
    # ----------------------------------------------------------------------
    def _get_kernel(
        self,
        N: int,
        D: int,
        H: int,
        W: int,
        dtype: torch.dtype,
    ):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kern_cache[key] = _build_conv3d_fused_kernel(
                N,
                self.weight.shape[1],
                D,
                H,
                W,
                self.weight.shape[0],
                self.Kd,
                self.Kh,
                self.Kw,
                self.clamp_min,
                self.clamp_max,
                dtype=tl_dtype,
            )
        return self._kern_cache[key]

    # ----------------------------------------------------------------------
    # Forward --------------------------------------------------------------
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        N, C_in, D, H, W = x_f16.shape

        kernel = self._get_kernel(N, D, H, W, x_f16.dtype)
        out_f16 = kernel(x_f16, w_f16, b_f16)

        return out_f16.to(orig_dtype)