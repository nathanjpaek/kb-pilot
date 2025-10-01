"""
Problem Name: 20_Conv3d_Subtract_Mish
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.32 runtime_stats={'mean': 4.32, 'std': 0.0664, 'min': 4.21, 'max': 4.67, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.2, 'std': 0.0346, 'min': 4.14, 'max': 4.35, 'num_trials': 100}, 'speedup_ratio': 0.972}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# -----------------------------------------------------------------------------
# Kernel builders
# -----------------------------------------------------------------------------
def _build_conv3d_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    F: int,
    KD: int,
    KH: int,
    KW: int,
    *,
    block_size: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1
    total_out = N * F * OD * OH * OW

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv3d_kernel(
        X: T.Tensor((N, C, D, H, W), dtype),
        Wt: T.Tensor((F, C, KD, KH, KW), dtype),
        B: T.Tensor((F,), dtype),
        Y: T.Tensor((N, F, OD, OH, OW), dtype),
    ):
        with T.Kernel(T.ceildiv(total_out, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx

            if idx < total_out:
                tmp = idx
                ow = tmp % OW
                tmp //= OW
                oh = tmp % OH
                tmp //= OH
                od = tmp % OD
                tmp //= OD
                f = tmp % F
                n = tmp // F

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for c in T.serial(C):
                    for kd in T.serial(KD):
                        for kh in T.serial(KH):
                            for kw in T.serial(KW):
                                id_ = od + kd
                                ih_ = oh + kh
                                iw_ = ow + kw
                                inp_val = X[n, c, id_, ih_, iw_]
                                w_val = Wt[f, c, kd, kh, kw]
                                acc[0] += inp_val.astype(accum_dtype) * w_val.astype(
                                    accum_dtype
                                )

                acc[0] += B[f].astype(accum_dtype)
                Y[n, f, od, oh, ow] = T.Cast(dtype, acc[0])

    return conv3d_kernel


def _build_transform_kernel(
    numel: int,
    *,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def transform_kernel(
        X: T.Tensor((numel,), dtype),
        mean_t: T.Tensor((1,), dtype),
        Out: T.Tensor((numel,), dtype),
    ):
        one_const = T.Cast(accum_dtype, 1)
        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                val = X[idx].astype(accum_dtype) - mean_t[0].astype(accum_dtype)
                softplus = T.log(one_const + T.exp(val))
                mish_val = val * T.tanh(softplus)
                Out[idx] = T.Cast(dtype, mish_val)

    return transform_kernel


# -----------------------------------------------------------------------------
# PyTorch wrapper module
# -----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Optimized 3D Conv → mean subtract → Mish activation using TileLang.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple):
        super().__init__()

        if isinstance(kernel_size, int):
            KD = KH = KW = int(kernel_size)
        else:
            KD, KH, KW = kernel_size

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.KD, self.KH, self.KW = KD, KH, KW

        # ----- parameters (same init as nn.Conv3d) --------------------------------------
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, KD, KH, KW)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        self.bias = nn.Parameter(torch.empty(out_channels))
        bound = 1 / math.sqrt(in_channels * KD * KH * KW)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # ----- kernel caches ------------------------------------------------------------
        self._conv_cache = {}        # keyed by (N, D, H, W, dtype)
        self._trans_cache = {}       # keyed by (numel, dtype)

    # -----------------------------------------------------------------------------
    # helpers
    # -----------------------------------------------------------------------------
    def _get_conv_kernel(
        self, N: int, D: int, H: int, W: int, dtype: torch.dtype
    ):
        key = (N, D, H, W, dtype)
        if key not in self._conv_cache:
            tl_dtype = "float16"
            self._conv_cache[key] = _build_conv3d_kernel(
                N,
                self.in_channels,
                D,
                H,
                W,
                self.out_channels,
                self.KD,
                self.KH,
                self.KW,
                dtype=tl_dtype,
            )
        return self._conv_cache[key]

    def _get_trans_kernel(self, numel: int, dtype: torch.dtype):
        key = (numel, dtype)
        if key not in self._trans_cache:
            tl_dtype = "float16"
            self._trans_cache[key] = _build_transform_kernel(numel, dtype=tl_dtype)
        return self._trans_cache[key]

    # -----------------------------------------------------------------------------
    # forward
    # -----------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16)
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16)
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16)

        N, C, D, H, W = x_f16.shape
        OD = D - self.KD + 1
        OH = H - self.KH + 1
        OW = W - self.KW + 1
        out_numel = N * self.out_channels * OD * OH * OW

        # ----- conv ------------------------------------------------------------------
        conv_kernel = self._get_conv_kernel(N, D, H, W, x_f16.dtype)
        y_f16 = conv_kernel(x_f16, w_f16, b_f16)

        # ----- mean ------------------------------------------------------------------
        mean_val = float(y_f16.to(torch.float32).mean().item())
        mean_tensor = torch.tensor([mean_val], dtype=torch.float16, device="cuda")

        # ----- transform (subtract mean + Mish) --------------------------------------
        trans_kernel = self._get_trans_kernel(out_numel, y_f16.dtype)
        y_final = trans_kernel(y_f16.view(-1), mean_tensor).view_as(y_f16)

        return y_final.to(orig_dtype)