"""
Problem Name: 60_Conv3d_Mean_Subtract
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=17.3 runtime_stats={'mean': 17.3, 'std': 0.0932, 'min': 16.8, 'max': 17.4, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.78, 'std': 0.0176, 'min': 1.76, 'max': 1.87, 'num_trials': 100}, 'speedup_ratio': 0.103}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# -------------------------------------------------------------------------
# Kernel factories
# -------------------------------------------------------------------------
def _build_conv3d_kernel(
    N,
    Cin,
    Din,
    Hin,
    Win,
    Cout,
    Kd,
    Kh,
    Kw,
    block_size: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    Dout = Din - Kd + 1
    Hout = Hin - Kh + 1
    Wout = Win - Kw + 1
    total_out = N * Cout * Dout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def conv3d_kernel(
        X: T.Tensor((N, Cin, Din, Hin, Win), dtype),
        Wt: T.Tensor((Cout, Cin, Kd, Kh, Kw), dtype),
        B: T.Tensor((Cout,), dtype),
        Y: T.Tensor((N, Cout, Dout, Hout, Wout), dtype),
    ):
        grid = T.ceildiv(total_out, block_size)
        with T.Kernel(grid, threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            out_idx = bx * block_size + tx
            if out_idx < total_out:
                # Decompose flat index → (n, co, z, y, x)
                remaining = out_idx
                n = remaining // (Cout * Dout * Hout * Wout)
                remaining = remaining % (Cout * Dout * Hout * Wout)
                co = remaining // (Dout * Hout * Wout)
                remaining = remaining % (Dout * Hout * Wout)
                z = remaining // (Hout * Wout)
                remaining = remaining % (Hout * Wout)
                y = remaining // Wout
                x = remaining % Wout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, B[co])

                for ci in range(Cin):
                    for kd in range(Kd):
                        zi = z + kd
                        for kh in range(Kh):
                            yi = y + kh
                            for kw in range(Kw):
                                xi = x + kw
                                vx = X[n, ci, zi, yi, xi].astype(accum_dtype)
                                vw = Wt[co, ci, kd, kh, kw].astype(accum_dtype)
                                acc[0] += vx * vw

                Y[n, co, z, y, x] = T.Cast(dtype, acc[0])

    return conv3d_kernel


def _build_mean_sub_kernel(
    N,
    C,
    D,
    H,
    W,
    block_size: int = 128,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    voxels = D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def mean_sub_kernel(
        Y_in: T.Tensor((N, C, D, H, W), dtype),
        Y_out: T.Tensor((N, C, D, H, W), dtype),
    ):
        with T.Kernel(N * C, threads=block_size) as b:
            tid = T.get_thread_binding(0)
            n = b // C
            c = b % C

            # shared buffer for sum / mean
            sm = T.alloc_shared((1,), accum_dtype)
            if tid == 0:
                sm[0] = T.Cast(accum_dtype, 0)
            T.tvm_storage_sync("shared")

            # partial sums
            for idx_blk in T.serial(T.ceildiv(voxels, block_size)):
                lin = idx_blk * block_size + tid
                if lin < voxels:
                    dz = lin // (H * W)
                    rem = lin % (H * W)
                    dy = rem // W
                    dx = rem % W
                    val = Y_in[n, c, dz, dy, dx].astype(accum_dtype)
                    T.atomic_add(sm[0], val)

            T.tvm_storage_sync("shared")

            # compute mean (thread 0)
            if tid == 0:
                sm[0] = sm[0] / T.Cast(accum_dtype, voxels)
            T.tvm_storage_sync("shared")

            # subtract mean and write out
            for idx_blk in T.serial(T.ceildiv(voxels, block_size)):
                lin = idx_blk * block_size + tid
                if lin < voxels:
                    dz = lin // (H * W)
                    rem = lin % (H * W)
                    dy = rem // W
                    dx = rem % W
                    val = Y_in[n, c, dz, dy, dx]
                    Y_out[n, c, dz, dy, dx] = val - T.Cast(dtype, sm[0])

    return mean_sub_kernel


# -------------------------------------------------------------------------
# Optimized PyTorch wrapper
# -------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    TileLang-accelerated 3D convolution followed by mean subtraction.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kd = self.kh = self.kw = int(kernel_size)

        # Parameters identical to nn.Conv3d defaults
        self.weight = nn.Parameter(
            torch.empty(
                self.out_channels,
                self.in_channels,
                self.kd,
                self.kh,
                self.kw,
            )
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = self.in_channels * self.kd * self.kh * self.kw
        bound = 1 / math.sqrt(fan_in)
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # kernel caches
        self._conv_cache = {}
        self._mean_cache = {}

    # ------------------------------------------------------------------
    # kernel getters
    # ------------------------------------------------------------------
    def _get_conv_kernel(self, shape, dtype: torch.dtype):
        (
            N,
            _,
            Din,
            Hin,
            Win,
        ) = shape
        key = (N, Din, Hin, Win, dtype)
        if key not in self._conv_cache:
            self._conv_cache[key] = _build_conv3d_kernel(
                N,
                self.in_channels,
                Din,
                Hin,
                Win,
                self.out_channels,
                self.kd,
                self.kh,
                self.kw,
                dtype="float16",
            )
        return self._conv_cache[key]

    def _get_mean_kernel(self, shape, dtype: torch.dtype):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype)
        if key not in self._mean_cache:
            self._mean_cache[key] = _build_mean_sub_kernel(
                N,
                C,
                D,
                H,
                W,
                dtype="float16",
            )
        return self._mean_cache[key]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_f16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        # conv3d
        conv_kernel = self._get_conv_kernel(x_f16.shape, x_f16.dtype)
        y_f16 = conv_kernel(x_f16, w_f16, b_f16)

        # mean subtraction
        mean_kernel = self._get_mean_kernel(y_f16.shape, y_f16.dtype)
        out_f16 = mean_kernel(y_f16)

        return out_f16.to(orig_dtype)