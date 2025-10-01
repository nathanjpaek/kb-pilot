"""
Problem Name: 76_Conv3d_Mean_Subtract_Conv3d
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.16 runtime_stats={'mean': 2.16, 'std': 0.00586, 'min': 2.15, 'max': 2.17, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.09, 'std': 0.0356, 'min': 2.06, 'max': 2.38, 'num_trials': 100}, 'speedup_ratio': 0.968}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# TileLang kernel factory : subtract spatial mean per (n,c)                   #
# --------------------------------------------------------------------------- #
def _build_mean_sub_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    threads: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    DHW = D * H * W
    grid = N * C
    inv_DHW = 1.0 / float(DHW)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), dtype),          # input
        Y: T.Tensor((N, C, D, H, W), dtype),          # auto-allocated output
    ):
        inv_val = T.Cast(accum_dtype, inv_DHW)

        with T.Kernel(grid, threads=threads) as bc:   # one block per (n,c)
            tid = T.get_thread_binding(0)
            n   = bc // C
            c   = bc %  C

            # ----------------- parallel reduction -------------------------
            part = T.alloc_local((1,), accum_dtype)
            part[0] = T.Cast(accum_dtype, 0)

            steps = T.ceildiv(DHW, threads)
            for s in T.serial(steps):
                idx = s * threads + tid
                if idx < DHW:
                    d  = idx // (H * W)
                    hw = idx %  (H * W)
                    h  = hw // W
                    w  = hw %  W
                    part[0] += T.Cast(accum_dtype, X[n, c, d, h, w])

            total = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        part[0],
                        True,
                        total[0],
                        tid,
                        dtype="handle",
                    )
                )

            mean_val = total[0] * inv_val

            # ----------------- subtract mean ------------------------------
            for s in T.serial(steps):
                idx = s * threads + tid
                if idx < DHW:
                    d  = idx // (H * W)
                    hw = idx %  (H * W)
                    h  = hw // W
                    w  = hw %  W
                    val = T.Cast(accum_dtype, X[n, c, d, h, w]) - mean_val
                    Y[n, c, d, h, w] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#                           PyTorch wrapper                                   #
# --------------------------------------------------------------------------- #
class ModelNew(torch.nn.Module):
    """
    Conv3d  →  subtract spatial mean  →  Conv3d
    (mean subtraction implemented in TileLang)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()

        k = kernel_size if isinstance(kernel_size, int) else kernel_size
        # ---------------- Conv-1 parameters --------------------------------
        w1_shape = (out_channels, in_channels, k, k, k)
        self.weight1 = torch.nn.Parameter(torch.empty(w1_shape))
        torch.nn.init.kaiming_uniform_(self.weight1, a=math.sqrt(5))

        fan_in1 = in_channels * k ** 3
        bound1  = 1 / math.sqrt(fan_in1)
        self.bias1 = torch.nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.bias1, -bound1, bound1)

        # ---------------- Conv-2 parameters --------------------------------
        w2_shape = (out_channels, out_channels, k, k, k)
        self.weight2 = torch.nn.Parameter(torch.empty(w2_shape))
        torch.nn.init.kaiming_uniform_(self.weight2, a=math.sqrt(5))

        fan_in2 = out_channels * k ** 3
        bound2  = 1 / math.sqrt(fan_in2)
        self.bias2 = torch.nn.Parameter(torch.empty(out_channels))
        torch.nn.init.uniform_(self.bias2, -bound2, bound2)

        # ---------------- kernel cache ------------------------------------
        self._kernels: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, C, D, H, W, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_mean_sub_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kernels[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------------- prepare tensors & params -----------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        w1 = self.weight1.to(device="cuda", dtype=torch.float16)
        b1 = self.bias1.to(device="cuda", dtype=torch.float16)

        w2 = self.weight2.to(device="cuda", dtype=torch.float16)
        b2 = self.bias2.to(device="cuda", dtype=torch.float16)

        # ---------------- first convolution ------------------------------
        x_fp16 = F.conv3d(x_fp16, w1, b1, stride=1, padding=0)

        # ---------------- mean-subtract kernel ---------------------------
        N, C, D, H, W = x_fp16.shape
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        x_fp16 = kernel(x_fp16.contiguous())

        # ---------------- second convolution -----------------------------
        x_fp16 = F.conv3d(x_fp16, w2, b2, stride=1, padding=0)

        return x_fp16.to(orig_dtype)