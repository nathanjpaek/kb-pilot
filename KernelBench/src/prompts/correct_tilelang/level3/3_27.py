"""
Problem Name: 27_RegNet
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.25 runtime_stats={'mean': 2.25, 'std': 0.00943, 'min': 2.23, 'max': 2.32, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.2, 'std': 0.0119, 'min': 2.19, 'max': 2.3, 'num_trials': 100}, 'speedup_ratio': 0.978}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# -------------------------------------------------------------------- #
#                       TileLang kernel factories                      #
# -------------------------------------------------------------------- #
def _build_mean_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    HW = H * W
    grid = (N * C + block - 1) // block
    inv_hw = 1.0 / float(HW)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, H, W), dtype),
        Out: T.Tensor((N, C),        dtype),
    ):
        inv_const = T.Cast(accum_dtype, inv_hw)

        with T.Kernel(grid, threads=block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < N * C:
                n = idx // C
                c = idx - n * C

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for h in T.serial(H):
                    for w in T.serial(W):
                        acc[0] += T.Cast(accum_dtype, X[n, c, h, w])

                mean_val = acc[0] * inv_const
                Out[n, c] = T.Cast(dtype, mean_val)

    return kernel


def _build_gemm_kernel(
    N: int,
    C: int,
    O: int,
    block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT  = N * O
    grid = (TOT + block - 1) // block

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A:   T.Tensor((N, C), dtype),        # mean activations
        W:   T.Tensor((O, C), dtype),        # weight (out, in)
        B:   T.Tensor((O,),   dtype),        # bias
        Out: T.Tensor((N, O), dtype),
    ):
        with T.Kernel(grid, threads=block) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block + tx
            if idx < TOT:
                o = idx % O
                n = idx // O

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for k in T.serial(C):
                    acc[0] += (
                        T.Cast(accum_dtype, A[n, k])
                        * T.Cast(accum_dtype, W[o, k])
                    )

                acc[0] += T.Cast(accum_dtype, B[o])
                Out[n, o] = T.Cast(dtype, acc[0])

    return kernel


# -------------------------------------------------------------------- #
#                          PyTorch wrapper                             #
# -------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    RegNet-like backbone (conv/bn/relu/maxpool blocks) followed by
    global-average-pool + Linear classification, where the last two
    ops are replaced by high-performance TileLang kernels.
    """

    def __init__(
        self,
        input_channels: int,
        stages: int,
        block_widths,
        output_classes: int,
    ):
        super().__init__()

        # ---------------- feature extractor (unchanged) ---------------- #
        self.stages = stages
        self.block_widths = block_widths

        layers = []
        cur_ch = input_channels
        for i in range(stages):
            layers.append(self._make_stage(cur_ch, block_widths[i]))
            cur_ch = block_widths[i]

        self.feature_extractor = nn.Sequential(*layers).cuda().half()

        # ---------------- fully-connected parameters ------------------- #
        in_feat = block_widths[-1]
        self.weight = nn.Parameter(torch.empty(output_classes, in_feat))
        self.bias   = nn.Parameter(torch.empty(output_classes))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(in_feat)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        # ---------------- kernel caches -------------------------------- #
        self._mean_kern_cache: Dict[Tuple[int, int, int, int, str], callable] = {}
        self._gemm_kern_cache: Dict[Tuple[int, int, int, str], callable] = {}

    # ---------------------------------------------------------------- #
    def _make_stage(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1).half().cuda(),
            nn.BatchNorm2d(out_c).half().cuda(),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1).half().cuda(),
            nn.BatchNorm2d(out_c).half().cuda(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    # ---------------------------------------------------------------- #
    def _get_mean_kernel(self, N, C, H, W, dtype):
        key = (N, C, H, W, dtype)
        if key not in self._mean_kern_cache:
            self._mean_kern_cache[key] = _build_mean_kernel(N, C, H, W, dtype=dtype)
        return self._mean_kern_cache[key]

    def _get_gemm_kernel(self, N, C, O, dtype):
        key = (N, C, O, dtype)
        if key not in self._gemm_kern_cache:
            self._gemm_kern_cache[key] = _build_gemm_kernel(N, C, O, dtype=dtype)
        return self._gemm_kern_cache[key]

    # ---------------------------------------------------------------- #
    def forward(self, x: torch.Tensor):
        orig_dtype = x.dtype
        x = x.to(device="cuda", dtype=torch.float16)

        # ------------- feature extractor (fp16) ----------------------- #
        y = self.feature_extractor(x)               # (N, C, H, W)
        N, C, H, W = y.shape

        # ------------- mean pooling kernel --------------------------- #
        mean_kernel = self._get_mean_kernel(N, C, H, W, "float16")
        mean_act = mean_kernel(y.contiguous())      # (N, C)

        # ------------- GEMM kernel ----------------------------------- #
        W_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        B_fp16 = self.bias .to(device="cuda", dtype=torch.float16).contiguous()
        gemm_kernel = self._get_gemm_kernel(N, C, W_fp16.shape[0], "float16")
        out_fp16 = gemm_kernel(mean_act, W_fp16, B_fp16)

        return out_fp16.to(orig_dtype)