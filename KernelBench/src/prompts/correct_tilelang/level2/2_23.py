"""
Problem Name: 23_Conv3d_GroupNorm_Mean
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.1 runtime_stats={'mean': 1.1, 'std': 0.0079, 'min': 1.09, 'max': 1.16, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.596, 'std': 0.0103, 'min': 0.588, 'max': 0.69, 'num_trials': 100}, 'speedup_ratio': 0.542}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Kernel factory : mean across (C, D, H, W) for each sample                  #
# --------------------------------------------------------------------------- #
def _build_mean_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    TOT_PER_SAMPLE = C * D * H * W
    scale_const = 1.0 / TOT_PER_SAMPLE

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def mean_kernel(
        X_flat: T.Tensor((N * TOT_PER_SAMPLE,), dtype),
        Out:    T.Tensor((N,), dtype),       # auto-allocated
    ):
        scale_c = T.Cast(accum_dtype, scale_const)

        with T.Kernel(N, threads=block_size) as bn:    # one block per sample
            tx = T.get_thread_binding(0)
            part = T.alloc_local((1,), accum_dtype)
            part[0] = T.Cast(accum_dtype, 0)

            for it in T.serial(T.ceildiv(TOT_PER_SAMPLE, block_size)):
                idx = it * block_size + tx
                if idx < TOT_PER_SAMPLE:
                    part[0] += T.Cast(
                        accum_dtype,
                        X_flat[bn * TOT_PER_SAMPLE + idx],
                    )

            total = T.alloc_local((1,), accum_dtype)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.Cast(accum_dtype, 0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        part[0],
                        True,
                        total[0],
                        tx,
                        dtype="handle",
                    )
                )

            if tx == 0:
                Out[bn] = T.Cast(dtype, total[0] * scale_c)

    return mean_kernel


# --------------------------------------------------------------------------- #
# Optimised module                                                            #
# --------------------------------------------------------------------------- #
class ModelNew(torch.nn.Module):
    """
    Conv3d → GroupNorm → Mean  (mean executed as a TileLang reduction)
    """

    def __init__(self, in_channels, out_channels, kernel_size, num_groups, eps: float = 1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.num_groups = num_groups
        self.eps = eps

        # --------------------- Conv3d parameters ---------------------------
        w_shape = (out_channels, in_channels, *self.kernel_size)
        self.weight = torch.nn.Parameter(torch.empty(w_shape))
        self.bias   = torch.nn.Parameter(torch.empty(out_channels))

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_channels * math.prod(self.kernel_size)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # --------------------- GroupNorm parameters ------------------------
        self.gn_weight = torch.nn.Parameter(torch.ones(out_channels))
        self.gn_bias   = torch.nn.Parameter(torch.zeros(out_channels))

        # --------------------- Kernel cache --------------------------------
        self._kern_cache: Dict[Tuple[int, int, int, int, int, str], callable] = {}

    # --------------------------------------------------------------------- #
    # Helper: get / compile reduction kernel                                #
    # --------------------------------------------------------------------- #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype_str: str):
        key = (N, C, D, H, W, dtype_str)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_mean_kernel(N, C, D, H, W, dtype=dtype_str)
        return self._kern_cache[key]

    # --------------------------------------------------------------------- #
    # Forward                                                               #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move to CUDA / fp16 early
        x = x.to(device="cuda", dtype=torch.float16)

        # --------------------- Conv3d -------------------------------------
        x = F.conv3d(
            x,
            self.weight.to(dtype=torch.float16, device="cuda"),
            self.bias.to(dtype=torch.float16, device="cuda"),
        )  # stride=1, padding=0, dilation=1

        # --------------------- GroupNorm ----------------------------------
        N, C, D, H, W = x.shape
        G = self.num_groups
        x_reshaped = x.view(N, G, -1).to(torch.float32)

        mean = x_reshaped.mean(dim=2, keepdim=True)
        var  = x_reshaped.var(dim=2, unbiased=False, keepdim=True)

        x_norm = ((x_reshaped - mean) / torch.sqrt(var + self.eps)).view(N, C, D, H, W)
        x_norm = x_norm.to(torch.float16)

        x_norm = (
            x_norm
            * self.gn_weight.to(dtype=torch.float16, device="cuda").view(1, C, 1, 1, 1)
            + self.gn_bias.to(dtype=torch.float16, device="cuda").view(1, C, 1, 1, 1)
        )

        # --------------------- Mean reduction via TileLang ----------------
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        out_fp16 = kernel(x_norm.contiguous().view(-1))  # (N,)

        return out_fp16.to(x.dtype)