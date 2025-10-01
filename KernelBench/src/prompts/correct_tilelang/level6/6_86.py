"""
Problem Name: 86_Conv3d_Min_Max
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.87 runtime_stats={'mean': 1.87, 'std': 0.0208, 'min': 1.86, 'max': 2.07, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.21, 'std': 0.0257, 'min': 2.19, 'max': 2.44, 'num_trials': 100}, 'speedup_ratio': 1.18}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                       TileLang kernel factory :  min+max                    #
# --------------------------------------------------------------------------- #
def _build_minmax_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = N * D * H * W       # one thread per output element
    grid  = (total + block_size - 1) // block_size

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, C, D, H, W), dtype),               # conv output
        Y: T.Tensor((N, 1, D, H, W), dtype),               # allocated by jit
    ):
        big_pos = T.Cast(accum_dtype, 3.4e38)
        big_neg = -big_pos

        with T.Kernel(grid, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                d_ = t2 % D
                n  = t2 // D

                mn = T.alloc_local((1,), accum_dtype)
                mx = T.alloc_local((1,), accum_dtype)
                mn[0] = big_pos
                mx[0] = big_neg

                for c in range(C):
                    val = X[n, c, d_, h, w].astype(accum_dtype)
                    mn[0] = T.min(mn[0], val)
                    mx[0] = T.max(mx[0], val)

                Y[n, 0, d_, h, w] = T.Cast(dtype, mn[0] + mx[0])

    return kernel


# --------------------------------------------------------------------------- #
#                              PyTorch wrapper                                #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Conv3d  →  (min + max across channel)   implemented with TileLang.
    Output shape : (N,1,D,H,W)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        # keep original Conv3d (identical init)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)

        # kernel cache  :  {(N,D,H,W,dtype) : compiled kernel}
        self._kern_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, D: int, H: int, W: int, dtype: str):
        key = (N, D, H, W, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_minmax_kernel(
                N, C, D, H, W, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # 1) convolution (high precision)
        y = self.conv(x)

        # 2) move to CUDA / fp16 for reduction kernel
        y_fp16 = y.to(device="cuda", dtype=torch.float16).contiguous()
        N, C, D, H, W = y_fp16.shape

        # 3) min+max reduction
        kernel = self._get_kernel(N, C, D, H, W, "float16")
        out_fp16 = kernel(y_fp16)                   # (N,1,D,H,W)

        # 4) cast back
        return out_fp16.to(orig_dtype)