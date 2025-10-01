"""
Problem Name: 8_Average_Pooling_2D
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0888 runtime_stats={'mean': 0.0888, 'std': 0.00204, 'min': 0.0861, 'max': 0.0991, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.125, 'std': 0.00781, 'min': 0.122, 'max': 0.201, 'num_trials': 100}, 'speedup_ratio': 1.41}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_avgpool2d_kernel(
    N: int,
    C: int,
    H: int,
    W: int,
    K: int,
    S: int,
    P: int,
    *,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    HO = (H + 2 * P - K) // S + 1
    WO = (W + 2 * P - K) // S + 1
    numel = N * C * HO * WO
    denom = float(K * K)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def avgpool2d_kernel(
        X: T.Tensor((N, C, H, W), dtype),
        Y: T.Tensor((N, C, HO, WO), dtype),
    ):
        denom_const = T.Cast(accum_dtype, denom)

        with T.Kernel(T.ceildiv(numel, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel:
                # Decode flat index -> (n, c, ho, wo)
                tmp = idx
                wo = tmp % WO
                tmp //= WO
                ho = tmp % HO
                tmp //= HO
                c = tmp % C
                n = tmp // C

                h_start = ho * S - P
                w_start = wo * S - P

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for kh in T.serial(K):
                    h_in = h_start + kh
                    for kw in T.serial(K):
                        w_in = w_start + kw
                        in_bounds = (
                            (h_in >= 0)
                            and (h_in < H)
                            and (w_in >= 0)
                            and (w_in < W)
                        )
                        if in_bounds:
                            acc[0] += T.Cast(
                                accum_dtype, X[n, c, h_in, w_in]
                            )
                        else:
                            acc[0] += T.Cast(accum_dtype, 0)

                Y[n, c, ho, wo] = T.Cast(dtype, acc[0] / denom_const)

    return avgpool2d_kernel


class ModelNew(nn.Module):
    """
    TileLang-optimized 2D Average Pooling (NCHW layout, count_include_pad=True)
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else self.kernel_size
        self.padding = int(padding)

        # Cache compiled kernels keyed by (N, C, H, W, dtype)
        self._kern_cache = {}

    # ------------------------------------------------------------------ #
    # Kernel cache retrieval                                             #
    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, C: int, H: int, W: int, dtype: torch.dtype):
        key = (N, C, H, W, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kern_cache[key] = _build_avgpool2d_kernel(
                N,
                C,
                H,
                W,
                self.kernel_size,
                self.stride,
                self.padding,
                dtype=tl_dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    # Forward pass                                                       #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f16 = x.to(device="cuda", dtype=torch.float16, copy=False).contiguous()

        N, C, H, W = x_f16.shape
        kernel = self._get_kernel(N, C, H, W, x_f16.dtype)
        y_f16 = kernel(x_f16)

        return y_f16.to(orig_dtype)