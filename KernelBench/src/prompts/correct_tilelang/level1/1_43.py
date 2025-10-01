"""
Problem Name: 43_Max_Pooling_3D
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.175 runtime_stats={'mean': 0.175, 'std': 0.00978, 'min': 0.169, 'max': 0.258, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.482, 'std': 0.00673, 'min': 0.478, 'max': 0.542, 'num_trials': 100}, 'speedup_ratio': 2.75}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _build_maxpool3d_kernel(
    N: int,
    C: int,
    D_in: int,
    H_in: int,
    W_in: int,
    K: int,
    S: int,
    P: int,
    Dil: int,
    block_elems: int = 256,
    dtype: str = "float16",
):
    # Output dimensions (ceil_mode==False)
    D_out = (D_in + 2 * P - Dil * (K - 1) - 1) // S + 1
    H_out = (H_in + 2 * P - Dil * (K - 1) - 1) // S + 1
    W_out = (W_in + 2 * P - Dil * (K - 1) - 1) // S + 1

    O = N * C * D_out * H_out * W_out

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def maxpool3d(
        X: T.Tensor((N, C, D_in, H_in, W_in), dtype),
        Out: T.Tensor((N, C, D_out, H_out, W_out), dtype),
    ):
        with T.Kernel(T.ceildiv(O, block_elems), threads=block_elems) as bx:
            tx = T.get_thread_binding(0)
            gidx = bx * block_elems + tx
            if gidx < O:
                w_out = gidx % W_out
                t1 = gidx // W_out
                h_out = t1 % H_out
                t2 = t1 // H_out
                d_out = t2 % D_out
                t3 = t2 // D_out
                c = t3 % C
                n = t3 // C

                max_val = T.alloc_local((1,), dtype)
                max_val[0] = -T.infinity(dtype)

                d_start = d_out * S - P
                h_start = h_out * S - P
                w_start = w_out * S - P

                for kd in T.serial(K):
                    d_in = d_start + kd * Dil
                    for kh in T.serial(K):
                        h_in = h_start + kh * Dil
                        for kw in T.serial(K):
                            w_in = w_start + kw * Dil
                            in_bounds = (
                                (d_in >= 0) and (d_in < D_in) and
                                (h_in >= 0) and (h_in < H_in) and
                                (w_in >= 0) and (w_in < W_in)
                            )
                            val = T.if_then_else(
                                in_bounds,
                                X[n, c, d_in, h_in, w_in],
                                -T.infinity(dtype),
                            )
                            max_val[0] = T.max(max_val[0], val)

                Out[n, c, d_out, h_out, w_out] = max_val[0]

    return maxpool3d


class ModelNew(nn.Module):
    """TileLang replacement for nn.MaxPool3d (no indices, ceil_mode=False)."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ):
        super().__init__()
        assert not return_indices, "return_indices=True not supported"
        assert not ceil_mode, "ceil_mode=True not supported"
        self.K = int(kernel_size)
        self.S = int(stride) if stride is not None else int(kernel_size)
        self.P = int(padding)
        self.Dil = int(dilation)
        self._cache = {}

    # ------------------------------------------------------------------
    def _get_kernel(self, shape, dtype_str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype_str)
        if key not in self._cache:
            self._cache[key] = _build_maxpool3d_kernel(
                N,
                C,
                D,
                H,
                W,
                self.K,
                self.S,
                self.P,
                self.Dil,
                dtype=dtype_str,
            )
        return self._cache[key]

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        kernel = self._get_kernel(x_fp16.shape, "float16")
        y_fp16 = kernel(x_fp16)
        return y_fp16.to(orig_dtype)