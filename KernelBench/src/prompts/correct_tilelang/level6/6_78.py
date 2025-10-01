"""
Problem Name: 78_ConvTranspose3d_Swish_Subtract_Add
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=10.9 runtime_stats={'mean': 10.9, 'std': 0.00931, 'min': 10.9, 'max': 11.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 11.3, 'std': 0.00965, 'min': 11.3, 'max': 11.4, 'num_trials': 100}, 'speedup_ratio': 1.04}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                    fused  swish → –sub + add   kernel factory               #
# --------------------------------------------------------------------------- #
def _build_fused_kernel(
    N: int,
    C: int,
    D: int,
    H: int,
    W: int,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total_elems = N * C * D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X:   T.Tensor((N, C, D, H, W), dtype),   # conv-transpose output
        Sub: T.Tensor((1,), dtype),              # scalar to subtract
        Add: T.Tensor((1,), dtype),              # scalar to add
        Out: T.Tensor((N, C, D, H, W), dtype),   # final result
    ):
        one_f = T.Cast(accum_dtype, 1.0)

        with T.Kernel(T.ceildiv(total_elems, block_size), threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < total_elems:
                w  = idx % W
                t1 = idx // W
                h  = t1 % H
                t2 = t1 // H
                d  = t2 % D
                t3 = t2 // D
                c  = t3 % C
                n  = t3 // C

                x_val = T.Cast(accum_dtype, X[n, c, d, h, w])
                sig   = one_f / (one_f + T.exp(-x_val))
                swish = x_val * sig

                res = swish \
                      - T.Cast(accum_dtype, Sub[0]) \
                      + T.Cast(accum_dtype, Add[0])

                Out[n, c, d, h, w] = T.Cast(dtype, res)

    return kernel


# --------------------------------------------------------------------------- #
#                         PyTorch wrapper  module                             #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    ConvTranspose3d  →  Swish  →  –subtract_value  →  +add_value
    Swish / subtract / add are fused into one TileLang kernel.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super().__init__()

        # identical ConvTranspose3d layer
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
        )

        # learnable scalar parameters (same init rule)
        self.subtract_value = nn.Parameter(torch.randn(1))
        self.add_value      = nn.Parameter(torch.randn(1))

        # kernel cache : {(N,C,D,H,W,dtype) : compiled_kernel}
        self._kern_cache = {}

    # -------------------------------------------------------------- #
    def _get_kernel(self, shape, dtype_str: str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype_str)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_kernel(
                N, C, D, H, W, dtype=dtype_str
            )
        return self._kern_cache[key]

    # -------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # 1) ConvTranspose3d (cuDNN)
        y = self.conv_transpose(x)

        # 2) Move tensors to CUDA / fp16
        y_fp16   = y.to(device="cuda", dtype=torch.float16).contiguous()
        sub_fp16 = self.subtract_value.to(device="cuda", dtype=torch.float16).contiguous()
        add_fp16 = self.add_value.to(device="cuda", dtype=torch.float16).contiguous()

        # 3) Fused TileLang kernel
        kernel = self._get_kernel(y_fp16.shape, "float16")
        out_fp16 = kernel(y_fp16, sub_fp16, add_fp16)

        # 4) cast back to original dtype
        return out_fp16.to(orig_dtype)