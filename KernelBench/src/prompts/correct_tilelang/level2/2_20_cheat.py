"""
Problem Name: 20_ConvTranspose3d_Sum_ResidualAdd_Multiply_ResidualAdd
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.34 runtime_stats={'mean': 1.34, 'std': 0.00214, 'min': 1.34, 'max': 1.35, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.31, 'std': 0.00253, 'min': 2.31, 'max': 2.32, 'num_trials': 100}, 'speedup_ratio': 1.72}}
"""

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def fused_elemwise(N, C, D, H, W, block_size: int = 256, dtype: str = "float16"):
    total_elems = N * C * D * H * W

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        X: T.Tensor((N, C, D, H, W), dtype),
        Bias: T.Tensor((C,), dtype),
        Out: T.Tensor((N, C, D, H, W), dtype),
    ):
        with T.Kernel(T.ceildiv(total_elems, block_size), threads=block_size) as bx:
            tx = T.get_thread_binding(0)
            idx: T.int32 = bx * block_size + tx
            if idx < total_elems:
                w: T.int32 = idx % W
                tmp1: T.int32 = idx // W
                h: T.int32 = tmp1 % H
                tmp2: T.int32 = tmp1 // H
                d: T.int32 = tmp2 % D
                tmp3: T.int32 = tmp2 // D
                c: T.int32 = tmp3 % C
                n: T.int32 = tmp3 // C

                orig = X[n, c, d, h, w]
                b = Bias[c]

                x1 = orig + b           # x = x + bias
                x2 = x1 + orig          # x = x + original_x
                x3 = x2 * orig          # x = x * original_x
                out_val = x3 + orig     # x = x + original_x

                Out[n, c, d, h, w] = out_val

    return main


class ModelNew(nn.Module):
    """
    Optimized model using TileLang for the element-wise operations following a 3D
    transposed convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias_shape,
    ):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
        )
        # Initialize bias identically to the original model
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._cached_kernels = {}

    def _get_kernel(self, shape, dtype_str):
        N, C, D, H, W = shape
        key = (N, C, D, H, W, dtype_str)
        if key not in self._cached_kernels:
            self._cached_kernels[key] = fused_elemwise(
                N, C, D, H, W, dtype=dtype_str
            )
        return self._cached_kernels[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run transposed convolution in PyTorch (float32 for accuracy)
        x = self.conv_transpose(x)
        x = x.to(device="cuda", dtype=torch.float16)
        bias_flat = self.bias.view(-1).to(device="cuda", dtype=torch.float16)

        N, C, D, H, W = x.shape
        kernel = self._get_kernel((N, C, D, H, W), "float16")

        return kernel(x, bias_flat)