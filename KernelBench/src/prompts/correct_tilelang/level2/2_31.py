"""
Problem Name: 31_Conv2d_Min_Add_Multiply
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0579 runtime_stats={'mean': 0.0579, 'std': 0.00119, 'min': 0.0565, 'max': 0.0642, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0644, 'std': 0.00255, 'min': 0.0623, 'max': 0.0797, 'num_trials': 100}, 'speedup_ratio': 1.11}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_fused_conv_kernel(
    N: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    K: int,
    const_val: float,
    scale_val: float,
    dtype: str = "float16",
    accum_dtype: str = "float",
    threads_per_block: int = 256,
):
    Hout = Hin - K + 1
    Wout = Win - K + 1
    TOTAL = N * Cout * Hout * Wout
    GRID = (TOTAL + threads_per_block - 1) // threads_per_block

    const_c = float(const_val)
    scale_c = float(scale_val)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, Cin, Hin, Win), dtype),
        Wt: T.Tensor((Cout, Cin, K, K), dtype),
        B_conv: T.Tensor((Cout,), dtype),
        B_add: T.Tensor((Cout,), dtype),
        Out: T.Tensor((N, Cout, Hout, Wout), dtype),
    ):
        const_val_f = T.Cast(accum_dtype, const_c)
        scale_f = T.Cast(accum_dtype, scale_c)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOTAL:
                ow = idx % Wout
                tmp = idx // Wout
                oh = tmp % Hout
                tmp //= Hout
                oc = tmp % Cout
                n = tmp // Cout

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, 0)

                for ic in T.serial(Cin):
                    for kh in T.serial(K):
                        ih = oh + kh
                        for kw in T.serial(K):
                            iw = ow + kw
                            acc[0] += (
                                T.Cast(accum_dtype, X[n, ic, ih, iw])
                                * T.Cast(accum_dtype, Wt[oc, ic, kh, kw])
                            )

                acc[0] += T.Cast(accum_dtype, B_conv[oc])

                # Fuse element-wise ops
                acc[0] = T.min(acc[0], const_val_f)
                acc[0] = acc[0] + T.Cast(accum_dtype, B_add[oc])
                acc[0] = acc[0] * scale_f

                Out[n, oc, oh, ow] = T.Cast(dtype, acc[0])

    return kernel


class ModelNew(nn.Module):
    """
    Conv2d → min(constant) → bias add → scaling, fused with TileLang
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        constant_value: float,
        bias_shape,
        scaling_factor: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = int(kernel_size)
        self.constant_value = float(constant_value)
        self.scaling_factor = float(scaling_factor)

        # Conv2d parameters identical to PyTorch defaults
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * self.kernel_size * self.kernel_size
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.conv_bias, -bound, bound)

        # Extra bias added after min()
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Kernel cache
        self._kernel_cache = {}

    def _get_kernel(self, N: int, H: int, W: int, dtype: str = "float16"):
        key = (N, H, W, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_fused_conv_kernel(
                N,
                self.in_channels,
                H,
                W,
                self.out_channels,
                self.kernel_size,
                self.constant_value,
                self.scaling_factor,
                dtype=dtype,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, H, W = x_fp16.shape
        kernel = self._get_kernel(N, H, W, "float16")

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_conv_fp16 = self.conv_bias.to(device="cuda", dtype=torch.float16).contiguous()
        b_add_fp16 = self.bias.to(device="cuda", dtype=torch.float16).view(-1).contiguous()

        y_fp16 = kernel(x_fp16, w_fp16, b_conv_fp16, b_add_fp16)
        return y_fp16.to(orig_dtype)