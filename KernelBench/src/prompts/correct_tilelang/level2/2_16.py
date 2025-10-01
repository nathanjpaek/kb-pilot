"""
Problem Name: 16_ConvTranspose2d_Mish_Add_Hardtanh_Scaling
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.64 runtime_stats={'mean': 2.64, 'std': 0.00905, 'min': 2.63, 'max': 2.72, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.167, 'std': 0.0248, 'min': 0.158, 'max': 0.408, 'num_trials': 100}, 'speedup_ratio': 0.0633}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_fused_deconv_kernel(
    N: int,
    IC: int,
    H_in: int,
    W_in: int,
    OC: int,
    K: int,
    stride: int,
    padding: int,
    output_padding: int,
    add_val: float,
    scale_val: float,
    threads_per_block: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    H_out = (H_in - 1) * stride - 2 * padding + K + output_padding
    W_out = (W_in - 1) * stride - 2 * padding + K + output_padding
    TOTAL = N * OC * H_out * W_out
    GRID = (TOTAL + threads_per_block - 1) // threads_per_block

    add_c = float(add_val)
    scale_c = float(scale_val)

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((N, IC, H_in, W_in), dtype),
        Wt: T.Tensor((IC, OC, K, K), dtype),
        B: T.Tensor((OC,), dtype),
        Out: T.Tensor((N, OC, H_out, W_out), dtype),
    ):
        add_const = T.Cast(accum_dtype, add_c)
        scale_const = T.Cast(accum_dtype, scale_c)
        one_f = T.Cast(accum_dtype, 1.0)
        neg_one_f = T.Cast(accum_dtype, -1.0)

        with T.Kernel(GRID, threads=threads_per_block) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads_per_block + tx
            if idx < TOTAL:
                wo = idx % W_out
                tmp = idx // W_out
                ho = tmp % H_out
                tmp //= H_out
                oc = tmp % OC
                n = tmp // OC

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = 0.0

                for ic in T.serial(IC):
                    for kh in T.serial(K):
                        h_nom = ho + padding - kh
                        if (h_nom % stride) == 0:
                            hi = h_nom // stride
                            if (0 <= hi) and (hi < H_in):
                                for kw in T.serial(K):
                                    w_nom = wo + padding - kw
                                    if (w_nom % stride) == 0:
                                        wi = w_nom // stride
                                        if (0 <= wi) and (wi < W_in):
                                            acc[0] += (
                                                T.Cast(accum_dtype, X[n, ic, hi, wi])
                                                * T.Cast(accum_dtype, Wt[ic, oc, kh, kw])
                                            )

                acc[0] += T.Cast(accum_dtype, B[oc])

                # Mish activation
                sp = T.log(one_f + T.exp(acc[0]))
                val = acc[0] * T.tanh(sp)

                # Add constant, clamp, scale
                val = val + add_const
                val = T.max(val, neg_one_f)
                val = T.min(val, one_f)
                val = val * scale_const

                Out[n, oc, ho, wo] = T.Cast(dtype, val)

    return kernel


class ModelNew(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        output_padding: int,
        add_value: float,
        scale: float,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = float(add_value)
        self.scale = float(scale)

        # Conv-transpose parameters (match PyTorch defaults)
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        with torch.no_grad():
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            fan_in = in_channels * kernel_size * kernel_size
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

        self._kern_cache = {}

    def _get_kernel(self, N: int, H_in: int, W_in: int, dtype: str = "float16"):
        key = (N, H_in, W_in, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_fused_deconv_kernel(
                N,
                self.in_channels,
                H_in,
                W_in,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.add_value,
                self.scale,
                dtype=dtype,
            )
        return self._kern_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, _, H_in, W_in = x_fp16.shape
        kernel = self._get_kernel(N, H_in, W_in)

        w_fp16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b_fp16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        y_fp16 = kernel(x_fp16, w_fp16, b_fp16)
        return y_fp16.to(orig_dtype)