"""
Problem Name: 77_ConvTranspose2d_Sigmoid_BiasAdd_Sigmoid
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.38 runtime_stats={'mean': 4.38, 'std': 0.0347, 'min': 4.33, 'max': 4.46, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.344, 'std': 0.0441, 'min': 0.289, 'max': 0.496, 'num_trials': 100}, 'speedup_ratio': 0.0785}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------
# TileLang kernel factory
# ----------------------------------------------------------------------
def _build_deconv_sigmoid_bias_sigmoid_kernel(
    B: int,
    Cin: int,
    Hin: int,
    Win: int,
    Cout: int,
    KH: int,
    KW: int,
    *,
    block_size: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    Hout = Hin + KH - 1
    Wout = Win + KW - 1
    numel_out = B * Cout * Hout * Wout

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((B, Cin, Hin, Win), dtype),
        Wt: T.Tensor((Cin, Cout, KH, KW), dtype),        # transposed-conv weights
        Bconv: T.Tensor((Cout,), dtype),                 # conv-transpose bias
        Badd: T.Tensor((Cout,), dtype),                  # extra bias after first sigmoid
        Out: T.Tensor((B, Cout, Hout, Wout), dtype),     # created by TileLang
    ):
        grid = T.ceildiv(numel_out, block_size)

        one_f32  = T.Cast(accum_dtype, 1)
        zero_f32 = T.Cast(accum_dtype, 0)

        with T.Kernel(grid, threads=block_size) as bx:
            tx  = T.get_thread_binding(0)
            idx = bx * block_size + tx
            if idx < numel_out:
                # ----------------------------------------------------------
                # unravel linear index -> (n, oc, oh, ow)
                # ----------------------------------------------------------
                sp_stride   = Hout * Wout
                batch_stride = Cout * sp_stride

                n  = idx // batch_stride
                r1 = idx - n * batch_stride
                oc = r1 // sp_stride
                r2 = r1 - oc * sp_stride
                oh = r2 // Wout
                ow = r2 - oh * Wout

                # ----------------------------------------------------------
                # main accumulation : full-conv (deconv / conv-transpose)
                # ----------------------------------------------------------
                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = zero_f32

                for ic in T.serial(Cin):
                    for kh in T.serial(KH):
                        ih = oh - kh
                        if (ih >= 0) and (ih < Hin):
                            for kw in T.serial(KW):
                                iw = ow - kw
                                if (iw >= 0) and (iw < Win):
                                    x_val = T.Cast(
                                        accum_dtype, X[n, ic, ih, iw]
                                    )
                                    w_val = T.Cast(
                                        accum_dtype, Wt[ic, oc, kh, kw]
                                    )
                                    acc[0] = acc[0] + x_val * w_val

                # add conv-transpose bias
                acc[0] = acc[0] + T.Cast(accum_dtype, Bconv[oc])

                # ----------------------------------------------------------
                # first sigmoid
                # ----------------------------------------------------------
                sig1 = one_f32 / (one_f32 + T.exp(-acc[0]))

                # add second bias
                sig1 = sig1 + T.Cast(accum_dtype, Badd[oc])

                # second sigmoid
                sig2 = one_f32 / (one_f32 + T.exp(-sig1))

                # store
                Out[n, oc, oh, ow] = T.Cast(dtype, sig2)

    return kernel


# ----------------------------------------------------------------------
# PyTorch wrapper module
# ----------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    ConvTranspose2d ➜ Sigmoid ➜ BiasAdd ➜ Sigmoid   fused TileLang kernel
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias_shape: tuple,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)

        # --- conv-transpose parameters ---------------------------------
        self.weight = nn.Parameter(
            torch.empty(
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        fan_in = self.in_channels * self.kernel_size * self.kernel_size
        bound = 1 / math.sqrt(fan_in)
        self.bias_conv = nn.Parameter(
            torch.empty(self.out_channels)
        )
        nn.init.uniform_(self.bias_conv, -bound, bound)

        # --- second bias (after first sigmoid) -------------------------
        self.bias_add = nn.Parameter(torch.randn(bias_shape).view(-1))

        # Kernel cache  : keyed by (B, H, W, dtype)
        self._kern_cache = {}

    # ------------------------------------------------------------------
    # kernel retrieval / compile
    # ------------------------------------------------------------------
    def _get_kernel(self, B: int, H: int, W: int, dtype: torch.dtype):
        key = (B, H, W, dtype)
        if key not in self._kern_cache:
            tl_dtype = "float16" if dtype in (torch.float16, torch.float32) else "bfloat16"
            self._kern_cache[key] = _build_deconv_sigmoid_bias_sigmoid_kernel(
                B=B,
                Cin=self.in_channels,
                Hin=H,
                Win=W,
                Cout=self.out_channels,
                KH=self.kernel_size,
                KW=self.kernel_size,
                dtype=tl_dtype,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, Cin, H, W)
        returns : (B, Cout, H + K - 1, W + K - 1)
        """
        B, Cin, H, W = x.shape
        assert Cin == self.in_channels, "in_channels mismatch"

        x_f16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w_f16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        bconv_f16 = self.bias_conv.to(device="cuda", dtype=torch.float16).contiguous()
        badd_f16 = self.bias_add.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(B, H, W, x_f16.dtype)
        y_f16 = kernel(x_f16, w_f16, bconv_f16, badd_f16)

        return y_f16.to(x.dtype)