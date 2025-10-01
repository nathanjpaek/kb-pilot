"""
Problem Name: 58_conv_transposed_3D__asymmetric_input__asymmetric_kernel
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.79 runtime_stats={'mean': 3.79, 'std': 0.0523, 'min': 3.78, 'max': 4.24, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.6, 'std': 0.0201, 'min': 2.56, 'max': 2.73, 'num_trials': 100}, 'speedup_ratio': 0.686}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# ----------------------------------------------------------------------------- #
def _build_deconv3d_kernel(
    N: int,
    C_in: int,
    D_in: int,
    H_in: int,
    W_in: int,
    C_out: int,
    Kd: int,
    Kh: int,
    Kw: int,
    block_M: int = 64,
    block_N: int = 64,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    D_out = D_in + Kd - 1       # stride = 1 , pad = 0 , output_pad = 0
    H_out = H_in + Kh - 1
    W_out = W_in + Kw - 1

    M_total = N * D_out * H_out * W_out
    K_total = C_in * Kd * Kh * Kw

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def deconv3d_kernel(
        X: T.Tensor((N, D_in, H_in, W_in, C_in), dtype),          # NDHWC
        Wt: T.Tensor((Kd, Kh, Kw, C_in, C_out), dtype),           # DHWCI-CO
        Y: T.Tensor((N, D_out, H_out, W_out, C_out), dtype),      # NDHWC
    ):
        with T.Kernel(
            T.ceildiv(C_out, block_N),
            T.ceildiv(M_total, block_M),
            threads=128,
        ) as (bx, by):
            A_sh = T.alloc_shared((block_M, block_K), dtype)
            B_sh = T.alloc_shared((block_K, block_N), dtype)
            C_fr = T.alloc_fragment((block_M, block_N), accum_dtype)
            Y_sh = T.alloc_shared((block_M, block_N), dtype)

            W_flat = T.Tensor((K_total, C_out), dtype, Wt.data)
            Y_flat = T.Tensor((M_total, C_out), dtype, Y.data)

            T.clear(C_fr)

            for ko in T.Pipelined(T.ceildiv(K_total, block_K), num_stages=num_stages):
                # ---------------- load X tile (im2col) ----------------
                for mi, ki in T.Parallel(block_M, block_K):
                    m = by * block_M + mi
                    k = ko * block_K + ki

                    if (m < M_total) and (k < K_total):
                        n_idx   = m // (D_out * H_out * W_out)
                        rem1    = m %  (D_out * H_out * W_out)
                        d_out   = rem1 // (H_out * W_out)
                        rem2    = rem1 %  (H_out * W_out)
                        h_out   = rem2 // W_out
                        w_out   = rem2 %  W_out

                        kd_idx  = k // (Kh * Kw * C_in)
                        remk1   = k %  (Kh * Kw * C_in)
                        kh_idx  = remk1 // (Kw * C_in)
                        remk2   = remk1 %  (Kw * C_in)
                        kw_idx  = remk2 // C_in
                        c_idx   = remk2 %  C_in

                        d_in = d_out - kd_idx
                        h_in = h_out - kh_idx
                        w_in = w_out - kw_idx

                        valid = (
                            (d_in >= 0) and (d_in < D_in) and
                            (h_in >= 0) and (h_in < H_in) and
                            (w_in >= 0) and (w_in < W_in)
                        )

                        A_sh[mi, ki] = T.if_then_else(
                            valid,
                            X[n_idx, d_in, h_in, w_in, c_idx],
                            T.Cast(dtype, 0),
                        )
                    else:
                        A_sh[mi, ki] = T.Cast(dtype, 0)

                # ---------------- load W tile ----------------
                T.copy(
                    W_flat[ko * block_K, bx * block_N],
                    B_sh,
                )

                # ---------------- GEMM ----------------
                T.gemm(A_sh, B_sh, C_fr)

            # ---------------- store ----------------
            T.copy(C_fr, Y_sh)
            T.copy(Y_sh, Y_flat[by * block_M, bx * block_N])

    return deconv3d_kernel


# ----------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    TileLang-optimised ConvTranspose3d for stride = 1, padding = 0,
    output_padding = 0, groups = 1, bias = False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple,
        stride: tuple = (1, 1, 1),
        padding: tuple = (0, 0, 0),
        output_padding: tuple = (0, 0, 0),
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        assert stride == (1, 1, 1) and padding == (0, 0, 0) and output_padding == (0, 0, 0), \
            "Only stride=1, padding=0, output_padding=0 supported."
        assert groups == 1, "groups != 1 not supported."
        assert not bias,   "bias not supported."

        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.Kd, self.Kh, self.Kw = kernel_size

        # ---------------- parameters ----------------
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, self.Kd, self.Kh, self.Kw)
        )
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # cache
        self._kern_cache = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, N: int, D_in: int, H_in: int, W_in: int, dtype: str):
        key = (N, D_in, H_in, W_in, dtype)
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_deconv3d_kernel(
                N,
                self.in_channels,
                D_in,
                H_in,
                W_in,
                self.out_channels,
                self.Kd,
                self.Kh,
                self.Kw,
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()

        N, Cin, D_in, H_in, W_in = x_fp16.shape

        # reorder tensors
        x_ncdhwc = x_fp16.permute(0, 2, 3, 4, 1).contiguous()
        w_perm = (
            self.weight.to(device="cuda", dtype=torch.float16)
            .permute(2, 3, 4, 0, 1)  # Kd,Kh,Kw,Cin,Cout
            .contiguous()
        )

        kernel = self._get_kernel(N, D_in, H_in, W_in, str(x_fp16.dtype))
        y_ncdhwc = kernel(x_ncdhwc, w_perm)

        y = y_ncdhwc.permute(0, 4, 1, 2, 3).contiguous()
        return y.to(orig_dtype)