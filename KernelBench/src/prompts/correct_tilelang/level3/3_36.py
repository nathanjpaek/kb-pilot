"""
Problem Name: 36_LTSMHn
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=487.0 runtime_stats={'mean': 487.0, 'std': 33.1, 'min': 433.0, 'max': 546.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 35.0, 'std': 0.834, 'min': 34.0, 'max': 38.5, 'num_trials': 100}, 'speedup_ratio': 0.0719}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                             TileLang  GEMM(+Bias)                           #
# --------------------------------------------------------------------------- #
def _build_linear_kernel(
    batch: int,
    in_dim: int,
    out_dim: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Y = X @ Wᵀ + Bias
    Shapes:
        X : (batch, in_dim)
        W : (out_dim, in_dim)
        B : (out_dim,)
        Y : (batch, out_dim)
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((batch, in_dim), dtype),
        W: T.Tensor((out_dim, in_dim), dtype),
        B: T.Tensor((out_dim,), dtype),
        Y: T.Tensor((batch, out_dim), dtype),
    ):
        grid_x = T.ceildiv(out_dim, block_N)
        grid_y = T.ceildiv(batch, block_M)

        with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_N, block_K), dtype)
            Bias_s = T.alloc_shared((block_N,), dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            # copy bias slice once
            T.copy(B[bx * block_N : (bx + 1) * block_N], Bias_s)
            T.clear(C_frag)

            k_tiles = T.ceildiv(in_dim, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(
                    X[
                        by * block_M : (by + 1) * block_M,
                        ko * block_K : (ko + 1) * block_K,
                    ],
                    A_s,
                )
                T.copy(
                    W[
                        bx * block_N : (bx + 1) * block_N,
                        ko * block_K : (ko + 1) * block_K,
                    ],
                    B_s,
                )
                T.gemm(A_s, B_s, C_frag, transpose_B=True)

            # add bias and store
            for ii, jj in T.Parallel(block_M, block_N):
                gi = by * block_M + ii
                gj = bx * block_N + jj
                if (gi < batch) and (gj < out_dim):
                    val = C_frag[ii, jj] + Bias_s[jj].astype(accum_dtype)
                    Y[gi, gj] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
#                               Model — LSTM                                  #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    Pure-Python stacked LSTM (no torch.nn layers) accelerated by TileLang GEMM
    kernels.  Returns final hidden state  h_n  with shape (num_layers, batch, H).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,          # unused but kept for signature parity
        dropout: float = 0.0,      # dropout ignored (==0.0 in original)
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)

        # ---- LSTM parameters ----
        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        self.bias_ih = nn.ParameterList()
        self.bias_hh = nn.ParameterList()

        stdv = 1.0 / math.sqrt(hidden_size)

        for layer in range(num_layers):
            in_dim = self.input_size if layer == 0 else self.hidden_size

            w_ih = nn.Parameter(torch.empty(4 * hidden_size, in_dim))
            w_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
            b_ih = nn.Parameter(torch.empty(4 * hidden_size))
            b_hh = nn.Parameter(torch.empty(4 * hidden_size))

            # uniform initialization
            nn.init.uniform_(w_ih, -stdv, stdv)
            nn.init.uniform_(w_hh, -stdv, stdv)
            nn.init.uniform_(b_ih, -stdv, stdv)
            nn.init.uniform_(b_hh, -stdv, stdv)

            # forget-gate bias = 1
            b_ih.data[hidden_size : 2 * hidden_size] = 1.0
            b_hh.data[hidden_size : 2 * hidden_size] = 1.0

            self.weight_ih.append(w_ih)
            self.weight_hh.append(w_hh)
            self.bias_ih.append(b_ih)
            self.bias_hh.append(b_hh)

        # ---- initial states (lazy) ----
        self.register_buffer("h0_init", torch.empty(0))
        self.register_buffer("c0_init", torch.empty(0))

        # ---- kernel cache  {(B,in_dim,out_dim,dtype) : primfunc } ----
        self._kernels: Dict[Tuple[int, int, int, str], callable] = {}

    # ----------------------------------------------------------------- #
    #                         kernel retrieval                           #
    # ----------------------------------------------------------------- #
    def _get_kernel(self, batch: int, in_dim: int, out_dim: int, dtype: str = "float16"):
        key = (batch, in_dim, out_dim, dtype)
        if key not in self._kernels:
            self._kernels[key] = _build_linear_kernel(
                batch=batch,
                in_dim=in_dim,
                out_dim=out_dim,
                dtype=dtype,
                accum_dtype="float32",
            )
        return self._kernels[key]

    # ----------------------------------------------------------------- #
    #                             forward                               #
    # ----------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x :  (batch, seq_len, input_size)
        Returns
        -------
        h_n : (num_layers, batch, hidden_size)
        """
        device = torch.device("cuda")
        x_f16 = x.to(device=device, dtype=torch.float16).contiguous()
        batch, seq_len, _ = x_f16.shape

        # create / resize initial states if necessary
        if self.h0_init.numel() == 0 or self.h0_init.shape[1] != batch:
            self.h0_init = torch.randn(
                self.num_layers, batch, self.hidden_size, device=device, dtype=torch.float32
            )
            self.c0_init = torch.randn_like(self.h0_init)

        h_list = []
        c_list = []
        prev_layer_output = x_f16  # (B, S, D)

        for layer in range(self.num_layers):
            w_ih = self.weight_ih[layer].to(device=device, dtype=torch.float16).contiguous()
            w_hh = self.weight_hh[layer].to(device=device, dtype=torch.float16).contiguous()
            b_ih = self.bias_ih[layer].to(device=device, dtype=torch.float16).contiguous()
            b_hh = self.bias_hh[layer].to(device=device, dtype=torch.float16).contiguous()

            h_t = self.h0_init[layer].to(dtype=torch.float32)
            c_t = self.c0_init[layer].to(dtype=torch.float32)

            out_seq = []

            in_dim = prev_layer_output.shape[2]

            k_input = self._get_kernel(batch, in_dim, 4 * self.hidden_size, "float16")
            k_hidden = self._get_kernel(batch, self.hidden_size, 4 * self.hidden_size, "float16")

            for t in range(seq_len):
                x_t = prev_layer_output[:, t, :]  # (B, in_dim)

                gates_x = k_input(x_t, w_ih, b_ih)          # (B, 4H) float16
                gates_h = k_hidden(h_t.to(torch.float16), w_hh, b_hh)  # (B, 4H)

                gates = gates_x.to(torch.float32) + gates_h.to(torch.float32)

                i, f, g, o = torch.chunk(gates, 4, dim=1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)

                c_t = f * c_t + i * g
                h_t = o * torch.tanh(c_t)

                out_seq.append(h_t.to(torch.float16).unsqueeze(1))

            prev_layer_output = torch.cat(out_seq, dim=1)  # (B, S, H)
            h_list.append(h_t.unsqueeze(0))  # (1,B,H)
            c_list.append(c_t.unsqueeze(0))

        h_n = torch.cat(h_list, dim=0)  # (L,B,H)
        return h_n.to(x.dtype)