"""
Problem Name: 42_GRUBidirectionalHidden
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=685.0 runtime_stats={'mean': 685.0, 'std': 0.15, 'min': 685.0, 'max': 686.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 71.4, 'std': 2.71, 'min': 70.3, 'max': 81.2, 'num_trials': 100}, 'speedup_ratio': 0.104}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
#                               TileLang kernels                              #
# --------------------------------------------------------------------------- #
def _build_linear_bias_kernel(
    B: int,
    K: int,
    O: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    num_stages: int = 2,
    dtype_in: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Y[B,O] = X[B,K] @ W[O,K]ᵀ + Bias[O]
    """

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((B, K), dtype_in),
        W: T.Tensor((O, K), dtype_in),
        Bias: T.Tensor((O,), dtype_in),
        Y: T.Tensor((B, O), dtype_in),  # auto-allocated
    ):
        with T.Kernel(
            T.ceildiv(O, block_N),
            T.ceildiv(B, block_M),
            threads=128,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype_in)
            B_s = T.alloc_shared((block_N, block_K), dtype_in)
            Bias_s = T.alloc_shared((block_N,), dtype_in)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            # copy bias once
            T.copy(Bias[bx * block_N:(bx + 1) * block_N], Bias_s)
            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(
                    X[by * block_M:(by + 1) * block_M,
                      ko * block_K:(ko + 1) * block_K],
                    A_s,
                )
                T.copy(
                    W[bx * block_N:(bx + 1) * block_N,
                      ko * block_K:(ko + 1) * block_K],
                    B_s,
                )
                T.gemm(A_s, B_s, C_loc, transpose_B=True)

            # add bias and store
            for i, j in T.Parallel(block_M, block_N):
                gi = by * block_M + i
                gj = bx * block_N + j
                if (gi < B) and (gj < O):
                    val = C_loc[i, j] + Bias_s[j].astype(accum_dtype)
                    Y[gi, gj] = T.Cast(dtype_in, val)

    return kernel


def _build_gru_elem_kernel(
    B: int,
    H: int,
    block_M: int = 128,
    dtype_in: str = "float16",
    accum_dtype: str = "float32",
):
    """
    Element-wise part of GRU:
        r = σ(i_r + h_r)
        z = σ(i_z + h_z)
        n = tanh(i_n + r * h_n)
        h  = (1 – z) * n + z * h_prev
    """

    threeH = 3 * H

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        I: T.Tensor((B, threeH), dtype_in),     # from input GEMM (i_r,i_z,i_n)
        Hpart: T.Tensor((B, threeH), dtype_in),  # from hidden GEMM (h_r,h_z,h_n)
        Hprev: T.Tensor((B, H), dtype_in),
        Hnew: T.Tensor((B, H), dtype_in),        # auto-allocated
    ):
        one = T.Cast(accum_dtype, 1.0)

        with T.Kernel(T.ceildiv(B, block_M), threads=block_M) as blk:
            tx = T.get_thread_binding(0)
            row = blk * block_M + tx
            if row < B:
                for j in range(H):
                    ir = I[row, j].astype(accum_dtype)
                    iz = I[row, H + j].astype(accum_dtype)
                    inn = I[row, 2 * H + j].astype(accum_dtype)

                    hr = Hpart[row, j].astype(accum_dtype)
                    hz = Hpart[row, H + j].astype(accum_dtype)
                    hnn = Hpart[row, 2 * H + j].astype(accum_dtype)

                    hp = Hprev[row, j].astype(accum_dtype)

                    r = one / (one + T.exp(-(ir + hr)))
                    z = one / (one + T.exp(-(iz + hz)))
                    n = T.tanh(inn + r * hnn)
                    h_out = (one - z) * n + z * hp

                    Hnew[row, j] = T.Cast(dtype_in, h_out)

    return kernel


# --------------------------------------------------------------------------- #
#                              PyTorch wrapper                                #
# --------------------------------------------------------------------------- #
class _GRUParamBank(nn.Module):
    """
    Container that holds all GRU weights under the exact same names as
    `nn.GRU`, enabling transparent `state_dict` copying.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        bias: bool,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.bidirectional = bidirectional
        stdv = 1.0 / math.sqrt(hidden_size)

        def _new_param(*shape):
            p = nn.Parameter(torch.empty(*shape))
            nn.init.uniform_(p, -stdv, stdv)
            return p

        for layer in range(num_layers):
            for direction in range(2 if bidirectional else 1):
                suffix = "_reverse" if direction == 1 else ""
                layer_input_size = (
                    input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
                )

                # W_ih , W_hh
                setattr(
                    self,
                    f"weight_ih_l{layer}{suffix}",
                    _new_param(3 * hidden_size, layer_input_size),
                )
                setattr(
                    self,
                    f"weight_hh_l{layer}{suffix}",
                    _new_param(3 * hidden_size, hidden_size),
                )

                if bias:
                    setattr(
                        self,
                        f"bias_ih_l{layer}{suffix}",
                        _new_param(3 * hidden_size),
                    )
                    setattr(
                        self,
                        f"bias_hh_l{layer}{suffix}",
                        _new_param(3 * hidden_size),
                    )


class ModelNew(nn.Module):
    """
    Bidirectional multi-layer GRU implemented with TileLang kernels.
    Returns only `h_n`, identical to the reference code.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 3,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bias_flag = bool(bias)
        self.batch_first = bool(batch_first)
        self.num_directions = 2

        # ---- learnable parameters ----
        self.gru = _GRUParamBank(
            self.input_size,
            self.hidden_size,
            self.num_layers,
            bias,
            bidirectional=True,
        )

        # ---- kernel caches ----
        self._gemm_cache: Dict[Tuple[int, int, int, torch.dtype], callable] = {}
        self._elem_cache: Dict[Tuple[int, int, torch.dtype], callable] = {}

    # ------------------------------------------------------------------ #
    #                          Kernel getters                            #
    # ------------------------------------------------------------------ #
    def _get_gemm(
        self,
        batch: int,
        in_dim: int,
        out_dim: int,
        dtype: torch.dtype,
    ):
        key = (batch, in_dim, out_dim, dtype)
        if key not in self._gemm_cache:
            self._gemm_cache[key] = _build_linear_bias_kernel(
                B=batch, K=in_dim, O=out_dim, dtype_in=str(dtype).split(".")[-1]
            )
        return self._gemm_cache[key]

    def _get_elem(
        self,
        batch: int,
        hidden: int,
        dtype: torch.dtype,
    ):
        key = (batch, hidden, dtype)
        if key not in self._elem_cache:
            self._elem_cache[key] = _build_gru_elem_kernel(
                B=batch, H=hidden, dtype_in=str(dtype).split(".")[-1]
            )
        return self._elem_cache[key]

    # ------------------------------------------------------------------ #
    #                             forward                                #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x :  (seq_len, batch, input)   when batch_first=False
              (batch, seq_len, input)  when batch_first=True
        Returns
        -------
        h_n : (num_layers * num_directions , batch , hidden)
        """
        orig_dtype = x.dtype
        device = torch.device("cuda")
        fp16 = torch.float16

        # arrange shape => (seq_len , batch , feat)
        if self.batch_first:
            x = x.permute(1, 0, 2).contiguous()

        seq_len, batch, _ = x.shape
        x = x.to(device=device, dtype=fp16).contiguous()

        # convenience lambda to fetch parameter tensors in fp16 / device.
        def _param(name):
            return getattr(self.gru, name).to(device=device, dtype=fp16)

        h_n_out = []

        # ---------- iterate over layers ----------
        layer_input = x
        for layer in range(self.num_layers):
            layer_outputs_fwd = []
            layer_outputs_bwd = []

            # ---- initial hidden zeros ----
            h_fwd = torch.zeros(batch, self.hidden_size, device=device, dtype=fp16)
            h_bwd = torch.zeros_like(h_fwd)

            # ---- select correct input size for this layer ----
            in_dim = (
                self.input_size
                if layer == 0
                else self.hidden_size * self.num_directions
            )

            # ---- weight access helpers ----
            def _w(s):  # suffix
                base = f"l{layer}{s}"
                return (
                    _param(f"weight_ih_{base}"),
                    _param(f"weight_hh_{base}"),
                    _param(f"bias_ih_{base}") if self.bias_flag else torch.zeros(
                        3 * self.hidden_size, device=device, dtype=fp16
                    ),
                    _param(f"bias_hh_{base}") if self.bias_flag else torch.zeros(
                        3 * self.hidden_size, device=device, dtype=fp16
                    ),
                )

            # ---------------- forward direction ----------------
            w_ih_f, w_hh_f, b_ih_f, b_hh_f = _w("")
            gemm_in_f = self._get_gemm(batch, in_dim, 3 * self.hidden_size, fp16)
            gemm_h_f = self._get_gemm(batch, self.hidden_size, 3 * self.hidden_size, fp16)
            elem_f = self._get_elem(batch, self.hidden_size, fp16)

            for t in range(seq_len):
                Xi = gemm_in_f(layer_input[t], w_ih_f, b_ih_f)
                Hpart = gemm_h_f(h_fwd, w_hh_f, b_hh_f)
                h_fwd = elem_f(Xi, Hpart, h_fwd)
                layer_outputs_fwd.append(h_fwd)

            h_n_out.append(h_fwd)  # last time step forward

            # ---------------- backward direction ----------------
            w_ih_b, w_hh_b, b_ih_b, b_hh_b = _w("_reverse")
            gemm_in_b = self._get_gemm(batch, in_dim, 3 * self.hidden_size, fp16)
            gemm_h_b = self._get_gemm(batch, self.hidden_size, 3 * self.hidden_size, fp16)
            elem_b = self._get_elem(batch, self.hidden_size, fp16)

            for t in reversed(range(seq_len)):
                Xi_b = gemm_in_b(layer_input[t], w_ih_b, b_ih_b)
                Hpart_b = gemm_h_b(h_bwd, w_hh_b, b_hh_b)
                h_bwd = elem_b(Xi_b, Hpart_b, h_bwd)
                layer_outputs_bwd.append(h_bwd)

            h_n_out.append(h_bwd)  # last time step backward

            # ---------- prepare input for next layer ----------
            layer_outputs_bwd.reverse()  # align timestep order
            concat_out = torch.cat(
                [torch.stack(layer_outputs_fwd), torch.stack(layer_outputs_bwd)], dim=2
            )  # (seq, batch, 2*hidden)
            layer_input = concat_out.contiguous()

        # assemble h_n   (num_layers*2 , batch , hidden)
        h_n = torch.stack(h_n_out, dim=0).to(orig_dtype)
        return h_n