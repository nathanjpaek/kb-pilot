"""
Problem Name: 41_GRUBirectional
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=733.0 runtime_stats={'mean': 733.0, 'std': 5.0, 'min': 725.0, 'max': 759.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 69.3, 'std': 0.379, 'min': 68.1, 'max': 70.3, 'num_trials': 100}, 'speedup_ratio': 0.0945}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_linear_kernel(
    M: int,
    K: int,
    N: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def linear_kernel(
        X: T.Tensor((M, K), dtype),
        W: T.Tensor((N, K), dtype),
        B: T.Tensor((N,), dtype),
        Y: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], W_shared)
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                if (global_m < M) and (global_n < N):
                    val = C_local[i, j] + B[global_n]
                    Y[global_m, global_n] = T.Cast(dtype, val)

    return linear_kernel


class ModelNew(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 3,
        bias: bool = True,
        batch_first: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias_flag = bias
        self.batch_first = batch_first
        self.num_directions = 2

        self.weight_ih = nn.ParameterList()
        self.weight_hh = nn.ParameterList()
        if self.bias_flag:
            self.bias_ih = nn.ParameterList()
            self.bias_hh = nn.ParameterList()

        stdv = 1.0 / math.sqrt(hidden_size)
        for layer in range(num_layers):
            layer_input_size = (
                input_size if layer == 0 else hidden_size * self.num_directions
            )
            for _ in range(self.num_directions):
                w_ih = nn.Parameter(torch.empty(3 * hidden_size, layer_input_size))
                w_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
                torch.nn.init.uniform_(w_ih, -stdv, stdv)
                torch.nn.init.uniform_(w_hh, -stdv, stdv)
                self.weight_ih.append(w_ih)
                self.weight_hh.append(w_hh)

                if self.bias_flag:
                    b_ih = nn.Parameter(torch.empty(3 * hidden_size))
                    b_hh = nn.Parameter(torch.empty(3 * hidden_size))
                    torch.nn.init.uniform_(b_ih, -stdv, stdv)
                    torch.nn.init.uniform_(b_hh, -stdv, stdv)
                    self.bias_ih.append(b_ih)
                    self.bias_hh.append(b_hh)

        global batch_size
        self.register_buffer(
            "h0", torch.randn(num_layers * self.num_directions, batch_size, hidden_size)
        )

        self._kernel_cache = {}

    def _get_linear_kernel(self, M: int, K: int, N: int, dtype: str = "float16"):
        key = (M, K, N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_linear_kernel(M, K, N, dtype=dtype)
        return self._kernel_cache[key]

    def _linear(self, X: torch.Tensor, W: torch.Tensor, B: torch.Tensor):
        M, K = X.shape
        N = W.shape[0]
        kernel = self._get_linear_kernel(M, K, N, "float16")
        X_fp16 = X.to(dtype=torch.float16, device="cuda").contiguous()
        W_fp16 = W.to(dtype=torch.float16, device="cuda").contiguous()
        B_fp16 = B.to(dtype=torch.float16, device="cuda").contiguous()
        Y_fp16 = kernel(X_fp16, W_fp16, B_fp16)
        return Y_fp16.to(dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        if self.batch_first:
            x = x.permute(1, 0, 2)

        x = x.to(dtype=torch.float32, device="cuda").contiguous()
        seq_len, batch_size, _ = x.shape

        h_prev_layers = self.h0[:, : batch_size].to(
            dtype=torch.float32, device="cuda"
        ).contiguous()

        layer_input = x
        for layer in range(self.num_layers):
            layer_outputs = []
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                w_ih = self.weight_ih[idx]
                w_hh = self.weight_hh[idx]
                if self.bias_flag:
                    b_ih = self.bias_ih[idx]
                    b_hh = self.bias_hh[idx]
                else:
                    b_ih = torch.zeros(
                        3 * self.hidden_size, device=w_ih.device, dtype=w_ih.dtype
                    )
                    b_hh = torch.zeros_like(b_ih)

                dir_input = (
                    layer_input
                    if direction == 0
                    else torch.flip(layer_input, [0]).contiguous()
                )

                X2d = dir_input.reshape(seq_len * batch_size, -1)
                x_lin = self._linear(X2d, w_ih, b_ih).reshape(
                    seq_len, batch_size, 3 * self.hidden_size
                )

                h_prev = h_prev_layers[idx]

                outputs_dir = []
                for t in range(seq_len):
                    gates_x = x_lin[t]
                    gates_h = self._linear(h_prev, w_hh, b_hh)
                    gates = gates_x + gates_h

                    r = torch.sigmoid(gates[:, : self.hidden_size])
                    z = torch.sigmoid(
                        gates[:, self.hidden_size : 2 * self.hidden_size]
                    )

                    n_pre = (
                        gates_x[:, 2 * self.hidden_size :]
                        + r * gates_h[:, 2 * self.hidden_size :]
                    )
                    n = torch.tanh(n_pre)

                    h_new = (1 - z) * n + z * h_prev
                    outputs_dir.append(h_new)
                    h_prev = h_new

                if direction == 1:
                    outputs_dir.reverse()

                layer_outputs.append(torch.stack(outputs_dir, dim=0))

            layer_input = torch.cat(layer_outputs, dim=2)

        output = layer_input
        if self.batch_first:
            output = output.permute(1, 0, 2)

        return output