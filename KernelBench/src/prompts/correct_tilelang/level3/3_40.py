"""
Problem Name: 40_GRUHidden
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1480.0 runtime_stats={'mean': 1480.0, 'std': 102.0, 'min': 1140.0, 'max': 1870.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 40.0, 'std': 2.64, 'min': 35.1, 'max': 44.4, 'num_trials': 100}, 'speedup_ratio': 0.027}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def build_matmul_kernel(M, N, K, block_M=64, block_N=64, block_K=32,
                        dtype="float16", accum_dtype="float"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_acc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_acc)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_acc)
            T.copy(C_acc, C[by * block_M, bx * block_N])

    return kernel


class GRULayer(nn.Module):
    """
    A single GRU layer implemented with TileLang matmul kernels
    """
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        gate_size = 3 * hidden_size

        # Parameters follow PyTorch's GRU layout: (3*hidden_size, input_size/hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(gate_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(gate_size, hidden_size))

        if bias:
            self.bias_ih = nn.Parameter(torch.empty(gate_size))
            self.bias_hh = nn.Parameter(torch.empty(gate_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        # Initialize as in PyTorch: U(-1/sqrt(hidden_size), 1/sqrt(hidden_size))
        stdv = 1.0 / math.sqrt(hidden_size)
        for p in self.parameters():
            if p is not None:
                p.data.uniform_(-stdv, stdv)

        # Kernel cache: keyed by (batch, in_dim, out_dim, dtype)
        self._kernel_cache = {}

    def _get_kernel(self, M, K, N, dtype):
        key = (M, K, N, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = build_matmul_kernel(M, N, K, dtype=dtype)
        return self._kernel_cache[key]

    def forward(self, x, h_prev):
        """
        x: (batch, input_size)   - float16 cuda
        h_prev: (batch, hidden_size) - float16 cuda
        returns h_next (batch, hidden_size)
        """
        batch_size = x.size(0)
        hidden_size = h_prev.size(1)
        gate_size = 3 * hidden_size

        # Prepare kernels
        dtype = "float16"
        k_in = self._get_kernel(batch_size, x.size(1), gate_size, dtype)
        k_h = self._get_kernel(batch_size, hidden_size, gate_size, dtype)

        # Matmul input and hidden contributions
        gi = k_in(x, self.weight_ih.to(x.device, torch.float16).t().contiguous())
        gh = k_h(h_prev, self.weight_hh.to(x.device, torch.float16).t().contiguous())

        if self.bias_ih is not None:
            gi = gi + self.bias_ih.to(x.device, torch.float16)
            gh = gh + self.bias_hh.to(x.device, torch.float16)

        # Split gates: order r, z, n (PyTorch GRU convention)
        gi_r, gi_z, gi_n = gi.split(hidden_size, dim=1)
        gh_r, gh_z, gh_n = gh.split(hidden_size, dim=1)

        r = torch.sigmoid(gi_r + gh_r)
        z = torch.sigmoid(gi_z + gh_z)
        n = torch.tanh(gi_n + r * gh_n)
        h_next = (1 - z) * n + z * h_prev
        return h_next


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, bias=True, batch_first=False):
        super().__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first

        layers = []
        for layer_idx in range(num_layers):
            in_feat = input_size if layer_idx == 0 else hidden_size
            layers.append(GRULayer(in_feat, hidden_size, bias))
        self.layers = nn.ModuleList(layers)

        # h0 buffer, will be broadcasted to batch size at runtime
        self.register_buffer("h0", torch.randn(num_layers, 1, hidden_size))

    def forward(self, x):
        """
        x: (seq_len, batch, input) if not batch_first else (batch, seq, input)
        returns h_n: (num_layers, batch, hidden)
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # (seq, batch, feature)

        seq_len, batch_size, _ = x.size()
        device = x.device

        # Initial hidden state
        h_t = self.h0.repeat(1, batch_size, 1).to(device, dtype=torch.float16)

        # Ensure input is float16 on cuda for TileLang
        x = x.to(device, dtype=torch.float16)

        for t in range(seq_len):
            inp = x[t]
            for l, layer in enumerate(self.layers):
                h_prev = h_t[l]
                h_next = layer(inp, h_prev)
                h_t[l] = h_next
                inp = h_next  # input for next layer

        return h_t.to(torch.float32)