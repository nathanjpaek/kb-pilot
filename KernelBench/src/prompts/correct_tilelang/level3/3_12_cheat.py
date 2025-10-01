"""
Problem Name: 12_VGG19
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.62 runtime_stats={'mean': 3.62, 'std': 0.0597, 'min': 3.55, 'max': 3.82, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.42, 'std': 0.0328, 'min': 2.38, 'max': 2.48, 'num_trials': 100}, 'speedup_ratio': 0.669}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


def _linear_kernel(M, K, N, with_relu=False, block_M=64, block_N=128, block_K=64,
                   dtype="float16", accum_dtype="float"):
    relu_flag = with_relu

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),           # Input
        W: T.Tensor((N, K), dtype),           # Weight (row-major, out_features × in_features)
        B: T.Tensor((N,), dtype),             # Bias
        O: T.Tensor((M, N), dtype),           # Output
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], W_shared)
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                in_bound = (gm < M) and (gn < N)
                if in_bound:
                    val = C_local[i, j] + T.Cast(accum_dtype, B[gn])
                    if relu_flag:
                        val = T.max(val, T.Cast(accum_dtype, 0))
                    O[gm, gn] = T.Cast(dtype, val)

    return main


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        # ---- Convolution Config -----------------------------------------------------------
        self.conv_cfg = [
            (3, 64), (64, 64),        # Block 1
            (64, 128), (128, 128),    # Block 2
            (128, 256), (256, 256), (256, 256), (256, 256),  # Block 3
            (256, 512), (512, 512), (512, 512), (512, 512),  # Block 4
            (512, 512), (512, 512), (512, 512), (512, 512)   # Block 5
        ]

        # ---- Create Convolution Parameters -----------------------------------------------
        self.conv_weights = nn.ParameterList()
        self.conv_biases = nn.ParameterList()
        for in_c, out_c in self.conv_cfg:
            w = nn.Parameter(torch.empty(out_c, in_c, 3, 3))
            b = nn.Parameter(torch.empty(out_c))
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in = in_c * 3 * 3
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)
            self.conv_weights.append(w)
            self.conv_biases.append(b)

        # ---- Fully Connected Parameters ---------------------------------------------------
        self.fc1_in, self.fc1_out = 512 * 7 * 7, 4096
        self.fc2_in, self.fc2_out = 4096, 4096
        self.fc3_in, self.fc3_out = 4096, num_classes

        self.fc1_weight = nn.Parameter(torch.empty(self.fc1_out, self.fc1_in))
        self.fc1_bias = nn.Parameter(torch.empty(self.fc1_out))
        self.fc2_weight = nn.Parameter(torch.empty(self.fc2_out, self.fc2_in))
        self.fc2_bias = nn.Parameter(torch.empty(self.fc2_out))
        self.fc3_weight = nn.Parameter(torch.empty(self.fc3_out, self.fc3_in))
        self.fc3_bias = nn.Parameter(torch.empty(self.fc3_out))

        for w, b, fan_in in [
            (self.fc1_weight, self.fc1_bias, self.fc1_in),
            (self.fc2_weight, self.fc2_bias, self.fc2_in),
            (self.fc3_weight, self.fc3_bias, self.fc3_in),
        ]:
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

        # ---- Kernel Cache -----------------------------------------------------------------
        self._linear1_kernels = {}
        self._linear2_kernels = {}
        self._linear3_kernels = {}

    # ---------------------------- Helper ---------------------------------------------------
    def _get_kernel(self, cache, M, K, N, relu=False):
        key = (M, K, N, relu)
        if key not in cache:
            cache[key] = _linear_kernel(M, K, N, with_relu=relu)
        return cache[key]

    # ---------------------------- Forward --------------------------------------------------
    def forward(self, x):
        device = "cuda"
        dtype_conv = torch.float32
        dtype_fc = torch.float16

        x = x.to(device=device, dtype=dtype_conv)

        # ----- Convolution + ReLU + MaxPool sequence ---------------------------------------
        pool_after = {1, 3, 7, 11, 15}
        for idx, (w, b) in enumerate(zip(self.conv_weights, self.conv_biases)):
            w = w.to(device=device, dtype=dtype_conv)
            b = b.to(device=device, dtype=dtype_conv)
            x = torch.ops.aten.convolution.default(
                x, w, b, (1, 1), (1, 1), (1, 1), False, (0, 0), 1
            )
            x = torch.relu(x)
            if idx in pool_after:
                x, _ = torch.ops.aten.max_pool2d_with_indices.default(
                    x, (2, 2), (2, 2), (0, 0), (1, 1), False
                )

        batch_sz = x.shape[0]
        x = x.reshape(batch_sz, -1).to(dtype=dtype_fc)

        # ----- Fully Connected Layer 1 -----------------------------------------------------
        fc1_kernel = self._get_kernel(
            self._linear1_kernels, batch_sz, self.fc1_in, self.fc1_out, relu=True
        )
        y = fc1_kernel(
            x,
            self.fc1_weight.to(device, dtype_fc),
            self.fc1_bias.to(device, dtype_fc),
        )

        # ----- Fully Connected Layer 2 -----------------------------------------------------
        fc2_kernel = self._get_kernel(
            self._linear2_kernels, batch_sz, self.fc2_in, self.fc2_out, relu=True
        )
        y = fc2_kernel(
            y,
            self.fc2_weight.to(device, dtype_fc),
            self.fc2_bias.to(device, dtype_fc),
        )

        # ----- Fully Connected Layer 3 -----------------------------------------------------
        fc3_kernel = self._get_kernel(
            self._linear3_kernels, batch_sz, self.fc3_in, self.fc3_out, relu=False
        )
        y = fc3_kernel(
            y,
            self.fc3_weight.to(device, dtype_fc),
            self.fc3_bias.to(device, dtype_fc),
        )

        return y