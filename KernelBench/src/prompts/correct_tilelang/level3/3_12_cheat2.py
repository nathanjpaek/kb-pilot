"""
Problem Name: 12_VGG19
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.89 runtime_stats={'mean': 2.89, 'std': 0.00824, 'min': 2.88, 'max': 2.91, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.53, 'std': 0.00986, 'min': 2.51, 'max': 2.56, 'num_trials': 100}, 'speedup_ratio': 0.875}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _build_linear_kernel(M, N, K, fuse_relu=False, block_M=64, block_N=128, block_K=32):
    dtype = "float16"
    accum_dtype = "float"

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),  # input
        W: T.Tensor((N, K), dtype),  # weight – note (N, K) so we will use transpose_B=True
        B: T.Tensor((N,), dtype),    # bias
        Out: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            W_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                # Copy tiles
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(W[bx * block_N, ko * block_K], W_shared)
                # GEMM
                T.gemm(A_shared, W_shared, C_local, transpose_B=True)

            # Write back with bias (and ReLU if fused)
            for i, j in T.Parallel(block_M, block_N):
                global_m = by * block_M + i
                global_n = bx * block_N + j
                if (global_m < M) and (global_n < N):
                    val = C_local[i, j] + B[global_n]
                    if fuse_relu:
                        val = T.max(val, T.Cast(accum_dtype, 0))
                    Out[global_m, global_n] = T.Cast(dtype, val)

    return kernel


class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        super(ModelNew, self).__init__()
        device = "cuda"
        weight_dtype = torch.float16

        # ---------------------------------------------------------------------
        # Convolution layers (weights + bias only, we will use aten.conv2d call)
        # ---------------------------------------------------------------------
        conv_cfgs = [
            (3, 64), (64, 64),          # Block 1
            (64, 128), (128, 128),      # Block 2
            (128, 256), (256, 256), (256, 256), (256, 256),  # Block 3
            (256, 512), (512, 512), (512, 512), (512, 512),  # Block 4
            (512, 512), (512, 512), (512, 512), (512, 512),  # Block 5
        ]
        self.conv_weights = nn.ParameterList()
        self.conv_biases = nn.ParameterList()
        for in_c, out_c in conv_cfgs:
            w = nn.Parameter(torch.empty(out_c, in_c, 3, 3, device=device, dtype=weight_dtype))
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            fan_in = in_c * 3 * 3
            bound = 1 / math.sqrt(fan_in)
            b = nn.Parameter(torch.empty(out_c, device=device, dtype=weight_dtype))
            nn.init.uniform_(b, -bound, bound)
            self.conv_weights.append(w)
            self.conv_biases.append(b)

        # ---------------------------------------------------------------------
        # Fully-connected layers (weights + bias)
        # ---------------------------------------------------------------------
        self.fc1_weight = nn.Parameter(torch.empty(4096, 512 * 7 * 7, device=device, dtype=weight_dtype))
        self.fc1_bias = nn.Parameter(torch.empty(4096, device=device, dtype=weight_dtype))
        self.fc2_weight = nn.Parameter(torch.empty(4096, 4096, device=device, dtype=weight_dtype))
        self.fc2_bias = nn.Parameter(torch.empty(4096, device=device, dtype=weight_dtype))
        self.fc3_weight = nn.Parameter(torch.empty(num_classes, 4096, device=device, dtype=weight_dtype))
        self.fc3_bias = nn.Parameter(torch.empty(num_classes, device=device, dtype=weight_dtype))

        nn.init.kaiming_uniform_(self.fc1_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc2_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fc3_weight, a=math.sqrt(5))

        for w, b in [
            (self.fc1_weight, self.fc1_bias),
            (self.fc2_weight, self.fc2_bias),
            (self.fc3_weight, self.fc3_bias),
        ]:
            fan_in = w.shape[1]
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b, -bound, bound)

        # ---------------------------------------------------------------------
        # Kernel caches for different batch sizes
        # ---------------------------------------------------------------------
        self._fc1_kernels = {}
        self._fc2_kernels = {}
        self._fc3_kernels = {}

    # -------------------------------------------------------------------------
    # Helper to fetch / compile kernels
    # -------------------------------------------------------------------------
    def _get_linear_kernel(self, cache, M, N, K, fuse_relu):
        key = M
        if key not in cache:
            cache[key] = _build_linear_kernel(M, N, K, fuse_relu=fuse_relu)
        return cache[key]

    # -------------------------------------------------------------------------
    # Forward pass
    # -------------------------------------------------------------------------
    def forward(self, x):
        x = x.to(device="cuda", dtype=torch.float16)
        cw_iter = iter(zip(self.conv_weights, self.conv_biases))
        pool_positions = {1, 3, 7, 11, 15}
        for idx in range(16):
            w, b = next(cw_iter)
            x = torch.ops.aten.conv2d(x, w, b, (1, 1), (1, 1), (1, 1), 1)
            x = torch.relu(x)
            if idx in pool_positions:
                x = torch.ops.aten.max_pool2d(x, (2, 2), (2, 2), (0, 0), (1, 1), False)

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)  # flatten
        # FC1 + ReLU
        fc1_kernel = self._get_linear_kernel(self._fc1_kernels, batch_size, 4096, 512 * 7 * 7, fuse_relu=True)
        x = fc1_kernel(x, self.fc1_weight, self.fc1_bias)
        # FC2 + ReLU
        fc2_kernel = self._get_linear_kernel(self._fc2_kernels, batch_size, 4096, 4096, fuse_relu=True)
        x = fc2_kernel(x, self.fc2_weight, self.fc2_bias)
        # FC3
        fc3_kernel = self._get_linear_kernel(self._fc3_kernels, batch_size, self.fc3_weight.shape[0], 4096, fuse_relu=False)
        x = fc3_kernel(x, self.fc3_weight, self.fc3_bias)
        return x.to(torch.float32)