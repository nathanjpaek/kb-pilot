"""
Problem Name: 5_AlexNet
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.613 runtime_stats={'mean': 0.613, 'std': 0.0034, 'min': 0.609, 'max': 0.635, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.468, 'std': 0.0142, 'min': 0.451, 'max': 0.535, 'num_trials': 100}, 'speedup_ratio': 0.763}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


class ModelNew(nn.Module):
    """AlexNet-like model with TileLang-accelerated Linear layers."""

    # ------------------------------------------------------------------
    # Static helper: TileLang GEMM kernel factory
    # ------------------------------------------------------------------
    @staticmethod
    def _build_linear_kernel(M, N, K, fuse_relu=False,
                              block_M=64, block_N=128, block_K=32,
                              dtype="float16", accum_dtype="float"):
        """Return Y = A @ W.T + B  (opt ReLU)   Shapes:  A[M,K]  W[N,K]  B[N]"""
        @tilelang.jit(out_idx=-1)
        @T.prim_func
        def kernel(
            A: T.Tensor((M, K), dtype),   # Input batch
            W: T.Tensor((N, K), dtype),   # Weight (N,K)
            B: T.Tensor((N,),    dtype),   # Bias
            Out: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                A_s = T.alloc_shared((block_M, block_K), dtype)
                W_s = T.alloc_shared((block_N, block_K), dtype)
                C_f = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_f)

                for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                    T.copy(A[by * block_M, ko * block_K], A_s)
                    T.copy(W[bx * block_N, ko * block_K], W_s)
                    T.gemm(A_s, W_s, C_f, transpose_B=True)

                for i, j in T.Parallel(block_M, block_N):
                    gm = by * block_M + i
                    gn = bx * block_N + j
                    if (gm < M) and (gn < N):
                        val = C_f[i, j] + B[gn]
                        if fuse_relu:
                            val = T.max(val, T.Cast(accum_dtype, 0))
                        Out[gm, gn] = T.Cast(dtype, val)
        return kernel

    # ------------------------------------------------------------------
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        # ---------------- Convolutional part (keep PyTorch ops) ----------
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)

        # ---------------- TileLang-based Linear layers -------------------
        device = "cuda"
        w_dtype = torch.float16

        # fc1 9216 -> 4096  (fuse ReLU)
        self.fc1_weight = nn.Parameter(torch.empty(4096, 256 * 6 * 6, device=device, dtype=w_dtype))
        self.fc1_bias   = nn.Parameter(torch.empty(4096,               device=device, dtype=w_dtype))
        # fc2 4096 -> 4096  (fuse ReLU)
        self.fc2_weight = nn.Parameter(torch.empty(4096, 4096, device=device, dtype=w_dtype))
        self.fc2_bias   = nn.Parameter(torch.empty(4096, device=device, dtype=w_dtype))
        # fc3 4096 -> num_classes
        self.fc3_weight = nn.Parameter(torch.empty(num_classes, 4096, device=device, dtype=w_dtype))
        self.fc3_bias   = nn.Parameter(torch.empty(num_classes, device=device, dtype=w_dtype))

        # --- identical initialization to nn.Linear defaults -------------
        for weight, bias in [
            (self.fc1_weight, self.fc1_bias),
            (self.fc2_weight, self.fc2_bias),
            (self.fc3_weight, self.fc3_bias),
        ]:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(weight.shape[1])
            nn.init.uniform_(bias, -bound, bound)

        # Kernel caches keyed by (batch_size, fuse_relu flag)
        self._fc1_kernels = {}
        self._fc2_kernels = {}
        self._fc3_kernels = {}

    # ------------------------------------------------------------------
    def _get_kernel(self, cache, M, N, K, fuse_relu):
        key = M  # batch size is the only dynamic dimension
        if key not in cache:
            cache[key] = self._build_linear_kernel(M, N, K, fuse_relu)
        return cache[key]

    # ------------------------------------------------------------------
    def forward(self, x):
        # ---------- convolutions (float32) -----------------------------
        x = self.maxpool1(self.relu1(self.conv1(x)))
        x = self.maxpool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.maxpool3(self.relu5(self.conv5(x)))

        # Flatten
        x = torch.flatten(x, 1)

        # Cast to fp16 for TileLang GEMM
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        B = x_fp16.shape[0]

        # ------------------ fc1 + ReLU ---------------------------------
        fc1_kernel = self._get_kernel(self._fc1_kernels, B, 4096, 256 * 6 * 6, True)
        x_fp16 = fc1_kernel(x_fp16, self.fc1_weight, self.fc1_bias)

        # ------------------ fc2 + ReLU ---------------------------------
        fc2_kernel = self._get_kernel(self._fc2_kernels, B, 4096, 4096, True)
        x_fp16 = fc2_kernel(x_fp16, self.fc2_weight, self.fc2_bias)

        # ------------------ fc3 ---------------------------------------
        fc3_kernel = self._get_kernel(self._fc3_kernels, B, self.fc3_weight.shape[0], 4096, False)
        x_fp16 = fc3_kernel(x_fp16, self.fc3_weight, self.fc3_bias)

        # Return in original dtype (float32)
        return x_fp16.to(torch.float32)