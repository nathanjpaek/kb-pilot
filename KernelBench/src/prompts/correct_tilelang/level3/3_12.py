"""
Problem Name: 12_VGG19
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=2.78 runtime_stats={'mean': 2.78, 'std': 0.00783, 'min': 2.76, 'max': 2.83, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.39, 'std': 0.00794, 'min': 2.38, 'max': 2.46, 'num_trials': 100}, 'speedup_ratio': 0.86}}
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# Helper:  build a TileLang GEMM kernel  (optionally fused ReLU)
# --------------------------------------------------------------------------- #

def _build_linear_kernel(
    M: int,                 # batch size (dynamic)
    N: int,                 # out_features
    K: int,                 # in_features
    fuse_relu: bool,
    block_M: int = 64,
    block_N: int = 128,
    block_K: int = 32,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    """Return a compiled TileLang kernel computing  Y = X @ W.T + B (+ReLU)."""

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),      # input
        W: T.Tensor((N, K), dtype),      # weight (row-major)
        B: T.Tensor((N,), dtype),        # bias
        Y: T.Tensor((M, N), dtype),      # output
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_N, block_K), dtype)
            C_f = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_f)

            # ---- main K reduction (pipelined) -----------------------------
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(X[by * block_M, ko * block_K], A_s)
                T.copy(W[bx * block_N, ko * block_K], B_s)
                T.gemm(A_s, B_s, C_f, transpose_B=True)

            # ---- bias, activation, store ----------------------------------
            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = C_f[i, j] + B[gn]
                    if fuse_relu:
                        val = T.max(val, T.Cast(accum_dtype, 0))
                    Y[gm, gn] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------------- #
# VGG-19 with TileLang-accelerated classifier                                 #
# --------------------------------------------------------------------------- #

class ModelNew(nn.Module):
    """VGG-19 where the three Linear layers are replaced by TileLang kernels."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # ----------------- feature extractor  (unchanged) ------------------
        self.features = nn.Sequential(
            # Block-1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block-2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block-3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block-4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block-5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # ----------------- TileLang Linear parameters ----------------------
        device = "cuda"
        w_dtype = torch.float16

        self.fc1_weight = nn.Parameter(torch.empty(4096, 512 * 7 * 7, device=device, dtype=w_dtype))
        self.fc1_bias   = nn.Parameter(torch.empty(4096,               device=device, dtype=w_dtype))
        self.fc2_weight = nn.Parameter(torch.empty(4096, 4096, device=device, dtype=w_dtype))
        self.fc2_bias   = nn.Parameter(torch.empty(4096, device=device, dtype=w_dtype))
        self.fc3_weight = nn.Parameter(torch.empty(num_classes, 4096, device=device, dtype=w_dtype))
        self.fc3_bias   = nn.Parameter(torch.empty(num_classes, device=device, dtype=w_dtype))

        # ---- identical initialisation to nn.Linear ------------------------
        for weight, bias in [
            (self.fc1_weight, self.fc1_bias),
            (self.fc2_weight, self.fc2_bias),
            (self.fc3_weight, self.fc3_bias),
        ]:
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            bound = 1 / math.sqrt(weight.shape[1])
            nn.init.uniform_(bias, -bound, bound)

        # kernel caches keyed by (batch, fuse_relu)
        self._fc1_kernels = {}
        self._fc2_kernels = {}
        self._fc3_kernels = {}

        # dropout (keep identical behaviour, p=0)
        self.drop1 = nn.Dropout(p=0.0)
        self.drop2 = nn.Dropout(p=0.0)

    # ---------------------------------------------------------------------
    def _get_kernel(self, cache, M, N, K, fuse_relu):
        key = M  # batch size only dynamic dimension
        if key not in cache:
            cache[key] = _build_linear_kernel(M, N, K, fuse_relu)
        return cache[key]

    # ---------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- convolutional backbone (float32) ---------------------------
        x = self.features(x)
        x = torch.flatten(x, 1)

        # ---- move to fp16 / CUDA for GEMMs ------------------------------
        x_fp16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        B = x_fp16.shape[0]

        # ------------------- FC-1 + ReLU ---------------------------------
        k1 = self._get_kernel(self._fc1_kernels, B, 4096, 512 * 7 * 7, True)
        x_fp16 = k1(x_fp16, self.fc1_weight, self.fc1_bias)
        x_fp16 = self.drop1(x_fp16)  # p=0, keeps training semantics

        # ------------------- FC-2 + ReLU ---------------------------------
        k2 = self._get_kernel(self._fc2_kernels, B, 4096, 4096, True)
        x_fp16 = k2(x_fp16, self.fc2_weight, self.fc2_bias)
        x_fp16 = self.drop2(x_fp16)

        # ------------------- FC-3 (logits) -------------------------------
        k3 = self._get_kernel(self._fc3_kernels, B, self.fc3_weight.shape[0], 4096, False)
        x_fp16 = k3(x_fp16, self.fc3_weight, self.fc3_bias)

        return x_fp16.to(dtype=torch.float32)