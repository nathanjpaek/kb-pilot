"""
Problem Name: 4_LeNet5
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.216 runtime_stats={'mean': 0.216, 'std': 0.0263, 'min': 0.193, 'max': 0.406, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.192, 'std': 0.0301, 'min': 0.171, 'max': 0.447, 'num_trials': 100}, 'speedup_ratio': 0.889}}
"""

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import tilelang
import tilelang.language as T


# --------------------------------------------------------------------------- #
# -------------------------  GEMM  Kernel Factory  -------------------------- #
# --------------------------------------------------------------------------- #
def _build_linear_kernel(
    M: int,
    N: int,
    K: int,
    fuse_relu: bool,
    threads: int = 256,
    dtype: str = "float16",
    accum_dtype: str = "float32",
):
    total = M * N
    grid = (total + threads - 1) // threads
    relu_flag = 1 if fuse_relu else 0  # baked-in compile-time const

    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def kernel(
        X: T.Tensor((M, K), dtype),          # (batch, in_features)
        W: T.Tensor((N, K), dtype),          # (out ,  in_features)
        B: T.Tensor((N,),     dtype),        # bias
        Y: T.Tensor((M, N), dtype),          # output (auto-allocated)
    ):
        zero_f = T.Cast(accum_dtype, 0)

        with T.Kernel(grid, threads=threads) as bx:
            tx = T.get_thread_binding(0)
            idx = bx * threads + tx
            if idx < total:
                m = idx // N
                n = idx - m * N

                acc = T.alloc_local((1,), accum_dtype)
                acc[0] = T.Cast(accum_dtype, B[n])

                for k in T.serial(K):
                    acc[0] += (
                        T.Cast(accum_dtype, X[m, k])
                        * T.Cast(accum_dtype, W[n, k])
                    )

                if relu_flag == 1:
                    acc[0] = T.max(acc[0], zero_f)

                Y[m, n] = T.Cast(dtype, acc[0])

    return kernel


# --------------------------------------------------------------------------- #
# ------------------------------   ModelNew   ------------------------------- #
# --------------------------------------------------------------------------- #
class ModelNew(nn.Module):
    """
    LeNet-5 with fully-connected layers replaced by fused TileLang GEMM kernels.
    Convolutional blocks remain on cuDNN.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # ------------------- Convolutional layers (unchanged) ----------------
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        # ------------------- Fully-connected parameters ----------------------
        # fc1 : 400 → 120
        self.fc1_w = nn.Parameter(torch.empty(120, 400))
        self.fc1_b = nn.Parameter(torch.empty(120))
        # fc2 : 120 → 84
        self.fc2_w = nn.Parameter(torch.empty(84, 120))
        self.fc2_b = nn.Parameter(torch.empty(84))
        # fc3 : 84  → num_classes
        self.fc3_w = nn.Parameter(torch.empty(num_classes, 84))
        self.fc3_b = nn.Parameter(torch.empty(num_classes))

        # identical initialisation to nn.Linear
        for w, b in [
            (self.fc1_w, self.fc1_b),
            (self.fc2_w, self.fc2_b),
            (self.fc3_w, self.fc3_b),
        ]:
            torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            bound = 1 / math.sqrt(w.shape[1])
            torch.nn.init.uniform_(b, -bound, bound)

        # ------------------- kernel caches -----------------------------------
        # key: (layer_id, M, dtype, relu_flag)
        self._kern_cache: Dict[Tuple[str, int, str, int], callable] = {}

    # ------------------------------------------------------------------ #
    def _get_kernel(self, layer_id: str, M: int, N: int, K: int, fuse_relu: bool, dtype: str):
        key = (layer_id, M, dtype, int(fuse_relu))
        if key not in self._kern_cache:
            self._kern_cache[key] = _build_linear_kernel(
                M, N, K, fuse_relu, dtype=dtype
            )
        return self._kern_cache[key]

    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype

        # ---------------- Convolutional feature extractor ----------------
        x = x.to(device="cuda", dtype=torch.float16)
        w1 = self.conv1.weight.to(device="cuda", dtype=torch.float16)
        b1 = self.conv1.bias.to(device="cuda", dtype=torch.float16)
        x = F.relu(F.conv2d(x, w1, b1, stride=1))

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        w2 = self.conv2.weight.to(device="cuda", dtype=torch.float16)
        b2 = self.conv2.bias.to(device="cuda", dtype=torch.float16)
        x = F.relu(F.conv2d(x, w2, b2, stride=1))

        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # flatten
        batch_size = x.shape[0]
        x = x.contiguous().view(batch_size, -1)           # (B, 400)

        # ---------- fc1 (ReLU fused) ------------------------------------
        ker = self._get_kernel(
            "fc1", batch_size, 120, 400, True, "float16"
        )
        y = ker(
            x,
            self.fc1_w.to(device="cuda", dtype=torch.float16),
            self.fc1_b.to(device="cuda", dtype=torch.float16),
        )  # (B,120)

        # ---------- fc2 (ReLU fused) ------------------------------------
        ker = self._get_kernel(
            "fc2", batch_size, 84, 120, True, "float16"
        )
        y = ker(
            y,
            self.fc2_w.to(device="cuda", dtype=torch.float16),
            self.fc2_b.to(device="cuda", dtype=torch.float16),
        )  # (B,84)

        # ---------- fc3 (no activation) ---------------------------------
        ker = self._get_kernel(
            "fc3", batch_size, self.fc3_w.shape[0], 84, False, "float16"
        )
        y = ker(
            y,
            self.fc3_w.to(device="cuda", dtype=torch.float16),
            self.fc3_b.to(device="cuda", dtype=torch.float16),
        )  # (B,num_classes)

        return y.to(orig_dtype)