"""
Problem Name: 23_EfficientNetB1
Generated using DSPy RAG with openai/o3
RAG Examples: 8
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H200', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.89 runtime_stats={'mean': 1.89, 'std': 0.0539, 'min': 1.81, 'max': 2.13, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.88, 'std': 0.0449, 'min': 1.8, 'max': 1.98, 'num_trials': 100}, 'speedup_ratio': 0.995}}
"""

import math
import torch
import torch.nn as nn
import tilelang
import tilelang.language as T


# --------------------------------------------------------------------- #
# TileLang kernel factory for Linear  (M=batch, K=1280, N=num_classes)
# --------------------------------------------------------------------- #
def _build_linear_kernel(
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
    def kernel(
        A: T.Tensor((M, K), dtype),      # input
        W: T.Tensor((N, K), dtype),      # weight (row-major)
        B: T.Tensor((N,), dtype),        # bias
        Y: T.Tensor((M, N), dtype),      # output
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128
        ) as (bx, by):
            As = T.alloc_shared((block_M, block_K), dtype)
            Ws = T.alloc_shared((block_N, block_K), dtype)
            Cf = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(Cf)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], As)
                T.copy(W[bx * block_N, ko * block_K], Ws)
                T.gemm(As, Ws, Cf, transpose_B=True)

            for i, j in T.Parallel(block_M, block_N):
                gm = by * block_M + i
                gn = bx * block_N + j
                if (gm < M) and (gn < N):
                    val = Cf[i, j] + B[gn]
                    Y[gm, gn] = T.Cast(dtype, val)

    return kernel


# --------------------------------------------------------------------- #
# PyTorch Model wrapper with TileLang-accelerated FC
# --------------------------------------------------------------------- #
class ModelNew(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()

        # ----------------------- Stem -----------------------------------
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # -------------------- MBConv blocks -----------------------------
        def _make_mbconv(in_c, out_c, stride, expand):
            hidden = round(in_c * expand)
            return nn.Sequential(
                nn.Conv2d(in_c, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
                nn.Conv2d(
                    hidden,
                    hidden,
                    3,
                    stride=stride,
                    padding=1,
                    groups=hidden,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden, out_c, 1, bias=False),
                nn.BatchNorm2d(out_c),
            )

        self.mbconv1 = _make_mbconv(32, 16, 1, 1)
        self.mbconv2 = _make_mbconv(16, 24, 2, 6)
        self.mbconv3 = _make_mbconv(24, 40, 2, 6)
        self.mbconv4 = _make_mbconv(40, 80, 2, 6)
        self.mbconv5 = _make_mbconv(80, 112, 1, 6)
        self.mbconv6 = _make_mbconv(112, 192, 2, 6)
        self.mbconv7 = _make_mbconv(192, 320, 1, 6)

        # ---------------------- Head ------------------------------------
        self.conv2 = nn.Conv2d(320, 1280, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)

        # ---------------- TileLang fully-connected ----------------------
        self.in_features = 1280
        self.out_features = num_classes

        self.weight = nn.Parameter(torch.empty(num_classes, 1280))
        self.bias = nn.Parameter(torch.empty(num_classes))

        # ---- identical to nn.Linear default init ----------------------
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.bias, -bound, bound)

        self._kernel_cache = {}  # key: (batch, dtype)

    # ----------------------------------------------------------------- #
    def _get_kernel(self, batch: int, dtype: str = "float16"):
        key = (batch, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _build_linear_kernel(
                batch, self.in_features, self.out_features, dtype=dtype
            )
        return self._kernel_cache[key]

    # ----------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ----------- Backbone (float32) --------------------------------
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        x = self.mbconv3(x)
        x = self.mbconv4(x)
        x = self.mbconv5(x)
        x = self.mbconv6(x)
        x = self.mbconv7(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)                     # (B, 1280)

        # ----------- TileLang FC (fp16 I/O) ----------------------------
        B = x.size(0)
        x16 = x.to(device="cuda", dtype=torch.float16).contiguous()
        w16 = self.weight.to(device="cuda", dtype=torch.float16).contiguous()
        b16 = self.bias.to(device="cuda", dtype=torch.float16).contiguous()

        kernel = self._get_kernel(B, "float16")
        y16 = kernel(x16, w16, b16)

        return y16.to(dtype=x.dtype)