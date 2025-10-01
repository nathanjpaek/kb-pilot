"""
Problem Name: 84_Gemm_BatchNorm_Scaling_Softmax
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.2 runtime_stats={'mean': 0.2, 'std': 0.00439, 'min': 0.195, 'max': 0.228, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.104, 'std': 0.00414, 'min': 0.0994, 'max': 0.131, 'num_trials': 100}, 'speedup_ratio': 0.52}}
"""

import math
import torch
import torch.nn as nn

import tilelang
import tilelang.language as T


def _make_gemm_kernel(M, N, K, block_M=128, block_N=128, block_K=32, num_stages=2, threads=128,
                      in_dtype="float16", accum_dtype="float32"):
    @tilelang.jit(out_idx=-1)
    @T.prim_func
    def gemm_kernel(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), in_dtype),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=threads,
        ) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), in_dtype)
            B_s = T.alloc_shared((block_K, block_N), in_dtype)
            C_loc = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_loc)

            k_tiles = T.ceildiv(K, block_K)
            for ko in T.Pipelined(k_tiles, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_loc)

            T.copy(C_loc, C[by * block_M, bx * block_N])

    return gemm_kernel


class ModelNew(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bn_eps=1e-5,
        bn_momentum=0.1,
        scale_shape=(1,),
        block_M=128,
        block_N=128,
        block_K=32,
    ):
        super(ModelNew, self).__init__()
        # -------- Linear parameters --------
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

        # -------- BatchNorm parameters --------
        self.bn_weight = nn.Parameter(torch.ones(out_features))
        self.bn_bias = nn.Parameter(torch.zeros(out_features))
        self.register_buffer("running_mean", torch.zeros(out_features))
        self.register_buffer("running_var", torch.ones(out_features))
        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum

        # -------- Scale parameter --------
        self.scale = nn.Parameter(torch.ones(scale_shape))

        # -------- Kernel cache --------
        self._kernel_cache = {}
        self._block_M = block_M
        self._block_N = block_N
        self._block_K = block_K

    def _get_kernel(self, M, K, N, dtype):
        key = (M, N, K, dtype)
        if key not in self._kernel_cache:
            self._kernel_cache[key] = _make_gemm_kernel(
                M,
                N,
                K,
                block_M=self._block_M,
                block_N=self._block_N,
                block_K=self._block_K,
            )
        return self._kernel_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = torch.device("cuda")
        # -------- Prepare Input & Weight --------
        x_fp16 = x.to(device=device, dtype=torch.float16, non_blocking=True)
        W_t_fp16 = self.weight.t().contiguous().to(device=device, dtype=torch.float16)
        M, K = x_fp16.shape
        N = self.out_features

        gemm_kernel = self._get_kernel(M, K, N, x_fp16.dtype)
        y = gemm_kernel(x_fp16, W_t_fp16)
        y = y + self.bias.to(device=device)

        # -------- Batch Normalization --------
        if self.training:
            batch_mean = y.mean(dim=0)
            batch_var = y.var(dim=0, unbiased=False)
            with torch.no_grad():
                self.running_mean.mul_(1 - self.bn_momentum).add_(self.bn_momentum * batch_mean)
                self.running_var.mul_(1 - self.bn_momentum).add_(self.bn_momentum * batch_var)
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
        y = (y - mean) / torch.sqrt(var + self.bn_eps)
        y = y * self.bn_weight + self.bn_bias

        # -------- Scaling --------
        y = self.scale * y

        # -------- Softmax --------
        y = torch.softmax(y, dim=1)
        return y