"""
Problem Name: 76_Gemm_Add_ReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.175 runtime_stats={'mean': 0.175, 'std': 0.0062, 'min': 0.166, 'max': 0.197, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0663, 'std': 0.00875, 'min': 0.0572, 'max': 0.109, 'num_trials': 100}, 'speedup_ratio': 0.379}}
"""

# ---------------------------------------------------------------------------
#  CuTe implementation of  Y = ReLU( X @ Wᵀ + b )
# Based on working 2_12.py, simplified (no multiplier, ReLU instead of LeakyReLU)
# ---------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _linear_bias_relu_kernel(
    gA     : cute.Tensor,           # (B, K)
    gBTv   : cute.Tensor,           # ((1,V), (K, O/V))
    gBiasv : cute.Tensor,           # ((1,V), (O/V))
    gCv    : cute.Tensor,           # ((1,V), (B, O/V))
    K      : cutlass.Int32,
):
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi = bidy
    og = bidx * bdimx + tidx

    B  = gCv.shape[1][0]
    Og = gCv.shape[1][1]

    if bi < B and og < Og:
        c_out = gCv[(None, (bi, og))]

        b_frag       = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag    = cute.make_fragment_like(c_out, gA.element_type)
        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()

        # GEMM reduction
        for k in range(K):
            a_val_f32 = cutlass.Float32(gA[bi, k])
            b_vec_gmem = gBTv[(None, (k, og))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec_f32 = b_frag.load().to(cutlass.Float32)
            acc = acc + a_val_f32 * b_vec_f32

        # Add bias
        bias_vec_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_vec_gmem, bias_frag)
        bias_f32 = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # ReLU - must use TensorSSA for false_value, not scalar
        zero_scalar = cutlass.Float32(0.0)
        acc = cute.where(acc > zero_scalar, acc, acc * zero_scalar)

        # Store
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


@cute.jit
def _linear_bias_relu_host(
    mA    : cute.Tensor,
    mBT   : cute.Tensor,
    mBias : cute.Tensor,
    mC    : cute.Tensor,
    V     : cutlass.Constexpr,
    K     : cutlass.Constexpr,
):
    B, O = mA.shape[0], mBT.shape[1]

    gBTv   = cute.zipped_divide(mBT,   (1, V))
    gBiasv = cute.zipped_divide(mBias, (V,))
    gCv    = cute.zipped_divide(mC,    (1, V))

    threads_per_block = 256
    O_groups = O // V
    grid_x = cute.ceil_div(O_groups, threads_per_block)
    grid_y = B

    _linear_bias_relu_kernel(
        mA, gBTv, gBiasv, gCv, cutlass.Int32(K)
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


class ModelNew(nn.Module):
    """CuTe-accelerated: Y = ReLU(X @ W^T + bias)"""

    def __init__(self, in_features: int, out_features: int, bias_shape):
        super().__init__()
        assert bias_shape == (out_features,), f"Expected {(out_features,)}, got {bias_shape}"
        # Match reference Model EXACTLY: nn.Linear(bias=False) + separate bias Parameter
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self._cache = {}

    def _apply(self, fn):
        """Override _apply to clear cache when dtype/device changes."""
        super()._apply(fn)
        self._cache.clear()
        return self

    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (O % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Y = ReLU(X @ W^T + bias) using CuTe kernel
        Reference computes: ReLU(gemm(x) + bias) where gemm is nn.Linear
        """
        assert x.dim() == 2, "Input must be 2-D"
        B, K = x.shape
        O, Kw = self.gemm.weight.shape
        assert Kw == K, "in_features mismatch"

        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        # Convert input to match weight dtype
        if x.dtype != self.gemm.weight.dtype:
            x = x.to(dtype=self.gemm.weight.dtype)
        
        # Use parameters from gemm
        W = self.gemm.weight.detach().contiguous()
        b = self.bias.detach().contiguous()

        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, f"Vector width {V} must divide out_features {O}"

        y = torch.empty((B, O), dtype=x.dtype, device=x.device)
        WT = W.t().contiguous()

        mA    = from_dlpack(x,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT   = from_dlpack(WT, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(b,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC    = from_dlpack(y,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype, V, K, O)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _linear_bias_relu_host,
                mA, mBT, mBias, mC,
                V,
                K
            )

        self._cache[key](mA, mBT, mBias, mC)
        return y


batch_size   = 128
in_features  = 1024
out_features = 512
bias_shape   = (out_features,)


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, bias_shape]