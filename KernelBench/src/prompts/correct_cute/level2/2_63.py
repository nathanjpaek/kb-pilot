"""
Problem Name: 63_Gemm_ReLU_Divide
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.184 runtime_stats={'mean': 0.184, 'std': 0.00725, 'min': 0.174, 'max': 0.208, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0619, 'std': 0.00749, 'min': 0.0543, 'max': 0.0905, 'num_trials': 100}, 'speedup_ratio': 0.336}}
"""

import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
#                         Device kernel
# ---------------------------------------------------------------------------
@cute.kernel
def _linear_relu_div_kernel(
    gA      : cute.Tensor,           # (B, K)                   – activations
    gBTv    : cute.Tensor,           # ((1,V), (K, O/V))        – Wᵀ, tiled
    gBiasv  : cute.Tensor,           # ((1,V), (O/V))           – bias, tiled
    gCv     : cute.Tensor,           # ((1,V), (B, O/V))        – output
    K       : cutlass.Int32,         # reduction length
    inv_div : cutlass.Float32,       # 1 / divisor
):
    # ------------------------ CUDA indices ---------------------------------
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi  = bidy                       # batch index  (0 … B-1)
    og  = bidx * bdimx + tidx        # output-vector group (0 … O/V-1)

    B   = gCv.shape[1][0]
    Og  = gCv.shape[1][1]            # O / V

    if (bi < B) and (og < Og):
        # -------------------------------------------------------------------
        # Views / fragments
        c_out = gCv[(None, (bi, og))]                 # (1,V)

        b_frag     = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag  = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()                     # TensorSSA(Float32,(1,V))

        # ---------------------- GEMM reduction -----------------------------
        for k in range(K):
            a_val_f32  = cutlass.Float32(gA[bi, k])         # scalar
            b_vec_gmem = gBTv[(None, (k, og))]              # (1,V)
            cute.autovec_copy(b_vec_gmem, b_frag)           # gmem → regs
            b_vec_f32  = b_frag.load().to(cutlass.Float32)  # TensorSSA
            acc = acc + a_val_f32 * b_vec_f32               # FMA

        # ------------------------ Bias + ReLU + div -------------------------
        # Add bias
        bias_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_gmem, bias_frag)
        bias_f32  = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # ReLU - FIXED: must use TensorSSA for false_value, not scalar
        zero_scalar = cutlass.Float32(0.0)
        acc  = cute.where(acc > zero_scalar, acc, acc * zero_scalar)

        # Divide by constant  (multiply by reciprocal)
        acc = acc * inv_div

        # ------------------------ Store result -----------------------------
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# ---------------------------------------------------------------------------
#                         Host wrapper
# ---------------------------------------------------------------------------
@cute.jit
def _linear_relu_div_host(
    mA      : cute.Tensor,            # (B, K)
    mBT     : cute.Tensor,            # (K, O)  – contiguous Wᵀ
    mBias   : cute.Tensor,            # (O)
    mC      : cute.Tensor,            # (B, O)
    V       : cutlass.Constexpr,      # vector width (compile-time)
    K       : cutlass.Constexpr,      # K dimension  (compile-time)
    inv_div : cutlass.Float32,        # 1 / divisor
):
    B, O = mA.shape[0], mBT.shape[1]

    # Tile along the contiguous output dimension O with width V
    gBTv   = cute.zipped_divide(mBT,  (1, V))   # ((1,V),(K,O/V))
    gBiasv = cute.zipped_divide(mBias, (V,))    # ((1,V),(O/V))
    gCv    = cute.zipped_divide(mC,    (1, V))  # ((1,V),(B,O/V))

    threads_per_block = 256
    O_groups          = O // V

    grid_x = cute.ceil_div(O_groups, threads_per_block)
    grid_y = B

    _linear_relu_div_kernel(
        mA, gBTv, gBiasv, gCv,
        cutlass.Int32(K),
        inv_div,
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
#                         PyTorch façade
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Fused implementation of
        Y = ReLU( X @ Wᵀ + b ) / divisor
    Vectorised along the output feature dimension for 128-bit memory ops.
    """

    def __init__(self, in_features: int, out_features: int, divisor: float):
        super().__init__()
        self.linear   = nn.Linear(in_features, out_features)   # bias=True
        self.divisor  = float(divisor)
        self._cache   = {}

    # Helper: pick 128-bit vector width -------------------------------------
    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (O % V):
            V //= 2
        return V

    # Flush cache on .to() / .cuda() ----------------------------------------
    def _apply(self, fn):
        self._cache.clear()
        return super()._apply(fn)

    # Forward ---------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Input must be 2-D (batch, in_features)"
        B, K = x.shape
        O, K_w = self.linear.weight.shape
        assert K == K_w, "in_features mismatch"

        # Ensure CUDA & contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        # Cast input to parameter dtype if needed
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)

        # Get parameters as contiguous buffers
        W = self.linear.weight.detach().contiguous()
        b = self.linear.bias.detach().contiguous()
        W_T = W.t().contiguous()

        # Choose vector width
        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, "Vector width V must divide out_features"

        # Allocate output
        y = torch.empty((B, O), dtype=x.dtype, device=x.device)

        # Wrap tensors for CuTe
        mA    = from_dlpack(x,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT   = from_dlpack(W_T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(b,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC    = from_dlpack(y,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype, V, K, O)
        if key not in self._cache:
            inv_div_c = cutlass.Float32(1.0 / self.divisor)
            self._cache[key] = cute.compile(
                _linear_relu_div_host,
                mA, mBT, mBias, mC,
                V,                 # constexpr
                K,                 # constexpr
                inv_div_c,
            )

        # Launch
        inv_div_c = cutlass.Float32(1.0 / self.divisor)
        self._cache[key](
            mA, mBT, mBias, mC,
            inv_div_c,
        )
        return y


# ---------------------------------------------------------------------------
#                       Benchmark convenience vars
# ---------------------------------------------------------------------------
batch_size   = 128
in_features  = 1024
out_features = 512
divisor      = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, divisor]