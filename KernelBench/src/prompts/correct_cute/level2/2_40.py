"""
Problem Name: 40_Matmul_Scaling_ResidualAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.0908 runtime_stats={'mean': 0.0908, 'std': 0.00978, 'min': 0.0802, 'max': 0.124, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0841, 'std': 0.00947, 'min': 0.0739, 'max': 0.115, 'num_trials': 100}, 'speedup_ratio': 0.926}}
"""

import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ----------------------------------------------------------------------------
# 1) Device kernel
# ----------------------------------------------------------------------------
@cute.kernel
def _linear_scale_res_kernel(
    gA            : cute.Tensor,            # (B, K)                   – activations
    gBTv          : cute.Tensor,            # ((1,V), (K, O/V))        – Wᵀ tiled
    gBiasv        : cute.Tensor,            # ((1,V), (O/V))           – bias tiled
    gCv           : cute.Tensor,            # ((1,V), (B, O/V))        – output
    K             : cutlass.Int32,          # reduction length
    scale_p1_f32  : cutlass.Float32,        # 1 + scaling_factor
):
    # ---------------- CUDA indices ------------------------------------------
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi  = bidy                       # batch row   (0…B-1)
    og  = bidx * bdimx + tidx        # output-vector group (0…O/V-1)

    B   = gCv.shape[1][0]
    Og  = gCv.shape[1][1]            # O / V

    if (bi < B) and (og < Og):
        # ---------------- Views & fragments ---------------------------------
        c_out = gCv[(None, (bi, og))]            # (1,V)

        b_frag     = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag  = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()               # TensorSSA(Float32,(1,V))

        # ---------------- GEMM inner-product --------------------------------
        for k in range(K):
            a_val_f32  = cutlass.Float32(gA[bi, k])            # scalar
            b_vec_gmem = gBTv[(None, (k, og))]                 # (1,V)
            cute.autovec_copy(b_vec_gmem, b_frag)              # → regs
            b_vec_f32  = b_frag.load().to(cutlass.Float32)
            acc = acc + a_val_f32 * b_vec_f32                  # FMA

        # ---------------- Add bias ------------------------------------------
        bias_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_gmem, bias_frag)
        bias_f32  = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # ---------------- Scale-and-Residual --------------------------------
        acc = acc * scale_p1_f32        # (1+sf) * (matmul + bias)

        # ---------------- Store result --------------------------------------
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# ----------------------------------------------------------------------------
# 2) Host wrapper
# ----------------------------------------------------------------------------
@cute.jit
def _linear_scale_res_host(
    mA           : cute.Tensor,            # (B, K)
    mBT          : cute.Tensor,            # (K, O)   – Wᵀ contiguous
    mBias        : cute.Tensor,            # (O)
    mC           : cute.Tensor,            # (B, O)
    V            : cutlass.Constexpr,      # vector width (compile-time)
    K            : cutlass.Constexpr,      # reduction dim (compile-time)
    scale_p1_f32 : cutlass.Float32,        # 1 + scaling_factor
):
    B, O = mA.shape[0], mBT.shape[1]

    # ---- Tile along contiguous output-dim O with width V -------------------
    gBTv   = cute.zipped_divide(mBT,  (1, V))   # ((1,V),(K,O/V))
    gBiasv = cute.zipped_divide(mBias, (V,))    # ((1,V),(O/V))
    gCv    = cute.zipped_divide(mC,    (1, V))  # ((1,V),(B,O/V))

    # ---- Launch geometry ---------------------------------------------------
    threads_per_block = 256
    O_groups          = O // V
    grid_x = cute.ceil_div(O_groups, threads_per_block)
    grid_y = B

    _linear_scale_res_kernel(
        mA, gBTv, gBiasv, gCv,
        cutlass.Int32(K),
        scale_p1_f32,
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


# ----------------------------------------------------------------------------
# 3) PyTorch façade
# ----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Fused CuTe implementation of
        Y = (1 + scaling_factor) * (X @ Wᵀ + b)
    Vectorised along the output dimension for 128-bit memory accesses.
    """

    def __init__(self, in_features: int, out_features: int, scaling_factor: float):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)  # bias=True default
        self.scaling_factor = float(scaling_factor)
        self._cache = {}

    # --------- helper: pick 128-bit vector width ----------------------------
    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4  # 128-bit
        while V > 1 and (O % V):
            V //= 2
        return V

    # Flush cache on dtype/device change -------------------------------------
    def _apply(self, fn):
        self._cache.clear()
        return super()._apply(fn)

    # ------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Y = (1 + sf) * (X @ W^T + bias) with CuTe.
        """
        assert x.dim() == 2, "Input must be 2-D (batch, in_features)"
        B, K = x.shape
        O, K_w = self.matmul.weight.shape
        assert K_w == K,  "in_features mismatch"

        # ---- ensure CUDA & contiguous, cast to weight dtype -----------------
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        if x.dtype != self.matmul.weight.dtype:
            x = x.to(self.matmul.weight.dtype)

        # ---- grab parameters ------------------------------------------------
        W   = self.matmul.weight.detach().contiguous()
        b   = self.matmul.bias.detach().contiguous()
        W_T = W.t().contiguous()

        # ---- choose vector width -------------------------------------------
        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, "Vector width V must divide out_features"

        # ---- allocate output ------------------------------------------------
        y = torch.empty((B, O), dtype=x.dtype, device=x.device)

        # ---- wrap tensors for CuTe -----------------------------------------
        mA    = from_dlpack(x,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT   = from_dlpack(W_T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(b,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC    = from_dlpack(y,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype, V, K, O)
        if key not in self._cache:
            scale_p1 = cutlass.Float32(1.0 + self.scaling_factor)
            # compile once
            self._cache[key] = cute.compile(
                _linear_scale_res_host,
                mA, mBT, mBias, mC,
                V,                  # constexpr V
                K,                  # constexpr K
                scale_p1,
            )

        # ---- launch ---------------------------------------------------------
        self._cache[key](
            mA, mBT, mBias, mC,
            cutlass.Float32(1.0 + self.scaling_factor)
        )
        return y


# ----------------------------------------------------------------------------
# 4) Convenience variables for benchmark harness
# ----------------------------------------------------------------------------
batch_size     = 128
in_features    = 64
out_features   = 128
scaling_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]