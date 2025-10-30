"""
Problem Name: 81_Gemm_Swish_Divide_Clamp_Tanh_Clamp
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.141 runtime_stats={'mean': 0.141, 'std': 0.024, 'min': 0.134, 'max': 0.378, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.087, 'std': 0.00279, 'min': 0.0843, 'max': 0.108, 'num_trials': 100}, 'speedup_ratio': 0.617}}
"""

import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ===========================================================================
#  1) Device kernel
# ===========================================================================
@cute.kernel
def _linear_swish_div_clamp_tanh_clamp_kernel(
    gA       : cute.Tensor,           # (B, K)
    gBTv     : cute.Tensor,           # ((1,V), (K, O/V))
    gBiasv   : cute.Tensor,           # ((1,V), (O/V))
    gCv      : cute.Tensor,           # ((1,V), (B, O/V))
    K        : cutlass.Int32,         # reduction length
    inv_two  : cutlass.Float32,       # 0.5
    clamp_lo : cutlass.Float32,       # -1.0
    clamp_hi : cutlass.Float32,       #  1.0
):
    # -----------------------------------------------------------------------
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi  = bidy                       # batch index
    og  = bidx * bdimx + tidx        # output-vector group

    B  = gCv.shape[1][0]
    Og = gCv.shape[1][1]             # O / V

    if (bi < B) and (og < Og):

        # Vector slice handled by this thread
        c_out = gCv[(None, (bi, og))]            # (1,V)

        # Temporary fragments
        b_frag     = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag  = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()                # TensorSSA(Float32,(1,V))

        # --------------------------- GEMM -----------------------------------
        for k in range(K):
            a_val_f32  = cutlass.Float32(gA[bi, k])          # scalar
            b_vec_gmem = gBTv[(None, (k, og))]               # (1,V)
            cute.autovec_copy(b_vec_gmem, b_frag)            # gmem → regs
            b_vec_f32  = b_frag.load().to(cutlass.Float32)
            acc = acc + a_val_f32 * b_vec_f32

        # ----------------------- Add bias -----------------------------------
        bias_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_gmem, bias_frag)
        bias_f32  = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # ----------------------- Swish --------------------------------------
        one  = cutlass.Float32(1.0)
        neg1 = cutlass.Float32(-1.0)
        zero = cutlass.Float32(0.0)

        one_t  = acc * zero + one                 # broadcast 1
        neg_acc= acc * neg1                       # -acc
        exp_val= cute.exp(neg_acc)
        sigmoid= one_t / (one_t + exp_val)
        acc    = acc * sigmoid                    # swish

        # ----------------------- Divide by 2 --------------------------------
        acc = acc * inv_two                       # × 0.5

        # ----------------------- Clamp [-1,1] -------------------------------
        lo_t = acc * zero + clamp_lo              # broadcast -1
        hi_t = acc * zero + clamp_hi              # broadcast  +1
        acc  = cute.where(acc < lo_t, lo_t, acc)  # floor
        acc  = cute.where(acc > hi_t, hi_t, acc)  # ceil

        # ----------------------- tanh ---------------------------------------
        two   = cutlass.Float32(2.0)
        e2x   = cute.exp(acc * two)               # e^{2x}
        tanh  = (e2x - one_t) / (e2x + one_t)
        acc   = tanh

        # ----------------------- Final clamp [-1,1] --------------------------
        acc  = cute.where(acc < lo_t, lo_t, acc)
        acc  = cute.where(acc > hi_t, hi_t, acc)

        # ----------------------- Store back ---------------------------------
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# ===========================================================================
#  2) Host wrapper
# ===========================================================================
@cute.jit
def _linear_swish_div_clamp_tanh_clamp_host(
    mA       : cute.Tensor,            # (B, K)
    mBT      : cute.Tensor,            # (K, O)
    mBias    : cute.Tensor,            # (O)
    mC       : cute.Tensor,            # (B, O)
    V        : cutlass.Constexpr,      # vector width (compile-time)
    K        : cutlass.Constexpr,      # reduction dim (compile-time)
    inv_two  : cutlass.Float32,        # 0.5
    clamp_lo : cutlass.Float32,        # -1.0
    clamp_hi : cutlass.Float32,        #  1.0
):
    B, O = mA.shape[0], mBT.shape[1]

    # Tile along contiguous O with width V
    gBTv   = cute.zipped_divide(mBT,  (1, V))   # ((1,V),(K,O/V))
    gBiasv = cute.zipped_divide(mBias, (V,))    # ((1,V),(O/V))
    gCv    = cute.zipped_divide(mC,    (1, V))  # ((1,V),(B,O/V))

    threads_per_block = 256
    O_groups          = O // V
    grid_x = cute.ceil_div(O_groups, threads_per_block)
    grid_y = B

    _linear_swish_div_clamp_tanh_clamp_kernel(
        mA, gBTv, gBiasv, gCv,
        cutlass.Int32(K),
        inv_two,
        clamp_lo,
        clamp_hi,
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


# ===========================================================================
#  3) PyTorch façade
# ===========================================================================
class ModelNew(nn.Module):
    """
    Fused CuTe implementation of:
        Y = clamp( tanh( clamp( swish(X @ Wᵀ + b) / 2 , -1, 1 ) ), -1, 1 )
    Vectorised along the output dimension for 128-bit memory accesses.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self._cache = {}

    # ------------------------- utilities ------------------------------------
    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4   # 128-bit
        while V > 1 and (O % V):
            V //= 2
        return V

    def _apply(self, fn):
        """Flush the CuTe JIT cache on dtype/device moves."""
        self._cache.clear()
        return super()._apply(fn)

    # ------------------------- forward --------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Input must be 2-D (batch, in_features)"
        B, K = x.shape
        O, K_w = self.gemm.weight.shape
        assert K == K_w, "in_features mismatch"

        # Ensure CUDA & contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        # Cast to parameter dtype if needed
        if x.dtype != self.gemm.weight.dtype:
            x = x.to(self.gemm.weight.dtype)

        # Parameters
        W   = self.gemm.weight.detach().contiguous()
        b   = self.gemm.bias.detach().contiguous() if self.gemm.bias is not None \
              else torch.zeros(O, dtype=W.dtype, device=W.device)
        W_T = W.t().contiguous()

        # Vector width
        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, "Vector width must divide out_features"

        # Output
        y = torch.empty((B, O), dtype=x.dtype, device=x.device)

        # ---------------- Wrap tensors for CuTe ------------------------------
        mA    = from_dlpack(x,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1))
        mBT   = from_dlpack(W_T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1))
        mBias = from_dlpack(b,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC    = from_dlpack(y,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,1))

        # ---------------- Compile / cache ------------------------------------
        key = (x.dtype, V, K, O)
        if key not in self._cache:
            inv_two  = cutlass.Float32(0.5)
            clamp_lo = cutlass.Float32(-1.0)
            clamp_hi = cutlass.Float32( 1.0)

            self._cache[key] = cute.compile(
                _linear_swish_div_clamp_tanh_clamp_host,
                mA, mBT, mBias, mC,
                V,               # constexpr V
                K,               # constexpr K
                inv_two,
                clamp_lo,
                clamp_hi,
            )

        # ---------------- Launch ---------------------------------------------
        self._cache[key](
            mA, mBT, mBias, mC,
            cutlass.Float32(0.5),     # inv_two
            cutlass.Float32(-1.0),    # clamp_lo
            cutlass.Float32( 1.0),    # clamp_hi
        )
        return y


# ===========================================================================
#  Benchmark convenience variables
# ===========================================================================
batch_size   = 128
in_features  = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]