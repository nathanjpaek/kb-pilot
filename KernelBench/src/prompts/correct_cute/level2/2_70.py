"""
Problem Name: 70_Gemm_Sigmoid_Scaling_ResidualAdd
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.174 runtime_stats={'mean': 0.174, 'std': 0.00285, 'min': 0.17, 'max': 0.191, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0612, 'std': 0.00262, 'min': 0.0592, 'max': 0.077, 'num_trials': 100}, 'speedup_ratio': 0.352}}
"""

import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ============================================================================
# 1) Device kernel: GEMM + sigmoid * scale + residual
# ============================================================================
@cute.kernel
def _linear_sigmoid_scale_res_kernel(
    gA          : cute.Tensor,            # (B, K)
    gBTv        : cute.Tensor,            # ((1,V), (K, O/V))
    gBiasv      : cute.Tensor,            # ((1,V), (O/V))
    gCv         : cute.Tensor,            # ((1,V), (B, O/V))
    K           : cutlass.Int32,          # reduction length
    scale_val   : cutlass.Float32,        # scaling_factor
):
    # ---- CUDA indices ------------------------------------------------------
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi  = bidy                       # batch index
    og  = bidx * bdimx + tidx        # output-vector group (0 … O/V-1)

    B   = gCv.shape[1][0]
    Og  = gCv.shape[1][1]            # O / V

    if (bi < B) and (og < Og):
        # ---- view handled by this thread ----------------------------------
        c_out = gCv[(None, (bi, og))]              # (1,V)

        b_frag     = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag  = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()                  # TensorSSA(Float32,(1,V))

        # ---------------- GEMM reduction -----------------------------------
        for k in range(K):
            a_val_f32  = cutlass.Float32(gA[bi, k])           # scalar
            b_vec_gmem = gBTv[(None, (k, og))]                # (1,V)
            cute.autovec_copy(b_vec_gmem, b_frag)             # → regs
            b_vec_f32  = b_frag.load().to(cutlass.Float32)
            acc = acc + a_val_f32 * b_vec_f32                 # FMA

        # ---------------- Add bias -----------------------------------------
        bias_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_gmem, bias_frag)
        bias_f32  = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # ---------------- Sigmoid * scale + residual -----------------------
        orig = acc                                           # residual path

        one  = cutlass.Float32(1.0)
        zero = cutlass.Float32(0.0)
        neg1 = cutlass.Float32(-1.0)

        one_tensor = acc * zero + one                        # broadcast 1
        sigmoid    = one_tensor / (one_tensor + cute.exp(acc * neg1))
        acc        = orig + scale_val * sigmoid              # final formula

        # ---------------- Store back ---------------------------------------
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# ============================================================================
# 2) Host wrapper
# ============================================================================
@cute.jit
def _linear_sigmoid_scale_res_host(
    mA        : cute.Tensor,            # (B, K)
    mBT       : cute.Tensor,            # (K, O)     – contiguous Wᵀ
    mBias     : cute.Tensor,            # (O)
    mC        : cute.Tensor,            # (B, O)
    V         : cutlass.Constexpr,      # vector width (compile-time)
    K         : cutlass.Constexpr,      # reduction dim (compile-time)
    scale_val : cutlass.Float32,        # scaling_factor (runtime)
):
    B, O = mA.shape[0], mBT.shape[1]

    # ---- tile contiguous O with width V ------------------------------------
    gBTv   = cute.zipped_divide(mBT,  (1, V))   # ((1,V),(K,O/V))
    gBiasv = cute.zipped_divide(mBias, (V,))    # ((1,V),(O/V))
    gCv    = cute.zipped_divide(mC,    (1, V))  # ((1,V),(B,O/V))

    # ---- launch geometry ---------------------------------------------------
    threads_per_block = 256
    O_groups          = O // V
    grid_x = cute.ceil_div(O_groups, threads_per_block)
    grid_y = B

    _linear_sigmoid_scale_res_kernel(
        mA, gBTv, gBiasv, gCv,
        cutlass.Int32(K),
        scale_val
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


# ============================================================================
# 3) PyTorch façade
# ============================================================================
class ModelNew(nn.Module):
    """
    Fused CuTe implementation of
        Y = (X @ Wᵀ + b) + scaling_factor * sigmoid(X @ Wᵀ + b)
    Vectorised along the output dimension for 128-bit memory transactions.
    """

    def __init__(self, input_size: int, hidden_size: int, scaling_factor: float):
        super().__init__()
        self.gemm = nn.Linear(input_size, hidden_size)   # bias=True by default
        self.scaling_factor = float(scaling_factor)
        self._cache = {}

    # ------------- helpers --------------------------------------------------
    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        # 128-bit vectors: 8×f16 or 4×f32
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (O % V):
            V //= 2
        return V

    def _apply(self, fn):
        """Flush CuTe JIT cache on dtype / device change."""
        self._cache.clear()
        return super()._apply(fn)

    # ------------- forward --------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Y = M + sf * sigmoid(M) where M = X @ Wᵀ + b
        using the fused CuTe kernel.
        """
        assert x.dim() == 2, "Input must be 2-D (batch, in_features)"
        B, K = x.shape
        O, Kw = self.gemm.weight.shape
        assert Kw == K, "in_features mismatch"

        # Ensure CUDA & contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        # Cast to parameter dtype if necessary
        if x.dtype != self.gemm.weight.dtype:
            x = x.to(self.gemm.weight.dtype)

        # Fetch parameters
        W = self.gemm.weight.detach().contiguous()
        b = self.gemm.bias.detach().contiguous()
        WT = W.t().contiguous()

        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, f"Vector width V={V} must divide hidden_size={O}"

        # Allocate output
        y = torch.empty((B, O), dtype=x.dtype, device=x.device)

        # ---------------- Wrap into CuTe tensors ----------------------------
        mA    = from_dlpack(x,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT   = from_dlpack(WT, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(b,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC    = from_dlpack(y,  assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype, V, K, O)
        if key not in self._cache:
            sf_c = cutlass.Float32(self.scaling_factor)
            # Compile once
            self._cache[key] = cute.compile(
                _linear_sigmoid_scale_res_host,
                mA, mBT, mBias, mC,
                V,              # constexpr
                K,              # constexpr
                sf_c,
            )

        # Launch fused kernel
        self._cache[key](
            mA, mBT, mBias, mC,
            cutlass.Float32(self.scaling_factor)
        )
        return y


# ============================================================================
# 4) Convenience variables for benchmark harness
# ============================================================================
batch_size     = 128
input_size     = 1024
hidden_size    = 512
scaling_factor = 2.0


def get_inputs():
    return [torch.randn(batch_size, input_size)]


def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]