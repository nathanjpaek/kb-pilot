"""
Problem Name: 9_Matmul_Subtract_Multiply_ReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=1.4 runtime_stats={'mean': 1.4, 'std': 0.127, 'min': 1.33, 'max': 2.65, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 0.0862, 'std': 0.00704, 'min': 0.0804, 'max': 0.129, 'num_trials': 100}, 'speedup_ratio': 0.0616}}
"""

import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# ---------------------------------------------------------------------------
# Device kernel: GEMM + (−sub) * mul + ReLU (vectorised along O)
# ---------------------------------------------------------------------------
@cute.kernel
def _linear_submul_relu_kernel(
    gA      : cute.Tensor,           # (B, K)            – input activations
    gBTv    : cute.Tensor,           # ((1,V), (K, O/V)) – tiled Wᵀ
    gBiasv  : cute.Tensor,           # ((1,V), (O/V))    – tiled bias
    gCv     : cute.Tensor,           # ((1,V), (B, O/V)) – output
    K       : cutlass.Int32,         # reduction length
    sub_val : cutlass.Float32,       # scalar to subtract
    mul_val : cutlass.Float32,       # scalar to multiply
):
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi  = bidy                       # batch row (0…B-1)
    og  = bidx * bdimx + tidx        # output-vector group (0…O/V-1)

    B   = gCv.shape[1][0]
    Og  = gCv.shape[1][1]            # O / V

    if bi < B and og < Og:
        # View of the V-wide slice this thread computes
        c_out = gCv[(None, (bi, og))]            # (1,V)

        b_frag    = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()                 # TensorSSA(Float32,(1,V))

        # --------------------- GEMM inner-product --------------------------
        for k in range(K):
            a_val_f32   = cutlass.Float32(gA[bi, k])           # scalar
            b_vec_gmem  = gBTv[(None, (k, og))]                # (1,V) in gmem
            cute.autovec_copy(b_vec_gmem, b_frag)              # gmem → regs
            b_vec_f32   = b_frag.load().to(cutlass.Float32)    # TensorSSA
            acc = acc + a_val_f32 * b_vec_f32                  # FMA

        # ----------------------- Epilogue -------------------------------
        # 1) add bias
        bias_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_gmem, bias_frag)
        bias_f32  = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # 2) subtract constant
        acc = acc - sub_val

        # 3) multiply
        acc = acc * mul_val

        # 4) ReLU - must use TensorSSA for false_value, not scalar
        zero_scalar = cutlass.Float32(0.0)
        acc  = cute.where(acc > zero_scalar, acc, acc * zero_scalar)

        # ----------------------- Store back -----------------------------
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))  # cast back to src dtype
        cute.autovec_copy(out_frag, c_out)


# ---------------------------------------------------------------------------
# Host wrapper: builds layouts, sets launch config, calls kernel
# ---------------------------------------------------------------------------
@cute.jit
def _linear_submul_relu_host(
    mA      : cute.Tensor,           # (B, K)
    mBT     : cute.Tensor,           # (K, O)     – Wᵀ contiguous
    mBias   : cute.Tensor,           # (O)
    mC      : cute.Tensor,           # (B, O)
    V       : cutlass.Constexpr,     # vector width (compile-time)
    K       : cutlass.Constexpr,     # reduction dim (compile-time)
    sub_val : cutlass.Float32,       # scalar subtraction
    mul_val : cutlass.Float32,       # scalar multiplication
):
    B, O = mA.shape[0], mBT.shape[1]

    # Tile along contiguous output dim O with width V
    gBTv   = cute.zipped_divide(mBT,  (1, V))    # ((1,V),(K,O/V))
    gBiasv = cute.zipped_divide(mBias, (V,))     # ((1,V),(O/V))
    gCv    = cute.zipped_divide(mC,   (1, V))    # ((1,V),(B,O/V))

    threads_per_block = 256
    O_groups          = O // V
    grid_x = cute.ceil_div(O_groups, threads_per_block)
    grid_y = B

    _linear_submul_relu_kernel(
        mA, gBTv, gBiasv, gCv,
        cutlass.Int32(K),
        sub_val,
        mul_val,
    ).launch(grid=(grid_x, grid_y, 1), block=(threads_per_block, 1, 1))


# ---------------------------------------------------------------------------
# PyTorch façade – fused Linear −sub *mul + ReLU
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """Fused implementation of  Y = relu( (X @ Wᵀ + b − sub) * mul )
    Vectorises along the output dimension for 128-bit gmem transactions."""

    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super().__init__()
        # Match reference Model EXACTLY: nn.Linear with default bias=True
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = float(subtract_value)
        self.multiply_value = float(multiply_value)
        self._cache = {}

    # pick 128-bit vector width
    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (O % V):
            V //= 2
        return V

    def _apply(self, fn):
        # Clear JIT cache on .to() / device change
        self._cache.clear()
        return super()._apply(fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Y = ReLU((X @ W^T + bias - subtract_value) * multiply_value)
        Reference computes: ReLU((linear(x) - subtract_value) * multiply_value)
        """
        assert x.dim() == 2, "input must be (batch, in_features)"
        B, K = x.shape
        O, K_w = self.linear.weight.shape
        assert K == K_w, "in_features mismatch"

        # ensure CUDA & contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()

        # match dtype with parameters
        if x.dtype != self.linear.weight.dtype:
            x = x.to(self.linear.weight.dtype)

        W  = self.linear.weight.detach().contiguous()
        b  = self.linear.bias.detach().contiguous()
        W_T = W.t().contiguous()

        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, "vector width V must divide out_features"

        y = torch.empty((B, O), dtype=x.dtype, device=x.device)

        # ---------------- Wrap into CuTe tensors ------------------------
        mA    = from_dlpack(x,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT   = from_dlpack(W_T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBias = from_dlpack(b,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mC    = from_dlpack(y,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        key = (x.dtype, V, K, O)
        if key not in self._cache:
            sub_c = cutlass.Float32(self.subtract_value)
            mul_c = cutlass.Float32(self.multiply_value)
            # compile once
            self._cache[key] = cute.compile(
                _linear_submul_relu_host,
                mA, mBT, mBias, mC,
                V,             # constexpr V
                K,             # constexpr K
                sub_c,
                mul_c,
            )

        # launch
        self._cache[key](
            mA, mBT, mBias, mC,
            cutlass.Float32(self.subtract_value),
            cutlass.Float32(self.multiply_value)
        )
        return y


# ---------------------------------------------------------------------------
# Convenience parameters for the benchmark harness
# ---------------------------------------------------------------------------
batch_size     = 1024
in_features    = 4096
out_features   = 2048
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]