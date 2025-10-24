"""
Problem Name: 12_Gemm_Multiply_LeakyReLU
Generated using DSPy RAG with openai/o3
RAG Examples: 5
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)', 'error_during_performance': RuntimeError('mat1 and mat2 must have the same dtype, but got Float and Half')} runtime=-1.0 runtime_stats={}
"""

# ---------------------------------------------------------------------------
#  Fused Linear → scale → LeakyReLU  (row-major 2-D tensors)
# ---------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ----------------------------------------------------------------------------
#  Device kernel
# ----------------------------------------------------------------------------
@cute.kernel
def _linear_scale_lrelu_kernel(
    gA   : cute.Tensor,           # (B, K)                – input activations
    gBTv : cute.Tensor,           # ((1,V), (K, O/V))     – tiled Wᵀ
    gBiasv : cute.Tensor,         # ((1,V), (O/V))        – tiled bias
    gCv  : cute.Tensor,           # ((1,V), (B, O/V))     – output
    K     : cutlass.Int32,        # reduction length
    mul   : cutlass.Float32,      # post-GEMM multiplier
    neg_slope : cutlass.Float32,  # LeakyReLU negative slope
):
    # ------------------------------------------------------------------------
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    bi     = bidy                        # batch row   (0 … B-1)
    og     = bidx * bdimx + tidx         # output-vector group (0 … O/V-1)

    B   = gCv.shape[1][0]
    Og  = gCv.shape[1][1]                # O / V

    if bi < B and og < Og:
        # --------------------------------------------------------------------
        # Views / fragments
        c_out = gCv[(None, (bi, og))]               # (1,V)

        b_frag     = cute.make_fragment_like(c_out, gA.element_type)
        bias_frag  = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()                   # TensorSSA(F32, (1,V))

        # --------------------------------------------------------------------
        # Main reduction  (accumulate (B[bi, :] @ Wᵀ) in fp32)
        for k in range(K):
            a_val_f32 = cutlass.Float32(gA[bi, k])

            # load vector from Wᵀ at row k
            b_vec_gmem = gBTv[(None, (k, og))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec_f32 = b_frag.load().to(cutlass.Float32)

            acc = acc + a_val_f32 * b_vec_f32

        # --------------------------------------------------------------------
        # Epilogue: bias → multiply → LeakyReLU
        # --------------------------------------------------------------------
        # 1. Add bias
        bias_vec_gmem = gBiasv[(None, og)]
        cute.autovec_copy(bias_vec_gmem, bias_frag)
        bias_f32 = bias_frag.load().to(cutlass.Float32)
        acc = acc + bias_f32

        # 2. Scalar multiply
        acc = acc * mul

        # 3. LeakyReLU
        zero = cutlass.Float32(0.0)
        pos  = acc > zero
        acc  = cute.where(pos, acc, acc * neg_slope)

        # --------------------------------------------------------------------
        # Store
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# ----------------------------------------------------------------------------
#  Host wrapper
# ----------------------------------------------------------------------------
@cute.jit
def _linear_scale_lrelu_host(
    mA     : cute.Tensor,             # (B, K)
    mBT    : cute.Tensor,             # (K, O)    Wᵀ
    mBias  : cute.Tensor,             # (O)
    mC     : cute.Tensor,             # (B, O)
    V      : cutlass.Constexpr,       # vector width
    K      : cutlass.Constexpr,       # K dimension
    mul    : cutlass.Float32,         # multiplier
    neg_slope : cutlass.Float32,      # negative slope
):
    B, O = mA.shape[0], mBT.shape[1]

    # Tile B^T and bias along contiguous dimension O
    gBTv   = cute.zipped_divide(mBT,   (1, V))      # ((1,V), (K, O/V))
    gBiasv = cute.zipped_divide(mBias, (V,))        # ((1,V), (O/V)) – 1-D → ((1,V), (Og))
    gCv    = cute.zipped_divide(mC,    (1, V))      # ((1,V), (B, O/V))

    threads_per_block = 256
    O_groups = O // V
    grid_x   = cute.ceil_div(O_groups, threads_per_block)
    grid_y   = B

    _linear_scale_lrelu_kernel(
        mA, gBTv, gBiasv, gCv,
        cutlass.Int32(K),
        mul,
        neg_slope
    ).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1)
    )


# ----------------------------------------------------------------------------
#  PyTorch reference implementation
# ----------------------------------------------------------------------------
class Model(nn.Module):
    """
    Reference PyTorch implementation:
        Y = LeakyReLU( (X @ Wᵀ + b) * multiplier )
    """
    def __init__(self,
                 in_features    : int,
                 out_features   : int,
                 multiplier     : float = 1.0,
                 negative_slope : float = 0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Simple manual matmul + bias + multiply + leaky_relu
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        x = x * self.multiplier
        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        return x


# ----------------------------------------------------------------------------
#  PyTorch façade (Optimized CuTe version)
# ----------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    Fused implementation of:
        Y = LeakyReLU( (X @ Wᵀ + b) * multiplier )
    Vectorises along the output dimension for 128-bit memory ops.
    """

    def __init__(self,
                 in_features    : int,
                 out_features   : int,
                 multiplier     : float = 1.0,
                 negative_slope : float = 0.01):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features))
        self.multiplier = float(multiplier)
        self.negative_slope = float(negative_slope)
        self.reset_parameters()

        self._cache = {}

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def _apply(self, fn):
        """Override _apply to clear cache when dtype/device changes."""
        super()._apply(fn)
        self._cache.clear()  # Clear cache on dtype/device change
        return self

    # ------------------------------
    # Helpers
    # ------------------------------
    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, O: int) -> int:
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (O % V):
            V //= 2
        return V

    # ------------------------------
    # Forward
    # ------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 2, "Input must be 2-D (batch, in_features)"
        B, K = x.shape
        O, Kw = self.weight.shape
        assert Kw == K, "in_features mismatch"

        # Ensure CUDA and contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        
        # Use the model's parameter dtype (respects .to() calls)
        # Convert input to match parameter dtype
        if x.dtype != self.weight.dtype:
            x = x.to(dtype=self.weight.dtype)
        
        # Use parameters as-is (they're already in the correct dtype from .to() calls)
        W = self.weight.detach().contiguous()
        b = self.bias.detach().contiguous()

        V = self._pick_vec_width(x.dtype, O)
        assert O % V == 0, "Vector width must divide out_features"

        # Output tensor matches working dtype
        y = torch.empty((B, O), dtype=x.dtype, device=x.device)
        
        # Transpose weight and ensure contiguous
        W_T = W.t().contiguous()

        # --------------------------------------------------------------------
        # Wrap tensors for CuTe  (row-major, dynamic leading stride)
        # --------------------------------------------------------------------
        mA   = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mBT  = from_dlpack(W_T, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))  # (K,O)
        mBias= from_dlpack(b, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))        # (O)
        mC   = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))

        # Cache compiled kernel with actual working dtype
        key = (x.dtype, V, K, O)
        if key not in self._cache:
            # Use plain float values directly
            mul_val = cutlass.Float32(self.multiplier)
            slope_val = cutlass.Float32(self.negative_slope)
            
            self._cache[key] = cute.compile(
                _linear_scale_lrelu_host,
                mA, mBT, mBias, mC,
                V,
                K,
                mul_val,
                slope_val
            )

        # --------------------------------------------------------------------
        # Launch
        # --------------------------------------------------------------------
        # Extract scalar values for kernel launch
        mul_val = cutlass.Float32(self.multiplier)
        slope_val = cutlass.Float32(self.negative_slope)
        
        self._cache[key](
            mA, mBT, mBias, mC,
            mul_val,
            slope_val
        )
        
        # Output is already in correct dtype, return as-is
        return y


# ----------------------------------------------------------------------------
#  Convenience inputs for the benchmark harness
# ----------------------------------------------------------------------------
batch_size     = 128
in_features    = 1024
out_features   = 512
multiplier     = 2.0
negative_slope = 0.1


def get_inputs():
    return [torch.randn(batch_size, in_features)]


def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]