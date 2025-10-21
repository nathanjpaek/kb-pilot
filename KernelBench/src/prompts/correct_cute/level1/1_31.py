"""
Problem Name: 31_ELU
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=4.27 runtime_stats={'mean': 4.27, 'std': 0.00644, 'min': 4.26, 'max': 4.3, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.25, 'std': 0.0266, 'min': 4.24, 'max': 4.51, 'num_trials': 100}, 'speedup_ratio': 0.995}}
"""

import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel: vectorized ELU
# y = x                if x > 0
#     alpha*(exp(x)-1) otherwise
# ---------------------------------------------------------------------------
@cute.kernel
def _elu_vec_kernel(
    gXv: cute.Tensor,                   # ((1,V), (B, D/V))
    gYv: cute.Tensor,                   # ((1,V), (B, D/V))
    alpha: cutlass.Float32,             # runtime alpha
):
    # CUDA indices
    tidx, _, _   = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _  = cute.arch.block_dim()

    bi = bidy                           # batch row
    ng = bidx * bdimx + tidx            # vector group along D

    B        = gXv.shape[1][0]
    D_groups = gXv.shape[1][1]          # D / V

    if bi < B and ng < D_groups:
        x_vec_gmem = gXv[(None, (bi, ng))]  # (1,V) view
        y_vec_gmem = gYv[(None, (bi, ng))]

        # Fragments for autovec copy
        x_frag = cute.make_fragment_like(x_vec_gmem, gXv.element_type)
        y_frag = cute.make_fragment_like(y_vec_gmem, gYv.element_type)

        # Load -> FP32 TensorSSA
        cute.autovec_copy(x_vec_gmem, x_frag)
        x_vec = x_frag.load().to(cutlass.Float32)

        # ELU computation in FP32
        zero = cutlass.Float32(0.0)
        one  = cutlass.Float32(1.0)
        pos_mask = x_vec > zero
        neg_val  = alpha * (cute.exp(x_vec) - one)
        y_vec    = cute.where(pos_mask, x_vec, neg_val)

        # Store back after casting to original dtype
        y_frag.store(y_vec.to(gXv.element_type))
        cute.autovec_copy(y_frag, y_vec_gmem)


# ---------------------------------------------------------------------------
# Host wrapper: tiling + launch
# ---------------------------------------------------------------------------
@cute.jit
def _elu_vec_host(
    mX: cute.Tensor,                    # (B, D)
    mY: cute.Tensor,                    # (B, D)
    V : cutlass.Constexpr,              # vector width (compile-time)
    alpha: cutlass.Float32,             # runtime alpha
):
    B, D = mX.shape

    # Vectorize along D with width V
    gXv = cute.zipped_divide(mX, (1, V))    # ((1,V), (B, D/V))
    gYv = cute.zipped_divide(mY, (1, V))    # ((1,V), (B, D/V))

    threads_per_block = 256
    D_groups = D // V

    grid_x = cute.ceil_div(D_groups, threads_per_block)
    grid_y = B

    _elu_vec_kernel(gXv, gYv, alpha).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe-accelerated ELU for (B, D) tensors with 128-bit vectorization.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)
        self._cache = {}

    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, D: int) -> int:
        # Prefer 128-bit transactions
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8*16 = 128 bits
        elif dtype == torch.float32:
            V = 4   # 4*32 = 128 bits
        else:
            V = 2   # fallback (e.g., float64)
        while V > 1 and (D % V != 0):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA + contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        assert x.dim() == 2, "Expected a 2-D (B, D) tensor"
        B, D = x.shape

        V = self._pick_vec_width(x.dtype, D)
        y = torch.empty_like(x)

        # Wrap tensors for CuTe (row-major)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            # Compile once per (dtype, V); alpha is runtime
            self._cache[key] = cute.compile(
                _elu_vec_host, mX, mY, V, cutlass.Float32(self.alpha)
            )

        # Launch with runtime alpha
        self._cache[key](mX, mY, cutlass.Float32(self.alpha))
        return y


# Suggested harness sizes (from original snippet)
batch_size = 4096
dim = 393216

def get_inputs():
    x = torch.rand(batch_size, dim)
    return [x]

def get_init_inputs():
    return [1.0]