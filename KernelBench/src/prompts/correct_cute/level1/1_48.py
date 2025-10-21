"""
Problem Name: 48_Mean_reduction_over_a_dimension
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=3.54 runtime_stats={'mean': 3.54, 'std': 0.0145, 'min': 3.51, 'max': 3.6, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.88, 'std': 0.0136, 'min': 2.86, 'max': 2.93, 'num_trials': 100}, 'speedup_ratio': 0.814}}
"""

import torch
import torch.nn as nn
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel: Mean reduction over dim=1 for (B, M, N) -> (B, N)
# Accumulates in FP32, vectorized along N with width V
# ---------------------------------------------------------------------------
@cute.kernel
def _mean_dim1_vec_kernel(
    gXv: cute.Tensor,              # ((1,1,V), (B, M, N/V))
    gYv: cute.Tensor,              # ((1,  V), (B,    N/V))
    scale: cutlass.Float32,        # 1.0 / M
):
    # CUDA indices
    tidx, _, _    = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _   = cute.arch.block_dim()

    b  = bidy
    ng = bidx * bdimx + tidx

    # Shapes after tiling
    B   = gXv.shape[1][0]
    M   = gXv.shape[1][1]
    Ng  = gXv.shape[1][2]          # N / V

    if (b < B) and (ng < Ng):
        # Initialize accumulation with first slice to define vector shape
        x0_gmem = gXv[(None, (b, 0, ng))]                 # (1,V) view
        x_frag  = cute.make_fragment_like(x0_gmem, gXv.element_type)
        cute.autovec_copy(x0_gmem, x_frag)
        acc = x_frag.load().to(cutlass.Float32)

        # Accumulate remaining M-1 rows
        for m in range(1, M):
            xm_gmem = gXv[(None, (b, m, ng))]
            cute.autovec_copy(xm_gmem, x_frag)
            acc = acc + x_frag.load().to(cutlass.Float32)

        # Compute mean
        y_vec_f32 = acc * scale

        # Store to output
        y_gmem = gYv[(None, (b, ng))]
        y_frag = cute.make_fragment_like(y_gmem, gYv.element_type)
        y_frag.store(y_vec_f32.to(gYv.element_type))
        cute.autovec_copy(y_frag, y_gmem)


# ---------------------------------------------------------------------------
# Host wrapper for dim=1 mean: tiles along N with V-wide vectors
# ---------------------------------------------------------------------------
@cute.jit
def _mean_dim1_vec_host(
    mX: cute.Tensor,               # (B, M, N)
    mY: cute.Tensor,               # (B, N)
    V : cutlass.Constexpr,         # vector width (compile-time)
):
    B, M, N = mX.shape

    # Vectorize along contiguous N
    gXv = cute.zipped_divide(mX, (1, 1, V))   # ((1,1,V), (B, M, N/V))
    gYv = cute.zipped_divide(mY, (1, V))      # ((1,V),   (B,    N/V))

    threads_per_block = 256
    N_groups = N // V

    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = B

    scale = cutlass.Float32(1.0 / float(M))

    _mean_dim1_vec_kernel(gXv, gYv, scale).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch front-end
# ---------------------------------------------------------------------------
class ModelNew(nn.Module):
    """
    CuTe-accelerated mean reduction over dim=1 for 3D tensors (B, M, N) -> (B, N).
    Vectorizes along N with 128-bit loads/stores when possible.
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim == 1, "This optimized implementation supports dim=1 for (B, M, N) inputs."
        self.dim = dim
        self._cache = {}

    @staticmethod
    def _pick_vec_width(dtype: torch.dtype, N: int) -> int:
        # Prefer 128-bit transactions
        if dtype in (torch.float16, torch.bfloat16):
            V = 8   # 8 x 16-bit = 128-bit
        elif dtype == torch.float32:
            V = 4   # 4 x 32-bit = 128-bit
        else:
            V = 2   # conservative fallback
        while V > 1 and (N % V):
            V //= 2
        return V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect 3-D input: (B, M, N); reduce over dim=1 -> (B, N)
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        assert x.dim() == 3 and self.dim == 1, "Expected 3-D (B,M,N) with dim=1"

        B, M, N = x.shape
        V = self._pick_vec_width(x.dtype, N)

        y = torch.empty((B, N), dtype=x.dtype, device=x.device)

        # Wrap for CuTe (row-major, dynamic leading dims)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (x.dtype, V)
        if key not in self._cache:
            self._cache[key] = cute.compile(_mean_dim1_vec_host, mX, mY, V)

        # Launch compiled callable
        self._cache[key](mX, mY)
        return y


# Suggested harness matching the original snippet
batch_size = 128
dim1 = 4096
dim2 = 4095

def get_inputs():
    x = torch.rand(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]