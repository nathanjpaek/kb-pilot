"""
Problem Name: 90_cumprod
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=20.2 runtime_stats={'mean': 20.2, 'std': 0.146, 'min': 19.8, 'max': 20.5, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 4.66, 'std': 0.00714, 'min': 4.65, 'max': 4.72, 'num_trials': 100}, 'speedup_ratio': 0.231}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _cumprod_dim1_kernel(
    gX: cute.Tensor,    # (B, N)
    gY: cute.Tensor,    # (B, N)
):
    # CUDA indices
    tidx, _, _   = cute.arch.thread_idx()
    bidx, _, _   = cute.arch.block_idx()
    bdimx, _, _  = cute.arch.block_dim()

    row = bidx * bdimx + tidx

    B = gX.shape[0]
    N = gX.shape[1]

    if row < B:
        # Accumulate in FP32 for stability
        prod = cutlass.Float32(1.0)
        for i in range(N):
            x_val = cutlass.Float32(gX[row, i])
            prod = prod * x_val
            # Cast back to destination dtype on store
            gY[row, i] = gY.element_type(prod)


@cute.jit
def _cumprod_dim1_host(
    mX: cute.Tensor,     # (B, N)
    mY: cute.Tensor,     # (B, N)
):
    B = mX.shape[0]

    threads_per_block = 256
    grid_x = cute.ceil_div(B, threads_per_block)

    _cumprod_dim1_kernel(mX, mY).launch(
        grid  = (grid_x, 1, 1),
        block = (threads_per_block, 1, 1),
    )


class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated cumulative product along dim=1 for 2D tensors (B, N).
    """
    def __init__(self, dim: int):
        super().__init__()
        assert dim == 1, "This implementation supports cumprod along dim=1."
        self.dim = dim
        self._cache = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect 2D input and reduction along dim=1
        assert x.dim() == 2, "Expected a 2-D tensor of shape (B, N)."

        # Ensure CUDA and contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        B, N = x.shape

        # Output tensor
        y = torch.empty_like(x)

        # Wrap tensors for CuTe (row-major with dynamic leading dims)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        # Compile once per dtype (shapes are dynamic)
        if self._cache is None:
            self._cache = cute.compile(_cumprod_dim1_host, mX, mY)

        # Launch compiled kernel
        self._cache(mX, mY)
        return y


# Example harness matching the provided sample
batch_size = 32768
input_shape = (32768,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]