"""
Problem Name: 46_Average_Pooling_3D
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=5.89 runtime_stats={'mean': 5.89, 'std': 0.00623, 'min': 5.88, 'max': 5.91, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 8.67, 'std': 0.0158, 'min': 8.66, 'max': 8.81, 'num_trials': 100}, 'speedup_ratio': 1.47}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Device kernel: scalar-per-thread 3D AvgPool with padding and stride
# count_include_pad=True (denominator = K^3)
# ---------------------------------------------------------------------------
@cute.kernel
def _avgpool3d_scalar_kernel(
    gX: cute.Tensor,             # (N, C, D, H, W)
    gY: cute.Tensor,             # (N, C, Do, Ho, Wo)
    K:  cutlass.Int32,           # kernel size (cubic)
    S:  cutlass.Int32,           # stride (equal in D,H,W)
    P:  cutlass.Int32,           # padding (equal in D,H,W)
):
    # CUDA indices
    tidx, _, _        = cute.arch.thread_idx()
    bidx, bidy, bidz  = cute.arch.block_idx()
    bdimx, _, _       = cute.arch.block_dim()

    # Shapes
    N  = gY.shape[0]
    C  = gY.shape[1]
    Do = gY.shape[2]
    Ho = gY.shape[3]
    Wo = gY.shape[4]

    D  = gX.shape[2]
    H  = gX.shape[3]
    W  = gX.shape[4]

    # Map blocks/threads to output coordinates
    wo = bidx * bdimx + tidx   # along Wo
    ho = bidy                  # along Ho

    # Flatten (N, C, Do) across grid.z
    NCDo = C * Do
    n  = bidz // NCDo
    r  = bidz - n * NCDo
    c  = r // Do
    do = r - c * Do

    if n < N and c < C and do < Do and ho < Ho and wo < Wo:
        # Top-left-front corner in input for this output position
        di0 = do * S - P
        hi0 = ho * S - P
        wi0 = wo * S - P

        acc = cutlass.Float32(0.0)

        # Sum over the KxKxK window with zero-padding
        for kd in range(K):
            di = di0 + kd
            if (di >= 0) and (di < D):
                for kh in range(K):
                    hi = hi0 + kh
                    if (hi >= 0) and (hi < H):
                        for kw in range(K):
                            wi = wi0 + kw
                            if (wi >= 0) and (wi < W):
                                acc = acc + cutlass.Float32(gX[n, c, di, hi, wi])

        # Average, count_include_pad=True => denominator is K^3
        inv_count = cutlass.Float32(1.0 / (K * K * K))
        y_val = acc * inv_count

        # Write back in input/output dtype
        gY[n, c, do, ho, wo] = y_val.to(gX.element_type)


# ---------------------------------------------------------------------------
# Host JIT: computes output shape, configures launch, and calls the kernel
# ---------------------------------------------------------------------------
@cute.jit
def _avgpool3d_host(
    mX: cute.Tensor,          # (N, C, D, H, W)
    mY: cute.Tensor,          # (N, C, Do, Ho, Wo)
    K:  cutlass.Constexpr,    # compile-time kernel size (cubic)
    S:  cutlass.Constexpr,    # compile-time stride
    P:  cutlass.Constexpr,    # compile-time padding
):
    N, C, D, H, W = mX.shape
    Do, Ho, Wo    = mY.shape[2], mY.shape[3], mY.shape[4]

    threads_per_block = 256
    grid_x = cute.ceil_div(Wo, threads_per_block)
    grid_y = Ho
    grid_z = N * C * Do

    _avgpool3d_scalar_kernel(mX, mY, cutlass.Int32(K), cutlass.Int32(S), cutlass.Int32(P)).launch(
        grid  = (grid_x, grid_y, grid_z),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
# PyTorch-facing module
# ---------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated AvgPool3d over NCDHW tensors.
    Assumes count_include_pad=True (PyTorch default).
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride) if stride is not None else int(kernel_size)
        self.padding = int(padding)
        self._cache = {}

    @staticmethod
    def _out_dim(in_size: int, k: int, s: int, p: int) -> int:
        # PyTorch AvgPool3d (ceil_mode=False)
        return (in_size + 2 * p - k) // s + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA & contiguous
        x = x.contiguous().cuda() if not x.is_cuda else x.contiguous()
        assert x.dim() == 5, "Expected input of shape (N, C, D, H, W)"

        N, C, D, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding

        Do = self._out_dim(D, K, S, P)
        Ho = self._out_dim(H, K, S, P)
        Wo = self._out_dim(W, K, S, P)

        y = torch.empty((N, C, Do, Ho, Wo), dtype=x.dtype, device=x.device)

        # Wrap tensors for CuTe (row-major NCDHW)
        mX = from_dlpack(x, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2, 3, 4)
        )
        mY = from_dlpack(y, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2, 3, 4)
        )

        key = (x.dtype, K, S, P)
        if key not in self._cache:
            self._cache[key] = cute.compile(_avgpool3d_host, mX, mY, K, S, P)

        self._cache[key](mX, mY)
        return y


# ---------------------------------------------------------------------------
# Harness helpers aligned with the original snippet
# ---------------------------------------------------------------------------
batch_size = 16
channels = 32
depth = 128
height = 128
width = 256
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    x = torch.rand(batch_size, channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]