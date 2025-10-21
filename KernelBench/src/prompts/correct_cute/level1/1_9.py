"""
Problem Name: 9_Tall_skinny_matrix_multiplication_
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=15.6 runtime_stats={'mean': 15.6, 'std': 0.00542, 'min': 15.6, 'max': 15.6, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.62, 'std': 0.0156, 'min': 2.59, 'max': 2.63, 'num_trials': 100}, 'speedup_ratio': 0.168}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

# -----------------------------------------------------------------------------
# Device kernel: each thread computes V adjacent output elements (along N)
# -----------------------------------------------------------------------------
@cute.kernel
def _tall_skinny_matmul_vecN_kernel(
    gA: cute.Tensor,     # (M, K)
    gBv: cute.Tensor,    # ((1,V), (K, N/V))
    gCv: cute.Tensor,    # ((1,V), (M, N/V))
    K:  cutlass.Int32,   # reduction size
):
    # CUDA indices
    tidx, _, _   = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _  = cute.arch.block_dim()

    # Map this thread to (row m, vector-group ng)
    m  = bidy
    ng = bidx * bdimx + tidx

    # From tiled shapes: gCv is ((1,V), (M, N_groups))
    M         = gCv.shape[1][0]
    N_groups  = gCv.shape[1][1]

    if m < M and ng < N_groups:
        # Output slice for this thread: static shape (1,V)
        c_out = gCv[(None, (m, ng))]

        # Register fragments
        b_frag       = cute.make_fragment_like(c_out, gA.element_type)
        acc_frag_f32 = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_frag_f32.fill(0.0)
        acc = acc_frag_f32.load()

        # K-loop reduction
        for k in range(K):
            a_val = cutlass.Float32(gA[m, k])

            b_vec_gmem = gBv[(None, (k, ng))]
            cute.autovec_copy(b_vec_gmem, b_frag)          # gmem -> regs
            b_vec = b_frag.load().to(cutlass.Float32)      # to FP32

            acc = acc + a_val * b_vec

        # Write back in original dtype
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# -----------------------------------------------------------------------------
# Host JIT wrapper: tiles along N, configures launch, and calls the kernel
# -----------------------------------------------------------------------------
@cute.jit
def _tall_skinny_matmul_vecN_host(
    mA: cute.Tensor,   # (M, K)
    mB: cute.Tensor,   # (K, N)
    mC: cute.Tensor,   # (M, N)
    V : cutlass.Constexpr,   # vector width
    K : cutlass.Constexpr,   # reduction size
):
    M = mA.shape[0]
    N = mB.shape[1]

    # Tile B and C along contiguous N dimension
    gBv = cute.zipped_divide(mB, (1, V))   # ((1,V), (K, N/V))
    gCv = cute.zipped_divide(mC, (1, V))   # ((1,V), (M, N/V))

    threads_per_block = 256
    N_groups = N // V

    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M

    _tall_skinny_matmul_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# -----------------------------------------------------------------------------
# PyTorch-facing module
# -----------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe vectorized GEMM for tall-and-skinny cases:
      C = A @ B, with A: (M,K), B: (K,N), C: (M,N).
    Vectorizes along N and parallelizes rows across the grid.
    """

    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, N: int) -> int:
        # Prefer 128-bit vector width
        if dtype in (torch.float16, torch.bfloat16):
            V = 8  # 8x16-bit
        else:
            V = 4  # 4x32-bit
        while V > 1 and (N % V != 0):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2D"
        assert A.shape[1] == B.shape[0], f"Incompatible shapes: A{A.shape} @ B{B.shape}"

        M, K = A.shape
        _, N = B.shape

        V = self._pick_vector_width(A.dtype, N)

        # Output tensor
        C = torch.empty((M, N), dtype=A.dtype, device=A.device)

        # Wrap PyTorch tensors into CuTe tensors (row-major, dynamic leading dim)
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (A.dtype, V, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _tall_skinny_matmul_vecN_host, mA, mB, mC, V, K
            )

        # Launch compiled kernel
        self._cache[key](mA, mB, mC)
        return C


# -----------------------------------------------------------------------------
# Convenience sizes and input helpers (optional)
# -----------------------------------------------------------------------------
M = 16384 * 2
N = 16 * 2

def get_inputs():
    # Example tall-skinny inner dimension: A (M,N), B (N,M) -> C (M,M)
    A = torch.rand(M, N)
    B = torch.rand(N, M)
    return [A, B]

def get_init_inputs():
    return []