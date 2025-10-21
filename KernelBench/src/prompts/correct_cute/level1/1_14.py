"""
Problem Name: 14_Matmul_for_upper_triangular_matrices
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=16.3 runtime_stats={'mean': 16.3, 'std': 0.0119, 'min': 16.2, 'max': 16.3, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.72, 'std': 0.00492, 'min': 2.71, 'max': 2.74, 'num_trials': 100}, 'speedup_ratio': 0.167}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# -----------------------------------------------------------------------------
# Device kernel: vectorized GEMM along N
# -----------------------------------------------------------------------------
@cute.kernel
def _utri_matmul_vecN_kernel(
    gA: cute.Tensor,     # (N, N)
    gBv: cute.Tensor,    # ((1,V), (N, N/V))
    gCv: cute.Tensor,    # ((1,V), (N, N/V))
    K:  cutlass.Int32,   # reduction dimension (= N)
):
    # CUDA indices
    tidx, _, _   = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    bdimx, _, _  = cute.arch.block_dim()

    # Output row and N-vector group this thread handles
    mi = bidy
    ng = bidx * bdimx + tidx

    N_rows   = gCv.shape[1][0]
    N_groups = gCv.shape[1][1]

    if mi < N_rows and ng < N_groups:
        # Output slice (static shape (1,V))
        c_out = gCv[(None, (mi, ng))]

        # Register fragments
        b_frag = cute.make_fragment_like(c_out, gA.element_type)

        acc_f32 = cute.make_fragment_like(c_out, cutlass.Float32)
        acc_f32.fill(0.0)
        acc = acc_f32.load()

        # K loop
        for k in range(K):
            a_val = cutlass.Float32(gA[mi, k])

            b_vec_gmem = gBv[(None, (k, ng))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec = b_frag.load().to(cutlass.Float32)

            acc = acc + a_val * b_vec

        # Store back in original dtype
        out_frag = cute.make_fragment_like(c_out, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_out)


# -----------------------------------------------------------------------------
# Host JIT wrapper: tiles along N and launches the kernel
# -----------------------------------------------------------------------------
@cute.jit
def _utri_matmul_vecN_host(
    mA: cute.Tensor,    # (N, N)
    mB: cute.Tensor,    # (N, N)
    mC: cute.Tensor,    # (N, N)
    V : cutlass.Constexpr,
    K : cutlass.Constexpr,   # = N
):
    N = mA.shape[0]

    # Tile B and C along column dimension
    gBv = cute.zipped_divide(mB, (1, V))   # ((1,V), (N, N/V))
    gCv = cute.zipped_divide(mC, (1, V))   # ((1,V), (N, N/V))

    threads_per_block = 256
    N_groups = N // V

    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = N

    _utri_matmul_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid  = (grid_x, grid_y, 1),
        block = (threads_per_block, 1, 1),
    )


# -----------------------------------------------------------------------------
# PyTorch-facing module
# -----------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated implementation of C = triu(A @ B) for upper-triangular (N,N) inputs.
    Vectorizes along N (columns) for coalesced memory and 128-bit transactions.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, N: int) -> int:
        # Prefer 128-bit gmem transactions
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (N % V):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA and contiguous
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2-D (N,N)"
        N, N2 = A.shape
        assert B.shape == (N, N2) and N == N2, "Inputs must be square and same size"

        V = self._pick_vector_width(A.dtype, N)

        # Output buffer
        C = torch.empty((N, N), dtype=A.dtype, device=A.device)

        # Wrap into CuTe tensors (row-major)
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        key = (A.dtype, V, N)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _utri_matmul_vecN_host, mA, mB, mC, V, N
            )

        # Launch matmul kernel
        self._cache[key](mA, mB, mC)

        # Upper-triangularize to match torch.triu(A @ B)
        C = torch.triu(C)
        return C


# -----------------------------------------------------------------------------
# Convenience for harness parity
# -----------------------------------------------------------------------------
N = 4096

def get_inputs():
    A = torch.triu(torch.rand(N, N))
    B = torch.triu(torch.rand(N, N))
    return [A, B]

def get_init_inputs():
    return []