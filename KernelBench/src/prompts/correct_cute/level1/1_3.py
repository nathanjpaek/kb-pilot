"""
Problem Name: 3_Batched_matrix_multiplication
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=48.4 runtime_stats={'mean': 48.4, 'std': 2.79, 'min': 34.7, 'max': 52.1, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 5.33, 'std': 0.00227, 'min': 5.33, 'max': 5.34, 'num_trials': 100}, 'speedup_ratio': 0.11}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _bmm_vecN_kernel(
    gA: cute.Tensor,     # (B, M, K)
    gBv: cute.Tensor,    # ((1,1,V), (B, K, N/V))
    gCv: cute.Tensor,    # ((1,1,V), (B, M, N/V))
    K:  cutlass.Int32,
):
    # CUDA indices
    tidx, _, _        = cute.arch.thread_idx()
    bidx, bidy, bidz  = cute.arch.block_idx()
    bdimx, _, _       = cute.arch.block_dim()

    b  = bidz                              # batch index
    m  = bidy                              # row index in C
    ng = bidx * bdimx + tidx               # group along N (N/V)

    Bsz = gCv.shape[1][0]                  # batch size
    M   = gCv.shape[1][1]
    Ng  = gCv.shape[1][2]                  # N_groups = N/V

    if (b < Bsz) and (m < M) and (ng < Ng):
        # View of V-wide output slice
        c_view = gCv[(None, (b, m, ng))]   # shape (1,1,V)

        # Register fragments
        b_frag   = cute.make_fragment_like(c_view, gA.element_type)
        acc_f32f = cute.make_fragment_like(c_view, cutlass.Float32)
        acc_f32f.fill(0.0)
        acc = acc_f32f.load()

        # Reduction over K
        for k in range(K):
            a_val = cutlass.Float32(gA[b, m, k])

            b_vec_gmem = gBv[(None, (b, k, ng))]
            cute.autovec_copy(b_vec_gmem, b_frag)
            b_vec = b_frag.load().to(cutlass.Float32)

            acc = acc + a_val * b_vec

        # Store result
        out_frag = cute.make_fragment_like(c_view, gA.element_type)
        out_frag.store(acc.to(gA.element_type))
        cute.autovec_copy(out_frag, c_view)


@cute.jit
def _bmm_vecN_host(
    mA: cute.Tensor,     # (B, M, K)
    mB: cute.Tensor,     # (B, K, N)
    mC: cute.Tensor,     # (B, M, N)
    V:  cutlass.Constexpr,
    K:  cutlass.Constexpr,
):
    Bsz = mA.shape[0]
    M   = mA.shape[1]
    N   = mB.shape[2]

    # Tile along N with vector width V
    gBv = cute.zipped_divide(mB, (1, 1, V))   # ((1,1,V), (B, K, N/V))
    gCv = cute.zipped_divide(mC, (1, 1, V))   # ((1,1,V), (B, M, N/V))

    threads_per_block = 256
    N_groups = N // V

    grid_x = cute.ceil_div(N_groups, threads_per_block)
    grid_y = M
    grid_z = Bsz

    _bmm_vecN_kernel(mA, gBv, gCv, cutlass.Int32(K)).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(threads_per_block, 1, 1)
    )


class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated batched matmul:
      C[b, m, :] = A[b, m, :] @ B[b, :, :]
    Vectorizes along N and parallelizes batch and rows.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    @staticmethod
    def _pick_vector_width(dtype: torch.dtype, N: int) -> int:
        # Prefer 128-bit transactions
        V = 8 if dtype in (torch.float16, torch.bfloat16) else 4
        while V > 1 and (N % V):
            V //= 2
        return V

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA + contiguous
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 3 and B.dim() == 3, "Inputs must be 3D"
        Bsz, M, K = A.shape
        Bb, Kb, N = B.shape
        assert Bb == Bsz and Kb == K, "Incompatible shapes for batched matmul"

        V = self._pick_vector_width(A.dtype, N)

        # Output
        C = torch.empty((Bsz, M, N), dtype=A.dtype, device=A.device)

        # Wrap tensors for CuTe (row-major)
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )

        # Compile/cache per (dtype, V, K)
        key = (A.dtype, V, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(_bmm_vecN_host, mA, mB, mC, V, K)

        # Launch
        self._cache[key](mA, mB, mC)
        return C