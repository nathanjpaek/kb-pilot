"""
Problem Name: 10_3D_tensor_matrix_multiplication
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=6.1 runtime_stats={'mean': 6.1, 'std': 0.0389, 'min': 5.97, 'max': 6.19, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.05, 'std': 0.00167, 'min': 1.05, 'max': 1.06, 'num_trials': 100}, 'speedup_ratio': 0.172}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
#                          Device Kernel
# ---------------------------------------------------------------------------
@cute.kernel
def _batched_matmul_vecL_kernel(
    gA: cute.Tensor,     # (N, M, K)  – unsliced
    gBv: cute.Tensor,    # ((1,V), (K, L/V))
    gCv: cute.Tensor,    # ((1,1,V), (N, M, L/V))
    K:  cutlass.Int32,
):
    # Thread & block indices -------------------------------------------------
    tidx, _, _      = cute.arch.thread_idx()
    bidx, bidy, bidz = cute.arch.block_idx()   # x, y, z
    bdimx, _, _     = cute.arch.block_dim()

    # Logical coordinates ----------------------------------------------------
    n  = bidz               # batch index
    mi = bidy               # row index
    lg = bidx * bdimx + tidx  # group of V columns (L_group)

    # Shapes after tiling ----------------------------------------------------
    N  = gCv.shape[1][0]
    M  = gCv.shape[1][1]
    Lg = gCv.shape[1][2]     # L_groups = L / V

    if (n < N) and (mi < M) and (lg < Lg):
        # -------------------------------------------------------------------
        # Slice out 1×V vector of C for this (n, m, lg)
        # gCv layout: ((1,1,V), (N, M, Lg))
        c_vec_gmem = gCv[(None, (n, mi, lg))]  # static shape (1,1,V)

        # Register fragments -------------------------------------------------
        b_frag = cute.make_fragment_like(c_vec_gmem, gA.element_type)

        acc_f32_frag = cute.make_fragment_like(c_vec_gmem, cutlass.Float32)
        acc_f32_frag.fill(0.0)
        acc = acc_f32_frag.load()              # TensorSSA FP32 accumulator

        # -------------------------------------------------------------------
        # Reduction over K
        for k in range(K):
            a_scalar  = cutlass.Float32(gA[n, mi, k])  # promote to FP32

            b_vec_gmem = gBv[(None, (k, lg))]          # ((1,V))
            cute.autovec_copy(b_vec_gmem, b_frag)      # load to registers
            b_vec = b_frag.load().to(cutlass.Float32)  # FP32 TensorSSA

            acc = acc + a_scalar * b_vec               # FMA in registers

        # -------------------------------------------------------------------
        # Write back
        out_frag = cute.make_fragment_like(c_vec_gmem, gA.element_type)
        out_frag.store(acc.to(gA.element_type))        # convert back
        cute.autovec_copy(out_frag, c_vec_gmem)        # store to gmem


# ---------------------------------------------------------------------------
#                          Host (JIT) Function
# ---------------------------------------------------------------------------
@cute.jit
def _batched_matmul_vecL_host(
    mA: cute.Tensor,    # (N,M,K)
    mB: cute.Tensor,    # (K,L)
    mC: cute.Tensor,    # (N,M,L)
    V:  cutlass.Constexpr,
    K:  cutlass.Constexpr,
):
    """
    Host configuration for batched matmul with vectorisation along L.
    V : vector width (compile-time)
    K : reduction size  (compile-time)
    """
    N, M, _ = mA.shape
    _, L    = mB.shape

    # 1) Tile B and C along the last (L) dimension --------------------------
    gBv = cute.zipped_divide(mB, (1, V))      # ((1,V), (K, L/V))
    gCv = cute.zipped_divide(mC, (1, 1, V))   # ((1,1,V), (N, M, L/V))

    # 2) Launch configuration ----------------------------------------------
    threads_per_block = 256
    L_groups          = L // V

    grid_x = cute.ceil_div(L_groups, threads_per_block)  # columns
    grid_y = M                                           # rows
    grid_z = N                                           # batches

    _batched_matmul_vecL_kernel(
        mA, gBv, gCv, cutlass.Int32(K)
    ).launch(
        grid  = (grid_x, grid_y, grid_z),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------------
#                          PyTorch Module
# ---------------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe accelerated 3-D batched matmul:
       C[n, m, :] = A[n, m, :] @ B
    Vectorises along L (columns) and parallelises N & M in the grid.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    # ---- vector width picker ----------------------------------------------
    def _pick_vector_width(self, dtype: torch.dtype, L: int) -> int:
        # 128-bit loads/stores
        if dtype in (torch.float16, torch.bfloat16):
            V = 8         # 8 × 16-bit = 128 bits
        elif dtype == torch.float32:
            V = 4         # 4 × 32-bit = 128 bits
        else:
            V = 4

        while V > 1 and (L % V != 0):
            V //= 2
        return V

    # ---- forward -----------------------------------------------------------
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA, contiguous -------------------------------------------------
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 3 and B.dim() == 2, "Shapes must be (N,M,K) & (K,L)"
        N, M, K = A.shape
        assert B.shape[0] == K, "Incompatible shapes"
        L = B.shape[1]

        V = self._pick_vector_width(A.dtype, L)

        # Output tensor ----------------------------------------------------------
        C = torch.zeros((N, M, L), dtype=A.dtype, device=A.device)

        # Wrap tensors for CuTe --------------------------------------------------
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2)
        )

        # Compile / cache per (dtype, V, K) -------------------------------------
        key = (A.dtype, V, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(
                _batched_matmul_vecL_host, mA, mB, mC, V, K
            )

        # Launch -----------------------------------------------------------------
        self._cache[key](mA, mB, mC)

        return C