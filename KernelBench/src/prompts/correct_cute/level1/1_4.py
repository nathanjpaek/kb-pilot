"""
Problem Name: 4_Matrix_vector_multiplication_
Generated using DSPy RAG with openai/o3
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=140.0 runtime_stats={'mean': 140.0, 'std': 0.0436, 'min': 140.0, 'max': 140.0, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 2.8, 'std': 0.00403, 'min': 2.79, 'max': 2.81, 'num_trials': 100}, 'speedup_ratio': 0.02}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------
# Device kernel: one CUDA thread computes one output row
# ---------------------------------------------------------------------
@cute.kernel
def _matvec_kernel(
    gA: cute.Tensor,   # (M, K)
    gB: cute.Tensor,   # (K, 1)
    gC: cute.Tensor,   # (M, 1)
    K:  cutlass.Int32, # reduction length
):
    # CUDA indices
    tidx, _, _  = cute.arch.thread_idx()
    bidx, _, _  = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    # Row this thread is responsible for
    mi = bidx * bdimx + tidx

    # Guard for partial blocks
    M = gA.shape[0]
    if mi < M:
        # Accumulator in FP32
        acc = cutlass.Float32(0.0)

        # Dot product over K
        for k in range(K):
            a_val = cutlass.Float32(gA[mi, k])
            b_val = cutlass.Float32(gB[k, 0])
            acc   = acc + a_val * b_val

        # Store result, converted back to original dtype
        gC[mi, 0] = acc.to(gA.element_type)


# ---------------------------------------------------------------------
# Host function: prepares launch configuration and calls the kernel
# ---------------------------------------------------------------------
@cute.jit
def _matvec_host(
    mA: cute.Tensor,
    mB: cute.Tensor,
    mC: cute.Tensor,
    K:  cutlass.Constexpr,   # compile-time constant
):
    M = mA.shape[0]

    threads_per_block = 256
    grid_x = cute.ceil_div(M, threads_per_block)

    _matvec_kernel(mA, mB, mC, cutlass.Int32(K)).launch(
        grid  = (grid_x, 1, 1),
        block = (threads_per_block, 1, 1),
    )


# ---------------------------------------------------------------------
# PyTorch-facing module
# ---------------------------------------------------------------------
class ModelNew(torch.nn.Module):
    """
    CuTe JIT-compiled GEMV  (C = A @ B)   where
        A : (M, K)
        B : (K, 1)
        C : (M, 1)
    """

    def __init__(self):
        super().__init__()
        self._cache = {}     # keyed by (dtype, K)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA & contiguous
        A = A.contiguous().cuda() if not A.is_cuda else A.contiguous()
        B = B.contiguous().cuda() if not B.is_cuda else B.contiguous()

        assert A.dim() == 2 and B.dim() == 2, "Inputs must be 2-D"
        assert B.shape[1] == 1,                "B must be (K,1)"
        assert A.shape[1] == B.shape[0],       "Incompatible shapes"

        M, K = A.shape
        device = A.device

        # Output tensor
        C = torch.zeros((M, 1), dtype=A.dtype, device=device)

        # Wrap as CuTe tensors (row-major, dynamic M)
        mA = from_dlpack(A, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mB = from_dlpack(B, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )
        mC = from_dlpack(C, assumed_align=16).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        )

        # Compile once per (dtype, K)
        key = (A.dtype, K)
        if key not in self._cache:
            self._cache[key] = cute.compile(_matvec_host, mA, mB, mC, K)

        # Launch
        self._cache[key](mA, mB, mC)

        return C