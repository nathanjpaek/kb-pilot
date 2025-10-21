"""
Problem Name: 95_CrossEntropyLoss
Generated using DSPy RAG with openai/gpt-5
RAG Examples: 7
Evaluation Result:
compiled=True correctness=True metadata={'hardware': 'NVIDIA H100 80GB HBM3', 'device': '0', 'correctness_trials': '(5 / 5)'} runtime=0.476 runtime_stats={'mean': 0.476, 'std': 0.0036, 'min': 0.469, 'max': 0.496, 'num_trials': 100, 'performance_comparison': {'original_pytorch_stats': {'mean': 1.46, 'std': 0.00344, 'min': 1.45, 'max': 1.47, 'num_trials': 100}, 'speedup_ratio': 3.07}}
"""

import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def _nll_gather_kernel(
    gLogP: cute.Tensor,   # (B, C)  log-probabilities
    gT:    cute.Tensor,   # (B,)    int32 class indices
    gLoss: cute.Tensor,   # (B,)    per-sample NLL
):
    # CUDA indices
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdimx, _, _ = cute.arch.block_dim()

    row = bidx * bdimx + tidx
    B = gLogP.shape[0]

    if row < B:
        cls = gT[row]                 # runtime class index (int32)
        val = gLogP[row, cls]         # scalar log-probability at target
        gLoss[row] = -val             # write negative log-likelihood


@cute.jit
def _nll_gather_host(
    mLogP: cute.Tensor,   # (B, C)
    mT:    cute.Tensor,   # (B,)
    mLoss: cute.Tensor,   # (B,)
):
    B = mLogP.shape[0]

    threads_per_block = 256
    grid_x = cute.ceil_div(B, threads_per_block)

    _nll_gather_kernel(mLogP, mT, mLoss).launch(
        grid=(grid_x, 1, 1),
        block=(threads_per_block, 1, 1),
    )


class ModelNew(torch.nn.Module):
    """
    CuTe-accelerated Cross Entropy:
      loss = mean( -log_softmax(predictions, dim=1)[range(B), targets] )
    Uses PyTorch for log_softmax, CuTe kernel for indexed gather and NLL writeback.
    """
    def __init__(self):
        super().__init__()
        self._cache = {}

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Ensure CUDA + contiguous
        X = predictions.contiguous().cuda() if not predictions.is_cuda else predictions.contiguous()
        T = targets.to(torch.int32).contiguous().cuda() if not targets.is_cuda else targets.to(torch.int32).contiguous()

        assert X.dim() == 2 and T.dim() == 1, "Shapes must be (B, C) and (B,)"
        B, C = X.shape
        assert T.shape[0] == B, "Targets length must match batch size"

        # Host: compute numerically stable log-softmax
        logP = torch.nn.functional.log_softmax(X, dim=1)

        # Output per-sample losses
        loss_vec = torch.empty(B, dtype=logP.dtype, device=logP.device)

        # Wrap as CuTe tensors (row-major / compact)
        mLogP = from_dlpack(logP,   assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1))
        mT    = from_dlpack(T,      assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))
        mLoss = from_dlpack(loss_vec, assumed_align=16).mark_compact_shape_dynamic(mode=0, stride_order=(0,))

        # Compile/cache by dtype
        key = (logP.dtype,)
        if key not in self._cache:
            self._cache[key] = cute.compile(_nll_gather_host, mLogP, mT, mLoss)

        # Launch kernel
        self._cache[key](mLogP, mT, mLoss)

        # Mean reduction to match F.cross_entropy default reduction='mean'
        return loss_vec.mean()


# Harness parity with the original snippet
batch_size = 32768
num_classes = 4096
input_shape = (num_classes,)
dim = 1

def get_inputs():
    return [torch.rand(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []