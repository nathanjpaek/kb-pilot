import torch
import torch.nn as nn
import tk_kernels

class ModelNew(nn.Module):
    """
    ThunderKittens-accelerated square matrix multiplication (C = A @ B)
    for KernelBench Level 1 Problem 1.
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, "Inner dimensions must match for matmul"

        C = torch.zeros((M, N), device=A.device, dtype=torch.float16).contiguous()

        # Call into TK pybind wrapper; bindings expect (A, B, C, M, K, N)
        tk_kernels.dispatch_micro(A, B, C, int(M), int(K), int(N))

        return C