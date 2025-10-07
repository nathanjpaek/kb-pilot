# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    """
    ThunderKittens accelerated single matmul: C = A @ B
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, "Inner dimensions must match"

        C = torch.zeros((M, N), dtype=torch.float16, device=A.device).contiguous()

        # Dispatch the ThunderKittens kernel
        tk_kernels.dispatch_micro(A, B, C, int(M), int(K), int(N))

        return C