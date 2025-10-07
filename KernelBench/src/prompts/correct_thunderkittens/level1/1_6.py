# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    """
    ThunderKittens-accelerated large-K matrix multiplication:  C = A @ B
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: (M, K)
        B: (K, N)
        """
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, "Inner dimensions must match"

        A = A.contiguous().cuda().to(torch.float16)
        B = B.contiguous().cuda().to(torch.float16)
        C = torch.zeros((M, N), dtype=torch.float16,
                        device=A.device).contiguous()

        tk_kernels.dispatch_micro(A, B, C, int(M), int(K), int(N))
        return C