# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    """
    ThunderKittens-accelerated matrix-vector multiply:  C = A @ B
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: (M, K)
        B: (K, 1)
        returns C: (M, 1)
        """
        assert A.dim() == 2 and B.dim() == 2
        M, K = A.shape
        Kb, one = B.shape
        assert K == Kb and one == 1, "B must be (K,1)"

        A = A.contiguous().cuda().to(torch.float16)
        B = B.contiguous().cuda().to(torch.float16)
        C = torch.zeros((M, 1), dtype=torch.float16,
                        device=A.device).contiguous()

        tk_kernels.dispatch_matvec(A, B, C,
                                  int(M), int(K))
        return C