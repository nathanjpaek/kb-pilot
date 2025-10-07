# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    """
    ThunderKittens-accelerated batched matmul: C = A @ B  (per batch)
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        A: (batch, M, K)
        B: (batch, K, N)
        """
        assert A.dim() == 3 and B.dim() == 3, "Expect 3-D tensors"
        batch, M, K = A.shape
        b2, Kb, N = B.shape
        assert batch == b2 and K == Kb, "Shape mismatch"

        A = A.contiguous().cuda().to(torch.float16)
        B = B.contiguous().cuda().to(torch.float16)
        C = torch.zeros((batch, M, N),
                        dtype=torch.float16,
                        device=A.device).contiguous()

        tk_kernels.dispatch_micro(A, B, C,
                                  int(batch), int(M), int(K), int(N))
        return C