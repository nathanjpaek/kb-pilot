# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    """
    ThunderKittens-accelerated matrix-scalar multiply: C = A * s
    """

    def __init__(self):
        super().__init__()

    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        assert A.dim() == 2, "A must be 2-D"
        M, N = A.shape

        A = A.contiguous().cuda().to(torch.float16)
        C = torch.zeros_like(A)

        # Call the TK dispatcher: (A, C, s, M, N)
        tk_kernels.dispatch_micro(A, C, float(s), int(M), int(N))

        return C