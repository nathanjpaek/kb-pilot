# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim  # 0 (rows) or 1 (cols)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16).contiguous().cuda()
        y = torch.empty_like(x)

        M = int(x.shape[0])
        N = int(x.shape[1]) if x.dim() > 1 else 1

        tk_kernels.dispatch_micro(x, y, M, N, int(self.dim))
        return y