# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda().to(torch.float16)
        M, N = x.shape
        y = torch.empty_like(x)
        tk_kernels.dispatch_micro(x, y, int(M), int(N))
        return y