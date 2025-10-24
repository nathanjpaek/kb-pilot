# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().to(torch.float16).cuda()
        mask = mask.contiguous().to(torch.float16).cuda()
        M, N = x.shape
        y = torch.empty_like(x)
        tk_kernels.dispatch_micro(x, mask, y, int(M), int(N))
        return y