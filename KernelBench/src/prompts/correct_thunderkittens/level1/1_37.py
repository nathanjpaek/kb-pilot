# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda().to(torch.float16)
        norm_val = torch.norm(x, p='fro').float().item()

        y = torch.empty_like(x)

        m = int(x.shape[0])
        n = int(x.shape[1]) if x.dim() > 1 else 1

        tk_kernels.dispatch_micro(
            x,
            y,
            float(norm_val),
            m,
            n
        )
        return y