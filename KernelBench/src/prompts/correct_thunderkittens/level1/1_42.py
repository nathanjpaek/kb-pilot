# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.dilation = int(dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous().cuda().to(torch.float16)
        B, C, H, W = x.shape
        K = self.kernel_size
        S = self.stride
        P = self.padding
        D = self.dilation
        H_out = (H + 2 * P - D * (K - 1) - 1) // S + 1
        W_out = (W + 2 * P - D * (K - 1) - 1) // S + 1
        y = torch.empty((B, C, H_out, W_out), dtype=x.dtype, device=x.device)
        tk_kernels.dispatch_micro(
            x, y,
            int(B), int(C), int(H), int(W),
            int(K), int(S), int(P), int(D)
        )
        return y