# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M, K = A.shape
        N = B.shape[1]
        A = A.contiguous().to(torch.float16).cuda()
        B = B.contiguous().to(torch.float16).cuda()
        C = torch.zeros((M, N), dtype=torch.float16, device=A.device).contiguous()
        tk_kernels.dispatch_micro(A, B, C, int(M), int(K), int(N))
        return C