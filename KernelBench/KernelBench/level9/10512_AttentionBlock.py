import torch
from torch import nn


class AttentionBlock(nn.Module):

    def __init__(self, in_nc, out_nc, nd, bias=False):
        super().__init__()
        self.in_nc = in_nc
        self.Wq = nn.Linear(in_nc, nd, bias=bias)
        self.Wk = nn.Linear(in_nc, nd, bias=bias)
        self.Wv = nn.Linear(in_nc, out_nc, bias=bias)

    def forward(self, x):
        B, N = x.shape
        N = N // self.in_nc
        x = x.view(B * N, self.in_nc)
        Q: 'torch.Tensor' = self.Wq(x).view(B, N, -1)
        K: 'torch.Tensor' = self.Wk(x).view(B, N, -1)
        V: 'torch.Tensor' = self.Wv(x).view(B, N, -1)
        A = Q @ K.transpose(1, 2)
        A = torch.softmax(A, dim=-1)
        out = A @ V
        out = out.view(B, -1)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_nc': 4, 'out_nc': 4, 'nd': 4}]
