import torch
import torch.nn as nn


class Bandpass(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.mean = nn.Parameter(torch.randn(1, input_dim, dtype=torch.float32)
            )
        self.icov = nn.Parameter(torch.eye(input_dim, input_dim, dtype=
            torch.float32) * 2)
        self.a = nn.Parameter(torch.tensor([2], dtype=torch.float32))

    def forward(self, x):
        self.a.data = torch.clamp(self.a.data, 0.01, 100)
        x = x - self.mean
        xm = torch.matmul(x.unsqueeze(1), self.icov)
        xm = torch.matmul(xm, x.unsqueeze(2)).squeeze(1)
        xm = torch.abs(xm)
        xm = torch.exp(-xm ** self.a)
        return xm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
