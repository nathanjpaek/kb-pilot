import torch
from torch import nn


class FRN(nn.Module):

    def __init__(self, num_features, eps=1e-06):
        super(FRN, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.t = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, x):
        miu2 = torch.pow(x, 2).mean(dim=(2, 3), keepdim=True)
        x = x * torch.rsqrt(miu2 + self.eps)
        return torch.max(self.gamma * x + self.beta, self.t)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
