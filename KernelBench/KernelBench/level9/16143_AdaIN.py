import torch
from torch import nn


class AdaIN(nn.Module):

    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'style_dim': 4, 'num_features': 4}]
