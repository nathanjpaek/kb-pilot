import torch
import torch.nn as nn


class AdaFRN(nn.Module):

    def __init__(self, style_dim, num_features, eps=1e-05):
        super(AdaFRN, self).__init__()
        self.tau = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.fc = nn.Linear(style_dim, num_features * 2)
        self.eps = eps

    def forward(self, x, s):
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=[2, 3], keepdim=True) +
            self.eps)
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, _beta = torch.chunk(h, chunks=2, dim=1)
        out = (1 + gamma) * x
        return torch.max(out, self.tau)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'style_dim': 4, 'num_features': 4}]
