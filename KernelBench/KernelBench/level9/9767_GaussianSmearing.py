import torch
import torch.nn as nn


class GaussianSmearing(nn.Module):

    def __init__(self, in_features, start=0, end=1, num_freqs=50):
        super(GaussianSmearing, self).__init__()
        self.num_freqs = num_freqs
        offset = torch.linspace(start, end, num_freqs)
        self.coeff = -0.5 / (offset[1] - offset[0]).item() ** 2
        self.offset = nn.Parameter(offset.view(-1, 1).repeat(1, in_features
            ).view(1, -1), requires_grad=False)

    def forward(self, x):
        x = x.repeat(1, self.num_freqs)
        x = x - self.offset
        return torch.exp(self.coeff * torch.pow(x, 2))


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4}]
