import torch
from torch import nn


class GaussianSmearing(nn.Module):

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50,
        trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable
        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter('coeff', nn.Parameter(coeff))
            self.register_parameter('offset', nn.Parameter(offset))
        else:
            self.register_buffer('coeff', coeff)
            self.register_buffer('offset', offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.
            num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
