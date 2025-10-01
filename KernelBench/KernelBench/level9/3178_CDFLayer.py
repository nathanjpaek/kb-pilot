import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class CDFLayer(nn.Module):

    def __init__(self, device='cpu'):
        super(CDFLayer, self).__init__()
        self.loc_scale = Parameter(torch.FloatTensor([0.0, 1.0]))

    def forward(self, x, dim=1):
        m = torch.distributions.Cauchy(self.loc_scale[0], self.loc_scale[1])
        return m.cdf(torch.cumsum(x, dim))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
