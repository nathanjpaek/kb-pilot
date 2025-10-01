import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.parameter import Parameter


class MeanReweightLayer(nn.Module):
    """Renamed to Attention-Bias (AB) layer in paper"""

    def __init__(self, channel):
        super(MeanReweightLayer, self).__init__()
        self.cfc = Parameter(torch.Tensor(channel))
        self.cfc.data.fill_(0)

    def forward(self, x):
        avg_y = torch.mean(x, dim=(2, 3), keepdim=True)
        avg_y = avg_y * self.cfc[None, :, None, None]
        return x + avg_y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channel': 4}]
