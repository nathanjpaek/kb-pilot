import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class SpatialSoftmax(nn.Module):

    def __init__(self, temperature=1, device='cpu'):
        super(SpatialSoftmax, self).__init__()
        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0

    def forward(self, feature):
        feature = feature.view(feature.shape[0], -1, feature.shape[1] *
            feature.shape[2])
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        return softmax_attention


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
