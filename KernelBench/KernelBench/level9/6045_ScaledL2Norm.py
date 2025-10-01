import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledL2Norm(nn.Module):

    def __init__(self, in_channels, initial_scale):
        super(ScaledL2Norm, self).__init__()
        self.in_channels = in_channels
        self.scale = nn.Parameter(torch.Tensor(in_channels))
        self.initial_scale = initial_scale
        self.reset_parameters()

    def forward(self, x):
        return F.normalize(x, p=2, dim=1) * self.scale.unsqueeze(0).unsqueeze(2
            ).unsqueeze(3)

    def reset_parameters(self):
        self.scale.data.fill_(self.initial_scale)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'initial_scale': 1.0}]
