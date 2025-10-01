import torch
from torch.nn import functional as F
import torch.nn as nn
from math import *


class HardAttn(nn.Module):
    """Hard Attention (Sec. 3.1.II)"""

    def __init__(self, in_channels):
        super(HardAttn, self).__init__()
        self.fc = nn.Linear(in_channels, 4 * 2)
        self.init_params()

    def init_params(self):
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([0, -0.75, 0, -0.25, 0, 0.25, 
            0, 0.75], dtype=torch.float))

    def forward(self, x):
        x = F.avg_pool2d(x, x.size()[2:]).view(x.size(0), x.size(1))
        theta = torch.tanh(self.fc(x))
        theta = theta.view(-1, 4, 2)
        return theta


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
