import torch
import torch.nn as nn
import torch.nn.functional as F


class CAModel(nn.Module):

    def __init__(self, env_d):
        super(CAModel, self).__init__()
        self.conv1 = nn.Conv2d(env_d * 3, 144, 1)
        self.conv2 = nn.Conv2d(144, env_d, 1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)


def get_inputs():
    return [torch.rand([4, 12, 64, 64])]


def get_init_inputs():
    return [[], {'env_d': 4}]
