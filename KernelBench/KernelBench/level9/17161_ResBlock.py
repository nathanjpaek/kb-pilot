import torch
from torch import nn
import torch.distributed


class ResBlock(nn.Module):

    def __init__(self, feature_size, action_size):
        super(ResBlock, self).__init__()
        self.lin_1 = nn.Linear(feature_size + action_size, feature_size)
        self.lin_2 = nn.Linear(feature_size + action_size, feature_size)

    def forward(self, x, action):
        res = nn.functional.leaky_relu(self.lin_1(torch.cat([x, action], 2)))
        res = self.lin_2(torch.cat([res, action], 2))
        return res + x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'action_size': 4}]
