import torch
import torch.nn as nn
import torch.nn.functional as F


class NNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(NNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 256)
        self.linear3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
