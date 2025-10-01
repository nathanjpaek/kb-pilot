import torch
import torch.nn as nn


class BiasConvFc2Net(nn.Module):

    def __init__(self, in_channels, groups, n_segment, kernel_size=3, padding=1
        ):
        super(BiasConvFc2Net, self).__init__()
        self.conv = nn.Conv1d(in_channels, 1, kernel_size, padding=padding)
        self.fc = nn.Linear(n_segment, n_segment)
        self.relu = nn.ReLU()
        self.lastlayer = nn.Linear(n_segment, groups)

    def forward(self, x):
        N, _C, T = x.shape
        x = self.conv(x)
        x = x.view(N, T)
        x = self.relu(self.fc(x))
        x = self.lastlayer(x)
        x = x.view(N, 1, -1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'groups': 1, 'n_segment': 4}]
