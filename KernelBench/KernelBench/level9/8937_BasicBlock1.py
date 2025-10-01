import torch
import torch.nn as nn


class BasicBlock1(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(BasicBlock1, self).__init__()
        self.ID = input_dim
        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=
            output_dim, kernel_size=1, padding=0, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        return torch.cat((x, out), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
