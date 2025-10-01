import torch
import torch.nn as nn


class CNNHead(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=2,
            kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
