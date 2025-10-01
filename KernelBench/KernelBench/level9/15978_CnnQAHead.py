import torch
from torch import nn


class CnnQAHead(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels=input_size, out_channels=2,
            kernel_size=1, padding=0)
        self.conv_3 = nn.Conv1d(in_channels=input_size, out_channels=2,
            kernel_size=3, padding=1)
        self.conv_5 = nn.Conv1d(in_channels=input_size, out_channels=2,
            kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()
        conv1_out = self.relu(self.conv_1(x).transpose(1, 2).contiguous().
            squeeze(-1))
        conv3_out = self.relu(self.conv_3(x).transpose(1, 2).contiguous().
            squeeze(-1))
        conv5_out = self.relu(self.conv_5(x).transpose(1, 2).contiguous().
            squeeze(-1))
        x = conv1_out + conv3_out + conv5_out
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
